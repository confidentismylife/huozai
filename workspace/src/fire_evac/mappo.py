from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


ROUTE_NORTH = 0
ROUTE_SOUTH = 1
ROUTE_WAIT = 2
FLOW_LEVELS = (0.25, 0.50, 0.75, 1.00)


def clipped_agent_surrogate_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one actor's clipped PPO loss, optionally over active steps only."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    surrogate = torch.minimum(unclipped, clipped)
    if mask is None:
        return -surrogate.mean()
    active = mask.to(dtype=surrogate.dtype, device=surrogate.device)
    return -(surrogate * active).sum() / active.sum().clamp_min(1.0)


def plan_supervision_targets(
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the active three-phase plan from the 50-feature observation."""
    if observations.ndim != 2 or observations.shape[1] < 50:
        raise ValueError("plan supervision requires 50-feature observations")
    durations = 12.0 * torch.stack(
        (observations[:, 32], observations[:, 37], observations[:, 42]), dim=1
    )
    age_steps = 12.0 * observations[:, 47]
    phase_index = (age_steps >= durations[:, 0]).long()
    phase_index += (age_steps >= durations[:, 0] + durations[:, 1]).long()
    phase_index = phase_index.clamp_max(2)
    batch_index = torch.arange(len(observations), device=observations.device)
    route_scores = torch.stack(
        (observations[:, 28:31], observations[:, 33:36], observations[:, 38:41]),
        dim=1,
    )
    releases = torch.stack(
        (observations[:, 31], observations[:, 36], observations[:, 41]), dim=1
    )
    route_targets = route_scores[batch_index, phase_index].argmax(dim=1)
    active_release = releases[batch_index, phase_index]
    levels = torch.as_tensor(FLOW_LEVELS, dtype=observations.dtype, device=observations.device)
    flow_targets = torch.abs(active_release[:, None] - levels[None, :]).argmin(dim=1)
    confidence = observations[:, 43].clamp(0.0, 1.0)
    contradiction = observations[:, 48].clamp(0.0, 1.0)
    route_weights = confidence * (1.0 - contradiction)
    flow_weights = route_weights * (route_targets != ROUTE_WAIT).to(observations.dtype)
    return route_targets, flow_targets, route_weights, flow_weights


def encode_factorized_action(route_action: int, flow_action: int) -> int:
    """Map the two heterogeneous agents to the original nine-action protocol."""
    route = int(route_action)
    flow = int(flow_action)
    if route not in (ROUTE_NORTH, ROUTE_SOUTH, ROUTE_WAIT):
        raise ValueError("route_action must be north, south, or wait")
    if flow not in range(len(FLOW_LEVELS)):
        raise ValueError("flow_action must select one of four release levels")
    return 8 if route == ROUTE_WAIT else route * len(FLOW_LEVELS) + flow


class FactorizedActionAdapter:
    """Present a route/flow pair while preserving the original environment."""

    def __init__(self, env: Any):
        import gymnasium as gym

        if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 9:
            raise ValueError("factorized MAPPO requires the original Discrete(9) environment")
        self.env = env
        self.action_space = gym.spaces.MultiDiscrete([3, 4])
        self.observation_space = env.observation_space

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        route, flow = np.asarray(action, dtype=int).tolist()
        return self.env.step(encode_factorized_action(route, flow))

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped


@dataclass(frozen=True)
class MAPPOConfig:
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    route_entropy_coefficient: float | None = None
    flow_entropy_coefficient: float | None = None
    plan_imitation_coefficient: float = 0.0
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 128


def _mlp(input_dim: int, output_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_dim),
    )


class FactorizedActorCentralCritic(nn.Module):
    """Two policy heads with one centralized team-value network."""

    def __init__(self, observation_dim: int, hidden_size: int = 128):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.route_actor = _mlp(self.observation_dim, 3, hidden_size)
        self.flow_actor = _mlp(self.observation_dim, len(FLOW_LEVELS), hidden_size)
        self.critic = _mlp(self.observation_dim, 1, hidden_size)

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        return Categorical(logits=self.route_actor(observations)), Categorical(logits=self.flow_actor(observations))

    def sample(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        routes, flows, route_log_prob, flow_log_prob, values = self.sample_components(
            observations, deterministic=deterministic
        )
        active_flow = (routes != ROUTE_WAIT).to(observations.dtype)
        return routes, flows, route_log_prob + active_flow * flow_log_prob, values

    def sample_components(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        route_distribution, flow_distribution = self.distributions(observations)
        if deterministic:
            routes = route_distribution.logits.argmax(dim=-1)
            flows = flow_distribution.logits.argmax(dim=-1)
        else:
            routes = route_distribution.sample()
            flows = flow_distribution.sample()
        values = self.critic(observations).squeeze(-1)
        return (
            routes,
            flows,
            route_distribution.log_prob(routes),
            flow_distribution.log_prob(flows),
            values,
        )

    def evaluate_actions(
        self, observations: torch.Tensor, routes: torch.Tensor, flows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_log_prob, flow_log_prob, route_entropy, flow_entropy, values = (
            self.evaluate_action_components(observations, routes, flows)
        )
        active_flow = (routes != ROUTE_WAIT).to(observations.dtype)
        return (
            route_log_prob + active_flow * flow_log_prob,
            route_entropy + active_flow * flow_entropy,
            values,
        )

    def evaluate_action_components(
        self, observations: torch.Tensor, routes: torch.Tensor, flows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        route_distribution, flow_distribution = self.distributions(observations)
        values = self.critic(observations).squeeze(-1)
        return (
            route_distribution.log_prob(routes),
            flow_distribution.log_prob(flows),
            route_distribution.entropy(),
            flow_distribution.entropy(),
            values,
        )


class FactorizedMAPPO:
    """Lightweight MAPPO for the route and release agents.

    Both actors receive the same controller observation so comparisons retain
    identical information.  The separate policies and shared team critic make
    the factorization explicit without pretending the agents are evacuees.
    """

    def __init__(
        self,
        observation_dim: int,
        config: MAPPOConfig | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.observation_dim = int(observation_dim)
        self.config = config or MAPPOConfig()
        self.seed = int(seed)
        self.device = torch.device(device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.network = FactorizedActorCentralCritic(
            self.observation_dim, self.config.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def act(self, observations: np.ndarray, deterministic: bool = False):
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[1] != self.observation_dim:
            raise ValueError(f"expected observation width {self.observation_dim}, got {array.shape[1]}")
        with torch.no_grad():
            routes, flows, log_prob, values = self.network.sample(
                torch.as_tensor(array, device=self.device), deterministic=deterministic
            )
        return tuple(item.cpu().numpy() for item in (routes, flows, log_prob, values))

    def _act_components(self, observations: np.ndarray, deterministic: bool = False):
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[1] != self.observation_dim:
            raise ValueError(f"expected observation width {self.observation_dim}, got {array.shape[1]}")
        with torch.no_grad():
            components = self.network.sample_components(
                torch.as_tensor(array, device=self.device), deterministic=deterministic
            )
        return tuple(item.cpu().numpy() for item in components)

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        routes = torch.as_tensor(batch["routes"], dtype=torch.long, device=self.device)
        flows = torch.as_tensor(batch["flows"], dtype=torch.long, device=self.device)
        old_route_log_prob = torch.as_tensor(
            batch["old_route_log_prob"], dtype=torch.float32, device=self.device
        )
        old_flow_log_prob = torch.as_tensor(
            batch["old_flow_log_prob"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        losses: list[tuple[float, float, float, float, float, float, float]] = []
        rng = np.random.default_rng(self.seed + int(len(batch["observations"])))
        for _ in range(self.config.update_epochs):
            order = rng.permutation(len(observations))
            for start in range(0, len(order), self.config.batch_size):
                indices = torch.as_tensor(order[start : start + self.config.batch_size], device=self.device)
                (
                    new_route_log_prob,
                    new_flow_log_prob,
                    route_entropy,
                    flow_entropy,
                    values,
                ) = self.network.evaluate_action_components(
                    observations[indices], routes[indices], flows[indices]
                )
                active_flow = (routes[indices] != ROUTE_WAIT).to(observations.dtype)
                route_loss = clipped_agent_surrogate_loss(
                    new_route_log_prob,
                    old_route_log_prob[indices],
                    advantages[indices],
                    self.config.clip_range,
                )
                flow_loss = clipped_agent_surrogate_loss(
                    new_flow_log_prob,
                    old_flow_log_prob[indices],
                    advantages[indices],
                    self.config.clip_range,
                    mask=active_flow,
                )
                policy_loss = route_loss + flow_loss
                value_loss = (returns[indices] - values).pow(2).mean()
                flow_entropy_mean = (flow_entropy * active_flow).sum() / active_flow.sum().clamp_min(1.0)
                route_entropy_mean = route_entropy.mean()
                route_entropy_coefficient = (
                    self.config.entropy_coefficient
                    if self.config.route_entropy_coefficient is None
                    else self.config.route_entropy_coefficient
                )
                flow_entropy_coefficient = (
                    self.config.entropy_coefficient
                    if self.config.flow_entropy_coefficient is None
                    else self.config.flow_entropy_coefficient
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - route_entropy_coefficient * route_entropy_mean
                    - flow_entropy_coefficient * flow_entropy_mean
                )
                plan_imitation_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
                if self.config.plan_imitation_coefficient > 0.0 and observations.shape[1] >= 50:
                    route_targets, flow_targets, route_weights, flow_weights = plan_supervision_targets(
                        observations[indices]
                    )
                    route_cross_entropy = F.cross_entropy(
                        self.network.route_actor(observations[indices]),
                        route_targets,
                        reduction="none",
                    )
                    flow_cross_entropy = F.cross_entropy(
                        self.network.flow_actor(observations[indices]),
                        flow_targets,
                        reduction="none",
                    )
                    route_imitation = (route_cross_entropy * route_weights).sum() / route_weights.sum().clamp_min(1.0)
                    flow_imitation = (flow_cross_entropy * flow_weights).sum() / flow_weights.sum().clamp_min(1.0)
                    plan_imitation_loss = route_imitation + flow_imitation
                    loss = loss + self.config.plan_imitation_coefficient * plan_imitation_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(
                    (
                        float(policy_loss.item()),
                        float(route_loss.item()),
                        float(flow_loss.item()),
                        float(value_loss.item()),
                        float(route_entropy_mean.item()),
                        float(flow_entropy_mean.item()),
                        float(plan_imitation_loss.item()),
                    )
                )
        mean = np.asarray(losses).mean(axis=0)
        return {
            "policy_loss": float(mean[0]),
            "route_policy_loss": float(mean[1]),
            "flow_policy_loss": float(mean[2]),
            "value_loss": float(mean[3]),
            "route_entropy": float(mean[4]),
            "flow_entropy": float(mean[5]),
            "entropy": float(mean[4] + mean[5]),
            "plan_imitation_loss": float(mean[6]),
        }

    def train(
        self,
        env: Any,
        total_timesteps: int,
        rollout_steps: int = 256,
        rollout_callback: Callable[["FactorizedMAPPO", int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if tuple(env.observation_space.shape) != (self.observation_dim,):
            raise ValueError("environment observation space does not match MAPPO")
        observation, _ = env.reset(seed=self.seed)
        completed = 0
        episodes = 0
        last_losses = {
            "policy_loss": np.nan,
            "route_policy_loss": np.nan,
            "flow_policy_loss": np.nan,
            "value_loss": np.nan,
            "route_entropy": np.nan,
            "flow_entropy": np.nan,
            "entropy": np.nan,
            "plan_imitation_loss": np.nan,
        }
        reward_total = 0.0
        route_counts = np.zeros(3, dtype=np.int64)
        flow_counts = np.zeros(len(FLOW_LEVELS), dtype=np.int64)
        rollout_history: list[dict[str, Any]] = []
        while completed < total_timesteps:
            size = min(int(rollout_steps), total_timesteps - completed)
            storage: dict[str, list] = {
                key: [] for key in (
                    "observations", "routes", "flows", "old_route_log_prob",
                    "old_flow_log_prob", "values", "rewards",
                    "next_values", "terminated", "episode_end",
                )
            }
            for _ in range(size):
                routes, flows, route_log_prob, flow_log_prob, values = self._act_components(observation)
                route, flow = int(routes[0]), int(flows[0])
                next_observation, reward, terminated, truncated, _ = env.step(
                    encode_factorized_action(route, flow)
                )
                with torch.no_grad():
                    next_value = float(
                        self.network.critic(
                            torch.as_tensor(next_observation[None, :], dtype=torch.float32, device=self.device)
                        ).item()
                    )
                for key, value in (
                    ("observations", np.asarray(observation, dtype=np.float32)),
                    ("routes", route), ("flows", flow),
                    ("old_route_log_prob", float(route_log_prob[0])),
                    ("old_flow_log_prob", float(flow_log_prob[0])),
                    ("values", float(values[0])), ("rewards", float(reward)),
                    ("next_values", next_value), ("terminated", bool(terminated)),
                    ("episode_end", bool(terminated or truncated)),
                ):
                    storage[key].append(value)
                reward_total += float(reward)
                observation = next_observation
                completed += 1
                if terminated or truncated:
                    observation, _ = env.reset()
                    episodes += 1
            rewards = np.asarray(storage["rewards"], dtype=np.float32)
            values = np.asarray(storage["values"], dtype=np.float32)
            next_values = np.asarray(storage["next_values"], dtype=np.float32)
            terminated = np.asarray(storage["terminated"], dtype=np.float32)
            episode_end = np.asarray(storage["episode_end"], dtype=np.float32)
            advantages = np.zeros(size, dtype=np.float32)
            running = 0.0
            for index in range(size - 1, -1, -1):
                delta = rewards[index] + self.config.gamma * next_values[index] * (1.0 - terminated[index]) - values[index]
                running = delta + self.config.gamma * self.config.gae_lambda * (1.0 - episode_end[index]) * running
                advantages[index] = running
            batch = {
                "observations": np.asarray(storage["observations"], dtype=np.float32),
                "routes": np.asarray(storage["routes"], dtype=np.int64),
                "flows": np.asarray(storage["flows"], dtype=np.int64),
                "old_route_log_prob": np.asarray(storage["old_route_log_prob"], dtype=np.float32),
                "old_flow_log_prob": np.asarray(storage["old_flow_log_prob"], dtype=np.float32),
                "returns": advantages + values,
                "advantages": advantages,
            }
            last_losses = self._update(batch)
            rollout_routes = np.asarray(storage["routes"], dtype=np.int64)
            rollout_flows = np.asarray(storage["flows"], dtype=np.int64)
            active_flow = rollout_routes != ROUTE_WAIT
            route_counts += np.bincount(rollout_routes, minlength=3)
            if active_flow.any():
                flow_counts += np.bincount(rollout_flows[active_flow], minlength=len(FLOW_LEVELS))
            record: dict[str, Any] = {
                "timesteps": completed,
                "route_action_fractions": (
                    np.bincount(rollout_routes, minlength=3) / max(1, len(rollout_routes))
                ).tolist(),
                "flow_action_fractions_nonwait": (
                    np.bincount(rollout_flows[active_flow], minlength=len(FLOW_LEVELS))
                    / max(1, int(active_flow.sum()))
                ).tolist(),
                **last_losses,
            }
            rollout_history.append(record)
            if rollout_callback is not None:
                rollout_callback(self, completed, record)
        return {
            "timesteps": float(completed),
            "episodes": float(episodes),
            "mean_step_reward": reward_total / max(1, completed),
            "route_action_fractions": (route_counts / max(1, int(route_counts.sum()))).tolist(),
            "flow_action_fractions_nonwait": (
                flow_counts / max(1, int(flow_counts.sum()))
            ).tolist(),
            "rollout_history": rollout_history,
            **last_losses,
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_dim": self.observation_dim,
                "config": asdict(self.config),
                "seed": self.seed,
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": dict(metadata or {}),
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple["FactorizedMAPPO", dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        algorithm = cls(
            observation_dim=int(payload["observation_dim"]),
            config=MAPPOConfig(**payload["config"]),
            seed=int(payload["seed"]),
            device=device,
        )
        algorithm.network.load_state_dict(payload["network"])
        algorithm.optimizer.load_state_dict(payload["optimizer"])
        return algorithm, dict(payload.get("metadata", {}))
