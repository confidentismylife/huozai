from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .confirmatory import JOINT_RELEASE_LEVELS, decode_absolute_action, encode_absolute_action
from .dynamic_llm_rl import NORTH, SOUTH


ACTION_MIRROR = np.asarray([4, 5, 6, 7, 0, 1, 2, 3, 8])


def physical_shield_observation(observation: np.ndarray) -> np.ndarray:
    """Use physical/raw forecast inputs only, independent of the advice arm."""
    raw = np.asarray(observation, dtype=np.float32).copy()
    if raw.shape != (34,):
        raise ValueError("shield requires a 34-dimensional observation")
    raw[28:33] = 0.0
    return raw


def effective_action_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Runtime mask, including the empty-set fallback used in validation.

    Preserve all tied maxima to avoid introducing a north/south bias into the
    mask. Selection of one executable action can still break exact ties.
    These classifier scores are not calibrated probabilities or risk bounds.
    """
    values = np.asarray(scores)
    if values.shape[-1] != 9 or not np.all(np.isfinite(values)):
        raise ValueError("nine finite action scores are required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    mask = values >= threshold
    empty = ~np.any(mask, axis=-1, keepdims=True)
    return mask | (empty & (values == np.max(values, axis=-1, keepdims=True)))


def action_mask_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels and scores must have matching state-by-action shapes")
    mask = effective_action_mask(scores, threshold)
    def rejection_rate(selector):
        return float(np.mean(~mask[selector])) if np.any(selector) else None
    top = labels[np.arange(len(labels)), np.argmax(scores, axis=1)]
    return {
        "states": len(labels),
        "unsafe_actions": int(np.sum(labels == 0)),
        "safe_actions": int(np.sum(labels == 1)),
        "indeterminate_actions": int(np.sum(labels < 0)),
        "unsafe_action_recall": rejection_rate(labels == 0),
        "safe_action_false_rejection_rate": rejection_rate(labels == 1),
        "indeterminate_action_rejection_rate": rejection_rate(labels < 0),
        "top_action_unsafe_rate": float(np.mean(top == 0)),
        "empty_raw_mask_rate": float(np.mean(~np.any(scores >= threshold, axis=1))),
        "mean_allowed_actions": float(mask.sum(axis=1).mean()),
    }


def mirror_shield_observation(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    for left, right in ((3, 4), (8, 9), (10, 11), (14, 15), (16, 17), (19, 20), (23, 24), (25, 26), (28, 29)):
        y[left], y[right] = x[right], x[left]
    return y


def mirror_action_values(values: np.ndarray) -> np.ndarray:
    mirrored = np.empty_like(values)
    mirrored[ACTION_MIRROR] = values
    return mirrored


def reactive_followup_action(observation: np.ndarray) -> int:
    """Frozen low-level fallback used after the counterfactual first action."""
    x = np.asarray(observation, dtype=float)
    hazard = x[8:10]
    occupancy = x[3:5]
    capacity = x[14:16]
    cost = 6.0 * hazard + 1.5 * occupancy + 2.0 * (capacity <= 1e-6)
    branch = int(np.argmin(cost))
    if capacity[branch] <= 1e-6 and x[18] > 0:
        return 8
    if capacity[branch] < 0.45:
        release = 0.25
    elif occupancy[branch] > 0.75:
        release = 0.50
    elif occupancy[branch] > 0.45:
        release = 0.75
    else:
        release = 1.00
    return encode_absolute_action(branch, release)


@dataclass(frozen=True)
class CounterfactualActionResult:
    action: int
    incapacitation_delta: float
    fed_delta: float
    congestion_per_step: float
    evacuation_delta: float
    total_reward: float
    rollout_steps: int = 0
    terminal_reached: bool = False
    first_actual_joint_action: int = -1
    first_action_blocked_by_hold: bool = False


def evaluate_counterfactual_actions(
    env: gym.Env,
    horizon: int = 20,
    continuation: Callable[[np.ndarray], int] = reactive_followup_action,
) -> tuple[CounterfactualActionResult, ...]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    start_info = env.unwrapped.domain_env._info()
    results = []
    for candidate in range(9):
        clone = copy.deepcopy(env)
        observation, reward, terminated, truncated, info = clone.step(candidate)
        first_actual = int(info["actual_joint_action"])
        first_hold = bool(info["action_blocked_by_sign_hold"])
        total_reward = float(reward)
        steps = 1
        while not (terminated or truncated) and steps < horizon:
            action = continuation(observation)
            observation, reward, terminated, truncated, info = clone.step(action)
            total_reward += float(reward)
            steps += 1
        results.append(
            CounterfactualActionResult(
                action=candidate,
                incapacitation_delta=float(info["incapacitation_rate"] - start_info["incapacitation_rate"]),
                fed_delta=float(info["mean_cumulative_fed"] - start_info["mean_cumulative_fed"]),
                congestion_per_step=float(
                    (info["cumulative_congestion"] - start_info["cumulative_congestion"]) / steps
                ),
                evacuation_delta=float(info["evacuation_rate"] - start_info["evacuation_rate"]),
                total_reward=total_reward,
                rollout_steps=steps,
                terminal_reached=bool(terminated or truncated),
                first_actual_joint_action=first_actual,
                first_action_blocked_by_hold=first_hold,
            )
        )
        clone.close()
    return tuple(results)


def admissible_action_mask(
    results: tuple[CounterfactualActionResult, ...],
    minimum_actions: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if len(results) != 9:
        raise ValueError("nine counterfactual action results are required")
    if not 1 <= minimum_actions <= 9:
        raise ValueError("minimum_actions must be in [1, 9]")
    # Incapacitation dominates, followed by FED, congestion, evacuation, and reward.
    ordered = sorted(
        range(9),
        key=lambda i: (
            results[i].incapacitation_delta,
            results[i].fed_delta,
            results[i].congestion_per_step,
            -results[i].evacuation_delta,
            -results[i].total_reward,
            i,
        ),
    )
    mask = np.zeros(9, dtype=np.float32)
    mask[ordered[:minimum_actions]] = 1.0
    rank = np.empty(9, dtype=np.float32)
    for position, action in enumerate(ordered):
        rank[action] = position / 8.0
    return mask, rank


def material_safety_labels(
    results: tuple[CounterfactualActionResult, ...],
    population: int = 240,
) -> np.ndarray:
    """Return 1=safe, 0=materially unsafe, -1=indeterminate.

    Indeterminate near-ties are intentionally excluded from classifier loss and
    validation's known-label denominators. The binary classifier cannot infer
    this third class: runtime may also exclude indeterminate actions. Report
    that rate explicitly; do not claim an uncertainty passthrough guarantee.
    """
    if len(results) != 9:
        raise ValueError("nine counterfactual action results are required")
    if population < 1:
        raise ValueError("population must be positive")
    incap = np.asarray([item.incapacitation_delta for item in results])
    fed = np.asarray([item.fed_delta for item in results])
    congestion = np.asarray([item.congestion_per_step for item in results])
    min_incap = float(incap.min())
    low_incap = incap <= min_incap + 1.0 / population
    min_fed = float(fed[low_incap].min())
    min_congestion = float(congestion[low_incap & (fed <= min_fed + 0.01)].min())
    safe = (
        (incap <= min_incap + 1.0 / population)
        & (fed <= min_fed + 0.01)
        & (congestion <= min_congestion + 0.15)
    )
    unsafe = (
        (incap >= min_incap + 0.01)
        | ((incap > min_incap + 1.0 / population) & (fed >= min_fed + 0.03))
        | ((incap <= min_incap + 1.0 / population) & (fed >= min_fed + 0.06))
    )
    labels = np.full(9, -1.0, dtype=np.float32)
    labels[unsafe] = 0.0
    labels[safe] = 1.0
    # The lexicographic winner must always remain admissible.
    winner = min(
        range(9),
        key=lambda i: (incap[i], fed[i], congestion[i], -results[i].evacuation_delta),
    )
    labels[winner] = 1.0
    return labels


class ShieldNetwork(nn.Module):
    def __init__(self, observation_dim: int = 34):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 128), nn.ReLU(),
            nn.Linear(128, 96), nn.ReLU(),
        )
        action_features = []
        for action in range(9):
            branch, release = decode_absolute_action(action)
            action_features.append(
                [float(branch == NORTH), float(branch == SOUTH), float(branch is None), release]
            )
        self.register_buffer("action_features", torch.as_tensor(action_features, dtype=torch.float32))
        self.action_head = nn.Sequential(
            nn.Linear(100, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.encoder(x)
        batch = state.shape[0]
        expanded_state = state[:, None, :].expand(batch, 9, state.shape[-1])
        expanded_action = self.action_features[None, :, :].expand(batch, 9, 4)
        return self.action_head(torch.cat([expanded_state, expanded_action], dim=-1)).squeeze(-1)


@dataclass
class ShieldPredictor:
    model: ShieldNetwork | tuple[ShieldNetwork, ...]
    mean: np.ndarray
    scale: np.ndarray
    threshold: float

    def _models(self) -> tuple[ShieldNetwork, ...]:
        return self.model if isinstance(self.model, tuple) else (self.model,)

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        raw = physical_shield_observation(observation)
        x = (raw - self.mean) / self.scale
        mirror_x = (mirror_shield_observation(raw) - self.mean) / self.scale
        predictions = []
        with torch.no_grad():
            for model in self._models():
                logits = model(torch.as_tensor(x[None, :], dtype=torch.float32))
                mirror_logits = model(torch.as_tensor(mirror_x[None, :], dtype=torch.float32))
                direct = torch.sigmoid(logits).cpu().numpy()[0]
                mirrored_back = mirror_action_values(torch.sigmoid(mirror_logits).cpu().numpy()[0])
                predictions.append(0.5 * (direct + mirrored_back))
        return np.mean(predictions, axis=0)

    def mask(self, observation: np.ndarray) -> np.ndarray:
        return effective_action_mask(self.probabilities(observation), self.threshold)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dicts": [model.state_dict() for model in self._models()],
                "format_version": 2,
                "input_contract": "physical_raw_forecast_34_zero_advice_28_33",
                "mean": torch.as_tensor(self.mean, dtype=torch.float32),
                "scale": torch.as_tensor(self.scale, dtype=torch.float32),
                "threshold": self.threshold,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, *, trusted_legacy: bool = False) -> "ShieldPredictor":
        # Legacy NumPy payloads require explicit opt-in for known local artifacts.
        # New files deserialize with the restricted weights-only loader.
        payload = torch.load(Path(path), map_location="cpu", weights_only=not trusted_legacy)
        if payload.get("format_version", 1) not in (1, 2):
            raise ValueError("unsupported shield checkpoint version")
        state_dicts = payload.get("state_dicts")
        if state_dicts is None:
            state_dicts = [payload["state_dict"]]
        models = []
        for state_dict in state_dicts:
            model = ShieldNetwork(len(payload["mean"]))
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        return cls(
            model=tuple(models),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            threshold=float(payload["threshold"]),
        )


class ConditionalSafetyShieldWrapper(gym.Wrapper):
    """Append a learned safe-action mask and optionally enforce it."""

    def __init__(self, env: gym.Env, predictor: ShieldPredictor, enforce: bool, expose: bool = True):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ConditionalSafetyShieldWrapper requires Box observations")
        self.predictor = predictor
        self.enforce = bool(enforce)
        self.expose = bool(expose)
        self.current_probabilities = np.zeros(9, dtype=np.float32)
        self.current_mask = np.ones(9, dtype=bool)
        low = np.concatenate([env.observation_space.low, np.zeros(18, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(18, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        self.current_probabilities = self.predictor.probabilities(observation)
        self.current_mask = effective_action_mask(self.current_probabilities, self.predictor.threshold)
        if self.expose:
            features = np.concatenate([self.current_probabilities, self.current_mask.astype(np.float32)])
        else:
            features = np.zeros(18, dtype=np.float32)
        return np.concatenate([observation, features]).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation), info

    def step(self, action: int):
        requested = int(action)
        applied = requested
        intervened = False
        if self.enforce and not self.current_mask[requested]:
            applied = int(np.argmax(np.where(self.current_mask, self.current_probabilities, -1.0)))
            intervened = True
        observation, reward, terminated, truncated, info = self.env.step(applied)
        enriched = dict(info)
        enriched.update(
            {
                "shield_requested_action": requested,
                "shield_applied_action": applied,
                "shield_intervened": intervened,
                "shield_safe_action_count": int(self.current_mask.sum()),
                "shield_allowed_action_count": int(self.current_mask.sum()),
                "shield_empty_raw_mask": bool(not np.any(self.current_probabilities >= self.predictor.threshold)),
                "shield_actual_joint_action": int(info["actual_joint_action"]),
                "shield_applied_actual_match": bool(applied == info["actual_joint_action"]),
            }
        )
        return self._augment(observation), reward, terminated, truncated, enriched
