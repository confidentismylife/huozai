from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import gymnasium as gym
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.mappo import (
    FLOW_LEVELS,
    ROUTE_WAIT,
    FactorizedMAPPO,
    MAPPOConfig,
    encode_factorized_action,
    clipped_agent_surrogate_loss,
    plan_supervision_targets,
)


def test_factorized_actions_cover_original_nine_actions() -> None:
    non_wait = {
        encode_factorized_action(route, flow)
        for route in (0, 1)
        for flow in range(len(FLOW_LEVELS))
    }
    waits = {encode_factorized_action(ROUTE_WAIT, flow) for flow in range(len(FLOW_LEVELS))}
    assert non_wait == set(range(8))
    assert waits == {8}


def test_wait_masks_flow_actor_gradient() -> None:
    algorithm = FactorizedMAPPO(observation_dim=34, config=MAPPOConfig(hidden_size=16))
    observations = torch.zeros((6, 34), dtype=torch.float32)
    routes = torch.full((6,), ROUTE_WAIT, dtype=torch.long)
    flows = torch.arange(6, dtype=torch.long) % len(FLOW_LEVELS)

    log_prob, entropy, _ = algorithm.network.evaluate_actions(observations, routes, flows)
    loss = -(log_prob.mean() + 0.01 * entropy.mean())
    loss.backward()

    flow_gradients = [parameter.grad for parameter in algorithm.network.flow_actor.parameters()]
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in flow_gradients)
    route_gradients = [parameter.grad for parameter in algorithm.network.route_actor.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in route_gradients)


def test_central_critic_reads_complete_state() -> None:
    algorithm = FactorizedMAPPO(observation_dim=34, config=MAPPOConfig(hidden_size=16))
    first_layer = next(layer for layer in algorithm.network.critic if isinstance(layer, torch.nn.Linear))
    assert first_layer.in_features == 34
    observations = np.zeros((3, 34), dtype=np.float32)
    routes, flows, log_prob, values = algorithm.act(observations, deterministic=True)
    assert routes.shape == flows.shape == log_prob.shape == values.shape == (3,)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    source = FactorizedMAPPO(observation_dim=34, config=MAPPOConfig(hidden_size=16), seed=7)
    checkpoint = tmp_path / "mappo.pt"
    source.save(checkpoint, metadata={"steps": 12})
    restored, metadata = FactorizedMAPPO.load(checkpoint)

    observations = np.ones((2, 34), dtype=np.float32)
    expected = source.act(observations, deterministic=True)[:2]
    actual = restored.act(observations, deterministic=True)[:2]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert metadata == {"steps": 12}


class TinyNineActionEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(34,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)
        self.step_index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(34, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        observation = np.full(34, min(1.0, self.step_index / 8), dtype=np.float32)
        reward = 1.0 if int(action) in (3, 7) else -0.1
        return observation, reward, self.step_index >= 8, False, {}


def test_short_training_rollout_updates_factorized_policy() -> None:
    algorithm = FactorizedMAPPO(
        observation_dim=34,
        config=MAPPOConfig(hidden_size=16, update_epochs=1, batch_size=8),
        seed=11,
    )
    before = [parameter.detach().clone() for parameter in algorithm.network.parameters()]
    metrics = algorithm.train(TinyNineActionEnv(), total_timesteps=32, rollout_steps=16)
    after = list(algorithm.network.parameters())

    assert metrics["timesteps"] == 32
    assert np.isfinite(metrics["policy_loss"])
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_training_reports_actor_diagnostics_and_calls_rollout_callback() -> None:
    algorithm = FactorizedMAPPO(
        observation_dim=34,
        config=MAPPOConfig(
            hidden_size=16,
            update_epochs=1,
            batch_size=8,
            route_entropy_coefficient=0.01,
            flow_entropy_coefficient=0.03,
        ),
        seed=13,
    )
    callback_steps: list[int] = []

    metrics = algorithm.train(
        TinyNineActionEnv(),
        total_timesteps=32,
        rollout_steps=16,
        rollout_callback=lambda _model, step, _record: callback_steps.append(step),
    )

    assert callback_steps == [16, 32]
    assert len(metrics["rollout_history"]) == 2
    assert len(metrics["route_action_fractions"]) == 3
    assert len(metrics["flow_action_fractions_nonwait"]) == 4
    assert sum(metrics["route_action_fractions"]) == pytest.approx(1.0)
    assert sum(metrics["flow_action_fractions_nonwait"]) == pytest.approx(1.0)
    assert np.isfinite(metrics["route_entropy"])
    assert np.isfinite(metrics["flow_entropy"])


def test_agent_surrogate_masks_inactive_flow_without_affecting_route() -> None:
    advantages = torch.tensor([1.0, -0.5])
    old = torch.zeros(2)
    route_loss = clipped_agent_surrogate_loss(torch.tensor([0.1, -0.1]), old, advantages, 0.2)
    flow_loss = clipped_agent_surrogate_loss(
        torch.tensor([5.0, -5.0]), old, advantages, 0.2, mask=torch.zeros(2)
    )
    assert torch.isfinite(route_loss)
    assert flow_loss.item() == 0.0


def test_plan_supervision_tracks_active_phase_and_masks_contradiction() -> None:
    observations = torch.zeros((3, 50), dtype=torch.float32)
    # Three phases: south@0.50 for 2 steps, north@0.75 for 3, wait thereafter.
    observations[:, 29] = 1.0
    observations[:, 31] = 0.50
    observations[:, 32] = 2 / 12
    observations[:, 33] = 1.0
    observations[:, 36] = 0.75
    observations[:, 37] = 3 / 12
    observations[:, 40] = 1.0
    observations[:, 41] = 0.25
    observations[:, 42] = 1 / 12
    observations[:, 43] = 0.8
    observations[:, 47] = torch.tensor([0.0, 3 / 12, 6 / 12])
    observations[2, 48] = 1.0

    routes, flows, route_weights, flow_weights = plan_supervision_targets(observations)

    assert routes.tolist() == [1, 0, ROUTE_WAIT]
    assert flows.tolist() == [1, 2, 0]
    assert route_weights.tolist() == pytest.approx([0.8, 0.8, 0.0])
    assert flow_weights.tolist() == pytest.approx([0.8, 0.8, 0.0])
