"""Frozen PPO continuations for an offline, privileged development teacher."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib

import gymnasium as gym
import numpy as np

from .source_world_planning import DynamicReferencePlan, PlanOutcome, plan_action


FROZEN_PPO_SEEDS = tuple(range(601, 606))
TRACE_FIELDS = ("elapsed", "requested_joint_action", "actual_joint_action",
                "actual_sign_action", "actual_release_fraction", "action_blocked_by_sign_hold")


def pure_actor_observation(observation):
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (49,) or not np.all(np.isfinite(raw)):
        raise ValueError("finite 49-dimensional history observation required")
    result = raw[:34].copy()
    result[28:33] = 0
    return result


def policy_parameter_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class FrozenPPOContinuation:
    training_seed: int
    model: object = field(repr=False)

    def __post_init__(self):
        if self.model.observation_space.shape != (34,):
            raise ValueError("frozen pure PPO requires 34 inputs")
        if not isinstance(self.model.action_space, gym.spaces.Discrete) or self.model.action_space.n != 9:
            raise ValueError("frozen pure PPO requires nine discrete actions")
        self.model.policy.set_training_mode(False)

    @property
    def name(self):
        return f"pure_ppo_{self.training_seed}"

    def action(self, observation):
        action, _ = self.model.predict(pure_actor_observation(observation), deterministic=True)
        result = np.asarray(action).item()
        if result != int(result) or int(result) not in range(9):
            raise ValueError("controller emitted an invalid joint action")
        return int(result)


def evaluate_continuation(env, observation, candidate, future_seed, trace=None):
    """Clone only the environment and run a full closed-loop candidate.

    Optional trace contains actual actuator feedback, including neutral-release
    action -1. Candidate actors are shared, deterministic, and never trained.
    """
    if int(future_seed) < 0:
        raise ValueError("future seed must be nonnegative")
    if not isinstance(candidate, (DynamicReferencePlan, FrozenPPOContinuation)):
        raise TypeError("manual plan or frozen PPO continuation required")
    x = np.asarray(observation).copy()
    pure_actor_observation(x)  # Validate the interface without changing input.
    clone = copy.deepcopy(env)
    try:
        clone.unwrapped.domain_env.rng = np.random.default_rng(future_seed)
        layer = clone
        while isinstance(layer, gym.Wrapper):
            if "rng" in vars(layer):
                layer.rng = np.random.default_rng(future_seed + 9_000_001)
            layer = layer.env
        config = clone.unwrapped.domain_env.config
        start_info = clone.unwrapped.domain_env._info()
        total_release = waits = hold_blocks = elapsed = 0
        term = trunc = False
        first_requested = first_actual = -1
        while not (term or trunc):
            action = (candidate.action(x) if isinstance(candidate, FrozenPPOContinuation)
                      else plan_action(candidate, x, elapsed, config.min_sign_hold_steps))
            x, _, term, trunc, info = clone.step(action)
            if elapsed == 0:
                first_requested = action
                first_actual = int(info["actual_joint_action"])
            total_release += float(info["actual_release_fraction"])
            waits += action == 8
            hold_blocks += bool(info["action_blocked_by_sign_hold"])
            if trace is not None:
                trace.append((elapsed, action, int(info["actual_joint_action"]),
                              int(info["actual_sign_action"]), float(info["actual_release_fraction"]),
                              int(bool(info["action_blocked_by_sign_hold"]))))
            elapsed += 1
            if elapsed > config.max_steps:
                raise RuntimeError("continuation exceeded episode boundary")
        return PlanOutcome(float(info["incapacitation_rate"]), float(info["mean_cumulative_fed"]),
                           float(info["evacuation_rate"]), float(info["remaining"] / config.population),
                           float(info["cumulative_congestion"] - start_info["cumulative_congestion"]),
                           elapsed, total_release / elapsed, int(waits), hold_blocks / elapsed,
                           first_requested, first_actual)
    finally:
        clone.close()
