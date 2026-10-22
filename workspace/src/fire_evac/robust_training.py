from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def worst_case_sampling_weights(
    mean_rewards: Mapping[str, float],
    behavior_order: Sequence[str],
    temperature: float = 0.10,
    exploration_mass: float = 0.20,
) -> np.ndarray:
    """Allocate more episodes to behaviors with lower validation reward."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= exploration_mass <= 1.0:
        raise ValueError("exploration_mass must be between zero and one")
    if set(mean_rewards) != set(behavior_order):
        raise ValueError("mean_rewards must contain exactly the requested behaviors")
    rewards = np.array([float(mean_rewards[name]) for name in behavior_order], dtype=float)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("validation rewards must be finite")
    logits = (-rewards / temperature) - np.max(-rewards / temperature)
    focused = np.exp(logits)
    focused /= focused.sum()
    uniform = np.full(len(behavior_order), 1.0 / len(behavior_order), dtype=float)
    return (1.0 - exploration_mass) * focused + exploration_mass * uniform


def worst_case_score(mean_rewards: Mapping[str, float]) -> float:
    if not mean_rewards:
        raise ValueError("at least one validation reward is required")
    return float(min(mean_rewards.values()))
