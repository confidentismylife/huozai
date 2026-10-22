"""Runtime interface for the development ternary action filter."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .terminal_shield import OBSERVATION_DIM, UNSAFE, TernaryShieldPredictor, ternary_action_mask


class TerminalShieldWrapper(gym.Wrapper):
    """Expose all class scores, allowed mask, and predicted abstentions.

    49 common observable inputs + 27 scores + 9 mask + 9 abstentions = 94.
    Study arms may zero these 45 derived fields using expose=False while
    preserving PPO input size. Explicit matched shield-only controls remain
    necessary to distinguish filter effects from advice effects.
    """

    def __init__(self, env: gym.Env, predictor: TernaryShieldPredictor, *, enforce: bool, expose: bool = True):
        super().__init__(env)
        if env.observation_space.shape != (OBSERVATION_DIM,):
            raise ValueError("ternary shield requires the 49-input history wrapper")
        self.predictor = predictor
        self.enforce = bool(enforce)
        self.expose = bool(expose)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, np.zeros(45)]).astype(np.float32),
            high=np.concatenate([env.observation_space.high, np.ones(45)]).astype(np.float32),
            dtype=np.float32,
        )
        self.current_probabilities = np.zeros((9, 3), dtype=np.float32)
        self.current_mask = np.ones(9, dtype=bool)
        self.current_abstain = np.zeros(9, dtype=bool)
        self.current_empty_raw = False

    def _augment(self, observation):
        self.current_probabilities = self.predictor.probabilities(observation)
        self.current_mask, self.current_abstain, empty = ternary_action_mask(self.current_probabilities, self.predictor.threshold)
        self.current_empty_raw = bool(empty)
        features = np.concatenate([self.current_probabilities.ravel(), self.current_mask, self.current_abstain]) if self.expose else np.zeros(45)
        return np.concatenate([observation, features]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation), info

    def step(self, action):
        requested = int(action)
        if not self.action_space.contains(requested):
            raise ValueError("joint action must be in [0, 8]")
        applied = requested
        if self.enforce and not self.current_mask[requested]:
            applied = int(np.argmin(np.where(self.current_mask, self.current_probabilities[:, UNSAFE], np.inf)))
        old_mask = self.current_mask.copy()
        old_empty = self.current_empty_raw
        old_abstain_count = int(self.current_abstain.sum())
        observation, reward, term, trunc, info = self.env.step(applied)
        actual = int(info["actual_joint_action"])
        enriched = dict(info)
        enriched.update({"shield_requested_action": requested, "shield_applied_action": applied,
                         "shield_actual_joint_action": actual, "shield_intervened": applied != requested,
                         "shield_applied_actual_match": applied == actual,
                         "shield_actual_in_allowed_set": bool(old_mask[actual]) if actual in range(9) else False,
                         "shield_allowed_action_count": int(old_mask.sum()),
                         "shield_predicted_abstention_count": old_abstain_count,
                         "shield_empty_raw_mask": old_empty})
        return self._augment(observation), reward, term, trunc, enriched
