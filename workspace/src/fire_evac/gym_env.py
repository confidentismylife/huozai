from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .behavior import BehaviorModel
from .config import EnvConfig
from .env import FireEvacuationEnv
from .hazard import FireScenario, HazardProvider


class GymFireEvacuationEnv(gym.Env[np.ndarray, int]):
    """Gymnasium adapter around the domain evacuation simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        behavior_model: BehaviorModel,
        config: EnvConfig | None = None,
        hazard_provider: HazardProvider | None = None,
    ):
        super().__init__()
        self.domain_env = FireEvacuationEnv(behavior_model, config=config, hazard_provider=hazard_provider)
        self.action_space = gym.spaces.Discrete(self.domain_env.action_count)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(14, dtype=np.float32),
            high=np.full(14, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        scenario: FireScenario | None = None
        if options is not None:
            candidate = options.get("scenario")
            if candidate is not None and not isinstance(candidate, FireScenario):
                raise TypeError("options['scenario'] must be a FireScenario")
            scenario = candidate
        observation, info = self.domain_env.reset(seed=seed, scenario=scenario)
        return observation.astype(np.float32, copy=False), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.domain_env.step(int(action))
        return observation.astype(np.float32, copy=False), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class HistoryStackWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Concatenate the latest observations to expose short-term dynamics."""

    def __init__(self, env: gym.Env[np.ndarray, int], history_length: int = 4):
        if history_length < 1:
            raise ValueError("history_length must be at least one")
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("HistoryStackWrapper requires a Box observation space")
        self.history_length = history_length
        self._history: list[np.ndarray] = []
        low = np.tile(env.observation_space.low, history_length)
        high = np.tile(env.observation_space.high, history_length)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _stack(self) -> np.ndarray:
        return np.concatenate(self._history).astype(np.float32, copy=False)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._history = [observation.copy() for _ in range(self.history_length)]
        return self._stack(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._history.pop(0)
        self._history.append(observation.copy())
        return self._stack(), reward, terminated, truncated, info
