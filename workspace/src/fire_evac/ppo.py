from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class PPOController:
    name = "ppo"

    def __init__(self, model: Any, history_length: int = 1):
        self.model = model
        if history_length < 1:
            raise ValueError("history_length must be at least one")
        self.history_length = history_length
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        self._history = []

    def act(self, observation: np.ndarray) -> int:
        if not self._history:
            self._history = [observation.copy() for _ in range(self.history_length)]
        else:
            self._history.pop(0)
            self._history.append(observation.copy())
        model_observation = np.concatenate(self._history).astype(np.float32, copy=False)
        action, _ = self.model.predict(model_observation, deterministic=True)
        return int(np.asarray(action).item())


def load_ppo_controller(path: str | Path, history_length: int = 1) -> PPOController:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Install the optional 'rl' dependencies before loading PPO") from exc
    return PPOController(PPO.load(str(path)), history_length=history_length)


class RecurrentPPOController:
    name = "recurrent_ppo"

    def __init__(self, model: Any):
        self.model = model
        self._state: Any = None
        self._episode_start = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        self._state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def act(self, observation: np.ndarray) -> int:
        action, self._state = self.model.predict(
            observation,
            state=self._state,
            episode_start=self._episode_start,
            deterministic=True,
        )
        self._episode_start = np.zeros((1,), dtype=bool)
        return int(np.asarray(action).item())


def load_recurrent_ppo_controller(path: str | Path) -> RecurrentPPOController:
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise RuntimeError("Install sb3-contrib before loading RecurrentPPO") from exc
    return RecurrentPPOController(RecurrentPPO.load(str(path)))
