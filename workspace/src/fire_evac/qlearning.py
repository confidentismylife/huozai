from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import FireEvacuationEnv


def discretize_observation(obs: np.ndarray) -> tuple[int, int, int, int]:
    occupancy = obs[:5]
    hazard = obs[5:10]
    previous_action = int(np.argmax(obs[10:13]))
    remaining = float(obs[-1])
    hazard_diff = float(hazard[3] - hazard[4])
    occupancy_diff = float(occupancy[3] - occupancy[4])
    h_bin = int(np.digitize(hazard_diff, [-0.08, -0.02, 0.02, 0.08]))
    o_bin = int(np.digitize(occupancy_diff, [-0.20, -0.05, 0.05, 0.20]))
    r_bin = int(np.digitize(remaining, [0.25, 0.50, 0.75]))
    return h_bin, o_bin, r_bin, previous_action


@dataclass
class QLearningController:
    q: dict[tuple[int, int, int, int], np.ndarray]
    name: str = "q_learning"

    def act(self, observation: np.ndarray) -> int:
        state = discretize_observation(observation)
        values = self.q.get(state)
        if values is None:
            return 2
        return int(np.argmax(values))


def train_q_learning(
    env: FireEvacuationEnv,
    episodes: int = 3000,
    seed: int = 42,
    alpha: float = 0.12,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> QLearningController:
    rng = np.random.default_rng(seed)
    q: dict[tuple[int, int, int, int], np.ndarray] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / episodes)
        done = False
        while not done:
            state = discretize_observation(obs)
            q.setdefault(state, np.zeros(env.action_count, dtype=float))
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.action_count))
            else:
                action = int(np.argmax(q[state]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_observation(next_obs)
            q.setdefault(next_state, np.zeros(env.action_count, dtype=float))
            target = reward
            if not (terminated or truncated):
                target += gamma * float(np.max(q[next_state]))
            q[state][action] += alpha * (target - q[state][action])
            obs = next_obs
            done = terminated or truncated
    return QLearningController(q=q)

