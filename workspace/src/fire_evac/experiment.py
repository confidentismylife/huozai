from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

import numpy as np

from .env import FireEvacuationEnv


class Controller(Protocol):
    name: str

    def act(self, observation: np.ndarray) -> int: ...


def run_episode(env: FireEvacuationEnv, controller: Controller, seed: int) -> dict[str, float | int | str]:
    observation, info = env.reset(seed=seed)
    reset_controller = getattr(controller, "reset", None)
    if callable(reset_controller):
        reset_controller()
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return {
        "controller": controller.name,
        "behavior_model": str(info["behavior_model"]),
        "active_behavior_model": str(info["active_behavior_model"]),
        "seed": seed,
        "steps": int(info["step"]),
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "remaining_rate": float(info["remaining"]) / env.config.population,
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": float(total_reward),
    }


def evaluate(
    env: FireEvacuationEnv,
    controllers: list[Controller],
    episodes: int,
    seed: int = 1000,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    episode_seeds = [seed + i for i in range(episodes)]
    for controller in controllers:
        for episode_seed in episode_seeds:
            rows.append(run_episode(env, controller, episode_seed))
    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["behavior_model"]), str(row["controller"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, float | str]] = []
    for (behavior, controller), items in sorted(groups.items()):
        summaries.append(
            {
                "behavior_model": behavior,
                "controller": controller,
                "episodes": float(len(items)),
                "evacuation_rate_mean": float(np.mean([float(x["evacuation_rate"]) for x in items])),
                "incapacitation_rate_mean": float(
                    np.mean([float(x["incapacitation_rate"]) for x in items])
                ),
                "remaining_rate_mean": float(np.mean([float(x["remaining_rate"]) for x in items])),
                "mean_cumulative_fed": float(np.mean([float(x["mean_cumulative_fed"]) for x in items])),
                "congestion_mean": float(np.mean([float(x["cumulative_congestion"]) for x in items])),
                "reward_mean": float(np.mean([float(x["total_reward"]) for x in items])),
            }
        )
    return summaries

