from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import gymnasium as gym
import numpy as np

from .dynamic_conflicts import (
    JOINT_RELEASE_LEVELS,
    DynamicConflictScenario,
    GymDynamicConflictEnv,
)
from .dynamic_llm_rl import DynamicLLMAdvice, NORTH, SOUTH


WAIT_ACTION = 8
ADVICE_MODES = ("none", "rule", "llm")


@dataclass(frozen=True)
class ObservationNoise:
    name: str = "none"
    occupancy_std: float = 0.0
    smoke_std: float = 0.0
    capacity_std: float = 0.0
    reopen_std_steps: float = 0.0
    delay_steps: int = 0

    def __post_init__(self) -> None:
        if min(self.occupancy_std, self.smoke_std, self.capacity_std, self.reopen_std_steps) < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if self.delay_steps not in (0, 1, 2):
            raise ValueError("delay_steps must be 0, 1, or 2")


NOISE_LEVELS: dict[str, ObservationNoise] = {
    "none": ObservationNoise(),
    "light": ObservationNoise("light", 0.01, 0.002, 0.03, 0.5, 0),
    "medium": ObservationNoise("medium", 0.03, 0.008, 0.08, 1.5, 1),
    "heavy": ObservationNoise("heavy", 0.06, 0.020, 0.15, 3.0, 2),
}


def encode_absolute_action(branch: int, release_fraction: float) -> int:
    if branch not in (NORTH, SOUTH):
        raise ValueError("branch must be north or south")
    try:
        level = JOINT_RELEASE_LEVELS.index(float(release_fraction))
    except ValueError as exc:
        raise ValueError(f"unsupported release fraction: {release_fraction}") from exc
    return level + 4 * branch


def decode_absolute_action(action: int) -> tuple[int | None, float]:
    parsed = int(action)
    if parsed == WAIT_ACTION:
        return None, 0.0
    if parsed not in range(8):
        raise ValueError("joint action must be in [0, 8]")
    return parsed // 4, JOINT_RELEASE_LEVELS[parsed % 4]


def frozen_rule_advice(scenario: DynamicConflictScenario) -> DynamicLLMAdvice:
    """Deterministic high-level rule using exactly the scenario forecast fields."""
    capacities = np.asarray([scenario.north_exit_capacity, scenario.south_exit_capacity], dtype=float)
    costs = 0.4 / np.maximum(capacities, 1.0)
    costs[scenario.forecast_source_branch] += 2.0
    if scenario.blocked_branch in (NORTH, SOUTH) and scenario.blockage_end_step > 0:
        costs[scenario.blocked_branch] += min(2.0, scenario.blockage_end_step / 8.0)
    target = int(np.argmin(costs))
    gap = float(abs(costs[NORTH] - costs[SOUTH]))
    confidence = float(np.clip(0.5 + 0.2 * gap, 0.5, 0.99))
    return DynamicLLMAdvice(target, confidence, ("avoid_predicted_smoke",))


class ConfirmatoryJointControlWrapper(gym.Wrapper):
    """Fair nine-action controller used by all confirmatory study arms.

    Actions have absolute semantics: north/south crossed with four release
    levels, plus wait. Every arm receives the same physical and raw forecast
    fields. Only the five fixed advice-channel values differ by study arm.
    """

    def __init__(
        self,
        env: GymDynamicConflictEnv,
        advice_mode: str,
        advice_by_scenario: Mapping[str, DynamicLLMAdvice] | None = None,
        noise: ObservationNoise | None = None,
        maximum_forecast_steps: int = 20,
    ):
        super().__init__(env)
        if advice_mode not in ADVICE_MODES:
            raise ValueError(f"advice_mode must be one of {ADVICE_MODES}")
        if maximum_forecast_steps < 1:
            raise ValueError("maximum_forecast_steps must be positive")
        self.advice_mode = advice_mode
        self.advice_by_scenario = dict(advice_by_scenario or {})
        self.noise = noise or NOISE_LEVELS["none"]
        self.maximum_forecast_steps = int(maximum_forecast_steps)
        self.current_scenario: DynamicConflictScenario | None = None
        self.current_advice: DynamicLLMAdvice | None = None
        self.rng = np.random.default_rng()
        self._observation_history: list[np.ndarray] = []
        self.action_space = gym.spaces.Discrete(9)
        # base physical 18 + reopen 1 + raw forecast 9 + advice 5 + release memory 1
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [env.observation_space.low, np.zeros(15, dtype=np.float32), np.asarray([-1.0], dtype=np.float32)]
            ),
            high=np.concatenate(
                [env.observation_space.high, np.full(15, np.inf, dtype=np.float32), np.asarray([1.0], dtype=np.float32)]
            ),
            dtype=np.float32,
        )

    def _resolve_advice(self, scenario: DynamicConflictScenario) -> DynamicLLMAdvice | None:
        if self.advice_mode == "none":
            return None
        if self.advice_mode == "rule":
            return frozen_rule_advice(scenario)
        if scenario.scenario_id not in self.advice_by_scenario:
            raise KeyError(f"no frozen LLM advice for scenario {scenario.scenario_id}")
        return self.advice_by_scenario[scenario.scenario_id]

    def _sensor_observation(self, observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32).copy()
        self._observation_history.append(raw)
        delayed_index = max(0, len(self._observation_history) - 1 - self.noise.delay_steps)
        sensed = self._observation_history[delayed_index].copy()
        sensed[0:5] = np.clip(
            sensed[0:5] + self.rng.normal(0.0, self.noise.occupancy_std, 5), 0.0, None
        )
        sensed[5:10] = np.clip(
            sensed[5:10] + self.rng.normal(0.0, self.noise.smoke_std, 5), 0.0, None
        )
        sensed[14:16] = np.clip(
            sensed[14:16] + self.rng.normal(0.0, self.noise.capacity_std, 2), 0.0, None
        )
        return sensed

    def _augment(self, observation: np.ndarray, info: dict) -> np.ndarray:
        if self.current_scenario is None:
            raise RuntimeError("reset must be called before observation augmentation")
        sensed = self._sensor_observation(observation)
        scenario = self.current_scenario
        remaining = float(info.get("blockage_steps_remaining", 0))
        remaining += float(self.rng.normal(0.0, self.noise.reopen_std_steps))
        reopen = np.clip(remaining / self.maximum_forecast_steps, 0.0, 1.0)
        forecast = np.asarray(
            [
                float(scenario.forecast_source_branch == NORTH),
                float(scenario.forecast_source_branch == SOUTH),
                np.clip(scenario.growth_rate / 0.05, 0.0, 1.0),
                np.clip(scenario.onset_step / 20.0, 0.0, 1.0),
                scenario.north_exit_capacity / 8.0,
                scenario.south_exit_capacity / 8.0,
                float(scenario.blocked_branch == NORTH),
                float(scenario.blocked_branch == SOUTH),
                np.clip(scenario.blockage_end_step / self.maximum_forecast_steps, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        if self.current_advice is None:
            advice_features = np.zeros(5, dtype=np.float32)
        else:
            target = self.current_advice.target_branch
            alternative = SOUTH if target == NORTH else NORTH
            target_hazard = float(sensed[8 + target])
            alternative_hazard = float(sensed[8 + alternative])
            contradiction = float(target_hazard > alternative_hazard + 0.01)
            advice_features = np.asarray(
                [
                    float(target == NORTH),
                    float(target == SOUTH),
                    self.current_advice.confidence,
                    1.0,
                    contradiction,
                ],
                dtype=np.float32,
            )
        full_capacity = max(1, int(self.unwrapped.domain_env.config.start_to_junction_capacity))
        release_memory = np.asarray(
            [np.clip(float(info.get("room_release_capacity", full_capacity)) / full_capacity, 0.0, 1.0)],
            dtype=np.float32,
        )
        return np.concatenate(
            [sensed, np.asarray([reopen], dtype=np.float32), forecast, advice_features, release_memory]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        scenario_payload = info.get("scenario")
        scenario = None if options is None else options.get("scenario")
        if not isinstance(scenario, DynamicConflictScenario):
            scenario_id = str(scenario_payload["scenario_id"])
            scenario = next(
                candidate
                for candidate in self.unwrapped.domain_env.conflict_provider.scenarios
                if candidate.scenario_id == scenario_id
            )
        self.current_scenario = scenario
        self.current_advice = self._resolve_advice(scenario)
        self.rng = np.random.default_rng(None if seed is None else seed + 9_000_001)
        self._observation_history = []
        enriched = dict(info)
        enriched.update(self._advice_info())
        return self._augment(observation, enriched), enriched

    def _advice_info(self) -> dict[str, object]:
        if self.current_advice is None:
            return {"advice_mode": self.advice_mode, "advice_valid": False, "advice_target": "none"}
        return {
            "advice_mode": self.advice_mode,
            "advice_valid": True,
            "advice_target": "north" if self.current_advice.target_branch == NORTH else "south",
            "advice_confidence": self.current_advice.confidence,
        }

    def step(self, action: int):
        requested_action = int(action)
        branch, release_fraction = decode_absolute_action(requested_action)
        if branch is None:
            self.unwrapped.domain_env.set_hold_rooms(True)
            mapped_action = 2
        else:
            self.unwrapped.domain_env.set_room_release_fraction(release_fraction)
            mapped_action = branch
        observation, reward, terminated, truncated, info = self.env.step(mapped_action)
        actual_sign = int(info["actual_sign_action"])
        if branch is None:
            actual_joint = WAIT_ACTION
        elif actual_sign not in (NORTH, SOUTH):
            # Signal hold can keep a neutral sign while rooms are released.
            # The resulting physical combination has no member in the nine-action set.
            actual_joint = -1
        else:
            actual_joint = encode_absolute_action(actual_sign, release_fraction)
        full_capacity = max(1, int(self.unwrapped.domain_env.config.start_to_junction_capacity))
        enriched = dict(info)
        enriched.update(
            {
                "requested_joint_action": requested_action,
                "requested_sign_action": mapped_action,
                "requested_release_fraction": release_fraction,
                "actual_joint_action": actual_joint,
                "actual_sign_action": actual_sign,
                "actual_release_fraction": float(info["room_release_capacity"]) / full_capacity,
                **self._advice_info(),
            }
        )
        return self._augment(observation, enriched), reward, terminated, truncated, enriched
