from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np

from .behavior import BehaviorModel
from .config import EnvConfig
from .dynamic_llm_rl import DynamicLLMAdvice, NORTH, SOUTH
from .env import FireEvacuationEnv
from .hazard import SyntheticHazardProvider


@dataclass(frozen=True)
class DynamicConflictScenario:
    scenario_id: str
    source_branch: int
    growth_rate: float
    onset_step: int
    background_hazard: float = 0.002
    conflict_type: str = "clear"
    north_exit_capacity: int = 8
    south_exit_capacity: int = 8
    blocked_branch: int = -1
    blockage_start_step: int = 0
    blockage_end_step: int = 0
    blockage_capacity: int = 0
    forecast_source_branch: int = 0


CONFLICT_TYPES = (
    "clear",
    "safe_exit_congested",
    "safe_exit_longer",
    "safe_exit_temporarily_blocked",
    "forecast_error",
    "combined_capacity_blockage",
)

TRUST_GATE_TYPES = (
    "clear",
    "safe_exit_short_block",
    "safe_exit_long_block",
    "safe_exit_congested",
    "forecast_error",
)

DOMAIN_RANDOMIZED_TYPES = (
    "clear",
    "safe_exit_short_block",
    "safe_exit_long_block",
    "safe_exit_congested",
    "forecast_error",
    "compound_conflict",
)


def make_dynamic_conflict_bank(n: int, seed: int, prefix: str) -> tuple[DynamicConflictScenario, ...]:
    if n < len(CONFLICT_TYPES):
        raise ValueError(f"at least {len(CONFLICT_TYPES)} scenarios are required")
    rng = np.random.default_rng(seed)
    scenarios: list[DynamicConflictScenario] = []
    for index in range(n):
        conflict_type = CONFLICT_TYPES[index % len(CONFLICT_TYPES)]
        source = index % 2
        safe = SOUTH if source == NORTH else NORTH
        north_capacity = south_capacity = 8
        blocked_branch = -1
        blockage_start = blockage_end = 0
        forecast_source = source
        if conflict_type == "safe_exit_congested":
            if safe == NORTH:
                north_capacity = 3
            else:
                south_capacity = 3
        elif conflict_type == "safe_exit_longer":
            if safe == NORTH:
                north_capacity = 4
            else:
                south_capacity = 4
        elif conflict_type == "safe_exit_temporarily_blocked":
            blocked_branch = safe
            blockage_end = int(rng.integers(4, 8))
        elif conflict_type == "forecast_error":
            forecast_source = safe
        elif conflict_type == "combined_capacity_blockage":
            if safe == NORTH:
                north_capacity = 4
            else:
                south_capacity = 4
            blocked_branch = safe
            blockage_end = int(rng.integers(3, 7))
        scenarios.append(
            DynamicConflictScenario(
                scenario_id=f"{prefix}{index + 1:03d}",
                source_branch=source,
                growth_rate=float(rng.uniform(0.010, 0.030)),
                onset_step=int(rng.integers(0, 8)),
                conflict_type=conflict_type,
                north_exit_capacity=north_capacity,
                south_exit_capacity=south_capacity,
                blocked_branch=blocked_branch,
                blockage_start_step=blockage_start,
                blockage_end_step=blockage_end,
                blockage_capacity=0,
                forecast_source_branch=forecast_source,
            )
        )
    order = rng.permutation(len(scenarios))
    return tuple(scenarios[int(index)] for index in order)


def make_trust_gate_bank(n: int, seed: int, prefix: str) -> tuple[DynamicConflictScenario, ...]:
    if n < len(TRUST_GATE_TYPES):
        raise ValueError(f"at least {len(TRUST_GATE_TYPES)} trust-gate scenarios are required")
    rng = np.random.default_rng(seed)
    scenarios: list[DynamicConflictScenario] = []
    for index in range(n):
        conflict_type = TRUST_GATE_TYPES[index % len(TRUST_GATE_TYPES)]
        source = index % 2
        safe = SOUTH if source == NORTH else NORTH
        north_capacity = south_capacity = 8
        blocked_branch = -1
        blockage_end = 0
        forecast_source = source
        if conflict_type == "safe_exit_short_block":
            blocked_branch = safe
            blockage_end = int(rng.integers(2, 5))
        elif conflict_type == "safe_exit_long_block":
            blocked_branch = safe
            blockage_end = int(rng.integers(8, 13))
        elif conflict_type == "safe_exit_congested":
            if safe == NORTH:
                north_capacity = 3
            else:
                south_capacity = 3
        elif conflict_type == "forecast_error":
            forecast_source = safe
        scenarios.append(
            DynamicConflictScenario(
                scenario_id=f"{prefix}{index + 1:03d}",
                source_branch=source,
                growth_rate=float(rng.uniform(0.010, 0.030)),
                onset_step=int(rng.integers(0, 8)),
                conflict_type=conflict_type,
                north_exit_capacity=north_capacity,
                south_exit_capacity=south_capacity,
                blocked_branch=blocked_branch,
                blockage_start_step=0,
                blockage_end_step=blockage_end,
                blockage_capacity=0,
                forecast_source_branch=forecast_source,
            )
        )
    order = rng.permutation(len(scenarios))
    return tuple(scenarios[int(index)] for index in order)


def make_domain_randomized_trust_bank(
    n: int, seed: int, prefix: str = "GDR"
) -> tuple[DynamicConflictScenario, ...]:
    """Create a broad training distribution over hazards, capacities and conflicts."""
    if n < len(DOMAIN_RANDOMIZED_TYPES):
        raise ValueError(f"at least {len(DOMAIN_RANDOMIZED_TYPES)} randomized scenarios are required")
    rng = np.random.default_rng(seed)
    scenarios: list[DynamicConflictScenario] = []
    for index in range(n):
        conflict_type = DOMAIN_RANDOMIZED_TYPES[index % len(DOMAIN_RANDOMIZED_TYPES)]
        source = int(rng.integers(0, 2))
        safe = SOUTH if source == NORTH else NORTH
        north_capacity = int(rng.integers(6, 11))
        south_capacity = int(rng.integers(6, 11))
        blocked_branch = -1
        blockage_end = 0
        forecast_source = source
        if conflict_type == "safe_exit_short_block":
            blocked_branch = safe
            blockage_end = int(rng.integers(1, 7))
        elif conflict_type == "safe_exit_long_block":
            blocked_branch = safe
            blockage_end = int(rng.integers(6, 19))
        elif conflict_type == "safe_exit_congested":
            if safe == NORTH:
                north_capacity = int(rng.integers(1, 7))
            else:
                south_capacity = int(rng.integers(1, 7))
        elif conflict_type == "forecast_error":
            forecast_source = safe
        elif conflict_type == "compound_conflict":
            # Combine imperfect forecasts with a randomly congested or blocked
            # nominally safe route, so no single cue identifies the right gate action.
            if rng.random() < 0.5:
                forecast_source = safe
            if rng.random() < 0.5:
                if safe == NORTH:
                    north_capacity = int(rng.integers(1, 6))
                else:
                    south_capacity = int(rng.integers(1, 6))
            else:
                blocked_branch = safe
                blockage_end = int(rng.integers(2, 15))
        scenarios.append(
            DynamicConflictScenario(
                scenario_id=f"{prefix}{index + 1:04d}",
                source_branch=source,
                growth_rate=float(rng.uniform(0.006, 0.045)),
                onset_step=int(rng.integers(0, 13)),
                background_hazard=float(rng.uniform(0.0005, 0.008)),
                conflict_type=conflict_type,
                north_exit_capacity=north_capacity,
                south_exit_capacity=south_capacity,
                blocked_branch=blocked_branch,
                blockage_start_step=0,
                blockage_end_step=blockage_end,
                blockage_capacity=0,
                forecast_source_branch=forecast_source,
            )
        )
    order = rng.permutation(len(scenarios))
    return tuple(scenarios[int(index)] for index in order)


class DynamicConflictHazardProvider(SyntheticHazardProvider):
    def __init__(self, scenarios: Sequence[DynamicConflictScenario]):
        if not scenarios:
            raise ValueError("scenarios cannot be empty")
        self.scenarios = tuple(scenarios)

    def sample_scenario(self, rng: np.random.Generator, prefix: str = "train") -> DynamicConflictScenario:
        del prefix
        return self.scenarios[int(rng.integers(0, len(self.scenarios)))]


def conflict_llm_advice(scenario: DynamicConflictScenario) -> DynamicLLMAdvice:
    """Reuse the audited v2 LLM policy under a possibly wrong fire forecast."""
    target = SOUTH if scenario.forecast_source_branch == NORTH else NORTH
    return DynamicLLMAdvice(
        target_branch=target,
        confidence=0.95,
        reason_codes=("avoid_predicted_smoke", "early_guidance"),
    )


class DynamicConflictFireEvacuationEnv(FireEvacuationEnv):
    def __init__(
        self,
        behavior_model: BehaviorModel,
        scenarios: Sequence[DynamicConflictScenario],
        config: EnvConfig | None = None,
    ):
        self.conflict_provider = DynamicConflictHazardProvider(scenarios)
        self.hold_rooms_for_next_step = False
        self.room_release_fraction_for_next_step = 1.0
        super().__init__(behavior_model, config=config, hazard_provider=self.conflict_provider)

    def reset(self, seed: int | None = None, scenario: DynamicConflictScenario | None = None):
        self.hold_rooms_for_next_step = False
        self.room_release_fraction_for_next_step = 1.0
        return super().reset(seed=seed, scenario=scenario)

    def set_hold_rooms(self, hold: bool = True) -> None:
        self.hold_rooms_for_next_step = bool(hold)

    def set_room_release_fraction(self, fraction: float) -> None:
        parsed = float(fraction)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError("room release fraction must be between zero and one")
        self.room_release_fraction_for_next_step = parsed

    def _start_to_junction_capacity(self) -> int:
        if self.hold_rooms_for_next_step:
            return 0
        return int(round(super()._start_to_junction_capacity() * self.room_release_fraction_for_next_step))

    def step(self, action: int):
        result = super().step(action)
        self.hold_rooms_for_next_step = False
        self.room_release_fraction_for_next_step = 1.0
        return result

    def _conflict_scenario(self) -> DynamicConflictScenario:
        if not isinstance(self.scenario, DynamicConflictScenario):
            raise TypeError("DynamicConflictFireEvacuationEnv requires a DynamicConflictScenario")
        return self.scenario

    def _branch_to_exit_capacities(self) -> np.ndarray:
        scenario = self._conflict_scenario()
        capacities = np.asarray(
            [scenario.north_exit_capacity, scenario.south_exit_capacity], dtype=int
        )
        if (
            scenario.blocked_branch in (NORTH, SOUTH)
            and scenario.blockage_start_step <= self.step_index < scenario.blockage_end_step
        ):
            capacities[scenario.blocked_branch] = scenario.blockage_capacity
        return capacities

    def _operational_features(self) -> np.ndarray:
        capacities = self._branch_to_exit_capacities().astype(np.float32)
        scale = max(1.0, float(self.config.branch_to_exit_capacity))
        blocked = (capacities <= 0).astype(np.float32)
        return np.asarray(
            [capacities[NORTH] / scale, capacities[SOUTH] / scale, blocked[NORTH], blocked[SOUTH]],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate([super()._observation(), self._operational_features()]).astype(
            np.float32, copy=False
        )

    def _info(self) -> dict[str, object]:
        info = super()._info()
        if self.scenario is not None:
            scenario = self._conflict_scenario()
            capacities = self._branch_to_exit_capacities()
            info.update(
                {
                    "conflict_type": scenario.conflict_type,
                    "forecast_source_branch": scenario.forecast_source_branch,
                    "north_exit_capacity": int(capacities[NORTH]),
                    "south_exit_capacity": int(capacities[SOUTH]),
                    "blockage_active": bool(
                        scenario.blocked_branch in (NORTH, SOUTH)
                        and scenario.blockage_start_step <= self.step_index < scenario.blockage_end_step
                    ),
                    "blockage_steps_remaining": max(0, scenario.blockage_end_step - self.step_index)
                    if scenario.blocked_branch in (NORTH, SOUTH)
                    else 0,
                }
            )
        return info


class GymDynamicConflictEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        behavior_model: BehaviorModel,
        scenarios: Sequence[DynamicConflictScenario],
        config: EnvConfig | None = None,
    ):
        super().__init__()
        self.domain_env = DynamicConflictFireEvacuationEnv(behavior_model, scenarios, config=config)
        self.action_space = gym.spaces.Discrete(self.domain_env.action_count)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(18, dtype=np.float32),
            high=np.full(18, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario = None if options is None else options.get("scenario")
        if scenario is not None and not isinstance(scenario, DynamicConflictScenario):
            raise TypeError("options['scenario'] must be a DynamicConflictScenario")
        observation, info = self.domain_env.reset(seed=seed, scenario=scenario)
        return observation.astype(np.float32, copy=False), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.domain_env.step(int(action))
        return observation.astype(np.float32, copy=False), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def conflict_scenario_records(scenarios: Sequence[DynamicConflictScenario]) -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in scenarios]


class HoldEnabledActionWrapper(gym.ActionWrapper):
    """Add action 3=hold rooms for one step; actions 0-2 retain sign semantics."""

    def __init__(self, env: GymDynamicConflictEnv):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)

    def action(self, action: int) -> int:
        parsed = int(action)
        if parsed not in (0, 1, 2, 3):
            raise ValueError("action must be 0=north, 1=south, 2=neutral, or 3=hold")
        if parsed == 3:
            self.unwrapped.domain_env.set_hold_rooms(True)
            return 2
        return parsed

    def step(self, action: int):
        observation, reward, terminated, truncated, info = super().step(action)
        enriched = dict(info)
        enriched["hold_action_requested"] = bool(int(action) == 3)
        return observation, reward, terminated, truncated, enriched


class ReopenForecastObservationWrapper(gym.Wrapper):
    """Append the predicted number of control steps until a blocked exit reopens."""

    def __init__(self, env: gym.Env, maximum_forecast_steps: int = 12):
        super().__init__(env)
        if maximum_forecast_steps < 1:
            raise ValueError("maximum_forecast_steps must be positive")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ReopenForecastObservationWrapper requires Box observations")
        self.maximum_forecast_steps = maximum_forecast_steps
        low = np.concatenate([env.observation_space.low, np.zeros(1, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(1, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray, info: dict) -> np.ndarray:
        remaining = min(
            self.maximum_forecast_steps,
            max(0, int(info.get("blockage_steps_remaining", 0))),
        )
        return np.concatenate(
            [observation, np.asarray([remaining / self.maximum_forecast_steps], dtype=np.float32)]
        ).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation, info), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation, info), reward, terminated, truncated, info


class LLMTrustGateWrapper(gym.Wrapper):
    """Meta-controller actions: 0=trust LLM, 1=override, 2=hold and wait."""

    def __init__(
        self,
        env: GymDynamicConflictEnv,
        advice_by_scenario: dict[str, DynamicLLMAdvice],
        maximum_forecast_steps: int = 12,
    ):
        super().__init__(env)
        self.advice_by_scenario = dict(advice_by_scenario)
        self.maximum_forecast_steps = maximum_forecast_steps
        self.current_advice: DynamicLLMAdvice | None = None
        self.action_space = gym.spaces.Discrete(3)
        low = np.concatenate([env.observation_space.low, np.zeros(4, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(4, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray, info: dict) -> np.ndarray:
        if self.current_advice is None:
            raise RuntimeError("reset must be called before using the trust gate")
        remaining = min(
            self.maximum_forecast_steps,
            max(0, int(info.get("blockage_steps_remaining", 0))),
        )
        features = np.asarray(
            [
                remaining / self.maximum_forecast_steps,
                float(self.current_advice.target_branch == NORTH),
                float(self.current_advice.target_branch == SOUTH),
                self.current_advice.confidence,
            ],
            dtype=np.float32,
        )
        return np.concatenate([observation, features]).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        scenario_id = str(info["scenario"]["scenario_id"])
        if scenario_id not in self.advice_by_scenario:
            raise KeyError(f"no LLM advice for scenario {scenario_id}")
        self.current_advice = self.advice_by_scenario[scenario_id]
        enriched = dict(info)
        enriched["llm_target_exit"] = "north" if self.current_advice.target_branch == NORTH else "south"
        return self._augment(observation, enriched), enriched

    def step(self, action: int):
        gate_action = int(action)
        if gate_action not in (0, 1, 2):
            raise ValueError("gate action must be 0=trust, 1=override, or 2=wait")
        if self.current_advice is None:
            raise RuntimeError("reset must be called before step")
        if gate_action == 0:
            mapped_action = self.current_advice.target_branch
            gate_name = "trust"
        elif gate_action == 1:
            mapped_action = SOUTH if self.current_advice.target_branch == NORTH else NORTH
            gate_name = "override"
        else:
            self.unwrapped.domain_env.set_hold_rooms(True)
            mapped_action = 2
            gate_name = "wait"
        observation, reward, terminated, truncated, info = self.env.step(mapped_action)
        enriched = dict(info)
        enriched.update(
            {
                "trust_gate_action": gate_name,
                "mapped_sign_action": mapped_action,
                "llm_target_exit": "north" if self.current_advice.target_branch == NORTH else "south",
            }
        )
        return self._augment(observation, enriched), reward, terminated, truncated, enriched


class ContradictionAwareLLMTrustGateWrapper(LLMTrustGateWrapper):
    """Append explicit smoke contradiction and target-route congestion evidence."""

    def __init__(
        self,
        env: GymDynamicConflictEnv,
        advice_by_scenario: dict[str, DynamicLLMAdvice],
        maximum_forecast_steps: int = 12,
        smoke_margin: float = 0.01,
    ):
        super().__init__(env, advice_by_scenario, maximum_forecast_steps)
        if smoke_margin < 0:
            raise ValueError("smoke_margin must be non-negative")
        self.smoke_margin = float(smoke_margin)
        low = np.concatenate(
            [self.observation_space.low, np.asarray([0.0, 0.0, -1.0, 0.0, -np.inf, -1.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [self.observation_space.high, np.asarray([np.inf, np.inf, 1.0, 1.0, np.inf, 1.0], dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray, info: dict) -> np.ndarray:
        base = super()._augment(observation, info)
        if self.current_advice is None:
            raise RuntimeError("reset must be called before using contradiction features")
        target = self.current_advice.target_branch
        alternative = SOUTH if target == NORTH else NORTH
        target_hazard = float(observation[8 + target])
        alternative_hazard = float(observation[8 + alternative])
        smoke_contrast = (target_hazard - alternative_hazard) / (
            abs(target_hazard) + abs(alternative_hazard) + 1e-6
        )
        contradiction = float(target_hazard > alternative_hazard + self.smoke_margin)
        target_capacity = float(observation[14 + target])
        alternative_capacity = float(observation[14 + alternative])
        target_occupancy = float(observation[3 + target])
        alternative_occupancy = float(observation[3 + alternative])
        evidence = np.asarray(
            [
                target_hazard,
                alternative_hazard,
                np.clip(smoke_contrast, -1.0, 1.0),
                contradiction,
                target_capacity - alternative_capacity,
                np.clip(target_occupancy - alternative_occupancy, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, evidence]).astype(np.float32, copy=False)


JOINT_RELEASE_LEVELS = (0.25, 0.50, 0.75, 1.00)


class JointRouteReleaseGateWrapper(ContradictionAwareLLMTrustGateWrapper):
    """Joint actions choose trust/override route and a graded room release capacity."""

    def __init__(
        self,
        env: GymDynamicConflictEnv,
        advice_by_scenario: dict[str, DynamicLLMAdvice],
        maximum_forecast_steps: int = 12,
        smoke_margin: float = 0.01,
    ):
        super().__init__(env, advice_by_scenario, maximum_forecast_steps, smoke_margin)
        self.action_space = gym.spaces.Discrete(9)
        low = np.concatenate([self.observation_space.low, np.zeros(1, dtype=np.float32)])
        high = np.concatenate([self.observation_space.high, np.ones(1, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray, info: dict) -> np.ndarray:
        base = super()._augment(observation, info)
        full_capacity = max(1, int(self.unwrapped.domain_env.config.start_to_junction_capacity))
        last_release = float(info.get("room_release_capacity", full_capacity)) / full_capacity
        return np.concatenate(
            [base, np.asarray([np.clip(last_release, 0.0, 1.0)], dtype=np.float32)]
        ).astype(np.float32, copy=False)

    def step(self, action: int):
        parsed = int(action)
        if parsed not in range(9):
            raise ValueError("joint gate action must be in [0, 8]")
        if self.current_advice is None:
            raise RuntimeError("reset must be called before step")
        if parsed == 8:
            route = "wait"
            release_fraction = 0.0
            self.unwrapped.domain_env.set_hold_rooms(True)
            mapped_action = 2
        else:
            route = "trust" if parsed < 4 else "override"
            release_fraction = JOINT_RELEASE_LEVELS[parsed % 4]
            self.unwrapped.domain_env.set_room_release_fraction(release_fraction)
            if route == "trust":
                mapped_action = self.current_advice.target_branch
            else:
                mapped_action = SOUTH if self.current_advice.target_branch == NORTH else NORTH
        observation, reward, terminated, truncated, info = self.env.step(mapped_action)
        actual_sign_action = int(info["actual_sign_action"])
        if parsed == 8:
            actual_gate_action = "wait"
            actual_joint_action = 8
        elif actual_sign_action not in (NORTH, SOUTH):
            actual_gate_action = "neutral"
            actual_joint_action = -1
        else:
            actual_gate_action = (
                "trust" if actual_sign_action == self.current_advice.target_branch else "override"
            )
            actual_joint_action = (
                JOINT_RELEASE_LEVELS.index(release_fraction)
                if actual_gate_action == "trust"
                else 4 + JOINT_RELEASE_LEVELS.index(release_fraction)
            )
        full_capacity = max(1, int(self.unwrapped.domain_env.config.start_to_junction_capacity))
        actual_release_fraction = float(info["room_release_capacity"]) / full_capacity
        enriched = dict(info)
        enriched.update(
            {
                # Keep the legacy names as explicit request fields for old output readers.
                "trust_gate_action": route,
                "joint_gate_action": parsed,
                "mapped_sign_action": mapped_action,
                "release_fraction": release_fraction,
                "requested_gate_action": route,
                "requested_joint_action": parsed,
                "requested_sign_action": mapped_action,
                "requested_release_fraction": release_fraction,
                "actual_gate_action": actual_gate_action,
                "actual_joint_action": actual_joint_action,
                "actual_sign_action": actual_sign_action,
                "actual_release_fraction": actual_release_fraction,
                "llm_target_exit": "north" if self.current_advice.target_branch == NORTH else "south",
            }
        )
        return self._augment(observation, enriched), reward, terminated, truncated, enriched
