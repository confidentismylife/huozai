from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np

from .hazard import FireScenario, SyntheticHazardProvider


NORTH = 0
SOUTH = 1


@dataclass(frozen=True)
class DynamicLLMAdvice:
    target_branch: int
    confidence: float
    reason_codes: tuple[str, ...]
    raw_response: str = ""


ALLOWED_DYNAMIC_REASONS = {
    "avoid_predicted_smoke",
    "early_guidance",
    "reduce_exposure",
    "balance_congestion",
}


def make_scenario_bank(n: int, seed: int, prefix: str) -> tuple[FireScenario, ...]:
    """Create a balanced, reproducible bank of synthetic fire forecasts."""
    if n < 2:
        raise ValueError("scenario bank must contain at least two scenarios")
    rng = np.random.default_rng(seed)
    branches = np.tile(np.array([NORTH, SOUTH], dtype=int), (n + 1) // 2)[:n]
    rng.shuffle(branches)
    scenarios = []
    for index, branch in enumerate(branches, 1):
        scenarios.append(
            FireScenario(
                scenario_id=f"{prefix}{index:03d}",
                source_branch=int(branch),
                growth_rate=float(rng.uniform(0.010, 0.030)),
                onset_step=int(rng.integers(0, 8)),
                background_hazard=0.002,
            )
        )
    return tuple(scenarios)


class ScenarioBankHazardProvider(SyntheticHazardProvider):
    """Sample only from a fixed scenario bank while retaining synthetic dynamics."""

    def __init__(self, scenarios: Sequence[FireScenario]):
        if not scenarios:
            raise ValueError("scenarios cannot be empty")
        identifiers = [scenario.scenario_id for scenario in scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scenario identifiers must be unique")
        self.scenarios = tuple(scenarios)

    def sample_scenario(self, rng: np.random.Generator, prefix: str = "train") -> FireScenario:
        del prefix
        return self.scenarios[int(rng.integers(0, len(self.scenarios)))]


def dynamic_advice_prompt(scenario: FireScenario, dt_seconds: float = 5.0) -> str:
    source = "north" if scenario.source_branch == NORTH else "south"
    return (
        "You are the high-level planner for a two-exit building evacuation. "
        "A lower-level reinforcement-learning controller will update the dynamic exit sign every control step. "
        "Choose one initial target exit from north or south using the fire forecast. "
        "The main safety objective is to minimize smoke exposure and incapacitation. "
        "The fire source corridor is the corridor where smoke originates and will become dangerous; "
        "it is not an evacuation origin or a recommended route. Because the two exits are otherwise equivalent, "
        "do not target the fire source corridor; target the opposite corridor. "
        "Return only JSON matching the requested schema.\n"
        f"Forecast fire source corridor: {source}.\n"
        f"Smoke growth rate: {scenario.growth_rate:.5f} FED increment per {dt_seconds:.1f} seconds.\n"
        f"Smoke growth begins at control step: {scenario.onset_step}.\n"
        "Initial occupancy is balanced between the two rooms; downstream congestion will be handled by the RL controller."
    )


def parse_dynamic_advice(payload: str | Mapping[str, object]) -> DynamicLLMAdvice:
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("dynamic LLM advice is not valid JSON") from exc
    required = {"target_exit", "confidence", "reason_codes"}
    if set(data) != required:
        raise ValueError("dynamic LLM advice schema mismatch")
    target = data["target_exit"]
    if target not in {"north", "south"}:
        raise ValueError("target_exit must be north or south")
    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")
    reasons = tuple(str(value) for value in data["reason_codes"])
    if not 1 <= len(reasons) <= 3 or any(reason not in ALLOWED_DYNAMIC_REASONS for reason in reasons):
        raise ValueError("invalid dynamic reason_codes")
    return DynamicLLMAdvice(
        target_branch=NORTH if target == "north" else SOUTH,
        confidence=confidence,
        reason_codes=reasons,
        raw_response=json.dumps(data, ensure_ascii=False),
    )


def advice_features(advice: DynamicLLMAdvice) -> np.ndarray:
    return np.asarray(
        [
            float(advice.target_branch == NORTH),
            float(advice.target_branch == SOUTH),
            advice.confidence,
        ],
        dtype=np.float32,
    )


class LLMGuidedObservationWrapper(gym.Wrapper):
    """Append a cached, episode-level LLM target to every dynamic observation.

    The base safety reward is intentionally unchanged. This isolates the value
    of the high-level forecast signal from reward shaping.
    """

    def __init__(self, env: gym.Env, advice_by_scenario: Mapping[str, DynamicLLMAdvice]):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("LLMGuidedObservationWrapper requires a Box observation space")
        self.advice_by_scenario = dict(advice_by_scenario)
        self.current_advice: DynamicLLMAdvice | None = None
        low = np.concatenate([env.observation_space.low, np.zeros(3, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(3, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        if self.current_advice is None:
            raise RuntimeError("reset must be called before observation augmentation")
        return np.concatenate([observation, advice_features(self.current_advice)]).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        scenario_id = str(info["scenario"]["scenario_id"])
        if scenario_id not in self.advice_by_scenario:
            raise KeyError(f"no cached LLM advice for scenario {scenario_id}")
        self.current_advice = self.advice_by_scenario[scenario_id]
        enriched = dict(info)
        enriched.update(self._advice_info())
        return self._augment(observation), enriched

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        enriched = dict(info)
        enriched.update(self._advice_info())
        enriched["action_matches_llm_target"] = bool(int(info["last_action"]) == self.current_advice.target_branch)
        return self._augment(observation), reward, terminated, truncated, enriched

    def _advice_info(self) -> dict[str, object]:
        if self.current_advice is None:
            raise RuntimeError("LLM advice is not initialized")
        return {
            "llm_target_exit": "north" if self.current_advice.target_branch == NORTH else "south",
            "llm_confidence": self.current_advice.confidence,
            "llm_reason_codes": self.current_advice.reason_codes,
        }


def scenario_records(scenarios: Sequence[FireScenario]) -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in scenarios]


def safe_forecast_target(scenario: FireScenario) -> int:
    """Reference label for advice auditing: choose the non-source corridor."""
    if scenario.source_branch not in (NORTH, SOUTH):
        raise ValueError("synthetic scenario must identify north or south as source_branch")
    return SOUTH if scenario.source_branch == NORTH else NORTH
