from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np


LEFT = 0
RIGHT = 1
NEUTRAL = -1


@dataclass(frozen=True)
class ConflictScenario:
    scenario_id: str
    distance_left: float
    distance_right: float
    smoke_left: float
    smoke_right: float
    crowd_direction: int
    crowd_size: int
    sign_direction: int
    warden_direction: int
    warden_authority: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LLMAdvice:
    recommended_exit: int
    confidence: float
    reason_codes: tuple[str, ...]
    raw_response: str = ""


ALLOWED_LLM_REASONS = {
    "avoid_smoke",
    "shorter_distance",
    "avoid_congestion",
    "follow_sign",
    "follow_crowd",
    "follow_warden",
    "resolve_conflict",
}


def generate_conflict_scenarios(n: int = 48, seed: int = 20260929) -> tuple[ConflictScenario, ...]:
    if n < 4:
        raise ValueError("at least four scenarios are required")
    rng = np.random.default_rng(seed)
    scenarios: list[ConflictScenario] = []
    seen: set[tuple[object, ...]] = set()
    smoke_patterns = ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    distances = ((3.0, 6.0), (6.0, 3.0), (4.0, 4.0), (3.0, 5.0), (5.0, 3.0), (6.0, 6.0))
    attempts = 0
    while len(scenarios) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError("could not generate enough unique scenarios")
        smoke_left, smoke_right = smoke_patterns[len(scenarios) % len(smoke_patterns)]
        distance_left, distance_right = distances[int(rng.integers(0, len(distances)))]
        crowd_direction = int(rng.choice([NEUTRAL, LEFT, RIGHT]))
        crowd_size = 0 if crowd_direction == NEUTRAL else int(rng.choice([1, 10, 20]))
        sign_direction = int(rng.choice([NEUTRAL, LEFT, RIGHT]))
        warden_direction = int(rng.choice([NEUTRAL, LEFT, RIGHT]))
        warden_authority = 0 if warden_direction == NEUTRAL else int(rng.choice([1, 2]))
        key = (
            distance_left,
            distance_right,
            smoke_left,
            smoke_right,
            crowd_direction,
            crowd_size,
            sign_direction,
            warden_direction,
            warden_authority,
        )
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(
            ConflictScenario(
                scenario_id=f"C{len(scenarios) + 1:03d}",
                distance_left=distance_left,
                distance_right=distance_right,
                smoke_left=smoke_left,
                smoke_right=smoke_right,
                crowd_direction=crowd_direction,
                crowd_size=crowd_size,
                sign_direction=sign_direction,
                warden_direction=warden_direction,
                warden_authority=warden_authority,
            )
        )
    order = rng.permutation(len(scenarios))
    return tuple(scenarios[int(i)] for i in order)


def crowd_counts(scenario: ConflictScenario) -> tuple[int, int]:
    if scenario.crowd_direction == LEFT:
        return scenario.crowd_size, 0
    if scenario.crowd_direction == RIGHT:
        return 0, scenario.crowd_size
    return 0, 0


def action_reward(scenario: ConflictScenario, action: int) -> float:
    if action not in (LEFT, RIGHT):
        raise ValueError("action must be LEFT or RIGHT")
    crowd_left, crowd_right = crowd_counts(scenario)
    if action == LEFT:
        smoke, distance, crowd = scenario.smoke_left, scenario.distance_left, crowd_left
    else:
        smoke, distance, crowd = scenario.smoke_right, scenario.distance_right, crowd_right
    return 1.0 - 4.0 * smoke - 0.08 * distance - 0.025 * crowd


def optimal_action(scenario: ConflictScenario) -> int:
    rewards = (action_reward(scenario, LEFT), action_reward(scenario, RIGHT))
    return int(rewards[RIGHT] > rewards[LEFT])


def paper_derived_logit_probability_left(scenario: ConflictScenario) -> float:
    """Fuse coefficients published across the two downloaded experiments.

    Distance/smoke use Lu et al. (2026) later-individual coefficients.
    Crowd/warden/authority use Lu et al. (2025) MNL coefficients.
    No sign coefficient is used because Tong et al. (2026) exact estimates
    are not available without the full paper. This is a transparent composite
    baseline, not a model published verbatim in either paper.
    """

    crowd_left, crowd_right = crowd_counts(scenario)
    utility_left = -0.341 * scenario.distance_left - 1.549 * scenario.smoke_left + 0.035 * crowd_left
    utility_right = 0.263 - 0.341 * scenario.distance_right - 1.549 * scenario.smoke_right + 0.035 * crowd_right
    if scenario.warden_direction == LEFT:
        utility_left += 2.739 + (0.559 if scenario.warden_authority == 2 else 0.0)
    elif scenario.warden_direction == RIGHT:
        utility_right += 2.739 + (0.559 if scenario.warden_authority == 2 else 0.0)
    difference = float(np.clip(utility_left - utility_right, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-difference))


def parse_llm_advice(payload: str | Mapping[str, object]) -> LLMAdvice:
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM advice is not valid JSON") from exc
    required = {"recommended_exit", "confidence", "reason_codes"}
    if set(data) != required:
        raise ValueError("LLM advice schema mismatch")
    exit_name = data["recommended_exit"]
    if exit_name not in {"left", "right"}:
        raise ValueError("recommended_exit must be left or right")
    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    reasons = tuple(str(value) for value in data["reason_codes"])
    if not reasons or len(reasons) > 4 or any(reason not in ALLOWED_LLM_REASONS for reason in reasons):
        raise ValueError("invalid reason_codes")
    return LLMAdvice(LEFT if exit_name == "left" else RIGHT, confidence, reasons, json.dumps(data, ensure_ascii=False))


def conflict_prompt(scenario: ConflictScenario) -> str:
    direction = {NEUTRAL: "none", LEFT: "left", RIGHT: "right"}
    authority = {0: "none", 1: "metro_staff", 2: "firefighter"}
    return (
        "You are making one high-level exit decision in a fire evacuation simulation. "
        "Prioritize avoiding visible smoke, then reduce travel distance and congestion. "
        "Exit signs, crowd movement, and wardens are information cues but may conflict or be wrong. "
        "Return only JSON matching the requested schema.\n"
        f"Left exit: distance={scenario.distance_left:.1f} m, smoke={scenario.smoke_left:.1f}.\n"
        f"Right exit: distance={scenario.distance_right:.1f} m, smoke={scenario.smoke_right:.1f}.\n"
        f"Crowd: direction={direction[scenario.crowd_direction]}, people={scenario.crowd_size}.\n"
        f"Exit sign points: {direction[scenario.sign_direction]}.\n"
        f"Warden points: {direction[scenario.warden_direction]}, authority={authority[scenario.warden_authority]}."
    )


def scenario_observation(scenario: ConflictScenario, advice: LLMAdvice | None = None) -> np.ndarray:
    crowd_left, crowd_right = crowd_counts(scenario)
    sign_left = float(scenario.sign_direction == LEFT)
    sign_right = float(scenario.sign_direction == RIGHT)
    warden_left = float(scenario.warden_direction == LEFT)
    warden_right = float(scenario.warden_direction == RIGHT)
    if advice is None:
        advice_left = advice_right = advice_confidence = 0.0
    else:
        advice_left = float(advice.recommended_exit == LEFT)
        advice_right = float(advice.recommended_exit == RIGHT)
        advice_confidence = advice.confidence
    return np.asarray(
        [
            scenario.distance_left / 10.0,
            scenario.distance_right / 10.0,
            scenario.smoke_left,
            scenario.smoke_right,
            crowd_left / 20.0,
            crowd_right / 20.0,
            sign_left,
            sign_right,
            warden_left,
            warden_right,
            scenario.warden_authority / 2.0,
            advice_left,
            advice_right,
            advice_confidence,
        ],
        dtype=np.float32,
    )


class ConflictChoiceEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: Sequence[ConflictScenario],
        advice: Mapping[str, LLMAdvice] | None = None,
        advice_alignment_bonus: float = 0.0,
    ):
        super().__init__()
        if not scenarios:
            raise ValueError("scenarios cannot be empty")
        self.scenarios = tuple(scenarios)
        self.advice = dict(advice or {})
        if advice_alignment_bonus < 0:
            raise ValueError("advice_alignment_bonus cannot be negative")
        self.advice_alignment_bonus = float(advice_alignment_bonus)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(14,), dtype=np.float32)
        self.current: ConflictScenario | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        super().reset(seed=seed)
        if options and "scenario_id" in options:
            target = str(options["scenario_id"])
            matches = [scenario for scenario in self.scenarios if scenario.scenario_id == target]
            if not matches:
                raise ValueError(f"unknown scenario_id: {target}")
            self.current = matches[0]
        else:
            self.current = self.scenarios[int(self.np_random.integers(0, len(self.scenarios)))]
        llm_advice = self.advice.get(self.current.scenario_id)
        return scenario_observation(self.current, llm_advice), {"scenario_id": self.current.scenario_id}

    def step(self, action: int):
        if self.current is None:
            raise RuntimeError("reset must be called before step")
        action = int(action)
        base_reward = action_reward(self.current, action)
        llm_advice = self.advice.get(self.current.scenario_id)
        advice_bonus = 0.0
        if llm_advice is not None and action == llm_advice.recommended_exit:
            advice_bonus = self.advice_alignment_bonus * llm_advice.confidence
        reward = base_reward + advice_bonus
        info = {
            "scenario_id": self.current.scenario_id,
            "chosen_exit": "left" if action == LEFT else "right",
            "optimal_exit": "left" if optimal_action(self.current) == LEFT else "right",
            "smoke_chosen": self.current.smoke_left if action == LEFT else self.current.smoke_right,
            "base_safety_reward": base_reward,
            "advice_alignment_bonus": advice_bonus,
        }
        return scenario_observation(self.current, self.advice.get(self.current.scenario_id)), reward, True, False, info


def evaluate_probability_policies(
    scenarios: Sequence[ConflictScenario],
    policy_probabilities_left: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for policy_name, probabilities in policy_probabilities_left.items():
        totals = {"reward": 0.0, "optimal": 0.0, "smoke": 0.0, "distance": 0.0, "sign": 0.0, "crowd": 0.0, "warden": 0.0}
        denominators = {"sign": 0, "crowd": 0, "warden": 0}
        for scenario in scenarios:
            p_left = float(np.clip(probabilities[scenario.scenario_id], 0.0, 1.0))
            p_right = 1.0 - p_left
            reward_left = action_reward(scenario, LEFT)
            reward_right = action_reward(scenario, RIGHT)
            optimal = optimal_action(scenario)
            crowd_left, crowd_right = crowd_counts(scenario)
            expected_reward = p_left * reward_left + p_right * reward_right
            optimal_probability = p_left if optimal == LEFT else p_right
            smoke_probability = p_left * scenario.smoke_left + p_right * scenario.smoke_right
            expected_distance = p_left * scenario.distance_left + p_right * scenario.distance_right
            totals["reward"] += expected_reward
            totals["optimal"] += optimal_probability
            totals["smoke"] += smoke_probability
            totals["distance"] += expected_distance
            if scenario.sign_direction != NEUTRAL:
                denominators["sign"] += 1
                totals["sign"] += p_left if scenario.sign_direction == LEFT else p_right
            if scenario.crowd_direction != NEUTRAL:
                denominators["crowd"] += 1
                totals["crowd"] += p_left if crowd_left > crowd_right else p_right
            if scenario.warden_direction != NEUTRAL:
                denominators["warden"] += 1
                totals["warden"] += p_left if scenario.warden_direction == LEFT else p_right
            rows.append(
                {
                    "policy": policy_name,
                    "scenario_id": scenario.scenario_id,
                    "probability_left": p_left,
                    "optimal_exit": "left" if optimal == LEFT else "right",
                    "expected_reward": expected_reward,
                    "optimal_choice_probability": optimal_probability,
                    "smoke_choice_probability": smoke_probability,
                    "expected_distance": expected_distance,
                }
            )
        n = len(scenarios)
        summary.append(
            {
                "policy": policy_name,
                "scenarios": n,
                "mean_reward": totals["reward"] / n,
                "optimal_choice_rate": totals["optimal"] / n,
                "smoke_choice_rate": totals["smoke"] / n,
                "mean_distance": totals["distance"] / n,
                "follow_sign_rate": totals["sign"] / max(1, denominators["sign"]),
                "follow_crowd_rate": totals["crowd"] / max(1, denominators["crowd"]),
                "follow_warden_rate": totals["warden"] / max(1, denominators["warden"]),
            }
        )
    return rows, summary


def save_scenarios(scenarios: Sequence[ConflictScenario], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([scenario.to_dict() for scenario in scenarios], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

