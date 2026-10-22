"""Action-relevant risk rankings for the v2 LLM/MAPPO interface.

The public contract ranks the original nine executable actions.  Scores are
bounded decision aids, not calibrated casualty probabilities and not masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from typing import Any

import numpy as np


ACTION_IDS = (
    "wait",
    "north_025", "north_050", "north_075", "north_100",
    "south_025", "south_050", "south_075", "south_100",
)
REASON_CODES = (
    "smoke_risk",
    "congestion_risk",
    "blockage_risk",
    "capacity_tradeoff",
    "wait_for_information",
)
FACTOR_LEVELS = {
    "severity": ("low", "high"),
    "trend": ("falling", "rising"),
    "hazard_side": ("north", "south"),
    "blockage": ("none", "north_closed"),
}


@dataclass(frozen=True)
class ActionRisk:
    action_id: str
    risk_score: float
    rank: int


@dataclass(frozen=True)
class ActionRanking:
    actions: tuple[ActionRisk, ...]
    reason_codes: tuple[str, ...]

    @property
    def best_action(self) -> str:
        return min(self.actions, key=lambda item: item.rank).action_id


def _ranking_from_scores(scores: Mapping[str, float], reasons: Sequence[str]) -> ActionRanking:
    if set(scores) != set(ACTION_IDS):
        raise ValueError("scores must cover the exact action set")
    clipped = {action: float(np.clip(scores[action], 0.0, 1.0)) for action in ACTION_IDS}
    ordered = sorted(ACTION_IDS, key=lambda action: (clipped[action], action))
    ranks = {action: index + 1 for index, action in enumerate(ordered)}
    return ActionRanking(
        tuple(ActionRisk(action, clipped[action], ranks[action]) for action in ACTION_IDS),
        tuple(reasons),
    )


def parse_action_ranking(payload: str | Mapping[str, Any]) -> ActionRanking:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    if set(data) != {"actions", "reason_codes"}:
        raise ValueError("action ranking schema mismatch")
    raw_actions = data["actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) != 9:
        raise ValueError("ranking must contain exactly nine actions")
    parsed: dict[str, ActionRisk] = {}
    ranks: list[int] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping) or set(raw) != {"action_id", "risk_score", "rank"}:
            raise ValueError("action entry schema mismatch")
        action = str(raw["action_id"])
        score = float(raw["risk_score"])
        rank = int(raw["rank"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("risk_score must be finite and between zero and one")
        if rank < 1 or rank > 9:
            raise ValueError("rank must be between one and nine")
        if action in parsed:
            raise ValueError("ranking must cover the exact action set")
        parsed[action] = ActionRisk(action, score, rank)
        ranks.append(rank)
    if set(parsed) != set(ACTION_IDS):
        raise ValueError("ranking must cover the exact action set")
    if sorted(ranks) != list(range(1, 10)):
        raise ValueError("rank values must be unique integers one through nine")
    ordered = sorted(parsed.values(), key=lambda item: item.rank)
    if any(ordered[index].risk_score > ordered[index + 1].risk_score + 1e-12
           for index in range(8)):
        raise ValueError("rank order must agree with nondecreasing risk scores")
    raw_reasons = data["reason_codes"]
    if (not isinstance(raw_reasons, list) or not 1 <= len(raw_reasons) <= 3
            or len(set(raw_reasons)) != len(raw_reasons)
            or any(str(value) not in REASON_CODES for value in raw_reasons)):
        raise ValueError("invalid reason_codes")
    return ActionRanking(tuple(parsed[action] for action in ACTION_IDS),
                         tuple(str(value) for value in raw_reasons))


def ranking_payload(ranking: ActionRanking) -> dict[str, Any]:
    return {
        "actions": [
            {"action_id": item.action_id, "risk_score": item.risk_score, "rank": item.rank}
            for item in ranking.actions
        ],
        "reason_codes": list(ranking.reason_codes),
    }


def _state_components(state: Mapping[str, Any]) -> tuple[list[float], list[float], tuple[int, int], float]:
    occupancy = [float(value) for value in state.get("occupancy", ())]
    smoke = [float(value) for value in state.get("smoke", ())]
    if len(occupancy) != 5 or len(smoke) != 5:
        raise ValueError("five-zone occupancy and smoke observations are required")
    capacities = (int(state.get("north_exit_capacity", 0)),
                  int(state.get("south_exit_capacity", 0)))
    context = dict(state.get("semantic_context") or {})
    trend = float(context.get("junction_occupancy_change_per_step", 0.0))
    return occupancy, smoke, capacities, trend


def numeric_action_ranking(state: Mapping[str, Any]) -> ActionRanking:
    """Frozen transparent continuous scorer using exactly the observable state."""
    occupancy, smoke, capacities, trend = _state_components(state)
    junction = float(np.clip(occupancy[2], 0.0, 1.0))
    positive_trend = float(np.clip(trend, 0.0, 0.10)) / 0.10
    scores: dict[str, float] = {
        "wait": float(np.clip(0.62 - 0.55 * junction - 0.18 * positive_trend, 0.05, 0.95))
    }
    for route_index, route in enumerate(("north", "south")):
        capacity = capacities[route_index]
        exit_smoke = float(np.clip(smoke[3 + route_index], 0.0, 0.12)) / 0.12
        for release in (0.25, 0.50, 0.75, 1.00):
            action = f"{route}_{int(release * 100):03d}"
            if capacity <= 0:
                scores[action] = 1.0
                continue
            congestion_pressure = junction + 0.45 * positive_trend
            score = (
                0.55 * exit_smoke
                + 0.28 * congestion_pressure * release
                + 0.10 / max(1, capacity)
                + 0.07 * (1.0 - release)
            )
            scores[action] = float(np.clip(score, 0.0, 0.98))
    return _ranking_from_scores(scores, ("smoke_risk", "congestion_risk", "capacity_tradeoff"))


def rule_action_ranking(state: Mapping[str, Any]) -> ActionRanking:
    """Frozen lexicographic-style rule comparator encoded on the same score scale."""
    occupancy, smoke, capacities, trend = _state_components(state)
    junction_high = occupancy[2] >= 0.35
    rising = trend > 0.0
    smoke_low = min(smoke[3], smoke[4])
    scores: dict[str, float] = {
        "wait": 0.18 if junction_high and rising else 0.72,
    }
    for route_index, route in enumerate(("north", "south")):
        blocked = capacities[route_index] <= 0
        smoky = smoke[3 + route_index] > smoke_low + 0.015
        for release in (0.25, 0.50, 0.75, 1.00):
            action = f"{route}_{int(release * 100):03d}"
            if blocked:
                score = 1.0
            else:
                score = 0.08 + (0.44 if smoky else 0.0)
                if release >= 0.75 and junction_high:
                    score += 0.18
                if release >= 0.75 and rising:
                    score += 0.14
                score += 0.05 * (1.0 - release)
                score += 0.03 / max(1, capacities[route_index])
            scores[action] = float(np.clip(score, 0.0, 1.0))
    return _ranking_from_scores(scores, ("blockage_risk", "smoke_risk", "congestion_risk"))


def build_ranking_factorial(
    base_states: Sequence[Mapping[str, Any]], seed: int = 20261023
) -> list[dict[str, Any]]:
    """Create a randomized complete 2^4 action-ranking gate within each block."""
    if not base_states:
        raise ValueError("at least one base state is required")
    rows: list[dict[str, Any]] = []
    for base in base_states:
        block = str(base["original_scenario_id"])
        original_occupancy = [float(value) for value in base["occupancy"]]
        if len(original_occupancy) != 5:
            raise ValueError("base occupancy must have five zones")
        total = sum(original_occupancy)
        corridor_total = original_occupancy[3] + original_occupancy[4]
        original_north = int(base["north_exit_capacity"])
        if original_north <= 0:
            raise ValueError("base north exit must be open before factorial blockage")
        for severity, trend, hazard_side, blockage in product(
            FACTOR_LEVELS["severity"], FACTOR_LEVELS["trend"],
            FACTOR_LEVELS["hazard_side"], FACTOR_LEVELS["blockage"],
        ):
            state = deepcopy(dict(base))
            junction = 0.20 if severity == "low" else 0.45
            rooms = total - corridor_total - junction
            if rooms < 0:
                raise ValueError("factorial junction level exceeds total occupancy")
            state["occupancy"] = [rooms / 2.0, rooms / 2.0, junction,
                                  original_occupancy[3], original_occupancy[4]]
            background = min(float(value) for value in base["smoke"])
            state["smoke"] = [background, background, background,
                              background + (0.055 if hazard_side == "north" else 0.005),
                              background + (0.055 if hazard_side == "south" else 0.005)]
            state["north_exit_capacity"] = 0 if blockage == "north_closed" else original_north
            state["blockage_active"] = blockage == "north_closed"
            state["blockage_steps_remaining"] = 4 if blockage == "north_closed" else 0
            change = -0.05 if trend == "falling" else 0.05
            context = dict(state.get("semantic_context") or {})
            context.update({
                "junction_occupancy_previous": junction - change,
                "junction_occupancy_change_per_step": change,
                "short_horizon_steps": 4,
                "forecast_source": "experimental sensor-fusion estimate",
                "future_outcomes_supplied": False,
            })
            state["semantic_context"] = context
            identity = {"block": block, "severity": severity, "trend": trend,
                        "hazard_side": hazard_side, "blockage": blockage}
            opaque = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
            state["scenario_id"] = f"RANK_{opaque}"
            rows.append({
                "design_id": f"{block}_{severity}_{trend}_{hazard_side}_{blockage}",
                "block_id": block,
                "severity": severity,
                "trend": trend,
                "hazard_side": hazard_side,
                "blockage": blockage,
                "state": state,
            })
    rng = np.random.default_rng(seed)
    return [rows[int(index)] for index in rng.permutation(len(rows))]


def _risk_map(ranking: ActionRanking | Mapping[str, Any]) -> dict[str, float]:
    parsed = ranking if isinstance(ranking, ActionRanking) else parse_action_ranking(ranking)
    return {item.action_id: float(item.risk_score) for item in parsed.actions}


def _release_gap(ranking: ActionRanking | Mapping[str, Any]) -> float:
    risks = _risk_map(ranking)
    high = [risks[f"{route}_{suffix}"] for route in ("north", "south") for suffix in ("075", "100")]
    low = [risks[f"{route}_{suffix}"] for route in ("north", "south") for suffix in ("025", "050")]
    return float(np.mean(high) - np.mean(low))


def _paired_differences(
    records: Sequence[Mapping[str, Any]], factor: str, low: str, high: str,
    measure,
) -> list[float]:
    other = [name for name in FACTOR_LEVELS if name != factor]
    groups: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["block_id"]), *(str(row[name]) for name in other))
        groups.setdefault(key, {})[str(row[factor])] = row
    if not groups or any(low not in pair or high not in pair for pair in groups.values()):
        raise ValueError(f"incomplete matched pairs for {factor}")
    return [float(measure(pair[high])) - float(measure(pair[low])) for pair in groups.values()]


def ranking_signature(ranking: ActionRanking) -> str:
    ordered = sorted(ranking.actions, key=lambda item: item.rank)
    return ">".join(item.action_id for item in ordered)


def ranking_sensitivity(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute the frozen action-relevant matched-pair gate metrics."""
    north_actions = tuple(action for action in ACTION_IDS if action.startswith("north_"))

    def north_risk(row):
        risks = _risk_map(row["ranking"])
        return float(np.mean([risks[action] for action in north_actions]))

    blockage = _paired_differences(records, "blockage", "none", "north_closed", north_risk)
    severity = _paired_differences(
        records, "severity", "low", "high", lambda row: _release_gap(row["ranking"])
    )
    trend = _paired_differences(
        records, "trend", "falling", "rising", lambda row: _release_gap(row["ranking"])
    )

    safe_flips: list[float] = []
    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        if str(row["blockage"]) != "none":
            continue
        key = (str(row["block_id"]), str(row["severity"]), str(row["trend"]))
        groups.setdefault(key, {})[str(row["hazard_side"])] = row
    if not groups or any("north" not in pair or "south" not in pair for pair in groups.values()):
        raise ValueError("incomplete unblocked hazard-side pairs")
    for pair in groups.values():
        north_hazard = pair["north"]["ranking"].best_action
        south_hazard = pair["south"]["ranking"].best_action
        safe_flips.append(float(north_hazard.startswith("south_") and south_hazard.startswith("north_")))

    return {
        "blockage": {
            "pairs": float(len(blockage)),
            "north_risk_increase": float(np.mean(blockage)),
        },
        "severity": {
            "pairs": float(len(severity)),
            "high_minus_low_release_gap": float(np.mean(severity)),
        },
        "trend": {
            "pairs": float(len(trend)),
            "high_minus_low_release_gap": float(np.mean(trend)),
        },
        "hazard_side": {
            "pairs_unblocked": float(len(safe_flips)),
            "safe_route_flip_fraction_unblocked": float(np.mean(safe_flips)),
        },
    }


def ranking_safety_violations(ranking: ActionRanking, state: Mapping[str, Any]) -> list[str]:
    best = ranking.best_action
    violations: list[str] = []
    if best.startswith("north_") and int(state.get("north_exit_capacity", 0)) <= 0:
        violations.append("best_action_targets_closed_north")
    if best.startswith("south_") and int(state.get("south_exit_capacity", 0)) <= 0:
        violations.append("best_action_targets_closed_south")
    return violations
