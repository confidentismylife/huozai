"""Frozen factorial design and response metrics for congestion-plan sensitivity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
from typing import Any

import numpy as np

from .deepseek_advice import EvacuationPlan


LEVELS = {
    "severity": ("low", "high"),
    "trend": ("falling", "rising"),
    "duration": ("short", "long"),
}


def build_factorial_states(
    base_states: Sequence[Mapping[str, Any]], seed: int = 20261022
) -> list[dict[str, Any]]:
    """Build and randomize a complete 2^3 design within every physical block."""
    if not base_states:
        raise ValueError("at least one base state is required")
    rows: list[dict[str, Any]] = []
    for base in base_states:
        block_id = str(base["original_scenario_id"])
        occupancy = [float(value) for value in base["occupancy"]]
        if len(occupancy) != 5:
            raise ValueError("base occupancy must contain five zones")
        total = sum(occupancy)
        corridor_total = occupancy[3] + occupancy[4]
        for severity, trend, duration in product(
            LEVELS["severity"], LEVELS["trend"], LEVELS["duration"]
        ):
            current = 0.20 if severity == "low" else 0.45
            change = -0.05 if trend == "falling" else 0.05
            remaining_rooms = total - corridor_total - current
            if remaining_rooms < 0:
                raise ValueError("factor level exceeds the base state's occupied population")
            state = deepcopy(dict(base))
            state["occupancy"] = [remaining_rooms / 2, remaining_rooms / 2, current,
                                  occupancy[3], occupancy[4]]
            context = dict(state.get("semantic_context") or {})
            context.update({
                "junction_occupancy_previous": current - change,
                "junction_occupancy_change_per_step": change,
                "congestion_forecast_steps_remaining": 2 if duration == "short" else 8,
                "congestion_forecast_source": "experimental sensor-fusion estimate",
                "future_outcomes_supplied": False,
            })
            state["semantic_context"] = context
            opaque = hashlib.sha256(json.dumps(
                {"block": block_id, "severity": severity, "trend": trend, "duration": duration},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:16]
            state["scenario_id"] = f"SENS_{opaque}"
            design_id = f"{block_id}_{severity}_{trend}_{duration}"
            rows.append({
                "design_id": design_id,
                "block_id": block_id,
                "severity": severity,
                "trend": trend,
                "duration": duration,
                "state": state,
            })
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


def plan_metrics(plan: EvacuationPlan) -> dict[str, Any]:
    """Encode capacity, waiting, timing and plan identity without simulation outcomes."""
    effective = [0.0 if phase.route == "wait" else float(phase.release_fraction)
                 for phase in plan.phases]
    durations = [int(phase.duration_steps) for phase in plan.phases]
    total = sum(durations)
    signature_payload = {
        "phases": [
            {"route": phase.route, "release_fraction": phase.release_fraction,
             "duration_steps": phase.duration_steps}
            for phase in plan.phases
        ],
        "confidence": plan.confidence,
        "reason_codes": list(plan.reason_codes),
        "replan_triggers": list(plan.replan_triggers),
    }
    return {
        "first_effective_release": effective[0],
        "weighted_effective_release": sum(value * duration for value, duration in zip(effective, durations)) / total,
        "wait_steps": sum(duration for phase, duration in zip(plan.phases, durations) if phase.route == "wait"),
        "total_plan_steps": total,
        "route_changes": sum(
            plan.phases[index].route != plan.phases[index - 1].route
            for index in range(1, len(plan.phases))
        ),
        "confidence": float(plan.confidence),
        "plan_signature": json.dumps(signature_payload, sort_keys=True, separators=(",", ":")),
    }


def factor_sensitivity(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute matched within-block response changes for each two-level factor."""
    results: dict[str, dict[str, float]] = {}
    factors = tuple(LEVELS)
    for factor in factors:
        low, high = LEVELS[factor]
        other = [name for name in factors if name != factor]
        groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
        for record in records:
            key = (str(record["block_id"]), str(record[other[0]]), str(record[other[1]]))
            groups.setdefault(key, {})[str(record[factor])] = record
        if not groups or any(low not in pair or high not in pair for pair in groups.values()):
            raise ValueError(f"incomplete matched pairs for {factor}")
        changes = []
        release_differences = []
        wait_differences = []
        for pair in groups.values():
            lower, upper = pair[low], pair[high]
            changes.append(str(lower["plan_signature"]) != str(upper["plan_signature"]))
            release_differences.append(
                float(upper["weighted_effective_release"])
                - float(lower["weighted_effective_release"])
            )
            wait_differences.append(float(upper["wait_steps"]) - float(lower["wait_steps"]))
        results[factor] = {
            "pairs": float(len(changes)),
            "paired_plan_change_fraction": float(np.mean(changes)),
            "weighted_release_high_minus_low": float(np.mean(release_differences)),
            "wait_steps_high_minus_low": float(np.mean(wait_differences)),
        }
    return results
