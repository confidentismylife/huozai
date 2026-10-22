"""Behavioral tests for the frozen congestion-sensitivity design."""

from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.congestion_sensitivity import (
    build_factorial_states,
    factor_sensitivity,
    plan_metrics,
)
from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase


def base_state(scenario_id: str, north_capacity: int) -> dict:
    return {
        "scenario_id": f"source-{scenario_id}",
        "original_scenario_id": scenario_id,
        "event_type": "congestion_threshold",
        "timestamp_step": 3,
        "forecast_source_branch": 1,
        "growth_rate": 0.02,
        "onset_step": 4,
        "occupancy": [0.375, 0.375, 0.2, 0.0125, 0.0125],
        "smoke": [0.005] * 5,
        "north_exit_capacity": north_capacity,
        "south_exit_capacity": 8,
        "blockage_active": False,
        "blockage_steps_remaining": 0,
        "last_release_fraction": 1.0,
        "semantic_context": {},
    }


def test_factorial_design_is_balanced_and_preserves_each_blocks_population() -> None:
    states = build_factorial_states([base_state("DSG0017", 10), base_state("DSG0018", 1)])

    assert len(states) == 16
    assert len({row["design_id"] for row in states}) == 16
    assert sum(row["severity"] == "high" for row in states) == 8
    assert sum(row["trend"] == "rising" for row in states) == 8
    assert sum(row["duration"] == "long" for row in states) == 8
    for row in states:
        state = row["state"]
        assert abs(sum(state["occupancy"]) - 0.975) < 1e-12
        expected = 0.45 if row["severity"] == "high" else 0.20
        assert state["occupancy"][2] == expected
        change = state["semantic_context"]["junction_occupancy_change_per_step"]
        assert change == (0.05 if row["trend"] == "rising" else -0.05)
        assert "incapacitation" not in state and "reward" not in state


def test_factorial_run_order_is_seeded_and_not_canonical() -> None:
    bases = [base_state("DSG0017", 10), base_state("DSG0018", 1)]
    first = build_factorial_states(bases, seed=19)
    second = build_factorial_states(bases, seed=19)
    canonical = sorted(row["design_id"] for row in first)

    assert [row["design_id"] for row in first] == [row["design_id"] for row in second]
    assert [row["design_id"] for row in first] != canonical


def test_plan_metrics_treat_wait_as_zero_effective_release() -> None:
    plan = EvacuationPlan(
        phases=(
            PlanPhase("north", 0.5, 2),
            PlanPhase("wait", 0.25, 1),
            PlanPhase("south", 1.0, 1),
        ),
        confidence=0.7,
        reason_codes=("balance_congestion",),
        replan_triggers=("congestion_threshold",),
    )

    metrics = plan_metrics(plan)

    assert metrics["first_effective_release"] == 0.5
    assert metrics["weighted_effective_release"] == 0.5
    assert metrics["wait_steps"] == 1
    assert metrics["total_plan_steps"] == 4
    assert metrics["route_changes"] == 2


def test_factor_sensitivity_uses_matched_pairs_within_block() -> None:
    rows = []
    for severity, trend, duration in product(("low", "high"), ("falling", "rising"), ("short", "long")):
        release = 0.25 if severity == "high" else 0.75
        rows.append({
            "block_id": "B1", "severity": severity, "trend": trend, "duration": duration,
            "plan_signature": f"severity-{severity}",
            "weighted_effective_release": release,
            "wait_steps": 0,
        })

    result = factor_sensitivity(rows)

    assert result["severity"]["paired_plan_change_fraction"] == 1.0
    assert result["trend"]["paired_plan_change_fraction"] == 0.0
    assert result["duration"]["paired_plan_change_fraction"] == 0.0
    assert result["severity"]["weighted_release_high_minus_low"] == -0.5
