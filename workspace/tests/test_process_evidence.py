"""Behavioral tests for the evacuation process evidence audit."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase
from fire_evac.process_evidence import (
    active_plan_decision,
    cyclically_permute_plans,
    select_training_event_states,
    summarize_episode_progress,
)


def test_progress_keeps_successful_evacuation_separate_from_early_incapacitation() -> None:
    records = [
        {"step": 1, "evacuated": 10, "incapacitated": 0, "remaining": 90},
        {"step": 2, "evacuated": 50, "incapacitated": 10, "remaining": 40},
        {"step": 3, "evacuated": 70, "incapacitated": 20, "remaining": 10},
    ]

    result = summarize_episode_progress(records, population=100, step_seconds=5)

    assert result == {
        "episode_end_time_seconds": 15.0,
        "final_successful_evacuation_rate": 0.7,
        "final_incapacitation_rate": 0.2,
        "final_remaining_rate": 0.1,
        "time_to_50pct_success_seconds": 10.0,
        "time_to_80pct_success_seconds": None,
        "time_to_90pct_success_seconds": None,
        "successful_evacuation_progress_auc": 13.0 / 30.0,
        "end_reason": "time_limit_or_truncation",
    }


def test_progress_rejects_population_balance_that_would_mix_outcomes() -> None:
    records = [{"step": 1, "evacuated": 40, "incapacitated": 20, "remaining": 30}]

    try:
        summarize_episode_progress(records, population=100)
    except ValueError as exc:
        assert "population balance" in str(exc)
    else:
        raise AssertionError("invalid outcome accounting must be rejected")


def test_active_plan_decision_uses_phase_timing_and_joint_action_semantics() -> None:
    plan = EvacuationPlan(
        phases=(
            PlanPhase("south", 0.50, 2),
            PlanPhase("north", 0.75, 3),
            PlanPhase("wait", 0.25, 4),
        ),
        confidence=0.8,
        reason_codes=("balance_congestion",),
        replan_triggers=("congestion_threshold",),
    )

    assert active_plan_decision(plan, age_steps=0) == {
        "plan_phase_index": 0,
        "plan_route": "south",
        "plan_release_fraction": 0.50,
        "plan_joint_action": 5,
    }
    assert active_plan_decision(plan, age_steps=2)["plan_joint_action"] == 2
    assert active_plan_decision(plan, age_steps=5)["plan_joint_action"] == 8


def test_training_event_selection_uses_first_congestion_state_per_scenario() -> None:
    records = [
        {"scenario_id": "DSG0002", "event_type": "congestion_threshold", "timestamp_step": 8},
        {"scenario_id": "DSG0001", "event_type": "smoke_reversal", "timestamp_step": 1},
        {"scenario_id": "DSG0002", "event_type": "congestion_threshold", "timestamp_step": 3},
        {"scenario_id": "DSG0001", "event_type": "congestion_threshold", "timestamp_step": 5},
    ]

    selected = select_training_event_states(records, maximum=2)

    assert [(row["scenario_id"], row["timestamp_step"]) for row in selected] == [
        ("DSG0001", 5),
        ("DSG0002", 3),
    ]


def test_plan_permutation_keeps_states_but_rotates_content() -> None:
    records = [
        ({"scenario_id": "A"}, "plan-A"),
        ({"scenario_id": "B"}, "plan-B"),
        ({"scenario_id": "C"}, "plan-C"),
    ]

    permuted = cyclically_permute_plans(records)

    assert permuted == [
        ({"scenario_id": "A"}, "plan-B"),
        ({"scenario_id": "B"}, "plan-C"),
        ({"scenario_id": "C"}, "plan-A"),
    ]
