from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase
from fire_evac.frozen_plan_library import FrozenPlanLibrary


def write_library(tmp_path: Path, valid: bool = True):
    states = [
        {
            "scenario_id": "A",
            "event_type": "initial",
            "forecast_source_branch": 0,
            "growth_rate": 0.01,
            "onset_step": 2,
            "occupancy": [0.4, 0.4, 0.05, 0.075, 0.075],
            "smoke": [0.002] * 5,
            "north_exit_capacity": 8,
            "south_exit_capacity": 4,
            "blockage_steps_remaining": 0,
            "last_release_fraction": 1.0,
        },
        {
            "scenario_id": "B",
            "event_type": "initial",
            "forecast_source_branch": 1,
            "growth_rate": 0.04,
            "onset_step": 8,
            "occupancy": [0.2, 0.2, 0.3, 0.15, 0.15],
            "smoke": [0.002, 0.002, 0.002, 0.01, 0.08],
            "north_exit_capacity": 3,
            "south_exit_capacity": 8,
            "blockage_steps_remaining": 0,
            "last_release_fraction": 0.25,
        },
    ]
    plans = [
        EvacuationPlan(
            (PlanPhase("south", 0.5, 2), PlanPhase("south", 0.75, 2), PlanPhase("north", 0.25, 1)),
            0.7,
            ("avoid_predicted_smoke",),
            ("smoke_reversal",),
        ),
        EvacuationPlan(
            (PlanPhase("north", 0.25, 2), PlanPhase("north", 0.5, 2), PlanPhase("wait", 0.25, 1)),
            0.8,
            ("balance_congestion",),
            ("congestion_threshold",),
        ),
    ]
    audit = []
    for state, plan in zip(states, plans):
        audit.append(
            {
                "scenario_id": state["scenario_id"],
                "event_type": state["event_type"],
                "valid_model_response": valid,
                "plan": {
                    "phases": [asdict(phase) for phase in plan.phases],
                    "confidence": plan.confidence,
                    "reason_codes": list(plan.reason_codes),
                    "replan_triggers": list(plan.replan_triggers),
                },
            }
        )
    states_path = tmp_path / "states.json"
    audit_path = tmp_path / "audit.json"
    states_path.write_text(json.dumps(states), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return states, plans, states_path, audit_path


def test_exact_state_retrieves_its_frozen_deepseek_plan(tmp_path: Path) -> None:
    states, plans, states_path, audit_path = write_library(tmp_path)
    library = FrozenPlanLibrary.from_files(states_path, audit_path)
    result = library(states[1])
    assert result == plans[1]
    assert library.last_match == {
        "query_scenario_id": "B",
        "matched_scenario_id": "B",
        "event_type": "initial",
        "squared_distance": 0.0,
        "candidate_count": 2,
    }


def test_library_rejects_non_model_fallback_records(tmp_path: Path) -> None:
    _, _, states_path, audit_path = write_library(tmp_path, valid=False)
    try:
        FrozenPlanLibrary.from_files(states_path, audit_path)
    except ValueError as exc:
        assert "valid" in str(exc).lower()
    else:
        raise AssertionError("invalid DeepSeek audit must be rejected")
