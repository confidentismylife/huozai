import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.action_risk_ranking_v2 import numeric_action_ranking
from fire_evac.semantic_constraints_v3 import (
    RouteConstraint,
    SemanticConstraints,
    build_semantic_gate_cases,
    compile_ranking_constraints,
    constraint_payload,
    parse_semantic_constraints,
    rule_extract_constraints,
    score_constraint_predictions,
)


def example() -> SemanticConstraints:
    return SemanticConstraints(
        north=RouteConstraint("closed", 0.0, False),
        south=RouteConstraint("restricted", 0.5, True),
        valid_for_steps=4,
        reason_codes=("explicit_closure", "capacity_limit", "sensor_fault"),
    )


def test_parser_enforces_status_capacity_semantics_and_exact_schema() -> None:
    parsed = parse_semantic_constraints(constraint_payload(example()))
    assert parsed == example()

    invalid = constraint_payload(example())
    invalid["north"]["max_release"] = 0.5
    with pytest.raises(ValueError, match="closed"):
        parse_semantic_constraints(invalid)

    invalid = constraint_payload(example())
    invalid["south"]["max_release"] = 1.0
    with pytest.raises(ValueError, match="restricted"):
        parse_semantic_constraints(invalid)


def test_gate_has_six_truth_blocks_four_styles_and_seeded_order() -> None:
    cases = build_semantic_gate_cases(seed=17)
    assert len(cases) == 24
    assert len({case["case_id"] for case in cases}) == 24
    assert {case["style"] for case in cases} == {"direct", "operational", "negation", "correction"}
    for truth_id in {case["truth_id"] for case in cases}:
        assert sum(case["truth_id"] == truth_id for case in cases) == 4
    assert [case["case_id"] for case in cases] != sorted(case["case_id"] for case in cases)
    serialized = json.dumps([
        {**case, "truth": constraint_payload(case["truth"])} for case in cases
    ]).lower()
    assert "reward" not in serialized and "incapacitation" not in serialized


def test_rule_parser_handles_direct_and_latest_update_examples() -> None:
    cases = build_semantic_gate_cases(seed=17)
    direct = next(case for case in cases if case["truth_id"] == "north_closed" and case["style"] == "direct")
    correction = next(case for case in cases if case["truth_id"] == "south_restricted" and case["style"] == "correction")
    assert rule_extract_constraints(direct["incident_report"]) == direct["truth"]
    assert rule_extract_constraints(correction["incident_report"]) == correction["truth"]


def test_compiler_never_prefers_closed_or_over_limit_actions() -> None:
    state = {
        "occupancy": [0.3, 0.3, 0.2, 0.09, 0.09],
        "smoke": [0.002, 0.002, 0.002, 0.001, 0.08],
        "north_exit_capacity": 8,
        "south_exit_capacity": 8,
        "semantic_context": {"junction_occupancy_change_per_step": 0.0},
    }
    base = numeric_action_ranking(state)
    compiled = compile_ranking_constraints(base, example())
    assert not compiled.best_action.startswith("north_")
    assert compiled.best_action in {"wait", "south_025", "south_050"}


def test_prediction_scoring_separates_exact_and_field_accuracy() -> None:
    truth = example()
    predictions = [
        {"truth": truth, "prediction": truth, "style": "direct"},
        {"truth": truth, "prediction": SemanticConstraints(
            truth.north, RouteConstraint("restricted", 0.25, True), 4, truth.reason_codes
        ), "style": "correction"},
    ]
    result = score_constraint_predictions(predictions)
    assert result["exact_accuracy"] == 0.5
    assert result["route_status_accuracy"] == 1.0
    assert result["max_release_accuracy"] == 0.75
    assert result["sensor_reliability_accuracy"] == 1.0
    assert result["correction_exact_accuracy"] == 0.0
