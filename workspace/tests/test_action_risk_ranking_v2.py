from itertools import product
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_evac.action_risk_ranking_v2 import (
    ACTION_IDS,
    ActionRanking,
    ActionRisk,
    build_ranking_factorial,
    numeric_action_ranking,
    parse_action_ranking,
    ranking_sensitivity,
    rule_action_ranking,
)


def base_state(block: str, north: int = 6, south: int = 10) -> dict:
    return {
        "scenario_id": block,
        "original_scenario_id": block,
        "event_type": "action_ranking_gate",
        "timestamp_step": 4,
        "forecast_source_branch": 0,
        "growth_rate": 0.03,
        "onset_step": 2,
        "occupancy": [0.35, 0.35, 0.20, 0.04, 0.04],
        "smoke": [0.002, 0.002, 0.002, 0.010, 0.030],
        "north_exit_capacity": north,
        "south_exit_capacity": south,
        "blockage_active": False,
        "blockage_steps_remaining": 0,
        "last_release_fraction": 0.75,
        "semantic_context": {},
    }


def valid_payload() -> dict:
    risks = {"wait": 0.60}
    for route, offset in (("north", 0.0), ("south", 0.4)):
        for index, suffix in enumerate(("025", "050", "075", "100")):
            risks[f"{route}_{suffix}"] = 0.10 + offset + 0.08 * index
    ordered = sorted(risks, key=lambda action: (risks[action], action))
    ranks = {action: rank + 1 for rank, action in enumerate(ordered)}
    return {
        "actions": [
            {"action_id": action, "risk_score": risks[action], "rank": ranks[action]}
            for action in ACTION_IDS
        ],
        "reason_codes": ["smoke_risk", "congestion_risk"],
    }


def test_parser_requires_complete_unique_ranking_consistent_with_scores() -> None:
    ranking = parse_action_ranking(json.dumps(valid_payload()))
    assert ranking.best_action == "north_025"
    assert tuple(item.action_id for item in ranking.actions) == ACTION_IDS

    missing = valid_payload()
    missing["actions"].pop()
    with pytest.raises(ValueError, match="nine"):
        parse_action_ranking(missing)

    duplicate = valid_payload()
    duplicate["actions"][-1]["action_id"] = duplicate["actions"][0]["action_id"]
    with pytest.raises(ValueError, match="exact action set"):
        parse_action_ranking(duplicate)

    inconsistent = valid_payload()
    inconsistent["actions"][0]["rank"] = 9
    with pytest.raises(ValueError, match="rank"):
        parse_action_ranking(inconsistent)


def test_rule_and_numeric_rankings_penalize_a_closed_exit_and_cover_same_actions() -> None:
    state = base_state("B1")
    state["north_exit_capacity"] = 0
    state["blockage_active"] = True
    for ranking in (rule_action_ranking(state), numeric_action_ranking(state)):
        assert tuple(item.action_id for item in ranking.actions) == ACTION_IDS
        assert not ranking.best_action.startswith("north_")
        north = [item.risk_score for item in ranking.actions if item.action_id.startswith("north_")]
        south = [item.risk_score for item in ranking.actions if item.action_id.startswith("south_")]
        assert min(north) > max(south)


def test_factorial_is_balanced_randomized_and_contains_no_outcomes() -> None:
    rows = build_ranking_factorial([base_state("B1"), base_state("B2", 4, 9)], seed=19)
    assert len(rows) == 32
    assert len({row["design_id"] for row in rows}) == 32
    for factor, high in (("severity", "high"), ("trend", "rising"),
                         ("hazard_side", "south"), ("blockage", "north_closed")):
        assert sum(row[factor] == high for row in rows) == 16
    canonical = sorted(row["design_id"] for row in rows)
    assert [row["design_id"] for row in rows] != canonical
    for row in rows:
        state = row["state"]
        assert abs(sum(state["occupancy"]) - 0.98) < 1e-12
        serialized = json.dumps(state).lower()
        assert "reward" not in serialized and "incapacitation" not in serialized


def _synthetic_ranking(severity: str, trend: str, hazard_side: str, blockage: str) -> ActionRanking:
    risks = {}
    for action in ACTION_IDS:
        if action == "wait":
            risks[action] = 0.45
            continue
        route, suffix = action.split("_")
        release = int(suffix) / 100
        risk = 0.10
        if route == hazard_side:
            risk += 0.20
        if blockage == "north_closed" and route == "north":
            risk += 0.40
        if release >= 0.75 and severity == "high":
            risk += 0.12
        if release >= 0.75 and trend == "rising":
            risk += 0.08
        risks[action] = min(risk, 1.0)
    ordered = sorted(ACTION_IDS, key=lambda action: (risks[action], action))
    ranks = {action: index + 1 for index, action in enumerate(ordered)}
    return ActionRanking(
        tuple(ActionRisk(action, risks[action], ranks[action]) for action in ACTION_IDS),
        ("smoke_risk",),
    )


def test_sensitivity_uses_matched_factorial_pairs_on_action_relevant_scores() -> None:
    records = []
    for severity, trend, hazard, blockage in product(
        ("low", "high"), ("falling", "rising"),
        ("north", "south"), ("none", "north_closed")
    ):
        ranking = _synthetic_ranking(severity, trend, hazard, blockage)
        records.append({
            "block_id": "B1", "severity": severity, "trend": trend,
            "hazard_side": hazard, "blockage": blockage, "ranking": ranking,
        })
    metrics = ranking_sensitivity(records)
    assert metrics["blockage"]["north_risk_increase"] == pytest.approx(0.40)
    assert metrics["severity"]["high_minus_low_release_gap"] == pytest.approx(0.12)
    assert metrics["trend"]["high_minus_low_release_gap"] == pytest.approx(0.08)
    assert metrics["hazard_side"]["safe_route_flip_fraction_unblocked"] == 1.0
