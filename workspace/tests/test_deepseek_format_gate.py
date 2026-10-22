from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase
from fire_evac.deepseek_format_gate import build_format_gate_states, plan_safety_violations


def test_format_gate_states_are_reproducible_and_cover_events() -> None:
    first = build_format_gate_states(12, seed=20261021)
    second = build_format_gate_states(12, seed=20261021)
    assert first == second
    assert len(first) == 12
    assert {state["event_type"] for state in first} == {
        "initial",
        "blockage_change",
        "smoke_reversal",
        "congestion_threshold",
    }
    assert len({state["scenario_id"] for state in first}) == 12


def test_format_gate_detects_immediate_blocked_route() -> None:
    state = next(
        state
        for state in build_format_gate_states(12, seed=20261021)
        if state["north_exit_capacity"] == 0 or state["south_exit_capacity"] == 0
    )
    blocked_route = "north" if state["north_exit_capacity"] == 0 else "south"
    plan = EvacuationPlan(
        phases=(
            PlanPhase(blocked_route, 1.0, 2),
            PlanPhase("wait", 0.25, 1),
            PlanPhase("wait", 0.25, 1),
        ),
        confidence=0.9,
        reason_codes=("respect_blockage",),
        replan_triggers=("blockage_change",),
    )
    assert plan_safety_violations(plan, state) == ["first_phase_targets_blocked_exit"]
