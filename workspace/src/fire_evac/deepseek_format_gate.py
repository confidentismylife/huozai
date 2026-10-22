from __future__ import annotations

from typing import Any, Mapping

from .deepseek_advice import EvacuationPlan
from .dynamic_conflicts import make_domain_randomized_trust_bank


EVENT_TYPES = ("initial", "blockage_change", "smoke_reversal", "congestion_threshold")


def build_format_gate_states(n: int = 12, seed: int = 20261021) -> list[dict[str, Any]]:
    """Build reproducible development-only snapshots covering all trigger types."""
    if n < len(EVENT_TYPES):
        raise ValueError("at least four states are required to cover the event types")
    scenarios = make_domain_randomized_trust_bank(n, seed, "DSG")
    states: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        event = EVENT_TYPES[index % len(EVENT_TYPES)]
        north_capacity = int(scenario.north_exit_capacity)
        south_capacity = int(scenario.south_exit_capacity)
        blocked_branch = int(scenario.blocked_branch)
        blockage_active = False
        blockage_remaining = 0
        if event == "blockage_change":
            if blocked_branch not in (0, 1):
                blocked_branch = 1 - int(scenario.forecast_source_branch)
            if blocked_branch == 0:
                north_capacity = 0
            else:
                south_capacity = 0
            blockage_active = True
            blockage_remaining = 3 + index % 5

        occupancy = [0.40, 0.40, 0.05, 0.075, 0.075]
        if event == "congestion_threshold":
            occupancy = [0.25, 0.25, 0.25, 0.125, 0.125]
        background = float(scenario.background_hazard)
        smoke = [background] * 5
        elapsed_after_onset = max(1, 4 + index % 4)
        actual_source = int(scenario.source_branch)
        smoke[3 + actual_source] = background + float(scenario.growth_rate) * elapsed_after_onset
        if event == "smoke_reversal":
            # An observed reversal intentionally conflicts with the forecast cue.
            observed_source = 1 - int(scenario.forecast_source_branch)
            smoke[3] = background
            smoke[4] = background
            smoke[3 + observed_source] = background + float(scenario.growth_rate) * elapsed_after_onset

        states.append(
            {
                "scenario_id": scenario.scenario_id,
                "event_type": event,
                "timestamp_step": 0 if event == "initial" else elapsed_after_onset,
                "forecast_source_branch": int(scenario.forecast_source_branch),
                "growth_rate": float(scenario.growth_rate),
                "onset_step": int(scenario.onset_step),
                "occupancy": occupancy,
                "smoke": smoke,
                "north_exit_capacity": north_capacity,
                "south_exit_capacity": south_capacity,
                "blockage_active": blockage_active,
                "blockage_steps_remaining": blockage_remaining,
                "last_release_fraction": (0.25, 0.50, 0.75, 1.00)[index % 4],
                "semantic_context": {
                    "warden_report": (
                        "The currently measured exit capacities and smoke sensors are authoritative."
                    ),
                    "control_period_seconds": 5,
                },
            }
        )
    return states


def plan_safety_violations(plan: EvacuationPlan, state: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    first_route = plan.phases[0].route
    if first_route == "north" and int(state.get("north_exit_capacity", 0)) <= 0:
        violations.append("first_phase_targets_blocked_exit")
    if first_route == "south" and int(state.get("south_exit_capacity", 0)) <= 0:
        violations.append("first_phase_targets_blocked_exit")
    return violations
