"""Auditable process metrics for evacuation-control experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .campaign_extension import plan_phase
from .deepseek_advice import EvacuationPlan
from .mappo import FLOW_LEVELS, ROUTE_WAIT, encode_factorized_action


def summarize_episode_progress(
    records: Sequence[Mapping[str, Any]],
    population: int,
    step_seconds: float = 5.0,
    thresholds: Sequence[float] = (0.5, 0.8, 0.9),
) -> dict[str, Any]:
    """Summarize successful evacuation without treating incapacitation as speed."""
    if population <= 0 or step_seconds <= 0 or not records:
        raise ValueError("population, step duration, and records must be positive")
    previous = (-1, -1, population + 1)
    progress: list[float] = []
    for expected_step, record in enumerate(records, start=1):
        step = int(record["step"])
        evacuated = int(record["evacuated"])
        incapacitated = int(record["incapacitated"])
        remaining = int(record["remaining"])
        if step != expected_step:
            raise ValueError("progress records must contain consecutive one-based steps")
        if evacuated + incapacitated + remaining != population:
            raise ValueError("population balance must equal the initial population")
        if evacuated < previous[0] or incapacitated < previous[1] or remaining > previous[2]:
            raise ValueError("evacuation outcomes must be monotonic")
        previous = (evacuated, incapacitated, remaining)
        progress.append(evacuated / population)

    last = records[-1]
    result: dict[str, Any] = {
        "episode_end_time_seconds": len(records) * float(step_seconds),
        "final_successful_evacuation_rate": int(last["evacuated"]) / population,
        "final_incapacitation_rate": int(last["incapacitated"]) / population,
        "final_remaining_rate": int(last["remaining"]) / population,
    }
    for threshold in thresholds:
        percent = int(round(100 * float(threshold)))
        reached = next((i for i, value in enumerate(progress, start=1) if value >= threshold), None)
        result[f"time_to_{percent}pct_success_seconds"] = (
            None if reached is None else reached * float(step_seconds)
        )
    result["successful_evacuation_progress_auc"] = sum(progress) / len(progress)
    result["end_reason"] = (
        "all_people_resolved" if int(last["remaining"]) == 0 else "time_limit_or_truncation"
    )
    return result


def active_plan_decision(plan: EvacuationPlan, age_steps: int) -> dict[str, Any]:
    """Return the active plan phase in the controller's nine-action vocabulary."""
    phase_index = plan_phase(plan, max(0, int(age_steps)))
    phase = plan.phases[phase_index]
    if phase.route == "wait":
        joint_action = encode_factorized_action(ROUTE_WAIT, 0)
    else:
        route = 0 if phase.route == "north" else 1
        flow = FLOW_LEVELS.index(float(phase.release_fraction))
        joint_action = encode_factorized_action(route, flow)
    return {
        "plan_phase_index": phase_index,
        "plan_route": phase.route,
        "plan_release_fraction": float(phase.release_fraction),
        "plan_joint_action": joint_action,
    }


def select_training_event_states(
    records: Sequence[Mapping[str, Any]], maximum: int = 12
) -> list[dict[str, Any]]:
    """Select the first observed congestion crossing per training scenario."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    congestion = [
        dict(record) for record in records
        if str(record.get("event_type")) == "congestion_threshold"
    ]
    congestion.sort(key=lambda row: (str(row["scenario_id"]), int(row["timestamp_step"])))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in congestion:
        scenario_id = str(record["scenario_id"])
        if scenario_id in seen:
            continue
        selected.append(record)
        seen.add(scenario_id)
        if len(selected) == maximum:
            break
    return selected


def cyclically_permute_plans(
    records: Sequence[tuple[Mapping[str, Any], Any]],
) -> list[tuple[Mapping[str, Any], Any]]:
    """Keep retrieval states fixed while rotating their associated plan content."""
    if len(records) < 2:
        raise ValueError("plan permutation requires at least two records")
    plans = [plan for _, plan in records]
    return [
        (state, plans[(index + 1) % len(plans)])
        for index, (state, _) in enumerate(records)
    ]
