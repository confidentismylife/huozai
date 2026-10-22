from __future__ import annotations

from dataclasses import replace

from .dynamic_conflicts import DynamicConflictScenario
from .dynamic_llm_rl import DynamicLLMAdvice, NORTH, SOUTH


PROMPT_VERSION = "confirmatory_v2_2_computed_target_guard"


def advice_prompt_v2(scenario: DynamicConflictScenario) -> str:
    source = "north" if scenario.forecast_source_branch == NORTH else "south"
    computed_target = "south" if scenario.forecast_source_branch == NORTH else "north"
    blocked = (
        "none"
        if scenario.blocked_branch not in (NORTH, SOUTH)
        else "north" if scenario.blocked_branch == NORTH else "south"
    )
    return (
        "You are a high-level safety advisor for a two-exit fire evacuation. "
        "Choose one target exit for a lower-level controller.\n"
        "SEMANTIC SAFETY RULE: forecast_fire_source is the corridor where smoke is predicted to originate. "
        "It is a danger location, not the evacuees' starting location and not a recommended exit. "
        "Do not choose forecast_fire_source. Choose the opposite exit. If that opposite exit is temporarily "
        "blocked or congested, keep it as the high-level target because the lower-level controller can wait, "
        "throttle release, or temporarily override. Never reinterpret the word source as origin of evacuees.\n"
        "MANDATORY TARGET MAPPING (higher priority than every capacity or blockage field):\n"
        "- If forecast_fire_source is north, target_exit MUST be south.\n"
        "- If forecast_fire_source is south, target_exit MUST be north.\n"
        "There are no exceptions to this mapping in this high-level task. Capacity and blockage fields are context "
        "for the lower-level controller and MUST NOT change target_exit.\n"
        "Return only one JSON object matching the supplied schema. Reason codes must be distinct.\n"
        f"forecast_fire_source={source}\n"
        f"smoke_growth_rate={scenario.growth_rate:.6f}\n"
        f"smoke_onset_step={scenario.onset_step}\n"
        f"north_exit_capacity={scenario.north_exit_capacity}\n"
        f"south_exit_capacity={scenario.south_exit_capacity}\n"
        f"forecast_blocked_exit={blocked}\n"
        f"forecast_reopen_step={scenario.blockage_end_step}\n"
        f"computed_safe_target={computed_target}\n"
        "FINAL OUTPUT CHECK: target_exit MUST exactly equal computed_safe_target. Copy that value without changing it.\n"
    )


def expected_semantic_target(scenario: DynamicConflictScenario) -> int:
    return SOUTH if scenario.forecast_source_branch == NORTH else NORTH


def safety_validate_advice(
    scenario: DynamicConflictScenario, advice: DynamicLLMAdvice
) -> tuple[DynamicLLMAdvice, bool]:
    """Correct a target that violates the frozen fire-source semantic invariant."""
    expected = expected_semantic_target(scenario)
    if advice.target_branch == expected:
        return advice, False
    corrected = DynamicLLMAdvice(
        target_branch=expected,
        confidence=min(0.60, advice.confidence),
        reason_codes=("avoid_predicted_smoke",),
        raw_response=advice.raw_response,
    )
    return corrected, True


def mirror_scenario(scenario: DynamicConflictScenario, suffix: str = "M") -> DynamicConflictScenario:
    blocked = (
        SOUTH
        if scenario.blocked_branch == NORTH
        else NORTH if scenario.blocked_branch == SOUTH else scenario.blocked_branch
    )
    return replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}{suffix}",
        source_branch=SOUTH if scenario.source_branch == NORTH else NORTH,
        north_exit_capacity=scenario.south_exit_capacity,
        south_exit_capacity=scenario.north_exit_capacity,
        blocked_branch=blocked,
        forecast_source_branch=(
            SOUTH if scenario.forecast_source_branch == NORTH else NORTH
        ),
    )
