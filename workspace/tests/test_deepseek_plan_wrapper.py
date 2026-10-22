from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase, REPLAN_TRIGGERS
from fire_evac.deepseek_plan_wrapper import DeepSeekPlanObservationWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank


def sample_plan() -> EvacuationPlan:
    return EvacuationPlan(
        phases=(
            PlanPhase("south", 0.50, 2),
            PlanPhase("south", 0.75, 3),
            PlanPhase("north", 0.25, 2),
        ),
        confidence=0.8,
        reason_codes=("respect_blockage",),
        replan_triggers=REPLAN_TRIGGERS,
    )


def test_plan_wrapper_replaces_redundant_advice_with_50_features() -> None:
    scenarios = make_trust_gate_bank(10, 81, "P")
    calls: list[dict] = []

    def provider(state):
        calls.append(dict(state))
        return sample_plan()

    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    env = DeepSeekPlanObservationWrapper(base, provider)
    observation, info = env.reset(seed=2, options={"scenario": scenarios[0]})

    assert observation.shape == (50,)
    assert env.observation_space.contains(observation)
    assert calls[0]["event_type"] == "initial"
    assert info["deepseek_plan_source"] == "provider"


def test_blockage_transition_triggers_at_most_one_replan() -> None:
    scenarios = make_trust_gate_bank(10, 91, "R")
    scenario = next(item for item in scenarios if item.conflict_type == "safe_exit_short_block")
    event_types: list[str] = []

    def provider(state):
        event_types.append(str(state["event_type"]))
        return sample_plan()

    base = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice_mode="none"
    )
    env = DeepSeekPlanObservationWrapper(base, provider, max_requests_per_episode=3)
    env.reset(seed=3, options={"scenario": scenario})
    for _ in range(scenario.blockage_end_step + 2):
        _, _, terminated, truncated, _ = env.step(8)
        if terminated or truncated:
            break

    assert event_types[0] == "initial"
    assert event_types.count("blockage_change") == 1
    assert len(event_types) <= 3
