from pathlib import Path
import sys
from dataclasses import replace
import copy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_evac.behavior import RuleBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.config import EnvConfig
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank
from fire_evac.terminal_shield import ObservableHistoryWrapper, mirror_history_observation
from fire_evac.counterfactual_shield import ACTION_MIRROR
from fire_evac.source_world_planning import (
    PLANS, compatible_source_worlds, source_world_scenarios, plan_action, evaluate_plan, select_reference_plan,
)


def environment():
    scenarios = make_trust_gate_bank(5, 11, "T")
    env = ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(RuleBehavior(), scenarios, EnvConfig(max_steps=10)), "none"))
    return env, scenarios


def test_source_support_uses_observation_and_forecasts_not_actual_source():
    env, scenarios = environment()
    scenario = replace(scenarios[0], onset_step=0)
    observation, _ = env.reset(seed=33, options={"scenario": scenario})
    hypotheses = source_world_scenarios(scenario)
    assert hypotheses == source_world_scenarios(replace(scenario, source_branch=1 - scenario.source_branch))
    support, _, mismatch = compatible_source_worlds(hypotheses, observation, 0)
    assert support.all() and not mismatch
    observation, *_ = env.step(3)
    support, _, mismatch = compatible_source_worlds(hypotheses, observation, 1)
    assert support.sum() == 1 and support[scenario.source_branch]
    modified = observation.copy()
    modified[5:10] += 1
    support, _, mismatch = compatible_source_worlds(hypotheses, modified, 1)
    assert support.all() and mismatch
    env.close()


def test_wait_is_limited_and_ends_when_smoke_identifies_source():
    env, _ = environment()
    observation, _ = env.reset(seed=34)
    observation[8:10] = .002
    wait = PLANS[3]
    assert plan_action(wait, observation, 0) == 8
    assert plan_action(wait, observation, 3) != 8
    observation[8] = .04
    assert plan_action(wait, observation, 0) != 8
    env.close()


def test_plan_request_reflection_including_ties_and_splits():
    env, _ = environment()
    observation, _ = env.reset(seed=35)
    observation[8:10] = .002
    observation[3:5] = 0
    observation[14:16] = 1
    mirrored = mirror_history_observation(observation)
    for plan in PLANS:
        for elapsed in range(16):
            assert plan_action(plan, mirrored, elapsed) == ACTION_MIRROR[plan_action(plan, observation, elapsed)]
    env.close()


def test_terminal_plan_evaluation_preserves_original_state_and_rng():
    env, _ = environment()
    observation, _ = env.reset(seed=36)
    reference = copy.deepcopy(env)
    result = evaluate_plan(env, observation, PLANS[3], 200)
    assert 1 <= result.steps <= 10
    assert result.final_remaining == 0 or result.steps == 10
    assert evaluate_plan(env, observation, PLANS[3], 200) == result
    actual = env.step(7)
    expected = reference.step(7)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    env.close()
    reference.close()


def test_point_mean_robust_and_oracle_use_declared_world_aggregation():
    risks = np.zeros((2, 2, 2, 11))
    risks[0, 0, :, 0] = 0
    risks[1, 0, :, 0] = .2
    risks[:, 1, :, 0] = .12
    assert select_reference_plan(risks, [True, True], 0, "point") == 0
    assert select_reference_plan(risks, [True, True], 0, "mean") == 0
    assert select_reference_plan(risks, [True, True], 0, "robust") == 1
    assert select_reference_plan(risks, [True, True], 0, "oracle", true_source=1) == 1
    assert select_reference_plan(risks, [False, True], 0, "point") == 1
