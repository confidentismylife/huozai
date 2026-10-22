import copy
from dataclasses import replace

import numpy as np
import pytest

from fire_evac.behavior import RuleBehavior, CalibratedLogitBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv
from fire_evac.counterfactual_direction_audit import (
    available_arms, intervention_action, clone_for_future, paired_step, terminal_metrics)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_env(behavior=None):
    scene = DynamicConflictScenario('CF_UNIT_ONLY', 1, .015, 0, .002,
                                    'clear', 8, 8, -1, 0, 0, 0, 1)
    env = TimestampedControlEnv(behavior or RuleBehavior(), (scene,))
    env.reset(seed=716, options={'scenario': scene})
    return env


def test_immediate_actions_respect_hold_and_release_limit():
    env = make_env()
    s = replace(env.snapshot, smoke=(0., 0., .01, .005, .04), sign=1, sign_hold_remaining=2)
    assert available_arms(s, env.constraints) == ['original_ppo', 'wait_one_then_ppo', 'wait_until_lower_then_ppo']
    s = replace(s, sign_hold_remaining=0, capacities=(3, 4))
    assert 'lower_release_25' in available_arms(s, env.constraints)
    assert 'lower_release_50' not in available_arms(s, env.constraints)


def test_clone_preserves_person_attributes_and_history_without_mutating_parent():
    env = make_env()
    env.step(4)
    original = copy.deepcopy(env.unwrapped.domain_env.people)
    a = clone_for_future(env, 9)
    b = clone_for_future(env, 9)
    assert a.unwrapped.domain_env.people == original
    paired_step(a, 4)
    paired_step(b, 4)
    np.testing.assert_array_equal(a.latest_observation, b.latest_observation)
    assert env.snapshot.step == 1
    assert env.unwrapped.domain_env.people == original


def test_uniforms_stay_paired_by_person_when_call_order_changes():
    behavior = CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
    env = make_env(behavior)
    a = clone_for_future(env, 71).unwrapped.domain_env.behavior_model
    b = clone_for_future(env, 71).unwrapped.domain_env.behavior_model
    from fire_evac.behavior import DecisionContext
    context = DecisionContext(np.array([.02, .04]), np.zeros(2), np.zeros(2), 0)
    people = env.unwrapped.domain_env.people[:8]
    x = {p.evacuee_id: a.choose_branch(p, context, None) for p in people}
    y = {p.evacuee_id: b.choose_branch(p, context, None) for p in people[::-1]}
    assert x == y


def test_wait_plan_is_bounded_and_uses_live_mask():
    env = make_env()
    assert intervention_action('wait_until_lower_then_ppo', env, 0, 3) == 8
    env.snapshot = replace(env.snapshot, sign=1, sign_hold_remaining=2,
                           smoke=(0., 0., .01, .005, .04))
    assert intervention_action('wait_until_lower_then_ppo', env, 1, 3) == 8
    assert intervention_action('wait_until_lower_then_ppo', env, 3, 3) is None
    with pytest.raises(ValueError):
        intervention_action('lower_release_25', env, 0, 3)
    env.snapshot = replace(env.snapshot, sign_hold_remaining=0)
    assert intervention_action('wait_until_lower_then_ppo', env, 1, 0) == 0


def test_failed_terminal_is_not_successful_evacuation_time():
    env = make_env()
    info = dict(mean_cumulative_fed=.5, cumulative_congestion=22,
                evacuation_rate=.5, incapacitation_rate=.5, remaining=0,
                step=40, evacuated=120)
    result = terminal_metrics(info, env.unwrapped.domain_env.config)
    assert result['episode_end_seconds'] == 200
    assert result['all_evacuated_seconds'] is None
