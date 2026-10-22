from dataclasses import replace
import numpy as np
import pytest

from fire_evac.direction_release_mechanism import preserve_base_release, set_diagnostic_hold_zero, step_zone_accounting
from fire_evac.config import EnvConfig
from fire_evac.behavior import RuleBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step


def test_route_change_preserves_actual_base_level_without_changing_direction():
    mask = np.ones(9, bool)
    action, status = preserve_base_release(1, 4, mask)
    assert action == 5 and status == 'preserved_base_release'
    assert action // 4 == 4 // 4 and action % 4 == 1 % 4


def test_impossible_recombination_keeps_legal_baseline():
    mask = np.ones(9, bool)
    mask[5] = False
    assert preserve_base_release(1, 4, mask) == (1, 'matched_release_infeasible_keep_base')
    with pytest.raises(ValueError, match='feasible'):
        preserve_base_release(5, 4, mask)


def test_waiting_is_not_mislabeled_as_preserved_release():
    assert preserve_base_release(8, 4, np.ones(9, bool)) == (4, 'waiting_not_factorizable')


def test_hold_zero_clears_both_constraints_and_only_changes_hold_config():
    scene = DynamicConflictScenario('hold', 0, .01, 5, forecast_source_branch=0)
    env = TimestampedControlEnv(RuleBehavior(), (scene,))
    env.reset(seed=771, options={'scenario': scene})
    paired_step(env, 1)
    assert env.snapshot.sign_hold_remaining == 2
    assert not env.constraints.action_mask(env.snapshot)[5]
    original = env.unwrapped.domain_env.config
    set_diagnostic_hold_zero(env)
    assert env.unwrapped.domain_env.config == replace(original, min_sign_hold_steps=0)
    assert env.snapshot.sign_hold_remaining == 0
    assert env.constraints.action_mask(env.snapshot)[5]
    _, _, _, _, info = paired_step(env, 5)
    assert info['actual_sign_action'] == 1 and info['actual_release_within_limit']
    env.close()


def test_zone_accounting_conserves_fed_and_does_not_change_trajectory():
    scene = DynamicConflictScenario('zones', 0, .01, 0, forecast_source_branch=0)
    a = TimestampedControlEnv(RuleBehavior(), (scene,))
    a.reset(seed=772, options={'scenario': scene})
    b = clone_for_future(a, 992)
    a = clone_for_future(a, 992)
    for action in [1, 1, 1, 5, 5]:
        before = a.unwrapped.domain_env.cumulative_fed
        result, fields = step_zone_accounting(a, action, paired_step)
        other = paired_step(b, action)
        assert sum(fields['zone_fed_increments']) == pytest.approx(a.unwrapped.domain_env.cumulative_fed - before)
        assert sum(fields['zone_person_steps']) >= 0
        np.testing.assert_array_equal(result[0], other[0])
        assert result[4]['mean_cumulative_fed'] == other[4]['mean_cumulative_fed']
    a.close()
    b.close()
