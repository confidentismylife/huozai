from pathlib import Path
from dataclasses import replace
import numpy as np

from fire_evac.behavior import CalibratedLogitBehavior, RuleBehavior
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv, Proposal
from fire_evac.counterfactual_direction_audit import clone_for_future, paired_step, PairedFutureBehavior
from fire_evac.advice_causal_audit import reseed_continuation, outcome_class, active_advice, ranked_outcomes

ROOT = Path(__file__).resolve().parents[1]


def make_env():
    scene = DynamicConflictScenario('CA_UNIT_ONLY', 1, .015, 6, .002,
                                    'clear', 8, 8, -1, 0, 0, 0, 1)
    actor = CalibratedLogitBehavior(ROOT / 'outputs/behavior_pipeline/behavior_model.json')
    env = TimestampedControlEnv(actor, (scene,))
    env.reset(seed=711, options={'scenario': scene})
    return clone_for_future(env, 51)


def test_reseeding_already_paired_state_replaces_the_inner_future():
    env = make_env()
    paired_step(env, 0)
    a, b, c = [reseed_continuation(env, seed) for seed in (71, 71, 72)]
    actor = a.unwrapped.domain_env.behavior_model
    assert isinstance(actor, PairedFutureBehavior)
    assert not isinstance(actor.base, PairedFutureBehavior)
    np.testing.assert_array_equal(actor.table, b.unwrapped.domain_env.behavior_model.table)
    assert not np.array_equal(actor.table, c.unwrapped.domain_env.behavior_model.table)
    assert a.unwrapped.domain_env.people == env.unwrapped.domain_env.people
    paired_step(a, 0)
    paired_step(b, 0)
    np.testing.assert_array_equal(a.latest_observation, b.latest_observation)
    assert env.snapshot.step == 1


def test_delayed_advice_never_refreshes_timestamp():
    env = make_env()
    s = env.snapshot
    p = Proposal.parse(dict(observation_time_s=0., preferred_exit='north', confidence=.75,
        candidate_actions=[0, 1], wait_steps=0, north_share=1.), s, 6., 'unit', 'unit')
    assert active_advice(env, p)[2] == 'pending'
    env.snapshot = replace(s, step=2, observation_time_s=10.)
    assert active_advice(env, p)[2] == 'accepted'
    env.snapshot = replace(s, step=4, observation_time_s=20.)
    _, decision, state = active_advice(env, p)
    assert state == 'rejected' and 'stale' in decision.reasons
    assert p.observation_time_s == 0.


def test_fed_reduction_cannot_hide_more_incapacitation():
    assert outcome_class(.004166, -.05) == 'harmful'
    assert outcome_class(-.004166, .05) == 'beneficial'
    assert outcome_class(0., -.002) == 'beneficial'
    assert outcome_class(0., .002) == 'harmful'
    assert outcome_class(0., .0002) == 'small_or_neutral'


def test_offline_rank_prioritizes_incapacitation_over_fed():
    means = {0: dict(incapacitation_rate=.1, mean_cumulative_fed=.4, cumulative_congestion=5.),
             1: dict(incapacitation_rate=.2, mean_cumulative_fed=.2, cumulative_congestion=2.)}
    assert ranked_outcomes(means) == [0, 1]
