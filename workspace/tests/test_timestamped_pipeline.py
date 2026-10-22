from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from fire_evac.behavior import RuleBehavior
from fire_evac.config import EnvConfig
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import (Snapshot, Proposal, CommonActionConstraints,
                                           TimestampedControlEnv, AsyncAdviceClient)


def snapshot(**changes):
    s = Snapshot('episode', 0, 0., '2026-10-07T00:00:00+00:00', (.5, .5, 0., 0., 0.),
                 (.002,) * 5, (8, 8), (False, False), 0, .02, 5, 0., 2, 0, 20)
    return replace(s, **changes)


def proposal(s, received=0., **changes):
    data = dict(observation_time_s=s.observation_time_s, preferred_exit='north', confidence=.9,
                candidate_actions=[0, 1, 4], wait_steps=0, north_share=.5)
    data.update(changes)
    return Proposal.parse(data, s, received, 'request', 'fixture')


def test_new_closure_checked_on_return():
    s = snapshot()
    p = proposal(s)
    latest = replace(s, step=1, observation_time_s=5, closed=(True, False), capacities=(0, 8))
    gate = CommonActionConstraints().validate_proposal(p, latest, 5)
    assert not gate.accepted and 'preferred_exit_unavailable' in gate.reasons
    assert gate.valid_candidate_actions == (4,)


def test_stale_timeout_and_future_are_distinct():
    rules = CommonActionConstraints()
    s = snapshot()
    assert rules.validate_proposal(proposal(s, 50), s, 50).reasons == ('stale',)
    assert set(rules.validate_proposal(proposal(s, 65), s, 65).reasons) == {'stale', 'timeout'}
    assert 'future_timestamp' in rules.validate_proposal(proposal(s, 5), s, 0).reasons
    assert 'episode_mismatch' in rules.validate_proposal(proposal(s), replace(s, episode_id='next'), 0).reasons


def test_bad_live_smoke_is_rejected_even_with_high_confidence():
    s = snapshot(smoke=(.002, .002, .002, .08, .005))
    result = CommonActionConstraints().validate_proposal(proposal(s), s, 0)
    assert not result.accepted and 'live_smoke_contradiction' in result.reasons


def test_capacity_throttle_and_no_route_fallback():
    constraints = CommonActionConstraints()
    s = snapshot(capacities=(3, 3))
    assert constraints.action_mask(s).tolist() == [True, False, False, False, True, False, False, False, True]
    assert constraints.filter_action(3, s) == 0
    both = replace(s, capacities=(0, 0), closed=(True, True))
    assert constraints.filter_action(7, both) == 8
    assert constraints.room_limit(replace(s, occupancy=(0, 0, .2, 0, 0))) == 0


def test_hold_mask_matches_actuator():
    s = snapshot(sign=0, sign_hold_remaining=2)
    mask = CommonActionConstraints().action_mask(s)
    assert not mask[4:8].any()
    assert CommonActionConstraints().filter_action(7, s) < 4


@pytest.mark.parametrize('changes', [dict(confidence=float('nan')), dict(confidence=True),
                                      dict(candidate_actions=[0, 0]), dict(candidate_actions=[True]),
                                      dict(observation_time_s=1), dict(wait_steps=1)])
def test_invalid_proposal(changes):
    with pytest.raises(ValueError):
        proposal(snapshot(), **changes)


def test_closed_corridor_no_entry_and_actual_room_capacity():
    # A late closure takes effect after the initial request, including sign hold.
    scene = DynamicConflictScenario('test', 0, .02, 20, north_exit_capacity=3,
                                    south_exit_capacity=3, blocked_branch=0,
                                    blockage_start_step=1, blockage_end_step=10)
    env = TimestampedControlEnv(RuleBehavior(), (scene,))
    env.reset(seed=1, options={'scenario': scene})
    _, _, _, _, first = env.step(3)
    assert first['constraint_applied_action'] == 0
    assert first['room_release_capacity'] <= 6
    for _ in range(4):
        _, _, _, _, info = env.step(3)
        assert info['unavailable_corridor_entries'] == 0
        assert info['actual_release_within_limit']
        assert not info['actual_direction_violation']
    env.close()


def test_snapshot_only_reads_observable_fields():
    scene = DynamicConflictScenario('test', 0, .02, 3)
    env = TimestampedControlEnv(RuleBehavior(), (scene,), EnvConfig(population=100))
    x, info = env.reset(seed=1, options={'scenario': scene})
    changed = dict(info, scenario={'source_branch': 1, 'arbitrary_secret': 'hidden'})
    s = Snapshot.from_sensor(x, changed, env.unwrapped.domain_env.config, 'e')
    assert s.forecast_source == 0 and s.population == 100
    assert 'source_branch' not in s.public_data() and 'scenario' not in s.public_data()


def test_async_nonblocking_late_response_and_latest_observation():
    release = threading.Event()
    s = snapshot()
    def transport(observed):
        release.wait(2)
        return dict(observation_time_s=observed.observation_time_s, preferred_exit='north', confidence=.9,
                    candidate_actions=[0], wait_steps=0, north_share=1.)
    client = AsyncAdviceClient(transport)
    try:
        assert client.submit(s, now_monotonic=100.)
        assert not client.submit(s, now_monotonic=100.)
        assert client.poll(s, CommonActionConstraints(), now_monotonic=101.)[2] == 'pending'
        assert client.poll(s, CommonActionConstraints(), now_monotonic=165.)[2] == 'timeout_pending_audit'
        release.set()
        # Wait for completion by joining the actual future, not arbitrary sleep.
        client.pending[0].result(timeout=2)
        latest = replace(s, closed=(True, False), capacities=(0, 8))
        p, gate, state = client.poll(latest, CommonActionConstraints(), now_monotonic=166.)
        assert state == 'rejected'
        assert {'timeout', 'stale', 'preferred_exit_unavailable'} <= set(gate.reasons)
    finally:
        release.set()
        client.close()
