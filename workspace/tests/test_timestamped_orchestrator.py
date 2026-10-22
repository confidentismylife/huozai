import sys
import threading
from pathlib import Path
import numpy as np
from fire_evac.behavior import RuleBehavior
from fire_evac.config import EnvConfig
from fire_evac.dynamic_conflicts import DynamicConflictScenario
from fire_evac.timestamped_pipeline import TimestampedControlEnv
from fire_evac.timestamped_orchestrator import execute_episode


def test_invalid_http_response_finishes_with_same_constrained_controller():
    scene = DynamicConflictScenario('invalid', 0, .01, 30)
    env = TimestampedControlEnv(RuleBehavior(), (scene,), EnvConfig(population=40, max_steps=25))
    env.reset(seed=2026, options={'scenario': scene})
    called = []
    model = object()
    def policy(m, observation, mask, proposal, decision, **kwargs):
        called.append(m)
        assert proposal is None and decision is None
        assert np.all(observation[28:33] == 0)
        allowed = np.flatnonzero(mask[:8])
        return (int(allowed[-1]) if len(allowed) else 8), 7
    def transport(_):
        return {'malformed': True}
    result = execute_episode(env, model, transport, control_interval_seconds=.0001, action_function=policy)
    assert result['final_metrics']['step'] > 1
    assert result['advice_final_state'].startswith('invalid_or_transport_error')
    assert all(m is model for m in called)
    assert all(not r['release_violation'] and not r['unavailable_entries'] for r in result['trace'])


def test_accepted_proposal_is_rechecked_until_expiry():
    scene = DynamicConflictScenario('expiry', 0, .01, 30)
    env = TimestampedControlEnv(RuleBehavior(), (scene,), EnvConfig(population=100, max_steps=25))
    env.reset(seed=1, options={'scenario': scene})
    states = []
    def policy(_, observation, mask, proposal, decision, **kwargs):
        if decision is not None:
            states.append((decision.accepted, decision.reasons, bool(observation[31])))
        return 8, 8
    def transport(s):
        return {'observation_time_s': s.observation_time_s, 'preferred_exit': 'north', 'confidence': .9,
                'candidate_actions': [0, 1], 'wait_steps': 0, 'north_share': 1.}
    result = execute_episode(env, object(), transport, control_interval_seconds=.0001, action_function=policy)
    assert any(accepted and exposed for accepted, _, exposed in states)
    assert any(not accepted and 'stale' in reasons and not exposed for accepted, reasons, exposed in states)

