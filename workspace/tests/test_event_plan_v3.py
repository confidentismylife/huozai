"""Behavioural regressions for the event-trigger correction."""
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.event_plan_v3 import EventCorrectedPlanWrapper
from fire_evac.deepseek_advice import EvacuationPlan, PlanPhase, REPLAN_TRIGGERS


class ObservationSequence(gym.Env):
    """Controlled sensor input; event detection and provider calls remain real."""
    def __init__(self, frames):
        self.frames = frames
        self.observation_space = gym.spaces.Box(0., np.inf, (34,), np.float32)
        self.action_space = gym.spaces.Discrete(9)

    def reading(self):
        people, north, south, blocked = self.frames[self.index]
        obs = np.zeros(34, dtype=np.float32)
        obs[2], obs[8], obs[9] = people / 240, north, south
        return obs, dict(step=self.index, blockage_active=blocked,
            north_exit_capacity=0 if blocked else 8, south_exit_capacity=8,
            scenario={'scenario_id': 'controlled-input'})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.reading()

    def step(self, action):
        self.index += 1
        obs, info = self.reading()
        return obs, 0., False, False, info


def build(frames, triggers=REPLAN_TRIGGERS, maximum=8):
    calls = []
    plan = EvacuationPlan((PlanPhase('north', .5, 3),) * 3, .7,
        ('balance_congestion',), tuple(triggers))
    def provider(state):
        calls.append(dict(state))
        return plan
    return EventCorrectedPlanWrapper(ObservationSequence(frames), provider,
        max_requests_per_episode=maximum), calls


def test_stable_safer_side_does_not_swallow_congestion_crossing():
    env, calls = build([(40, .01, .04, False), (46, .01, .04, False)])
    env.reset(seed=7)
    _, _, _, _, info = env.step(0)
    assert [s['event_type'] for s in calls] == ['initial', 'congestion_threshold']
    assert info['deepseek_requests_this_episode'] == 2
    assert calls[-1]['timestamp_step'] == 1


def test_congestion_staying_high_does_not_repeat_but_reentry_does():
    frames = [(40, .01, .04, False), (46, .01, .04, False),
              (50, .01, .04, False), (40, .01, .04, False), (47, .01, .04, False)]
    env, calls = build(frames)
    env.reset(seed=7)
    for _ in frames[1:]:
        env.step(0)
    assert [s['timestamp_step'] for s in calls] == [0, 1, 4]
    assert [s['event_type'] for s in calls[1:]] == ['congestion_threshold'] * 2


def test_smoke_reversal_replans_without_congestion():
    env, calls = build([(20, .01, .04, False), (20, .04, .01, False)])
    env.reset(seed=7)
    env.step(0)
    assert [s['event_type'] for s in calls] == ['initial', 'smoke_reversal']


def test_simultaneous_events_keep_one_blockage_priority_request():
    env, calls = build([(40, .01, .04, False), (46, .04, .01, True),
                        (48, .04, .01, True)])
    env.reset(seed=7)
    env.step(0)
    env.step(0)
    assert [s['event_type'] for s in calls] == ['initial', 'blockage_change']


def test_request_cap_includes_initial_and_reset_restores_allowance():
    frames = [(40, .01, .04, False), (46, .01, .04, False),
              (40, .04, .01, True), (48, .01, .04, False)]
    env, calls = build(frames, maximum=2)
    env.reset(seed=7)
    for _ in frames[1:]:
        env.step(0)
    assert [s['event_type'] for s in calls] == ['initial', 'congestion_threshold']
    env.reset(seed=8)
    _, _, _, _, info = env.step(0)
    assert len(calls) == 4
    assert info['deepseek_requests_this_episode'] == 2


def test_disabled_congestion_trigger_does_not_replan():
    env, calls = build([(40, .01, .04, False), (46, .01, .04, False)],
        triggers=('smoke_reversal',))
    env.reset(seed=7)
    env.step(0)
    assert len(calls) == 1


def test_congestion_also_triggers_with_ambiguous_smoke():
    env, calls = build([(40, .02, .02, False), (46, .02, .02, False)])
    env.reset(seed=7)
    env.step(0)
    assert calls[-1]['event_type'] == 'congestion_threshold'
