"""Offline paired interventions; never an online oracle or a trained shield.

All alternative actions are selected from public snapshots. Hidden state is
copied only by the experiment runner to evaluate terminal outcomes.
"""
from __future__ import annotations

import copy
import numpy as np

from .timestamped_pipeline import Proposal, ppo_action


IMMEDIATE_ARMS = tuple(f"lower_release_{p}" for p in (25, 50, 75, 100))
WAIT_ARMS = ("wait_one_then_ppo", "wait_until_lower_then_ppo")


def episode_key(record):
    return (record['scenario_id'], record['behavior'], int(record['population_seed']),
            int(record['training_seed']))


def state_id(record):
    return '|'.join(map(str, (*episode_key(record), int(record['step']))))


def lower_branch(snapshot):
    return int(snapshot.smoke[4] < snapshot.smoke[3])


def available_arms(snapshot, constraints):
    mask = constraints.action_mask(snapshot)
    b = lower_branch(snapshot)
    return ['original_ppo'] + [arm for i, arm in enumerate(IMMEDIATE_ARMS) if mask[b * 4 + i]] + list(WAIT_ARMS)


def intervention_action(arm, env, offset, original_level):
    """Return a feasible intervention, or None to resume the fixed PPO policy.

Waiting stops room release, not junction routing. The second waiting plan
waits at least once, at most three times, and then sends the currently lower
smoke branch at a legal release level no greater than the original level.
"""
    s = env.snapshot
    mask = env.constraints.action_mask(s)
    if arm == 'original_ppo':
        return None
    if arm in IMMEDIATE_ARMS:
        if offset:
            return None
        action = lower_branch(s) * 4 + IMMEDIATE_ARMS.index(arm)
        if not mask[action]:
            raise ValueError('immediate counterfactual action is not executable')
        return action
    if arm == 'wait_one_then_ppo':
        return 8 if offset == 0 else None
    if arm != 'wait_until_lower_then_ppo':
        raise ValueError('unknown intervention')
    if offset == 0:
        return 8
    candidates = [lower_branch(s) * 4 + level for level in range(original_level + 1)
                  if mask[lower_branch(s) * 4 + level]]
    if candidates:
        return max(candidates)
    return 8 if offset < 3 else None


class FixedUniform:
    """One per-person, per-step uniform for the calibrated Bernoulli choice."""
    def __init__(self, value):
        self.value = float(value)

    def random(self):
        return self.value


class PairedFutureBehavior:
    """Preserve choice probabilities while pairing draws by person and time."""
    def __init__(self, base, table):
        if base.name not in ('rule', 'calibrated_logit'):
            raise ValueError('only audited deterministic/Bernoulli behavior supported')
        self.base, self.table, self.step = base, table, 0
        self.name = base.name

    def choose_branch(self, evacuee, context, rng):
        del rng
        return self.base.choose_branch(evacuee, context, FixedUniform(self.table[self.step, evacuee.evacuee_id]))


def clone_for_future(env, future_seed):
    clone = copy.deepcopy(env)
    domain = clone.unwrapped.domain_env
    table = np.random.default_rng(future_seed).random((domain.config.max_steps + 1, domain.config.population))
    domain.behavior_model = PairedFutureBehavior(domain.behavior_model, table)
    return clone


def paired_step(env, action):
    domain = env.unwrapped.domain_env
    if isinstance(domain.behavior_model, PairedFutureBehavior):
        domain.behavior_model.step = env.snapshot.step
    return env.step(action)


def faulty_policy_action(env, model):
    s = env.snapshot
    p = gate = None
    if abs(s.smoke[3] - s.smoke[4]) > .01:
        worse = int(s.smoke[4] > s.smoke[3])
        payload = dict(observation_time_s=s.observation_time_s,
                       preferred_exit=('north', 'south')[worse], confidence=.99,
                       candidate_actions=[worse * 4, worse * 4 + 1],
                       wait_steps=0, north_share=float(worse == 0))
        p = Proposal.parse(payload, s, s.observation_time_s, f'cf-{s.step}', 'observable_fault_fixture')
        gate = env.constraints.validate_proposal(p, s, s.observation_time_s)
    return ppo_action(model, env.policy_observation(p, gate), env.constraints.action_mask(s), p, gate)[0]


def terminal_metrics(info, config):
    return dict(mean_cumulative_fed=float(info['mean_cumulative_fed']),
                cumulative_congestion=float(info['cumulative_congestion']),
                evacuation_rate=float(info['evacuation_rate']),
                incapacitation_rate=float(info['incapacitation_rate']),
                remaining=int(info['remaining']),
                episode_end_seconds=float(info['step'] * config.dt_seconds),
                all_evacuated_seconds=float(info['step'] * config.dt_seconds)
                if info['evacuated'] == config.population else None)
