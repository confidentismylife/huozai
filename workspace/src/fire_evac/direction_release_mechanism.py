"""Finite development interventions; no frozen controller is modified."""
from dataclasses import replace
import copy
import numpy as np


def preserve_base_release(base, hinted, mask):
    """Only recombine a feasible route change; waiting is explicitly separate.

    The nine actions cannot encode a new direction with zero room release.
    A wait-to-release transition therefore keeps the original hinted action
    and is marked unmatched, instead of hiding a simultaneous wait change.
    """
    base, hinted = int(base), int(hinted)
    if not mask[base] or not mask[hinted]:
        raise ValueError('base and hint must be feasible')
    if base == hinted:
        return base, 'unchanged'
    if 8 in (base, hinted):
        return hinted, 'waiting_not_factorizable'
    matched = (hinted // 4) * 4 + base % 4
    if not mask[matched]:
        return base, 'matched_release_infeasible_keep_base'
    return matched, 'preserved_base_release'


def set_diagnostic_hold_zero(env):
    """Clear both domain and wrapper hold, while preserving every other field."""
    domain = env.unwrapped.domain_env
    domain.config = replace(domain.config, min_sign_hold_steps=0)
    env._hold_remaining = 0
    env.snapshot = replace(env.snapshot, sign_hold_remaining=0)
    return env


def step_zone_accounting(env, action, step_function):
    """Observe actual individual increments offline, never policy features."""
    domain = env.unwrapped.domain_env
    zones = [list(z) for z in domain.zones]
    before = np.array([p.fed for p in domain.people])
    alive = [p.active for p in domain.people]
    result = step_function(env, action)
    after = np.array([p.fed for p in domain.people])
    delta = after - before
    if (delta < -1e-12).any():
        raise AssertionError('FED cannot decrease')
    fed = [float(delta[z].sum()) for z in zones]
    newly_incap = [sum(alive[i] and not domain.people[i].active for i in z) for z in zones]
    return result, dict(zone_person_steps=[len(z) for z in zones],
                        zone_fed_increments=fed, zone_new_incapacitations=newly_incap)
