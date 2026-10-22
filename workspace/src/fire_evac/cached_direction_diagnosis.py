"""Reconstruct frozen no-advice inputs and factor cached one-step effects.

Pure data transformations: no environment step, hidden people, or future
outcome enters an action. Missing history and incomplete legal cells fail.
"""
import numpy as np

from .config import EnvConfig


def no_advice_input_from_cache(state, scene, previous_room_capacity=None, config=None):
    config = config or EnvConfig()
    if state['scenario_id'] != scene['scenario_id']:
        raise ValueError('scenario mismatch')
    if int(state['step']) > 0 and previous_room_capacity is None:
        raise ValueError('previous actual release capacity required for release memory')
    x = np.zeros(34, dtype=np.float32)
    x[:5] = state['occupancy']
    x[5:10] = state['smoke']
    sign = int(state['sign'])
    if sign not in range(3):
        raise ValueError('invalid cached sign')
    x[10 + sign] = 1
    # Recover integer active counts before forming the float32 remaining rate.
    active = sum(int(round(float(v) * config.population)) for v in state['occupancy'])
    x[13] = active / config.population
    x[14:16] = np.asarray(state['capacities'], dtype=np.float32) / config.branch_to_exit_capacity
    x[16:18] = np.asarray(state['capacities']) <= 0
    if scene['blocked_branch'] in (0, 1):
        x[18] = np.clip(max(0, scene['blockage_end_step'] - state['step']) / 20, 0, 1)
    x[19 + scene['forecast_source_branch']] = 1
    x[21] = np.clip(scene['growth_rate'] / .05, 0, 1)
    x[22] = np.clip(scene['onset_step'] / 20, 0, 1)
    x[23] = scene['north_exit_capacity'] / 8
    x[24] = scene['south_exit_capacity'] / 8
    x[25] = scene['blocked_branch'] == 0
    x[26] = scene['blocked_branch'] == 1
    x[27] = np.clip(scene['blockage_end_step'] / 20, 0, 1)
    last = config.start_to_junction_capacity if state['step'] == 0 else previous_room_capacity
    x[33] = np.clip(last / config.start_to_junction_capacity, 0, 1)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite cached input')
    return x


def joint_effect_cells(base_action, hint_action, legal_actions):
    """Return direction x release cells; waiting is a separate mechanism."""
    if 8 in (base_action, hint_action):
        return None
    b, h = int(base_action), int(hint_action)
    cells = dict(base=b, direction_only=(h // 4) * 4 + b % 4,
                 release_only=(b // 4) * 4 + h % 4, full=h)
    if not all(a in legal_actions for a in cells.values()):
        return None
    return cells


def factor_effects(cells, candidate_means, metric):
    """Exact additive identity for cached terminal differences, no new rollout."""
    values = {k: candidate_means[a][metric] for k, a in cells.items()}
    direction = values['direction_only'] - values['base']
    release = values['release_only'] - values['base']
    interaction = values['full'] - values['direction_only'] - values['release_only'] + values['base']
    full = values['full'] - values['base']
    if not np.isclose(direction + release + interaction, full, atol=1e-12, rtol=0):
        raise ValueError('inconsistent factor decomposition')
    return dict(direction_at_base_release=direction, release_at_base_direction=release,
                direction_release_interaction=interaction, full_difference=full)
