import numpy as np
import pytest

from fire_evac.cached_direction_diagnosis import (
    no_advice_input_from_cache, joint_effect_cells, factor_effects,
)


def sample():
    state = dict(scenario_id='cache', step=0, sign=2, occupancy=[.5, .5, 0, 0, 0],
                 smoke=[.002] * 5, capacities=[10, 3])
    scene = dict(scenario_id='cache', blocked_branch=-1, blockage_end_step=0,
                 forecast_source_branch=0, growth_rate=.01, onset_step=6,
                 north_exit_capacity=10, south_exit_capacity=3)
    return state, scene


def test_reconstruction_recovers_fixed_forecast_and_clear_advice():
    state, scene = sample()
    x = no_advice_input_from_cache(state, scene)
    assert x.shape == (34,) and x.dtype == np.float32
    assert x[12] == x[13] == x[19] == x[33] == 1
    assert x[14] == 1.25 and x[15] == .375
    assert np.all(x[28:33] == 0)


def test_noninitial_input_requires_and_uses_real_release_history():
    state, scene = sample()
    state['step'] = 3
    state['occupancy'] = [119 / 240, 120 / 240, 0, 0, 0]
    with pytest.raises(ValueError, match='previous actual'):
        no_advice_input_from_cache(state, scene)
    x = no_advice_input_from_cache(state, scene, previous_room_capacity=5)
    assert x[33] == .25 and x[13] == np.float32(239 / 240)


def test_closed_capacity_and_reopening_encoding():
    state, scene = sample()
    state.update(step=2, capacities=[0, 3])
    scene.update(blocked_branch=0, blockage_end_step=7)
    x = no_advice_input_from_cache(state, scene, previous_room_capacity=0)
    assert x[16] == x[25] == 1 and x[18] == .25
    assert x[33] == 0


def test_factorization_separates_joint_effect_and_interaction():
    cells = joint_effect_cells(1, 4, [0, 1, 4, 5, 8])
    assert cells == dict(base=1, direction_only=5, release_only=0, full=4)
    means = {a: {'fed': v} for a, v in [(1, .3), (5, .4), (0, .5), (4, .9)]}
    d = factor_effects(cells, means, 'fed')
    assert d['full_difference'] == pytest.approx(.6)
    assert d['direction_at_base_release'] == pytest.approx(.1)
    assert d['release_at_base_direction'] == pytest.approx(.2)
    assert d['direction_release_interaction'] == pytest.approx(.3)


def test_missing_legal_cells_or_waiting_are_not_fabricated():
    assert joint_effect_cells(1, 4, [1, 4, 8]) is None
    assert joint_effect_cells(8, 4, [4, 8]) is None
    state, scene = sample()
    scene['scenario_id'] = 'other'
    with pytest.raises(ValueError, match='scenario mismatch'):
        no_advice_input_from_cache(state, scene)
