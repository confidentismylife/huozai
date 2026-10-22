"""Meaningful checks for geometry, units, dose bookkeeping and planner semantics."""
import json
from pathlib import Path
import numpy as np
import pytest

from fire_evac.safestep_baseline import (SmallMap, HazardField, SafeStepEpisode, shortest_path,
    future_fed_oracle, toxic_utility, controlled_hazard, evaluate_actions)

CFG = json.loads((Path(__file__).resolve().parents[1] / 'schemas/safestep_reproduction_20261014.json').read_text(encoding='utf-8'))


def constant_field(grid, rate):
    return HazardField(np.array([0., 2000.]), np.full((2, 24), rate), grid.labels, 'test-only constant')


def test_original_figure_geometry_and_step_counts():
    grid = SmallMap.paper_case_map()
    assert len(grid.labels) == 24
    right, _ = shortest_path(grid, grid.start)
    assert len(right) == 8
    state = grid.start
    for a in right: state = grid.transition(state, a)
    assert grid.labels[state] == 'Exit_B'
    # Physically block R15 to require the left route, without blocking Room B.
    values = np.zeros((2, 24))
    values[:, 14] = .01
    hazard = HazardField(np.array([0., 2000.]), values, grid.labels, 'test')
    left, blocked = shortest_path(grid, grid.start, hazard, 220)
    assert len(left) == 15 and 14 in blocked


def test_exposure_units_and_invalid_actions_advance_time():
    grid = SmallMap.paper_case_map()
    env = SafeStepEpisode(grid, constant_field(grid, .001), CFG['paper_settings'], 220, 128)
    state, reward, _, info = env.step(0)  # Wall to left of Room B.
    assert state == grid.start and info['invalid']
    assert info['path_reward_kind'] == 'impossible'
    assert env.fed == pytest.approx(.001 * 2 / .3)
    assert reward == pytest.approx(.1 * -1 + .2 * -100 * env.fed)
    _, _, _, second = env.step(2)
    assert second['absolute_time_s'] == pytest.approx(220 + 2 / .3)
    assert env.fed == pytest.approx(2 * .001 * 2 / .3)


def test_utility_boundaries_match_equation_7():
    settings = CFG['paper_settings']
    assert [toxic_utility(x, settings) for x in [0, .01, .299999, .3, .999999, 1]] == [0, -100, -100, -300, -300, -800]


def test_stationary_node_dose_is_not_personal_evacuation_dose():
    grid = SmallMap.paper_case_map()
    hazard = constant_field(grid, .001)
    assert np.allclose(hazard.stationary_fed(565), .565)
    path, blocked = shortest_path(grid, grid.start, hazard, 565)
    assert path is None and len(blocked) == 24
    outcome, _ = evaluate_actions(grid, hazard, CFG['paper_settings'], 565, 128, shortest_path(grid, grid.start)[0])
    assert outcome['exit_reached'] and outcome['personal_FED'] == pytest.approx(8 * 2 / .3 * .001)


def test_time_expanded_oracle_agrees_with_direct_dose_integral():
    grid = SmallMap.paper_case_map()
    hazard = controlled_hazard(grid, CFG['controlled_hazard'])
    path, minimum = future_fed_oracle(grid, hazard, CFG['paper_settings'], 220, 128)
    outcome, _ = evaluate_actions(grid, hazard, CFG['paper_settings'], 220, 128, path)
    assert outcome['exit_reached'] and outcome['personal_FED'] == pytest.approx(minimum)
    assert outcome['steps'] == 15


def test_zero_hazard_reduces_oracle_to_a_feasible_route_and_reward_is_path_only():
    grid = SmallMap.paper_case_map()
    hazard = constant_field(grid, 0)
    path, minimum = future_fed_oracle(grid, hazard, CFG['paper_settings'], 220, 128)
    outcome, _ = evaluate_actions(grid, hazard, CFG['paper_settings'], 220, 128, path)
    assert minimum == 0 and outcome['exit_reached'] and outcome['personal_FED'] == 0
    env = SafeStepEpisode(grid, hazard, CFG['paper_settings'], 220, 128)
    assert env.step(2)[1] == pytest.approx(-.025)


def test_profile_rejects_negative_rates_and_out_of_domain_times():
    grid = SmallMap.paper_case_map()
    with pytest.raises(ValueError): constant_field(grid, -.001)
    with pytest.raises(ValueError): constant_field(grid, .001).rate(0, 2001)
