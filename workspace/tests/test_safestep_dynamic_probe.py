import itertools
import numpy as np
import pytest
from fire_evac.safestep_baseline import SmallMap
from fire_evac.safestep_dynamic_probe import ControlTickField, future_j_plan, snapshot_path


def test_snapshot_equal_rates_reduces_to_shortest_and_forced_exit_respected():
    grid = SmallMap.paper_case_map()
    rates = np.full(24, .00005)
    assert len(snapshot_path(grid, grid.start, rates, 2/.3, .00125)) == 8
    assert len(snapshot_path(grid, grid.start, rates, 2/.3, .00125, 20)) == 15
    rates[14:20] = .05
    assert len(snapshot_path(grid, grid.start, rates, 2/.3, .00125)) == 15


def test_control_tick_field_does_not_expose_future_before_onset():
    rates = np.zeros((5, 24))
    rates[2:, 0] = .1
    field = ControlTickField(rates, 220, 2/.3)
    assert field.rate(0, 220 + 2/.3) == 0
    assert field.rate(0, 220 + 2*(2/.3)) == .1
    with pytest.raises(ValueError):
        field.rate(0, 219)


def test_future_reference_matches_exhaustive_legal_paths_on_tiny_graph():
    # Independent exhaustive paths, including backtracks; a terminal exit is final.
    grid = SmallMap(('start', 'right', 'exit_a', 'left', 'exit_b'),
        ((0, 0), (1, 0), (2, 0), (-1, 0), (-2, 0)), start=0, exits=(2, 4))
    rates = np.array([[.01, .5, 0, 0, 0], [.01, 0, 0, .25, 0],
                      [.01, .5, 0, .25, 0], [.01, 0, 0, .25, 0], [.01, 0, 0, .25, 0]])
    path, value = future_j_plan(grid, rates, 1, .001, 0)
    candidates = []
    for length in range(1, 5):
        for actions in itertools.product(range(4), repeat=length):
            state, cost = 0, 0.
            for k, action in enumerate(actions):
                if state in grid.exits:
                    break
                new = grid.transition(state, action)
                if new == state:
                    break
                cost += rates[k, state] + .001
                state = new
            else:
                if state in grid.exits:
                    candidates.append(cost)
    assert value == pytest.approx(min(candidates))
    assert path == [1, 1]
    assert snapshot_path(grid, 0, rates[0], 1, .001) == [0, 0]


def test_replanner_only_uses_supplied_snapshot():
    grid = SmallMap.paper_case_map()
    before = np.full(24, .00005)
    after = before.copy()
    after[14:20] = .05
    assert snapshot_path(grid, grid.start, before, 2/.3, .00125) == snapshot_path(
        grid, grid.start, before.copy(), 2/.3, .00125)
    assert snapshot_path(grid, grid.start, before, 2/.3, .00125) != snapshot_path(
        grid, grid.start, after, 2/.3, .00125)
