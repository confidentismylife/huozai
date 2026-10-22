import numpy as np
import pytest
from fire_evac.safestep_baseline import SmallMap
from fire_evac.safestep_q_cache_audit import exact_action_values, closed_walk_members, returns_to_go


def test_reference_accounts_for_forced_action_and_terminal_failure_penalty():
    grid = SmallMap(('start', 'middle', 'exit'), ((0, 0), (1, 0), (2, 0)), start=0, exits=(2,))
    rates = np.zeros((4, 3))
    q = exact_action_values(grid, rates, 1, .1, 20, 2)
    assert q[0, 0, 1] == pytest.approx(-4)  # Two moves to exit.
    assert q[2, 0, 1] == pytest.approx(-42)  # Only one move remains.
    assert q[2, 1, 1] == pytest.approx(-2)   # Exit, no failure penalty.
    assert q[1, 1, 0] == pytest.approx(-44)  # Forced return prevents timely exit.
    assert np.isnan(q[0, 0, 0])


def test_positional_cycle_members_exclude_pre_reveal_walks():
    pairs = [(0, 1), (1, 0), (0, 1), (1, 2), (2, 1)]
    trace = [{'from_state': a, 'to_state': b, 'from_cell': str(a), 'to_cell': str(b)} for a, b in pairs]
    member, witnesses = closed_walk_members(trace, 2)
    assert member.tolist() == [False, False, False, True, True]
    assert witnesses[0]['cells'] == ['1', '2', '1']
    assert witnesses[0]['begin_k'] == 3


def test_saved_suffix_return_includes_terminal_cost_once():
    assert returns_to_go([-.1, -.2, -40.3]).tolist() == pytest.approx([-40.6, -40.5, -40.3])
