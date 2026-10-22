import numpy as np
import pytest
from fire_evac.safestep_baseline import SmallMap
from fire_evac.safestep_execution_checks import select_action, policy_graph_witness


def test_mask_rejects_highest_wall_score_and_uses_only_geometry():
    grid = SmallMap.paper_case_map()
    state = grid.labels.index('Room_A')
    legal = [a for a, _ in grid.valid_neighbors(state)]
    assert legal == [3]
    assert select_action([100, 90, 80, -5]) == 0
    assert select_action([100, 90, 80, -5], legal) == 3


def test_ties_preserve_first_index_and_legal_raw_winner():
    assert select_action([3, 3, 1, 1], [1, 0]) == 0
    assert select_action([1, 5, 0, 0], [0, 1]) == select_action([1, 5, 0, 0])
    with pytest.raises(ValueError): select_action([1, 2, 3, 4], [])
    with pytest.raises(ValueError): select_action([1, 2, 3, np.nan], [0, 1])


def test_geometry_mask_cannot_guarantee_escape_from_a_legal_cycle():
    grid = SmallMap.paper_case_map()
    actions = [0] * 24
    actions[3] = 1
    actions[4] = 0
    witness = policy_graph_witness(grid, actions, 3)
    assert not witness['graph_reaches_exit']
    assert witness['cycle_type'] == 'legal_movement_cycle'
    assert witness['cycle_states'] == [3, 4]
    assert witness['cycle_length'] == 2


def test_wall_loop_and_exit_are_distinct_graph_outcomes():
    grid = SmallMap.paper_case_map()
    actions = [0] * 24
    assert policy_graph_witness(grid, actions, 23)['cycle_type'] == 'wall_self_loop'
    actions[0] = 2
    witness = policy_graph_witness(grid, actions, 1)
    assert witness['graph_reaches_exit'] and witness['exit_label'] == 'Exit_A'
