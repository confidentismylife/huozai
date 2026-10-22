import numpy as np
import pytest
from fire_evac.safestep_author_numerics import independent_author_integral, path_from_actions, shortest_to_exit


def test_author_integral_preserves_initial_sample_and_zero_exit():
    # One initial exposure, one intermediate exposure and no exit exposure.
    fed, trace = independent_author_integral([(0,0),(1,0),(2,0)],
                                            {(0,0):[.01]*10,(1,0):[.02]*10},dt=5)
    assert fed == pytest.approx(.15)
    assert [r["increment"] for r in trace] == pytest.approx([.05,.1,0])


def test_author_sample_uses_floor_index_and_caps_at_last_data_row():
    _, trace = independent_author_integral([(0,0)]*4,{(0,0):[1.,2.,3.]},dt=2/.3)
    assert [r["rate"] for r in trace] == [1.,2.,3.,3.]


def test_geometry_uses_four_moves_and_retains_invalid_action():
    grid = np.array([[0,0,1],[1,0,0]])
    assert shortest_to_exit(grid,(0,0),(2,1)) == [(0,0),(1,0),(1,1),(2,1)]
    assert path_from_actions(grid,(0,0),[0,1,3,1]) == [(0,0),(0,0),(1,0),(1,1),(2,1)]
