"""Check factor orientation, interactions, empirical ties and cell identities."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/run_crossed_teacher_rng_diagnostic.py"
spec = importlib.util.spec_from_file_location("crossed_teacher", path)
audit = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(path.parent))
spec.loader.exec_module(audit)


def test_population_only_variation_does_not_become_stream_or_interaction():
    gains = np.array([[[0., 1.], [0., 1.]], [[2., 4.], [2., 4.]]])
    population, stream, interaction = audit.contrasts(gains)
    np.testing.assert_array_equal(population, [2., 3.])
    np.testing.assert_array_equal(stream, [0., 0.])
    np.testing.assert_array_equal(interaction, [0., 0.])


def test_future_only_variation_does_not_become_population():
    gains = np.array([[[1.], [4.]], [[1.], [4.]]])
    population, stream, interaction = audit.contrasts(gains)
    np.testing.assert_array_equal(population, [0.])
    np.testing.assert_array_equal(stream, [3.])
    np.testing.assert_array_equal(interaction, [0.])


def test_crossed_sign_reversal_has_interaction_despite_zero_main_differences():
    gains = np.array([[[1.], [-1.]], [[-1.], [1.]]])
    population, stream, interaction = audit.contrasts(gains)
    np.testing.assert_array_equal(population, [0.])
    np.testing.assert_array_equal(stream, [0.])
    np.testing.assert_array_equal(interaction, [4.])


def test_pairwise_optimum_overlap_is_not_transitive():
    # Opposite diagonal sets A/B; each adjacent cell accepts both A and B.
    diagonal = np.array([[0., .01], [.01, 0.]])
    middle = np.array([0., 0.])
    assert audit.overlap(diagonal) == (0, 1.0)
    assert audit.overlap(np.stack((diagonal[0], middle)))[0] == 1
    assert audit.overlap(np.stack((diagonal[1], middle)))[0] == 1
    assert audit.overlap(diagonal, .20)[0] == 0


def test_pool_six_futures_before_maximizing_world_means():
    cube = np.zeros((2, 2, 2, 17, 3, 11))
    cube[:, 0, 0, :, :, 0] = 1
    cube[:, 1, 1, :, :, 0] = 1
    pooled = audit.pooled_scores(cube, [True, True])
    np.testing.assert_array_equal(pooled[:, :, 0], np.full((2, 17), .5))
    # Mean of the two separate worst-world costs would incorrectly give 1.


def test_cell_validation_rejects_swapped_future_stream_identity():
    group = {"supported": [True, True], "populations": [
        {"population_seed": 4, "planning_seeds": [10, 11, 12]},
        {"population_seed": 5, "planning_seeds": [20, 21, 22]}]}
    risks = np.zeros((2, 17, 3, 11))
    risks[..., 2] = 1
    cell = {"population_index": 0, "stream_index": 1, "population_seed": 4,
            "planning_seeds": np.array([10, 11, 12]), "supported": np.ones(2, bool),
            "risks": risks, "planning_rollouts": 102}
    with pytest.raises(AssertionError):
        audit.validate_cell(cell, group, 0, 1)
    cell["planning_seeds"] = np.array([20, 21, 22])
    audit.validate_cell(cell, group, 0, 1)
