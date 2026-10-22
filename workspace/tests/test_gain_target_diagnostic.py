from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from gain_target_diagnostic import decompose, gain_class, paired_future_interval


def test_worst_loss_difference_is_not_minimum_paired_world_gain():
    risks = np.array([[[[.10]], [[.02]]], [[[.03]], [[.08]]]])
    d = decompose(.01, risks, risks[1], np.array([True, True]), 1)
    assert d["robust_gain_pp"] == pytest.approx(2)
    assert d["minimum_world_gain_pp"] == pytest.approx(-5)
    assert d["different_worst_sources"]


def test_exact_decomposition_with_distinct_future_samples():
    risks = np.array([[[[.12],[.10]], [[.08],[.04]]], [[[.10],[.09]], [[.03],[.04]]]])
    actual = np.array([[[.13],[.10],[.11]], [[.02],[.03],[.01]]])
    d = decompose(.02, risks, actual, np.array([True, True]), 1)
    assert d["total_error_pp"] == pytest.approx(sum(d[k] for k in ("prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp")))


def test_single_supported_world_has_zero_aggregation_difference():
    risks = np.array([[[[.90]], [[.10]]], [[[.10]], [[.08]]]])
    d = decompose(.02, risks, risks[1], np.array([False, True]), 1)
    assert d["robust_to_true_pp"] == 0
    assert d["minimum_world_gain_pp"] == pytest.approx(2)


def test_baseline_duplicate_produces_zero_gain():
    risks = np.full((2,2,3,2), .07)
    d = decompose(0, risks, risks[0], np.array([True,True]), 0)
    assert all(d[k] == 0 for k in ("robust_gain_pp","actual_gain_pp","total_error_pp","minimum_world_gain_pp"))


def test_unsupported_truth_is_rejected():
    with pytest.raises(ValueError):
        decompose(0, np.zeros((2,2,3,1)), np.zeros((2,3,1)), np.array([True,False]), 1)


def test_invalid_shapes_or_nonfinite_are_rejected():
    with pytest.raises(ValueError):
        decompose(0, np.zeros((2,3,3,1)), np.zeros((2,3,1)), np.array([True,True]), 0)
    with pytest.raises(ValueError):
        decompose(np.nan, np.zeros((2,2,3,1)), np.zeros((2,3,1)), np.array([True,True]), 0)


def test_gain_class_uses_percentage_points_and_any_harm():
    assert [gain_class(v) for v in (-.001, 0, .1, .11)] == ["harmful","neutral_or_small","neutral_or_small","beneficial"]


def test_bootstrap_pairs_future_across_fixed_populations():
    x = np.array([[0., 2., 4.], [4., 2., 0.]])
    np.testing.assert_array_equal(paired_future_interval(x, 9), [2.,2.])
