"""Safeguards against overstating empirical label conflicts and neighbors."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/audit_observable_teacher_labels.py"
spec = importlib.util.spec_from_file_location("observable_label_audit", path)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_different_lexicographic_labels_can_share_primary_optima():
    # Both plans have the same primary incapacitation; a FED tie-break changes.
    costs = np.array([[0, 0], [0, 0]])
    shared, regret = audit.primary_overlap(costs)
    assert shared == 2 and regret == 0


def test_conflicting_primary_optima_need_positive_empirical_regret():
    costs = np.array([[0, .01], [.01, 0]])
    assert audit.primary_overlap(costs) == (0, 1.0)
    assert audit.primary_overlap(costs, tolerance_pp=1.0) == (2, 1.0)


def test_exact_observation_group_uses_actual_masked_student_features():
    first = np.zeros(49, dtype=np.float32)
    second = first.copy()
    second[28:33] = 1  # Fields hidden from the actual student.
    second[0] = -0.0
    assert audit.exact_key(first) == audit.exact_key(second)
    second[44] = .01  # Observed smoke history matters in the full view.
    assert audit.exact_key(first) != audit.exact_key(second)
    assert audit.exact_key(first, True) == audit.exact_key(second, True)


def test_close_observations_are_not_merged_as_exact():
    first = np.zeros(49, dtype=np.float32)
    second = first.copy()
    second[21] = np.nextafter(np.float32(0), np.float32(1))
    assert audit.exact_key(first) != audit.exact_key(second)


def test_nearest_neighbor_respects_family_and_initial_state_exclusions():
    references = np.array([[0, 0], [.2, 0], [.5, 0]])
    index, distance = audit.nearest_index(np.array([0, 0]), references,
                                          np.ones(2), np.array([True, False]),
                                          np.array([False, False, True]))
    assert index == 2 and distance == .5


def test_empty_eligible_reference_is_an_error_not_a_fake_prediction():
    with pytest.raises(ValueError):
        audit.nearest_index(np.zeros(2), np.zeros((1, 2)), np.ones(2),
                            np.ones(2, bool), np.zeros(1, bool))


def test_robust_cost_is_max_of_world_means_not_mean_of_future_maxima():
    # Crossed future losses: each world has mean .5; max per future would be 1.
    risks = np.zeros((2, 17, 2, 11))
    risks[0, :, 0, 0] = 1
    risks[1, :, 1, 0] = 1
    score = audit.priorities(risks, np.ones(2, bool))
    np.testing.assert_array_equal(score[:, 0], np.full(17, .5))


def test_incompatible_world_nan_scores_are_not_used():
    risks = np.zeros((2, 17, 3, 11))
    risks[1] = np.nan
    np.testing.assert_array_equal(audit.priorities(risks, np.array([True, False])),
                                  np.zeros((17, 6)))


def test_float32_cache_target_reconstruction_matches_frozen_double_precision():
    from fire_evac.plan_distillation import relative_priority_targets
    risks = np.random.default_rng(8).random((2, 17, 3, 11)).astype(np.float32)
    scores = audit.priorities(risks, np.array([True, True]))
    np.testing.assert_array_equal(scores - scores.min(axis=0),
                                  relative_priority_targets(risks, np.array([True, True])))
