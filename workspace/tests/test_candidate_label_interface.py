from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from candidate_label_interface import (PPO_NAMES, DIAGNOSTIC_ROLE, candidate_indices,
    planning_labels, choose_candidate, family_weights, require_training_partition)

NAMES = tuple(f"manual{i}" for i in range(12)) + PPO_NAMES


def source():
    risks = np.zeros((2, 17, 6, 1)); risks[0, 12] = .10; risks[1, 12] = .03
    risks[0, 13] = .02; risks[1, 13] = .08
    return {"observation": np.arange(49, dtype=float), "supported": np.array([True, True]), "planning_risks": risks}


def test_all_candidates_retained_when_old_nominee_is_baseline():
    data = source(); data["old_proposal"] = np.array(12)
    out = planning_labels(data, NAMES)
    assert out["features"].shape == (5, 54)
    np.testing.assert_array_equal(out["plan_indices"], range(12, 17))
    np.testing.assert_array_equal(out["features"][:, -5:], np.eye(5))
    assert out["target_R"][1] == pytest.approx(.02) and out["target_M"][1] == pytest.approx(-.05)


def test_registry_order_resolved_by_name():
    names = list(NAMES); names[13], names[16] = names[16], names[13]
    assert tuple(candidate_indices(names)) == (12, 16, 14, 15, 13)


def test_actual_results_true_source_and_old_predictions_cannot_change_labels():
    data = source(); first = planning_labels(data, NAMES)
    data.update({"evaluation_outcomes": "poison", "source_branch": object(), "predicted_gains": np.full(17, np.nan), "old_proposal": 999})
    second = planning_labels(data, NAMES)
    for key in first: np.testing.assert_array_equal(first[key], second[key])


def test_features_mask_existing_columns_and_do_not_mutate_source():
    data = source(); x = data["observation"].copy(); out = planning_labels(data, NAMES)
    assert not out["features"][:, 28:33].any()
    np.testing.assert_array_equal(data["observation"], x)


def test_supported_world_only_and_paired_future_baseline_zero():
    data = source(); data["supported"] = np.array([False, True]); out = planning_labels(data, NAMES)
    assert out["target_R"][1] == out["target_M"][1] == pytest.approx(-.05)
    assert out["target_R"][0] == out["target_M"][0] == 0
    assert not out["paired_future_gains"][:, 0].any()


def test_ties_threshold_and_retained_margin():
    assert choose_candidate(np.zeros(5)) == 0
    g = np.zeros(5); g[1] = .001; assert choose_candidate(g) == 0
    g[1] = g[3] = .02; assert choose_candidate(g) == 1
    assert choose_candidate(g, .0195) == 0
    assert choose_candidate(g, .018) == 1


def test_equal_total_weight_for_families_with_unequal_rows():
    f = np.array(["A", "A", "B"]); w = family_weights(f)
    assert w.sum() == pytest.approx(1) and w[f == "A"].sum() == w[f == "B"].sum() == pytest.approx(.5)


def test_viewed_GTEST_export_rejected_by_fit_guard():
    with pytest.raises(ValueError, match="diagnostic"):
        require_training_partition(DIAGNOSTIC_ROLE, ["GTEST0004"], ["GTEST0004"], [])


def test_fresh_partition_rejects_family_leakage_and_accepts_registered_training():
    with pytest.raises(ValueError): require_training_partition("fresh_family_disjoint_training", ["A"], ["A"], ["A"])
    with pytest.raises(ValueError): require_training_partition("fresh_family_disjoint_training", ["B"], ["A"], ["B"])
    assert require_training_partition("fresh_family_disjoint_training", ["A"], ["A"], ["B"])


def test_malformed_or_negative_loss_and_empty_support_rejected():
    data = source(); data["supported"][:] = False
    with pytest.raises(ValueError): planning_labels(data, NAMES)
    data = source(); data["planning_risks"][0, 13, 0, 0] = -.01
    with pytest.raises(ValueError): planning_labels(data, NAMES)
    with pytest.raises(ValueError): candidate_indices(NAMES[:-1])
