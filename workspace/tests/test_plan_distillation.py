from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.plan_distillation import (
    ObservablePlanStudent, PlanScoreNetwork, choose_priority, family_weights,
    fit_student, observable_features, relative_priority_targets, stratified_family_split,
)
from fire_evac.source_world_planning import select_reference_plan


def test_relative_targets_reproduce_minimax_including_ties_and_supported_worlds():
    rng = np.random.default_rng(114)
    risks = rng.random((2, 17, 3, 11))
    for support in ([True, True], [True, False], [False, True]):
        targets = relative_priority_targets(risks, support)
        assert choose_priority(targets) == select_reference_plan(risks, support, 0, "robust")
        np.testing.assert_array_equal(targets.min(axis=0), np.zeros(6))
    risks[..., 0] = 0
    risks[..., 1] = .2
    assert choose_priority(relative_priority_targets(risks, [True, True])) == select_reference_plan(risks, [True, True], 0, "robust")
    with pytest.raises(ValueError, match="nonempty"):
        relative_priority_targets(risks, [False, False])


def test_feature_contract_masks_advice_and_forecast_ablation_without_input_mutation():
    x = np.arange(98, dtype=np.float32).reshape(2, 49)
    original = x.copy()
    full, forecast = observable_features(x), observable_features(x, True)
    np.testing.assert_array_equal(full[:, 28:33], np.zeros((2, 5)))
    np.testing.assert_array_equal(forecast[:, 19:28], x[:, 19:28])
    changed = x.copy()
    changed[:, :19] = -99
    changed[:, 28:] = 900
    np.testing.assert_array_equal(observable_features(changed, True), forecast)
    np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match="49-input"):
        observable_features(np.zeros((2, 48)))


def test_family_weights_and_stratified_split_keep_repeats_together():
    weights = family_weights(["a", "a", "b"])
    assert weights[:2].sum() == weights[2] == .5
    records = [{"scenario_id": f"{kind}_{i}", "conflict_type": kind} for kind in ("A", "B") for i in range(6)]
    split = stratified_family_split(records)
    assert split == stratified_family_split(records)
    for kind in ("A", "B"):
        assert sum(split[f"{kind}_{i}"] == "train" for i in range(6)) == 4


def test_student_normalization_uses_train_only_and_checkpoint_is_reproducible(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(115)
    x = rng.random((6, 49), dtype=np.float32)
    y = rng.random((6, 17, 6), dtype=np.float32)
    families = np.asarray(["a", "a", "b", "b", "c", "c"])
    train, cal = np.arange(6) < 4, np.arange(6) >= 4
    student, _, info = fit_student(x, y, families, train, cal, 801, epochs=20)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[cal] *= 1000
    changed_y[cal] *= 1000
    other, _, _ = fit_student(changed_x, changed_y, families, train, cal, 801, epochs=20)
    np.testing.assert_array_equal(student.input_mean, other.input_mean)
    np.testing.assert_array_equal(student.input_std, other.input_std)
    np.testing.assert_array_equal(student.target_scale, other.target_scale)
    assert info["training_families"] == 2 and info["calibration_families"] == 1
    path = tmp_path / "student.pt"
    student.save(path)
    loaded = ObservablePlanStudent.load(path)
    np.testing.assert_array_equal(loaded.priorities(x[0]), student.priorities(x[0]))
    assert loaded.choose(x[0]) == student.choose(x[0])
    altered = x[0].copy()
    altered[28:33] = 33
    np.testing.assert_array_equal(student.priorities(altered), student.priorities(x[0]))
    with pytest.raises(ValueError, match="scenario family"):
        fit_student(x, y, np.asarray(["a"] * 6), train, cal, 801, epochs=20)


def test_plan_id_is_not_confused_with_nine_action_id():
    scores = np.ones((17, 6))
    scores[16, 0] = -1
    assert choose_priority(scores) == 16
    network = PlanScoreNetwork()
    student = ObservablePlanStudent(network, np.zeros(49, np.float32), np.ones(49, np.float32),
                                    np.ones(6, np.float32), False, 801, 10)
    assert student.priorities(np.zeros(49)).shape == (17, 6)
