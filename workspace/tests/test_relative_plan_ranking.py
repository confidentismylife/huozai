from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.relative_plan_ranking import (
    BASELINE_PLAN_INDEX, ObservableScoreStudent, PlanScoreNetwork,
    baseline_relative_targets, choose_priority, fit_relative_student,
    fit_score_student, ranking_loss, teacher_primary_regret,
)
from fire_evac.plan_distillation import relative_priority_targets
import fire_evac.relative_plan_ranking as relative_module


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def training_arrays():
    rng = np.random.default_rng(195)
    x = rng.random((5, 49), dtype=np.float32)
    y = rng.random((5, 17, 6), dtype=np.float32)
    y -= y.min(axis=1, keepdims=True)
    families = np.asarray(["a", "a", "b", "c", "c"])
    train, calibration = np.arange(5) < 3, np.arange(5) >= 3
    return x, y, families, train, calibration


def test_baseline_subtraction_preserves_every_lexicographic_plan_rank_and_zero():
    rng = np.random.default_rng(190)
    y = rng.integers(0, 6, size=(4, 17, 6)).astype(np.float32)
    original = y.copy()
    signed = baseline_relative_targets(y)
    for row in range(len(y)):
        assert sorted(range(17), key=lambda p: tuple(y[row, p]) + (p,)) == sorted(
            range(17), key=lambda p: tuple(signed[row, p]) + (p,))
    np.testing.assert_array_equal(signed[:, BASELINE_PLAN_INDEX], np.zeros((4, 6)))
    np.testing.assert_array_equal(y, original)
    assert np.any(signed < 0) and np.any(signed > 0)
    with pytest.raises(ValueError, match="predeclared baseline"):
        baseline_relative_targets(y, 13)


def test_gap_ranking_favors_correct_order_ignores_primary_ties_and_keeps_gradients():
    targets = torch.zeros((2, 17, 6))
    targets[0, :, 0] = torch.arange(17) / 4
    weights = torch.tensor([0.5, 0.5])
    correct = targets.clone().requires_grad_()
    reversed_scores = -targets
    good_loss, good = ranking_loss(correct, targets, weights)
    bad_loss, bad = ranking_loss(reversed_scores, targets, weights)
    assert good["pairwise_ranking"] < bad["pairwise_ranking"]
    assert good_loss < bad_loss
    ties_loss, ties = ranking_loss(torch.zeros_like(targets), torch.zeros_like(targets), weights)
    assert float(ties_loss) == 0 and float(ties["pairwise_ranking"]) == 0
    good_loss.backward()
    assert torch.isfinite(correct.grad).all() and torch.any(correct.grad != 0)
    common_shift = torch.ones_like(targets) * 4
    _, shifted = ranking_loss(correct.detach() + common_shift, targets, weights)
    torch.testing.assert_close(shifted["pairwise_ranking"], good["pairwise_ranking"])


@pytest.mark.parametrize("objective", ["mse", "relative_rank"])
def test_both_objectives_use_train_only_family_equal_normalizers(objective):
    x, y, families, train, cal = training_arrays()
    x[0], x[1], x[2] = 0, 2, 10
    student, curve, info = fit_score_student(x, y, families, train, cal, 801,
                                            epochs=20, objective=objective)
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[cal] *= 1000
    changed_y[cal] *= 1000
    other, _, _ = fit_score_student(changed_x, changed_y, families, train, cal, 801,
                                    epochs=20, objective=objective)
    np.testing.assert_array_equal(student.input_mean, other.input_mean)
    np.testing.assert_array_equal(student.input_std, other.input_std)
    np.testing.assert_array_equal(student.target_scale, other.target_scale)
    assert student.input_mean[0] == 5.5  # Equal family mass, rather than equal rows.
    assert info["training_families"] == 2 and info["calibration_families"] == 1
    assert len(curve) == info["actual_epochs"] == 20
    assert curve[0]["calibration_loss"] is None
    assert curve[9]["calibration_loss"] is not None
    assert info["checkpoint_selection"].endswith("ties_keep_earliest")


@pytest.mark.parametrize("objective", ["mse", "relative_rank"])
def test_common_regret_selection_keeps_earliest_equal_checkpoint(objective):
    x, y, families, train, cal = training_arrays()
    y[:, :, 0] = 0  # Every choice has zero primary teacher regret.
    selected = np.asarray([False, False, False, True, False])
    student, curve, info = fit_score_student(x, y, families, train, cal, 801,
                                            epochs=130, objective=objective,
                                            selection_mask=selected)
    assert student.selected_epoch == 10
    assert info["best_calibration_primary_regret"] == 0
    assert info["actual_epochs"] == 110
    assert info["selection_rows"] == 1
    assert sum(row["selected_checkpoint"] for row in curve) == 1
    assert info["best_calibration_loss"] == curve[9]["calibration_loss"]


def test_family_disjointness_excludes_audit_rows_and_invalid_selection():
    x, y, families, train, cal = training_arrays()
    with pytest.raises(ValueError, match="scenario family"):
        fit_relative_student(x, y, ["a"] * 5, train, cal, 801, epochs=10)
    incomplete = cal.copy()
    incomplete[-1] = False
    with pytest.raises(ValueError, match="only training and calibration"):
        fit_relative_student(x, y, families, train, incomplete, 801, epochs=10)
    with pytest.raises(ValueError, match="subset of calibration"):
        fit_score_student(x, y, families, train, cal, 801, epochs=10,
                          selection_mask=train)
    with pytest.raises(ValueError, match="subset of calibration"):
        fit_score_student(x, y, families, train, cal, 801, epochs=10,
                          selection_mask=np.zeros(5, bool))


@pytest.mark.parametrize("objective", ["mse", "relative_rank"])
def test_roundtrip_predictions_are_deterministic_masked_and_use_plan_ids(tmp_path, objective):
    x, y, families, train, cal = training_arrays()
    student, _, _ = fit_score_student(x, y, families, train, cal, 802,
                                     epochs=20, objective=objective)
    path = tmp_path / f"{objective}.pt"
    student.save(path)
    loaded = ObservableScoreStudent.load(path)
    prediction = student.priorities(x[0])
    np.testing.assert_array_equal(loaded.priorities(x[0]), prediction)
    np.testing.assert_array_equal(student.priorities(x[0]), prediction)
    altered = x[0].copy()
    altered[28:33] = 99
    np.testing.assert_array_equal(student.priorities(altered), prediction)
    assert loaded.objective == objective
    assert loaded.choose(x[0]) == student.choose(x[0])
    assert student.choose(x[0]) in range(17)
    if objective == "relative_rank":
        np.testing.assert_array_equal(prediction[BASELINE_PLAN_INDEX], np.zeros(6))
    scores = np.ones((17, 6))
    scores[16, 0] = -1
    assert choose_priority(scores) == 16


def test_signed_primary_regret_has_no_clipping_and_weights_families():
    targets = np.zeros((2, 17, 6))
    targets[0, 16, 0], targets[1, 15, 0] = -2, -4
    predictions = np.zeros_like(targets)  # Choose plan zero, despite negative best scores.
    assert teacher_primary_regret(predictions, targets, [.25, .75]) == 3.5
    shifted = targets + 10
    assert teacher_primary_regret(predictions, shifted, [.25, .75]) == 3.5


def test_forecast_only_student_is_invariant_to_all_other_observable_features():
    x, y, families, train, cal = training_arrays()
    student, _, _ = fit_relative_student(x, y, families, train, cal, 803,
                                        forecast_only=True, epochs=10)
    changed = x[0].copy()
    changed[:19] = -100
    changed[28:] = 100
    np.testing.assert_array_equal(student.priorities(x[0]), student.priorities(changed))


def test_network_contract_retains_all_seventeen_candidates():
    assert PlanScoreNetwork()(torch.zeros((3, 49))).shape == (3, 17, 6)


def test_remaining_and_congestion_keep_worst_world_means_after_baseline_centering():
    rng = np.random.default_rng(199)
    risks = rng.random((2, 17, 3, 11))
    risks[1, :, :, 3] += 2  # World 1 dominates final remaining.
    risks[0, :, :, 4] += 4  # World 0 dominates additional congestion.
    scores = baseline_relative_targets(relative_priority_targets(risks, [True, True]))
    for output_field, risk_field in ((4, 3), (5, 4)):
        world_means = risks[..., risk_field].mean(axis=2)
        worst = world_means.max(axis=0)
        np.testing.assert_allclose(scores[:, output_field], worst - worst[12], atol=2e-7)
        average = world_means.mean(axis=0)
        assert not np.allclose(scores[:, output_field], average - average[12])


@pytest.mark.parametrize("objective", ["mse", "relative_rank"])
def test_checkpoint_regret_receives_only_initial_selection_rows_with_family_equal_weights(monkeypatch, objective):
    rng = np.random.default_rng(200)
    x = rng.random((7, 49), dtype=np.float32)
    targets = rng.random((7, 17, 6), dtype=np.float32)
    targets -= targets.min(axis=1, keepdims=True)
    families = np.asarray(["a", "a", "b", "c", "c", "d", "d"])
    train, calibration = np.arange(7) < 3, np.arange(7) >= 3
    selection = np.asarray([False, False, False, True, False, True, False])
    captured = []
    original = relative_module.teacher_primary_regret

    def record_regret(predictions, teacher_targets, weights):
        captured.append((np.asarray(teacher_targets).copy(), np.asarray(weights).copy()))
        return original(predictions, teacher_targets, weights)

    monkeypatch.setattr(relative_module, "teacher_primary_regret", record_regret)
    _, _, info = fit_score_student(x, targets, families, train, calibration, 801,
                                   objective=objective, epochs=10, selection_mask=selection)
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0][0], targets[selection])
    np.testing.assert_array_equal(captured[0][1], [.5, .5])
    np.testing.assert_array_equal(captured[1][0], targets[calibration])
    np.testing.assert_array_equal(captured[1][1], [.25, .25, .25, .25])
    assert info["selection_rows"] == 2 and info["selection_families"] == 2
