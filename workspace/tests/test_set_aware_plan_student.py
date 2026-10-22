import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

path = Path(__file__).resolve().parents[1] / "scripts/set_aware_plan_student.py"
spec = importlib.util.spec_from_file_location("set_student", path)
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


def test_set_accepts_primary_near_ties_without_overwriting_single_lexical_label():
    y = np.zeros((1, 17, 6), np.float32)
    y[0, :, 0] = .02
    y[0, [1, 2], 0] = 0
    y[0, 3, 0] = .0005
    y[0, 1, 2] = .01
    _, one = s.training_targets(y, "single")
    gain, multiple = s.training_targets(y, "set")
    assert np.flatnonzero(one[0]).tolist() == [2]
    assert np.flatnonzero(multiple[0]).tolist() == [1, 2, 3]
    np.testing.assert_array_equal(gain[:, 12], np.zeros((1, 6)))


def test_set_loss_ignores_redistribution_within_acceptable_set():
    logits = torch.tensor([[3., 1., -1.]])
    acceptable = torch.tensor([[True, True, False]])
    np.testing.assert_allclose(s.set_mass_loss(logits, acceptable), s.set_mass_loss(logits[:, [1, 0, 2]], acceptable))
    assert s.set_mass_loss(logits, acceptable) < s.set_mass_loss(logits, torch.tensor([[False, True, False]]))


def test_all_plans_acceptable_has_zero_set_loss_but_empty_set_is_invalid():
    logits = torch.tensor([[3., -2., 7.]])
    assert abs(float(s.set_mass_loss(logits, torch.ones((1, 3), dtype=torch.bool)))) < 1e-6
    with pytest.raises(ValueError):
        s.set_mass_loss(logits, torch.zeros((1, 3), dtype=torch.bool))


def test_network_enforces_zero_baseline_in_every_gain_field():
    network = s.SetGainNetwork()
    gains, _ = network(torch.ones((2, 49)))
    torch.testing.assert_close(gains[:, 12], torch.zeros((2, 6)), rtol=0, atol=0)


def predictions():
    gains = np.zeros((5, 17, 6))
    probabilities = np.zeros((5, 17))
    probabilities[:, 3] = 1
    return gains, probabilities


def test_guard_rejects_uncertain_positive_mean_and_accepts_strong_gain():
    gains, probabilities = predictions()
    gains[:, 3, 0] = [.001, .003, .005, .007, .009]
    nominee, guarded, info = s.ensemble_decision(gains, probabilities, k=2)
    assert nominee == 3 and guarded == 12 and info["blocked_nomination"]
    gains[:, 3, 0] = .02
    assert s.ensemble_decision(gains, probabilities, k=2)[1] == 3


def test_guard_never_changes_a_baseline_nomination():
    gains, probabilities = predictions()
    probabilities[:] = 0
    probabilities[:, 12] = 1
    assert s.ensemble_decision(gains, probabilities)[0:2] == (12, 12)


def test_margin_checks_all_plans_and_takes_worst_state_within_family():
    gains = np.zeros((5, 3, 17, 6))
    actual = np.zeros((3, 17, 6))
    # Errors on a plan that need not be nominated must still affect the margin.
    actual[0, 16, 0] = -.003
    actual[1, 16, 0] = -.007
    k, info = s.calibrate_margin(gains, actual, np.array(["F0", "F0", "F1"]))
    assert k == 7 and info["family_max_optimistic_residuals"] == {"F0": 7., "F1": 0.}


def test_fitting_rejects_unassigned_calibration_rows_before_training():
    with pytest.raises(ValueError, match="training and checkpoint"):
        s.fit(np.zeros((3, 49)), np.zeros((3, 17, 6)), np.array(["train", "checkpoint", "margin"]),
              np.zeros(3), np.array([True, False, False]), np.array([False, True, False]), 801, "set")
