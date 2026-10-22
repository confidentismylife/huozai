import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from consistent_gain_policy import BASE, MIN_GAIN, winner, decide, ensemble_scores, family_margin, fit_gain, GainRidge


def test_negative_or_tied_predictions_keep_baseline():
    g = np.full(17, -.2); g[BASE] = 0
    assert winner(g) == BASE
    assert decide(g, .01)["execution"] == BASE
    assert winner(np.zeros(17)) == BASE


def test_ranking_and_override_use_same_adjusted_score():
    g = np.zeros(17); g[0] = .04; g[1] = .02
    penalty = np.full(17, .001); penalty[0] = .10
    result = decide(g, penalty)
    assert result["raw_gain_choice"] == 0
    assert result["consistent_nominee"] == result["execution"] == 1
    assert result["scores"][1] > MIN_GAIN


def test_strict_threshold_and_all_negative_adjusted_scores():
    g = np.zeros(17); g[0] = MIN_GAIN
    result = decide(g, 0)
    assert result["consistent_nominee"] == 0 and result["execution"] == BASE
    assert decide(g, .01)["consistent_nominee"] == BASE


def test_baseline_does_not_receive_uncertainty_penalty():
    result = decide(np.zeros(17), 100)
    assert result["scores"][BASE] == 0 and result["execution"] == BASE


def test_malformed_scores_and_penalties_rejected():
    g = np.zeros(17); g[BASE] = .001
    with pytest.raises(ValueError): winner(g)
    with pytest.raises(ValueError): decide(np.zeros(17), -.001)
    with pytest.raises(ValueError): decide(np.full(17, np.nan), 0)


def test_ensemble_can_select_less_uncertain_candidate():
    g = np.zeros((3, 17, 6)); g[:, 0, 0] = [0, .05, .10]; g[:, 1, 0] = .02
    result = ensemble_scores(g, 1)
    assert result["raw_gain_choice"] == 0 and result["execution"] == 1


def test_nominee_margin_uses_only_fixed_nominee_and_family_worst():
    g = np.zeros((4, 17)); g[:, 0] = [.01, .02, .03, .04]
    actual = g.copy(); actual[:, 0] -= [.01, .02, .03, .04]; actual[:, 1] = -100
    margin, info = family_margin(g, actual, np.array(["a", "a", "b", "b"]))
    assert info["family_max_nominee_optimism"] == {"a": .02, "b": .04}
    assert margin == .04


def test_ridge_learns_gain_and_preserves_zero_baseline(tmp_path):
    x = np.zeros((10, 49)); x[:, 0] = np.arange(10)
    y = np.zeros((10, 17)); y[:, 0] = .01*x[:, 0]
    model = fit_gain(x, y, np.array(["a"]*5 + ["b"]*5))
    pred = model.predict(x)
    assert np.corrcoef(pred[:, 0], y[:, 0])[0, 1] > .99
    np.testing.assert_array_equal(pred[:, BASE], np.zeros(10))
    model.save(tmp_path / "model.npz")
    np.testing.assert_array_equal(GainRidge.load(tmp_path / "model.npz").predict(x), pred)


def test_forbidden_family_cannot_enter_fitting():
    with pytest.raises(ValueError):
        fit_gain(np.zeros((2, 49)), np.zeros((2, 17)), np.array(["train", "heldout"]), ["heldout"])


def test_advice_features_removed_and_family_weights_used():
    x = np.zeros((3, 49)); x[:, 0] = [1, 1, 10]; x[:, 28:33] = 999
    model = fit_gain(x, np.zeros((3, 17)), np.array(["a", "a", "b"]))
    assert model.mean[0] == 5.5
    np.testing.assert_array_equal(model.mean[28:33], np.zeros(5))
