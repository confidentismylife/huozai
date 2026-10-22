from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from override_worth_model import proposal, meta_features, labels, filtered_choice, WorthModel, fit_worth


def test_proposal_threshold_baseline_and_ties():
    g = np.zeros(17); g[0] = .001
    assert proposal(g) == 12
    g[0] = .002; assert proposal(g) == 0
    assert proposal(np.zeros(17)) == 12


def test_meta_only_observables_and_proposal_properties():
    x = np.zeros(49); x[28:33] = 999
    g = np.zeros(17); g[0] = .02; g[1] = .01
    f = meta_features(x, g, 0)
    assert f.shape == (68,) and np.all(f[28:33] == 0)
    assert f[49] == 1 and f[-2] == .02 and f[-1] == .01


def test_wrong_or_baseline_meta_proposal_rejected():
    with pytest.raises(ValueError): meta_features(np.zeros(49), np.zeros(17), 12)
    g = np.zeros(17); g[0] = .02
    with pytest.raises(ValueError): meta_features(np.zeros(49), g, 1)


def test_classes_use_meaningful_benefit_and_any_harm():
    np.testing.assert_array_equal(labels(np.array([-.01, 0, .0005, .001, .002])), [0, 1, 1, 1, 2])


def test_filter_never_expands_existing_acceptance():
    assert filtered_choice(12, [.05, .05, .90]) == 12
    assert filtered_choice(0, [.05, .05, .90]) == 0
    assert filtered_choice(0, [.11, .04, .85]) == 12
    assert filtered_choice(0, [.05, .20, .75]) == 12


def test_invalid_class_scores_rejected():
    with pytest.raises(ValueError): filtered_choice(0, [-.1, .2, .9])
    with pytest.raises(ValueError): filtered_choice(0, [.1, .1, .1])


def test_model_softmax_and_roundtrip(tmp_path):
    model = WorthModel(np.zeros(68), np.ones(68), np.zeros((68, 3)), np.array([0., 0., 1.]))
    p = model.probabilities(np.zeros(68))
    assert p[2] > p[0] and np.isclose(p.sum(), 1)
    model.save(tmp_path/"m.npz")
    np.testing.assert_array_equal(WorthModel.load(tmp_path/"m.npz").probabilities(np.zeros(68)), p)


def test_missing_training_class_rejected():
    with pytest.raises(ValueError): fit_worth(np.zeros((3, 68)), np.zeros(3, int), np.array(["a", "b", "c"]))
