import numpy as np
import pytest

from fire_evac.safestep_public_benchmark import Graph, rollout
from fire_evac.safestep_causal_forecast import ObservationPrefix
from fire_evac.safestep_robust_gate import (
    delivered_history, build_bundle, gated_action, switch_threshold,
    region_distances, make_synthetic_truth, noisy_measurements, PlanBundle,
)

MODELS = [dict(id="trend30", window_s=30, projection_cap_s=60, slope_gain=1.),
          dict(id="half30", window_s=30, projection_cap_s=60, slope_gain=.5),
          dict(id="trend60", window_s=60, projection_cap_s=60, slope_gain=1.)]


def test_latency_retains_acquisition_timestamp_and_ignores_undelivered_future():
    t = np.arange(5., 101., 5.)
    x = np.column_stack([t, t * 2])
    a = delivered_history(t, x, 41.6667, 15.)
    assert a.times[-1] == 25.
    assert a.rates[-1].tolist() == [25., 50.]
    x[t > 26.6667] = 99999
    b = delivered_history(t, x, 41.6667, 15.)
    assert np.array_equal(a.rates, b.rates)
    with pytest.raises(ValueError):
        delivered_history(t, x, 10., 15.)


def test_constant_models_share_actions_and_gate_has_no_opportunity():
    g = Graph([(0, 0), (1, 0), (2, 0)], ((2, 0),))
    h = ObservationPrefix(np.arange(0., 61., 5.), np.tile([.2, .1, 0.], (13, 1)), 60.)
    bundle = build_bundle(g, h, 1., 4, MODELS)
    for method in bundle.first_actions:
        assert np.array_equal(bundle.first_actions[method], bundle.first_actions["snapshot_hold"])
    assert gated_action(bundle, 0) == (1, 2, False, False)


def test_gate_uses_worst_model_gain_and_never_truth():
    b = PlanBundle({"snapshot_hold": np.array([1]), "ensemble_mean": np.array([3])},
        np.array([9]), np.array([8]), np.array([.02]), np.array([[.4], [.02], [.3]]))
    assert gated_action(b, 0, .05) == (1, 9, True, False)
    assert gated_action(b, 0, .01) == (3, 8, True, True)
    assert switch_threshold(-1, 8) == .05
    assert switch_threshold(8, 8) == .01
    assert switch_threshold(9, 8) == .10


def test_generator_is_reproducible_nonnegative_and_preserves_pre_onset_source():
    t = np.arange(5., 901., 5.)
    source = np.ones((len(t), 3)) * .001
    distances = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :])
    for family in ("accelerating_local", "transient_local"):
        a, metadata = make_synthetic_truth(t, source, distances, 18101, family)
        b, _ = make_synthetic_truth(t, source, distances, 18101, family)
        assert np.array_equal(a, b)
        assert np.all(a >= source)
        assert np.array_equal(a[t <= 220], source[t <= 220])
        assert len(metadata['additions']) == 2
        noisy = noisy_measurements(a, 18101, .3)
        assert np.isfinite(noisy).all() and np.all(noisy >= 0)
        assert np.array_equal(noisy, noisy_measurements(a, 18101, .3))
        assert not np.array_equal(noisy, noisy_measurements(a, 18201, .3))


def test_region_distance_shared_cells_use_same_region_and_exits_no_shortcut():
    g = Graph([(0, 0), (1, 0), (2, 0), (3, 0)], ((3, 0),))
    d = region_distances(g, ['R1', 'R1', 'R2', 'EXIT'], ['R1', 'R2'])
    assert d.tolist() == [[0., 1.], [1., 0.]]


def test_future_data_cannot_change_any_bundle_value_or_gate():
    g = Graph([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], ((2, 0),))
    t = np.arange(5., 151., 5.)
    x = .1 + t[:, None] * np.arange(1, 7)[None, :] * .001
    x[:, 2] = 0
    h1 = delivered_history(t, x, 81.6667, 15.)
    x[t > 66.6667] *= 1e6
    h2 = delivered_history(t, x, 81.6667, 15.)
    a, b = build_bundle(g, h1, 1., 5, MODELS), build_bundle(g, h2, 1., 5, MODELS)
    for key in a.first_actions:
        assert np.array_equal(a.first_actions[key], b.first_actions[key])
    assert np.array_equal(a.relative_gain_by_model, b.relative_gain_by_model)
    assert gated_action(a, 0) == gated_action(b, 0)
