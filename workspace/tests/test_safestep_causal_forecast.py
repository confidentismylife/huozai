import numpy as np
import pytest

from fire_evac.safestep_causal_forecast import (
    ObservationPrefix, available_prefix, make_forecast, CappedLinearForecast,
    predicted_interval_costs, causal_first_actions, fixed_policy_values,
)
from fire_evac.safestep_public_benchmark import Graph, future_min_fed_policy, rollout


def test_off_grid_prefix_uses_no_future_endpoint_and_is_detached():
    times = np.arange(0., 31., 5.)
    measured = np.column_stack((times, times * 2))
    h = available_prefix(times, measured, 16.6666666667)
    assert h.times.tolist() == [0., 5., 10., 15.]
    assert h.rates[-1].tolist() == [15., 30.]
    assert not h.times.flags.writeable and not h.rates.flags.writeable
    measured[3] = 999
    assert h.rates[-1].tolist() == [15., 30.]
    with pytest.raises(ValueError):
        ObservationPrefix(np.array([15., 20.]), np.ones((2, 2)), 16.)


def test_future_suffix_perturbation_preserves_costs_and_actions():
    graph = Graph([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], ((2, 0),))
    times = np.arange(0., 101., 5.)
    measured = np.ones((len(times), 6)) + times[:, None] * .01
    altered = measured.copy()
    altered[times > 16.6667] *= np.arange(1, 7)[None, :] * 10000
    first = available_prefix(times, measured, 16.6667)
    second = available_prefix(times, altered, 16.6667)
    a = causal_first_actions(graph, first, 1., 5)
    b = causal_first_actions(graph, second, 1., 5)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(predicted_interval_costs(a[3], first.decision_time, 1., 5),
                          predicted_interval_costs(b[3], second.decision_time, 1., 5))


def test_capped_zero_crossing_integral_matches_independent_midpoint_sum():
    f = CappedLinearForecast(10., np.array([2., 3., 0., 1.]),
                             np.array([1., -1., -1., 0.]), 5.)
    assert f.integrate(10., 20.) == pytest.approx([57.5, 4.5, 0., 10.])
    assert f.sample(100.) == pytest.approx([7., 0., 0., 1.])
    x = np.linspace(1.3, 13.1, 200000, endpoint=False)
    step = (13.1 - 1.3) / len(x)
    # Independent expression, not calling the analytic primitive or sample.
    y = np.maximum(0., f.anchor[None, :] + f.slope[None, :]
                   * np.minimum(x[:, None] + step / 2, f.cap_s))
    assert f.integrate(11.3, 23.1) == pytest.approx(y.sum(axis=0) * step, abs=1e-8)
    with pytest.raises(ValueError):
        f.sample(9.)


def test_linear_ols_uses_fixed_history_and_is_anchored_to_latest():
    times = np.arange(0., 61., 5.)
    rates = np.column_stack((2. + times * .1, 10. - times * .1))
    rates[:5] += 100  # Outside last 30 seconds; must not affect slope.
    h = ObservationPrefix(times, rates, 63.)
    f = make_forecast(h)
    assert f.slope == pytest.approx([.1, -.1])
    assert f.sample(70.) == pytest.approx([9., 3.])
    half = make_forecast(h, slope_gain=.5)
    assert half.sample(70.) == pytest.approx([8.5, 3.5])


def test_constant_history_reduces_trend_to_snapshot():
    graph = Graph([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], ((2, 0),))
    h = ObservationPrefix(np.arange(0., 31., 5.),
                          np.tile([.1, 2., 0., .1, .1, .1], (7, 1)), 31.)
    snap = causal_first_actions(graph, h, 1., 5, slope_gain=0.)
    trend = causal_first_actions(graph, h, 1., 5, slope_gain=1.)
    assert np.array_equal(snap[0], trend[0])
    assert snap[1] == pytest.approx(trend[1])


def test_trend_handles_known_future_linear_reversal_and_common_deadline():
    graph = Graph([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], ((2, 0),))
    h = ObservationPrefix(np.array([0., 5.]),
                          np.array([[.1, .4, 0., 0., 0., 0.],
                                    [.1, .4, 0., .1, .1, .1]]), 5.)
    snap, _, _, _ = causal_first_actions(graph, h, 1., 5, slope_gain=0.)
    trend, _, _, _ = causal_first_actions(graph, h, 1., 5)
    assert snap[0] == 3  # Four-step detour wins if current rates stay fixed.
    assert trend[0] == 1  # Growing detour risk makes the two-step route better.
    constrained, _, _, _ = causal_first_actions(graph, h, 1., 2, slope_gain=0.)
    assert constrained[0] == 1  # Common deadline excludes the long detour.
    _, values, _, _ = causal_first_actions(graph, h, 1., 1)
    assert np.isinf(values[0])  # An unfinished route has no finite success value.


def test_fixed_continuation_values_match_rollout_and_keep_failure_infinite():
    graph = Graph([(0, 0), (1, 0), (2, 0)], ((2, 0),))
    costs = np.tile([.1, .2, 0.], (3, 1))
    policy, _, _ = future_min_fed_policy(graph, costs)
    values = fixed_policy_values(graph, policy, costs)
    assert values[0, 0] == pytest.approx(rollout(graph, policy, costs, 0)['FED'])
    bad = np.array([[1, 0, -1]] * 3)
    assert np.isinf(fixed_policy_values(graph, bad, costs)[0, 0])
