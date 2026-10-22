"""Past-only native-observation forecasting for the public SafeStep graph.

Online functions accept detached history prefixes, not the evaluator RateField.
The evaluator may use future truth for scoring; it must not pass it to a planner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from fire_evac.safestep_public_benchmark import Graph, future_min_fed_policy


@dataclass(frozen=True)
class ObservationPrefix:
    times: np.ndarray
    rates: np.ndarray
    decision_time: float

    def __post_init__(self):
        times = np.array(self.times, dtype=float, copy=True)
        rates = np.array(self.rates, dtype=float, copy=True)
        if (times.ndim != 1 or not len(times) or rates.ndim != 2
                or rates.shape[0] != len(times) or not np.isfinite(times).all()
                or not np.isfinite(rates).all() or np.any(rates < 0)
                or np.any(np.diff(times) <= 0) or not math.isfinite(self.decision_time)
                or times[-1] > self.decision_time + 1e-9):
            raise ValueError("Only finite nonnegative observed history at/before decision time")
        times.setflags(write=False)
        rates.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "rates", rates)


def available_prefix(native_times, measured_rates, decision_time):
    """Observation delivery adapter. No interpolation with unpublished samples."""
    times = np.asarray(native_times, dtype=float)
    rates = np.asarray(measured_rates, dtype=float)
    if times.ndim != 1 or rates.ndim != 2 or len(times) != len(rates):
        raise ValueError("Native observation dimensions disagree")
    end = int(np.searchsorted(times, decision_time + 1e-9, side="right"))
    if end == 0:
        raise ValueError("No observation at or before decision time")
    return ObservationPrefix(times[:end], rates[:end], float(decision_time))


def estimate_slope(history: ObservationPrefix, window_s: float):
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("Positive past-history window required")
    selected = history.times >= history.times[-1] - window_s - 1e-9
    t = history.times[selected]
    y = history.rates[selected]
    if len(t) < 2:
        return np.zeros(history.rates.shape[1])
    centered = t - t.mean()
    return (centered @ y) / float(centered @ centered)


@dataclass(frozen=True)
class CappedLinearForecast:
    origin: float
    anchor: np.ndarray
    slope: np.ndarray
    cap_s: float

    def __post_init__(self):
        anchor = np.array(self.anchor, dtype=float, copy=True)
        slope = np.array(self.slope, dtype=float, copy=True)
        if (anchor.ndim != 1 or anchor.shape != slope.shape or np.any(anchor < 0)
                or not np.isfinite(anchor).all() or not np.isfinite(slope).all()
                or not math.isfinite(self.origin) or not math.isfinite(self.cap_s)
                or self.cap_s <= 0):
            raiseError = "Invalid nonnegative anchor, finite slope or positive cap"
            raise ValueError(raiseError)
        anchor.setflags(write=False)
        slope.setflags(write=False)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "slope", slope)

    def sample(self, query_time):
        u = float(query_time) - self.origin
        if not math.isfinite(u) or u < -1e-9:
            raise ValueError("Forecast query precedes last observation")
        return np.maximum(0., self.anchor + self.slope * min(max(0., u), self.cap_s))

    def primitive(self, query_time):
        u = float(query_time) - self.origin
        if not math.isfinite(u) or u < -1e-9:
            raise ValueError("Forecast query precedes last observation")
        u = max(0., u)
        # A falling line contributes only until its first zero crossing.
        width = np.full_like(self.anchor, min(u, self.cap_s))
        falling = self.slope < 0
        width[falling] = np.minimum(width[falling],
                                   self.anchor[falling] / -self.slope[falling])
        area = self.anchor * width + .5 * self.slope * width**2
        held = np.maximum(0., self.anchor + self.slope * self.cap_s)
        return area + max(0., u - self.cap_s) * held

    def integrate(self, start, end):
        if end < start:
            raise ValueError("Reversed forecast interval")
        result = self.primitive(end) - self.primitive(start)
        if np.any(result < -1e-10):
            raise ValueError("Negative forecast integral")
        return np.maximum(result, 0.)


def make_forecast(history: ObservationPrefix, window_s=30., projection_cap_s=60., slope_gain=1.):
    if not math.isfinite(slope_gain) or slope_gain < 0:
        raise ValueError("Finite nonnegative slope gain required")
    slope = estimate_slope(history, window_s) * slope_gain
    return CappedLinearForecast(float(history.times[-1]), history.rates[-1], slope,
                                float(projection_cap_s))


def predicted_interval_costs(forecast, decision_time, dt_s, remaining_steps):
    if not math.isfinite(dt_s) or dt_s <= 0 or remaining_steps <= 0:
        raise ValueError("Positive duration and remaining steps required")
    boundaries = [forecast.primitive(decision_time + k * dt_s)
                  for k in range(remaining_steps + 1)]
    costs = np.diff(boundaries, axis=0)
    if np.any(costs < -1e-10):
        raise ValueError("Negative per-interval forecast cost")
    return np.maximum(costs, 0.)


def causal_first_actions(graph: Graph, history: ObservationPrefix, dt_s,
                         remaining_steps, window_s=30., projection_cap_s=60., slope_gain=1.):
    forecast = make_forecast(history, window_s, projection_cap_s, slope_gain)
    costs = predicted_interval_costs(forecast, history.decision_time, dt_s, remaining_steps)
    actions, values, moves = future_min_fed_policy(graph, costs)
    return actions[0].copy(), values[0].copy(), moves[0].copy(), forecast


def fixed_policy_values(graph: Graph, policies, true_costs):
    """Evaluator-only continuation dose of a frozen receding-horizon policy table.

    Infeasible deadline continuations remain infinite, not zero partial dose.
    """
    policies = np.asarray(policies, dtype=int)
    true_costs = np.asarray(true_costs, dtype=float)
    if policies.shape != true_costs.shape or true_costs.shape[1] != len(graph.cells):
        raise ValueError("Policy and cost dimensions disagree")
    horizon, n = true_costs.shape
    values = np.full((horizon + 1, n), math.inf)
    for e in graph.exit_ids:
        values[:, e] = 0.
    for k in range(horizon - 1, -1, -1):
        for i in range(n):
            if i in graph.exit_ids or policies[k, i] < 0:
                continue
            j = graph.destination(i, int(policies[k, i]))
            values[k, i] = true_costs[k, i] + values[k + 1, j]
    return values
