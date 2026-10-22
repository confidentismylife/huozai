"""Explicit native-time rate interpolation/integration for reconstruction work.

Rates must already be in reciprocal SI seconds. The curve model is piecewise
linear in RATE, not gas concentration. No extrapolation, reordering, silent
deduplication, or use of derived 5s samples as the integration authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass
class NativeRateCurve:
    times_s: Sequence[float]
    rates_per_s: Sequence[float]
    maximum_gap_s: float

    def __post_init__(self):
        self.times_s = np.asarray(self.times_s, dtype=float).copy()
        self.rates_per_s = np.asarray(self.rates_per_s, dtype=float).copy()
        t, r = self.times_s, self.rates_per_s
        if t.ndim != 1 or r.ndim != 1 or len(t) != len(r) or len(t) < 2:
            raise ValueError("At least two equally sized 1D native time/rate arrays required")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)) or np.any(t < 0) or np.any(r < 0):
            raise ValueError("Finite nonnegative fire times and per-second rates required")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Native times must be strictly increasing; no silent sort/deduplication")
        if not math.isfinite(self.maximum_gap_s) or self.maximum_gap_s <= 0:
            raise ValueError("Explicit positive finite maximum_gap_s required")
        if np.any(np.diff(t) > self.maximum_gap_s + 1e-9):
            raise ValueError("Native data gap exceeds the declared quality bound")

    def _within(self, points):
        a = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(a)) or np.any(a < self.times_s[0]) or np.any(a > self.times_s[-1]):
            raise ValueError("Requested times are outside native coverage; extrapolation prohibited")
        return a

    def sample(self, times_s):
        return np.interp(self._within(times_s), self.times_s, self.rates_per_s)

    def integrate(self, start_s: float, end_s: float) -> float:
        self._within([start_s, end_s])
        if end_s < start_s:
            raise ValueError("Reversed exposure interval")
        if start_s == end_s:
            return 0.0
        knots = np.concatenate(([start_s], self.times_s[(self.times_s > start_s) & (self.times_s < end_s)], [end_s]))
        rates = self.sample(knots)
        # Exact integral of the declared piecewise-linear RATE model.
        return float(np.sum(np.diff(knots)*(rates[:-1]+rates[1:])/2))

    def regular_samples(self, *, start_s: float, end_s: float, step_s: float = 5.0):
        if not math.isfinite(step_s) or step_s <= 0 or not math.isfinite(start_s) or not math.isfinite(end_s) or end_s < start_s:
            raise ValueError("Invalid regular grid")
        intervals = (end_s-start_s)/step_s
        if abs(intervals-round(intervals)) > 1e-9:
            raise ValueError("Requested grid end must lie on its regular grid")
        times = start_s+np.arange(round(intervals)+1)*step_s
        return times, self.sample(times)


def route_exposure(curves, segments):
    """Integrate ordered (region,start_fire_s,end_fire_s) segments, no gaps.

    Caller must specify a route and observation coverage; no hidden waiting,
    extrapolation, original-paper walking speeds or pre-evacuation exposure.
    """
    total = 0.0
    previous_end = None
    if not segments:
        raise ValueError("Explicit route segments required")
    for region, start, end in segments:
        if region not in curves:
            raise ValueError(f"Unknown region: {region}")
        if end < start or (previous_end is not None and abs(start-previous_end) > 1e-8):
            raise ValueError("Route segments must be ordered, contiguous and nonnegative in duration")
        total += curves[region].integrate(start, end)
        previous_end = end
    return total
