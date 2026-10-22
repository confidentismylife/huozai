"""Scenario-agreement routing gate, using observed history only.

The ensemble is a fixed engineering collection, not a probabilistic confidence
set. Gating compares identical proposed/comparator plans under every model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from fire_evac.safestep_public_benchmark import Graph, future_min_fed_policy
from fire_evac.safestep_causal_forecast import (
    ObservationPrefix, make_forecast, predicted_interval_costs, fixed_policy_values,
)


def delivered_history(times, measured, decision_time, latency_s=0.):
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError("Nonnegative delivery delay required")
    times = np.asarray(times, dtype=float)
    end = int(np.searchsorted(times + latency_s, decision_time + 1e-9, side="right"))
    if not end:
        raise ValueError("No delivered measurement")
    return ObservationPrefix(times[:end], np.asarray(measured)[:end], float(decision_time))


def policy_terminal_ids(graph: Graph, actions):
    """Exit of each open-loop fixed plan; -1 for failure by the deadline."""
    horizon, n = actions.shape
    exits = np.full((horizon + 1, n), -1, dtype=int)
    for e in graph.exit_ids:
        exits[:, e] = e
    for k in range(horizon - 1, -1, -1):
        for i in range(n):
            if i in graph.exit_ids or actions[k, i] < 0:
                continue
            exits[k, i] = exits[k + 1, graph.destination(i, int(actions[k, i]))]
    return exits[0]


@dataclass
class PlanBundle:
    first_actions: dict[str, np.ndarray]
    baseline_exit: np.ndarray
    nominee_exit: np.ndarray
    minimum_relative_gain: np.ndarray
    relative_gain_by_model: np.ndarray


def build_bundle(graph, history, dt_s, remaining, models):
    n = len(graph.cells)
    held = np.tile(history.rates[-1] * dt_s, (remaining, 1))
    baseline_policy, _, _ = future_min_fed_policy(graph, held)
    costs, first = [], {"snapshot_hold": baseline_policy[0].copy()}
    for spec in models:
        forecast = make_forecast(history, spec["window_s"], spec["projection_cap_s"], spec["slope_gain"])
        predicted = predicted_interval_costs(forecast, history.decision_time, dt_s, remaining)
        policy, _, _ = future_min_fed_policy(graph, predicted)
        costs.append(predicted)
        first[spec["id"]] = policy[0].copy()
    ensemble_policy, _, _ = future_min_fed_policy(graph, np.mean(costs, axis=0))
    first["ensemble_mean"] = ensemble_policy[0].copy()
    gains = np.full((len(costs), n), -math.inf)
    for m, predicted in enumerate(costs):
        baseline = fixed_policy_values(graph, baseline_policy, predicted)[0]
        nominee = fixed_policy_values(graph, ensemble_policy, predicted)[0]
        finite = np.isfinite(baseline) & np.isfinite(nominee)
        gains[m, finite] = ((baseline[finite] - nominee[finite])
                            / np.maximum(baseline[finite], 1e-12))
    return PlanBundle(first, policy_terminal_ids(graph, baseline_policy),
                      policy_terminal_ids(graph, ensemble_policy), gains.min(axis=0), gains)


def gated_action(bundle, state, threshold=.05):
    baseline = int(bundle.first_actions["snapshot_hold"][state])
    nominee = int(bundle.first_actions["ensemble_mean"][state])
    opportunity = baseline != nominee and baseline >= 0 and nominee >= 0
    accepted = bool(opportunity and bundle.minimum_relative_gain[state] >= threshold)
    selected = nominee if accepted else baseline
    selected_exit = int(bundle.nominee_exit[state] if accepted else bundle.baseline_exit[state])
    return selected, selected_exit, opportunity, accepted


def switch_threshold(previous_exit, nominee_exit, initial=.05, same=.01, different=.10):
    if previous_exit < 0:
        return initial
    return same if previous_exit == nominee_exit else different


def region_distances(graph: Graph, region_by_cell, region_names):
    """Minimum graph distance between region cells, including shared labels."""
    members = {name: [i for i, r in enumerate(region_by_cell) if r == name]
               for name in region_names}
    # Graph exit nodes are absorbing for movement, but region perturbations should
    # measure corridor distances using nonexit cells only, without exit shortcuts.
    answer = np.full((len(region_names), len(region_names)), math.inf)
    for r, name in enumerate(region_names):
        queue = deque((i, 0) for i in members[name])
        seen = set(members[name])
        while queue:
            state, distance = queue.popleft()
            if region_by_cell[state] in members:
                j = region_names.index(region_by_cell[state])
                answer[r, j] = min(answer[r, j], distance)
            for _, dest in graph.edges[state]:
                if dest not in graph.exit_ids and dest not in seen:
                    seen.add(dest)
                    queue.append((dest, distance + 1))
    return answer


def make_synthetic_truth(times, source_regions, distances, seed, family):
    """A mathematical stress field. Not a combustion/smoke simulation."""
    times = np.asarray(times, dtype=float)
    result = np.asarray(source_regions, dtype=float).copy()
    if family == "source":
        return result, {"family": "source", "additions": []}
    if family not in ("accelerating_local", "transient_local"):
        raise ValueError(family)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7041]))
    additions = []
    for _ in range(2):
        center = int(rng.integers(result.shape[1]))
        width = float(rng.uniform(2, 5))
        amplitude = float(rng.uniform(.003, .02))
        onset = float(rng.uniform(220, 560))
        rise, fall = float(rng.uniform(30, 90)), float(rng.uniform(30, 90))
        if family == "accelerating_local":
            temporal = np.clip((times - onset) / 120., 0, 1)**2
        else:
            temporal = np.maximum(0., np.minimum((times - onset) / rise,
                                                 (onset + rise + fall - times) / fall))
        spatial = np.exp(-.5 * (distances[center] / width)**2)
        result += amplitude * temporal[:, None] * spatial[None, :]
        additions.append(dict(center_region_index=center, width_graph_moves=width,
            amplitude_FER=amplitude, onset_fire_s=onset, rise_s=rise, fall_s=fall))
    assert np.isfinite(result).all() and np.all(result >= 0)
    return result, dict(family=family, additions=additions)


def noisy_measurements(truth, seed, coefficient_of_variation):
    if not math.isfinite(coefficient_of_variation) or coefficient_of_variation < 0:
        raise ValueError("Nonnegative noise coefficient required")
    truth = np.asarray(truth, dtype=float)
    if coefficient_of_variation == 0:
        return truth.copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9017]))
    z = rng.standard_normal(truth.shape)
    sigma = math.sqrt(math.log1p(coefficient_of_variation**2))
    return truth * np.exp(sigma * z - .5 * sigma**2)
