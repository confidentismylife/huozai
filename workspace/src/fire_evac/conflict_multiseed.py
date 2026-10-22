from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CORE_METRICS = (
    "mean_reward",
    "optimal_choice_rate",
    "smoke_choice_rate",
    "mean_distance",
)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260929,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean across training seeds."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def aggregate_seed_summaries(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str] = CORE_METRICS,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260929,
) -> list[dict[str, object]]:
    """Return tidy mean, between-seed SD, and bootstrap intervals by policy."""
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["policy"]), []).append(row)
    output: list[dict[str, object]] = []
    for policy, items in sorted(groups.items()):
        for metric_index, metric in enumerate(metrics):
            values = np.asarray([float(item[metric]) for item in items], dtype=float)
            lower, upper = bootstrap_mean_interval(
                values,
                samples=bootstrap_samples,
                seed=bootstrap_seed + metric_index,
            )
            output.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "training_seeds": len(values),
                    "mean": float(values.mean()),
                    "between_seed_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_ci95_lower": lower,
                    "bootstrap_ci95_upper": upper,
                }
            )
    return output


def paired_seed_differences(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str = "llm_guided_ppo",
    baseline: str = "ppo",
    metrics: Sequence[str] = CORE_METRICS,
) -> list[dict[str, object]]:
    """Compute within-training-seed candidate-minus-baseline differences."""
    indexed: dict[tuple[int, str], Mapping[str, object]] = {}
    for row in rows:
        key = (int(row["training_seed"]), str(row["policy"]))
        if key in indexed:
            raise ValueError(f"duplicate seed/policy row: {key}")
        indexed[key] = row
    seeds = sorted({seed for seed, policy in indexed if policy == candidate})
    baseline_seeds = sorted({seed for seed, policy in indexed if policy == baseline})
    if not seeds or seeds != baseline_seeds:
        raise ValueError("candidate and baseline must have the same non-empty seed set")
    output: list[dict[str, object]] = []
    for seed in seeds:
        candidate_row = indexed[(seed, candidate)]
        baseline_row = indexed[(seed, baseline)]
        for metric in metrics:
            output.append(
                {
                    "training_seed": seed,
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_value": float(candidate_row[metric]),
                    "baseline_value": float(baseline_row[metric]),
                    "difference": float(candidate_row[metric]) - float(baseline_row[metric]),
                }
            )
    return output


def aggregate_paired_differences(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260929,
) -> list[dict[str, object]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["metric"]), []).append(float(row["difference"]))
    output: list[dict[str, object]] = []
    for metric_index, (metric, values) in enumerate(sorted(groups.items())):
        array = np.asarray(values, dtype=float)
        lower, upper = bootstrap_mean_interval(
            array,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 100 + metric_index,
        )
        output.append(
            {
                "policy": "llm_guided_ppo_minus_ppo",
                "metric": metric,
                "training_seeds": len(array),
                "mean": float(array.mean()),
                "between_seed_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
            }
        )
    return output
