from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


NumericRow = dict[str, float | int | str]


def paired_bootstrap_comparisons(
    rows: Sequence[NumericRow],
    candidate: str,
    baselines: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int = 5000,
    seed: int = 20260928,
) -> list[dict[str, float | int | str]]:
    """Compare controllers using matched behavior/episode seeds.

    Differences are candidate minus baseline. Negative values are improvements
    for loss metrics such as incapacitation_rate and mean_cumulative_fed.
    """
    grouped: dict[tuple[str, str], dict[int, NumericRow]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["behavior_model"]), str(row["controller"]))][int(row["seed"])] = row
    behaviors = sorted({str(row["behavior_model"]) for row in rows})
    rng = np.random.default_rng(seed)
    output: list[dict[str, float | int | str]] = []
    for behavior in behaviors:
        candidate_rows = grouped.get((behavior, candidate), {})
        for baseline in baselines:
            baseline_rows = grouped.get((behavior, baseline), {})
            common_seeds = sorted(set(candidate_rows) & set(baseline_rows))
            if not common_seeds:
                continue
            for metric in metrics:
                differences = np.array(
                    [float(candidate_rows[s][metric]) - float(baseline_rows[s][metric]) for s in common_seeds],
                    dtype=float,
                )
                indices = rng.integers(0, differences.size, size=(bootstrap_samples, differences.size))
                boot_means = differences[indices].mean(axis=1)
                output.append(
                    {
                        "behavior_model": behavior,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_episodes": len(common_seeds),
                        "mean_difference": float(differences.mean()),
                        "ci_2_5": float(np.quantile(boot_means, 0.025)),
                        "ci_97_5": float(np.quantile(boot_means, 0.975)),
                        "probability_difference_below_zero": float(np.mean(boot_means < 0.0)),
                    }
                )
    return output


def hierarchical_bootstrap_multiseed(
    rows: Sequence[NumericRow],
    candidate: str,
    baselines: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int = 5000,
    seed: int = 20260928,
) -> list[dict[str, float | int | str]]:
    """Bootstrap training seeds first and paired evaluation episodes second."""
    baseline_lookup: dict[tuple[str, str, int], NumericRow] = {}
    candidate_lookup: dict[tuple[str, int, int], NumericRow] = {}
    for row in rows:
        behavior = str(row["behavior_model"])
        controller = str(row["controller"])
        episode_seed = int(row["seed"])
        if controller in baselines:
            baseline_lookup[(behavior, controller, episode_seed)] = row
        elif controller == candidate:
            training_seed = int(row["training_seed"])
            candidate_lookup[(behavior, training_seed, episode_seed)] = row

    behaviors = sorted({key[0] for key in candidate_lookup})
    training_seeds = sorted({key[1] for key in candidate_lookup})
    rng = np.random.default_rng(seed)
    output: list[dict[str, float | int | str]] = []
    for behavior in behaviors:
        for baseline in baselines:
            for metric in metrics:
                differences_by_training_seed: dict[int, np.ndarray] = {}
                for training_seed in training_seeds:
                    episode_seeds = sorted(
                        episode_seed
                        for b, ts, episode_seed in candidate_lookup
                        if b == behavior
                        and ts == training_seed
                        and (behavior, baseline, episode_seed) in baseline_lookup
                    )
                    if episode_seeds:
                        differences_by_training_seed[training_seed] = np.array(
                            [
                                float(candidate_lookup[(behavior, training_seed, episode_seed)][metric])
                                - float(baseline_lookup[(behavior, baseline, episode_seed)][metric])
                                for episode_seed in episode_seeds
                            ],
                            dtype=float,
                        )
                available_seeds = sorted(differences_by_training_seed)
                if not available_seeds:
                    continue
                per_seed_means = np.array([differences_by_training_seed[s].mean() for s in available_seeds])
                boot_means = np.empty(bootstrap_samples, dtype=float)
                for b in range(bootstrap_samples):
                    sampled_training_seeds = rng.choice(available_seeds, size=len(available_seeds), replace=True)
                    sampled_seed_means = []
                    for sampled_seed in sampled_training_seeds:
                        diffs = differences_by_training_seed[int(sampled_seed)]
                        sampled_seed_means.append(float(rng.choice(diffs, size=len(diffs), replace=True).mean()))
                    boot_means[b] = float(np.mean(sampled_seed_means))
                output.append(
                    {
                        "behavior_model": behavior,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "training_seeds": len(available_seeds),
                        "episodes_per_seed": min(len(differences_by_training_seed[s]) for s in available_seeds),
                        "mean_difference": float(per_seed_means.mean()),
                        "between_seed_std": float(per_seed_means.std(ddof=1)) if len(per_seed_means) > 1 else 0.0,
                        "ci_2_5": float(np.quantile(boot_means, 0.025)),
                        "ci_97_5": float(np.quantile(boot_means, 0.975)),
                        "probability_difference_below_zero": float(np.mean(boot_means < 0.0)),
                    }
                )
    return output
