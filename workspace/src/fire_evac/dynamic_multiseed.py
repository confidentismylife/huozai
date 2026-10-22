from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .conflict_multiseed import (
    aggregate_paired_differences,
    aggregate_seed_summaries,
    paired_seed_differences,
)


DYNAMIC_METRICS = (
    "evacuation_rate",
    "incapacitation_rate",
    "mean_cumulative_fed",
    "cumulative_congestion",
    "sign_switches",
    "total_reward",
)


def summarize_cross_episodes(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["test_behavior"]), int(row["training_seed"]), str(row["policy"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (behavior, seed, policy), items in sorted(groups.items()):
        record: dict[str, object] = {
            "test_behavior": behavior,
            "training_seed": seed,
            "policy": policy,
            "episodes": len(items),
        }
        for metric in DYNAMIC_METRICS:
            record[metric] = float(np.mean([float(item[metric]) for item in items]))
        output.append(record)
    return output


def cross_seed_statistics(
    per_seed_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260929,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    behaviors = sorted({str(row["test_behavior"]) for row in per_seed_rows})
    aggregate: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    paired_aggregate: list[dict[str, object]] = []
    for behavior_index, behavior in enumerate(behaviors):
        subset = [row for row in per_seed_rows if str(row["test_behavior"]) == behavior]
        behavior_aggregate = aggregate_seed_summaries(
            subset,
            metrics=DYNAMIC_METRICS,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + behavior_index * 1000,
        )
        for row in behavior_aggregate:
            row["test_behavior"] = behavior
            aggregate.append({"test_behavior": row.pop("test_behavior"), **row})
        behavior_paired = paired_seed_differences(
            subset,
            candidate="llm_guided_ppo",
            baseline="ppo",
            metrics=DYNAMIC_METRICS,
        )
        for row in behavior_paired:
            row["test_behavior"] = behavior
            paired.append({"test_behavior": row.pop("test_behavior"), **row})
        behavior_paired_aggregate = aggregate_paired_differences(
            behavior_paired,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + behavior_index * 1000,
        )
        for row in behavior_paired_aggregate:
            row["test_behavior"] = behavior
            paired_aggregate.append({"test_behavior": row.pop("test_behavior"), **row})
    return aggregate, paired, paired_aggregate
