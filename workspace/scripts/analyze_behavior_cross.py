from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank behavior-specific PPO policies across test behavior models.")
    parser.add_argument("--cross-matrix", type=Path, default=ROOT / "outputs" / "behavior_cross" / "cross_matrix.csv")
    parser.add_argument(
        "--baseline-results", type=Path, default=ROOT / "outputs" / "behavior_cross" / "cross_baseline_results.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "behavior_cross")
    return parser.parse_args()


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.cross_matrix.open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    variants = sorted({row["training_variant"] for row in matrix})
    ranking: list[dict[str, float | int | str]] = []
    for variant in variants:
        rows = [row for row in matrix if row["training_variant"] == variant]
        incapacitation = np.array([float(row["incapacitation_rate_mean"]) for row in rows])
        fed = np.array([float(row["mean_cumulative_fed_mean"]) for row in rows])
        reward = np.array([float(row["total_reward_mean"]) for row in rows])
        seed_std = np.array([float(row["incapacitation_rate_between_seed_std"]) for row in rows])
        ranking.append(
            {
                "rank": 0,
                "training_variant": variant,
                "test_behaviors": len(rows),
                "mean_incapacitation_rate": float(incapacitation.mean()),
                "worst_incapacitation_rate": float(incapacitation.max()),
                "mean_between_seed_std": float(seed_std.mean()),
                "mean_cumulative_fed": float(fed.mean()),
                "worst_total_reward": float(reward.min()),
                "mean_total_reward": float(reward.mean()),
            }
        )
    ranking.sort(key=lambda row: (float(row["worst_incapacitation_rate"]), float(row["mean_incapacitation_rate"])))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    write_csv(ranking, args.output_dir / "cross_robustness_ranking.csv")

    with args.baseline_results.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in baseline_rows:
        groups.setdefault((row["test_behavior"], row["controller"]), []).append(row)
    baseline_summary: list[dict[str, float | int | str]] = []
    for (test_behavior, controller), rows in sorted(groups.items()):
        baseline_summary.append(
            {
                "test_behavior": test_behavior,
                "controller": controller,
                "episodes": len(rows),
                "evacuation_rate_mean": float(np.mean([float(row["evacuation_rate"]) for row in rows])),
                "incapacitation_rate_mean": float(np.mean([float(row["incapacitation_rate"]) for row in rows])),
                "mean_cumulative_fed": float(np.mean([float(row["mean_cumulative_fed"]) for row in rows])),
                "total_reward_mean": float(np.mean([float(row["total_reward"]) for row in rows])),
            }
        )
    write_csv(baseline_summary, args.output_dir / "cross_baseline_summary.csv")

    print("Robustness ranking (lower worst-case incapacitation is better):")
    for row in ranking:
        print(
            f"{row['rank']}. {row['training_variant']:<9} "
            f"mean={row['mean_incapacitation_rate']:.4f} worst={row['worst_incapacitation_rate']:.4f} "
            f"seed_std={row['mean_between_seed_std']:.4f} fed={row['mean_cumulative_fed']:.4f}"
        )


if __name__ == "__main__":
    main()
