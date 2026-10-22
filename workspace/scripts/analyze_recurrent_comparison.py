from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.comparison import hierarchical_bootstrap_multiseed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired architecture comparison for recurrent and history-stack PPO.")
    parser.add_argument(
        "--recurrent-results",
        type=Path,
        default=ROOT / "outputs" / "recurrent_comparison" / "recurrent_episode_results.csv",
    )
    parser.add_argument(
        "--history-results",
        type=Path,
        default=ROOT / "outputs" / "behavior_cross" / "cross_episode_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "recurrent_comparison")
    return parser.parse_args()


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.recurrent_results.open("r", encoding="utf-8-sig", newline="") as handle:
        recurrent = list(csv.DictReader(handle))
    with args.history_results.open("r", encoding="utf-8-sig", newline="") as handle:
        history = [row for row in csv.DictReader(handle) if row["training_variant"] == "logit"]

    combined: list[dict[str, float | int | str]] = []
    for row in recurrent:
        normalized = dict(row)
        normalized["behavior_model"] = row["test_behavior"]
        normalized["controller"] = "recurrent_lstm"
        combined.append(normalized)
    for row in history:
        normalized = dict(row)
        normalized["behavior_model"] = row["test_behavior"]
        normalized["controller"] = "history_stack_4"
        combined.append(normalized)
    comparisons = hierarchical_bootstrap_multiseed(
        combined,
        candidate="recurrent_lstm",
        baselines=("history_stack_4",),
        metrics=("incapacitation_rate", "mean_cumulative_fed", "total_reward"),
        bootstrap_samples=5000,
        seed=20260928,
    )
    write_csv(comparisons, args.output_dir / "architecture_bootstrap_comparisons.csv")

    architectures = {"history_stack_4": history, "recurrent_lstm": recurrent}
    ranking: list[dict[str, float | int | str]] = []
    for architecture, rows in architectures.items():
        seed_behavior_means: dict[tuple[int, str], list[float]] = {}
        fed_means: dict[tuple[int, str], list[float]] = {}
        for row in rows:
            training_seed = int(row["training_seed"])
            test_behavior = str(row["test_behavior"])
            key = (training_seed, test_behavior)
            seed_behavior_means.setdefault(key, []).append(float(row["incapacitation_rate"]))
            fed_means.setdefault(key, []).append(float(row["mean_cumulative_fed"]))
        cell_incap = np.array([np.mean(values) for values in seed_behavior_means.values()])
        cell_fed = np.array([np.mean(values) for values in fed_means.values()])
        behavior_averages = []
        for behavior in sorted({key[1] for key in seed_behavior_means}):
            behavior_averages.append(
                np.mean([np.mean(values) for key, values in seed_behavior_means.items() if key[1] == behavior])
            )
        ranking.append(
            {
                "architecture": architecture,
                "training_seeds": len({key[0] for key in seed_behavior_means}),
                "mean_incapacitation_rate": float(cell_incap.mean()),
                "worst_behavior_incapacitation_rate": float(max(behavior_averages)),
                "mean_cumulative_fed": float(cell_fed.mean()),
            }
        )
    ranking.sort(key=lambda row: float(row["worst_behavior_incapacitation_rate"]))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    write_csv(ranking, args.output_dir / "architecture_robustness_ranking.csv")

    print("RecurrentPPO minus history-stack PPO (positive loss differences are worse):")
    for row in comparisons:
        if row["metric"] == "incapacitation_rate":
            print(
                f"{row['behavior_model']:<28} diff={row['mean_difference']:+.5f} "
                f"95% CI [{row['ci_2_5']:+.5f}, {row['ci_97_5']:+.5f}]"
            )
    print("Ranking:")
    for row in ranking:
        print(
            f"{row['rank']}. {row['architecture']:<16} mean={row['mean_incapacitation_rate']:.4f} "
            f"worst={row['worst_behavior_incapacitation_rate']:.4f} fed={row['mean_cumulative_fed']:.4f}"
        )


if __name__ == "__main__":
    main()
