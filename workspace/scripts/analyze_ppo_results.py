from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.comparison import paired_bootstrap_comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap analysis for PPO evaluation episodes.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "ppo" / "ppo_episode_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ppo" / "ppo_comparisons.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    comparisons = paired_bootstrap_comparisons(
        rows,
        candidate="ppo",
        baselines=("neutral", "hazard_aware_rule"),
        metrics=("incapacitation_rate", "mean_cumulative_fed", "total_reward"),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    for row in comparisons:
        if row["behavior_model"] == "calibrated_logit":
            print(
                f"{row['metric']:<24} PPO-{row['baseline']:<18} "
                f"diff={row['mean_difference']:+.5f} "
                f"95% CI [{row['ci_2_5']:+.5f}, {row['ci_97_5']:+.5f}]"
            )
    print(f"Comparisons: {args.output}")


if __name__ == "__main__":
    main()
