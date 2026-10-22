from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import LLMSurrogateBehavior, MixedLogitBehavior, RuleBehavior
from fire_evac.controllers import HazardAwareController, NeutralController, StaticNorthController
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate, summarize, write_csv
from fire_evac.qlearning import train_q_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimum viable evacuation-control experiment.")
    parser.add_argument("--train-episodes", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def print_summary(rows: list[dict[str, float | str]]) -> None:
    header = (
        f"{'behavior':<28} {'controller':<18} {'evac':>8} "
        f"{'incap':>8} {'fed':>9} {'congestion':>12} {'reward':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['behavior_model']:<28} {row['controller']:<18} "
            f"{row['evacuation_rate_mean']:>8.3f} {row['incapacitation_rate_mean']:>8.3f} "
            f"{row['mean_cumulative_fed']:>9.3f} {row['congestion_mean']:>12.3f} "
            f"{row['reward_mean']:>9.3f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_env = FireEvacuationEnv(MixedLogitBehavior())
    learned = train_q_learning(training_env, episodes=args.train_episodes, seed=args.seed)
    controllers = [
        NeutralController(),
        StaticNorthController(),
        HazardAwareController(),
        learned,
    ]

    all_rows: list[dict[str, float | int | str]] = []
    for behavior in [RuleBehavior(), MixedLogitBehavior(), LLMSurrogateBehavior()]:
        env = FireEvacuationEnv(behavior)
        all_rows.extend(evaluate(env, controllers, episodes=args.eval_episodes, seed=10_000))

    summary_rows = summarize(all_rows)
    write_csv(all_rows, args.output_dir / "prototype_episode_results.csv")
    with (args.output_dir / "prototype_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print_summary(summary_rows)
    print(f"\nEpisode results: {args.output_dir / 'prototype_episode_results.csv'}")
    print(f"Summary:         {args.output_dir / 'prototype_summary.csv'}")


if __name__ == "__main__":
    main()

