from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import CalibratedLogitBehavior, LLMSurrogateBehavior, MixedLogitBehavior, RuleBehavior
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate
from fire_evac.ppo import load_recurrent_ppo_controller
from fire_evac.ppo_experiment import PPOTrainingConfig, train_recurrent_ppo_seed


METRICS = ("evacuation_rate", "incapacitation_rate", "mean_cumulative_fed", "total_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RecurrentPPO with four-step history PPO.")
    parser.add_argument("--seeds", default="51,52,53")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "recurrent_comparison")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--history-summary",
        type=Path,
        default=ROOT / "outputs" / "behavior_cross" / "cross_per_seed_summary.csv",
    )
    return parser.parse_args()


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["test_behavior"]), int(row["training_seed"])), []).append(row)
    output: list[dict[str, float | int | str]] = []
    for (test_behavior, training_seed), items in sorted(groups.items()):
        record: dict[str, float | int | str] = {
            "architecture": "recurrent_lstm",
            "test_behavior": test_behavior,
            "training_seed": training_seed,
            "episodes": len(items),
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in items]))
        output.append(record)
    return output


def aggregate(per_seed: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in per_seed:
        groups.setdefault((str(row["architecture"]), str(row["test_behavior"])), []).append(row)
    output: list[dict[str, float | int | str]] = []
    for (architecture, test_behavior), items in sorted(groups.items()):
        record: dict[str, float | int | str] = {
            "architecture": architecture,
            "test_behavior": test_behavior,
            "training_seeds": len(items),
        }
        for metric in METRICS:
            values = np.array([float(row[f"{metric}_mean"]) for row in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_between_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PPOTrainingConfig(total_timesteps=args.timesteps, validation_episodes=args.validation_episodes)
    model_paths: dict[int, Path] = {}
    metadata = []
    for index, seed in enumerate(seeds, start=1):
        print(f"Training RecurrentPPO seed {seed} ({index}/{len(seeds)})...", flush=True)
        model_path, run_metadata = train_recurrent_ppo_seed(
            args.calibration_artifact, args.output_dir, seed, config
        )
        model_paths[seed] = model_path
        metadata.append(run_metadata)
        print(f"  selected={model_path.name} validation_reward={run_metadata['best_validation_reward']}", flush=True)

    test_factories = {
        "rule": RuleBehavior,
        "mixed_logit": MixedLogitBehavior,
        "calibrated_logit": lambda: CalibratedLogitBehavior(args.calibration_artifact),
        "llm_surrogate_unvalidated": LLMSurrogateBehavior,
    }
    episode_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        controller = load_recurrent_ppo_controller(model_paths[seed])
        for test_behavior, factory in test_factories.items():
            rows = evaluate(FireEvacuationEnv(factory()), [controller], args.eval_episodes, seed=40_000)
            for row in rows:
                row["architecture"] = "recurrent_lstm"
                row["test_behavior"] = test_behavior
                row["training_seed"] = seed
            episode_rows.extend(rows)
    write_csv(episode_rows, args.output_dir / "recurrent_episode_results.csv")
    recurrent_per_seed = summarize(episode_rows)

    with args.history_summary.open("r", encoding="utf-8-sig", newline="") as handle:
        history_source = [row for row in csv.DictReader(handle) if row["training_variant"] == "logit"]
    history_per_seed: list[dict[str, float | int | str]] = []
    for row in history_source:
        history_per_seed.append(
            {
                "architecture": "history_stack_4",
                "test_behavior": row["test_behavior"],
                "training_seed": int(row["training_seed"]),
                "episodes": int(row["episodes"]),
                **{f"{metric}_mean": float(row[f"{metric}_mean"]) for metric in METRICS},
            }
        )
    combined_per_seed = history_per_seed + recurrent_per_seed
    write_csv(combined_per_seed, args.output_dir / "architecture_per_seed_summary.csv")
    combined_aggregate = aggregate(combined_per_seed)
    write_csv(combined_aggregate, args.output_dir / "architecture_comparison.csv")
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "training_runs": metadata,
                "seeds": seeds,
                "timesteps": args.timesteps,
                "evaluation_episodes": args.eval_episodes,
                "comparison": "RecurrentPPO LSTM versus four-observation stack PPO-Logit",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nArchitecture comparison:")
    for row in combined_aggregate:
        print(
            f"{row['architecture']:<16} test={row['test_behavior']:<28} "
            f"incap={row['incapacitation_rate_mean']:.4f}±{row['incapacitation_rate_between_seed_std']:.4f} "
            f"fed={row['mean_cumulative_fed_mean']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
