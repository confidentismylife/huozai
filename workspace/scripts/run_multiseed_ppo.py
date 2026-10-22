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
from fire_evac.comparison import hierarchical_bootstrap_multiseed
from fire_evac.controllers import HazardAwareController, NeutralController, StaticNorthController
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate
from fire_evac.ppo import load_ppo_controller
from fire_evac.ppo_experiment import PPOTrainingConfig, train_ppo_seed


METRICS = ("evacuation_rate", "incapacitation_rate", "mean_cumulative_fed", "total_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PPO across independent training seeds.")
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ppo_multiseed")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    return parser.parse_args()


def write_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_per_seed(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["behavior_model"]), str(row["controller"]), str(row["training_seed"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, float | int | str]] = []
    for (behavior, controller, training_seed), items in sorted(groups.items()):
        record: dict[str, float | int | str] = {
            "behavior_model": behavior,
            "controller": controller,
            "training_seed": training_seed,
            "episodes": len(items),
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in items]))
        output.append(record)
    return output


def aggregate_ppo_seeds(per_seed: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, float | int | str]]] = {}
    for row in per_seed:
        if row["controller"] == "ppo":
            groups.setdefault(str(row["behavior_model"]), []).append(row)
    output: list[dict[str, float | int | str]] = []
    for behavior, items in sorted(groups.items()):
        record: dict[str, float | int | str] = {"behavior_model": behavior, "training_seeds": len(items)}
        for metric in METRICS:
            values = np.array([float(row[f"{metric}_mean"]) for row in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_between_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must contain unique integer seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PPOTrainingConfig(total_timesteps=args.timesteps, validation_episodes=args.validation_episodes)
    metadata = []
    model_paths: dict[int, Path] = {}
    for index, seed in enumerate(seeds, start=1):
        print(f"Training seed {seed} ({index}/{len(seeds)})...", flush=True)
        model_path, seed_metadata = train_ppo_seed(args.calibration_artifact, args.output_dir, seed, config)
        model_paths[seed] = model_path
        metadata.append(seed_metadata)
        print(
            f"  selected={model_path.name} steps={seed_metadata['actual_timesteps']} "
            f"validation_reward={seed_metadata['best_validation_reward']}",
            flush=True,
        )

    behavior_factories = {
        "rule": RuleBehavior,
        "mixed_logit": MixedLogitBehavior,
        "calibrated_logit": lambda: CalibratedLogitBehavior(args.calibration_artifact),
        "llm_surrogate_unvalidated": LLMSurrogateBehavior,
    }
    all_rows: list[dict[str, float | int | str]] = []
    baseline_controllers = [NeutralController(), StaticNorthController(), HazardAwareController()]
    for behavior_factory in behavior_factories.values():
        baseline_rows = evaluate(
            FireEvacuationEnv(behavior_factory()), baseline_controllers, args.eval_episodes, seed=30_000
        )
        for row in baseline_rows:
            row["training_seed"] = "baseline"
        all_rows.extend(baseline_rows)

    for seed in seeds:
        controller = load_ppo_controller(model_paths[seed])
        for behavior_factory in behavior_factories.values():
            ppo_rows = evaluate(
                FireEvacuationEnv(behavior_factory()), [controller], args.eval_episodes, seed=30_000
            )
            for row in ppo_rows:
                row["training_seed"] = seed
            all_rows.extend(ppo_rows)

    write_rows(all_rows, args.output_dir / "multiseed_episode_results.csv")
    per_seed = summarize_per_seed(all_rows)
    write_rows(per_seed, args.output_dir / "multiseed_per_seed_summary.csv")
    aggregate = aggregate_ppo_seeds(per_seed)
    write_rows(aggregate, args.output_dir / "multiseed_aggregate.csv")
    comparisons = hierarchical_bootstrap_multiseed(
        all_rows,
        candidate="ppo",
        baselines=("neutral", "hazard_aware_rule"),
        metrics=("incapacitation_rate", "mean_cumulative_fed", "total_reward"),
        bootstrap_samples=5000,
    )
    write_rows(comparisons, args.output_dir / "multiseed_comparisons.csv")
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps({"training_runs": metadata, "evaluation_episodes": args.eval_episodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nAcross-training-seed PPO results:")
    for row in aggregate:
        print(
            f"{row['behavior_model']:<28} evac={row['evacuation_rate_mean']:.4f}±{row['evacuation_rate_between_seed_std']:.4f} "
            f"incap={row['incapacitation_rate_mean']:.4f}±{row['incapacitation_rate_between_seed_std']:.4f} "
            f"fed={row['mean_cumulative_fed_mean']:.4f}±{row['mean_cumulative_fed_between_seed_std']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
