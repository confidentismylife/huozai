from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import (
    BehaviorModel,
    CalibratedLogitBehavior,
    EpisodeBehaviorEnsemble,
    LLMSurrogateBehavior,
    MixedLogitBehavior,
    RuleBehavior,
)
from fire_evac.controllers import HazardAwareController, NeutralController, StaticNorthController
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate
from fire_evac.ppo import load_ppo_controller
from fire_evac.ppo_experiment import PPOTrainingConfig, train_ppo_seed


METRICS = ("evacuation_rate", "incapacitation_rate", "mean_cumulative_fed", "total_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior-specific PPO policies and build a cross-test matrix.")
    parser.add_argument("--variants", default="rule,logit,llm,ensemble")
    parser.add_argument("--seeds", default="51,52,53")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "behavior_cross")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    return parser.parse_args()


def factories(calibration_artifact: Path) -> tuple[
    dict[str, Callable[[], BehaviorModel]], dict[str, Callable[[], BehaviorModel]]
]:
    test_factories: dict[str, Callable[[], BehaviorModel]] = {
        "rule": RuleBehavior,
        "mixed_logit": MixedLogitBehavior,
        "calibrated_logit": lambda: CalibratedLogitBehavior(calibration_artifact),
        "llm_surrogate_unvalidated": LLMSurrogateBehavior,
    }
    training_factories: dict[str, Callable[[], BehaviorModel]] = {
        "rule": RuleBehavior,
        "logit": lambda: CalibratedLogitBehavior(calibration_artifact),
        "llm": LLMSurrogateBehavior,
        "ensemble": lambda: EpisodeBehaviorEnsemble(tuple(test_factories.values())),
    }
    return training_factories, test_factories


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_cross(rows: list[dict[str, float | int | str]]) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    per_seed_groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["training_variant"]), str(row["test_behavior"]), str(row["training_seed"]))
        per_seed_groups.setdefault(key, []).append(row)
    per_seed: list[dict[str, float | int | str]] = []
    for (variant, test_behavior, training_seed), items in sorted(per_seed_groups.items()):
        record: dict[str, float | int | str] = {
            "training_variant": variant,
            "test_behavior": test_behavior,
            "training_seed": training_seed,
            "episodes": len(items),
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in items]))
        per_seed.append(record)

    aggregate_groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in per_seed:
        aggregate_groups.setdefault((str(row["training_variant"]), str(row["test_behavior"])), []).append(row)
    aggregate: list[dict[str, float | int | str]] = []
    for (variant, test_behavior), items in sorted(aggregate_groups.items()):
        record = {"training_variant": variant, "test_behavior": test_behavior, "training_seeds": len(items)}
        for metric in METRICS:
            values = np.array([float(row[f"{metric}_mean"]) for row in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_between_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(record)
    return per_seed, aggregate


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    requested_variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    training_factories, test_factories = factories(args.calibration_artifact)
    unknown = sorted(set(requested_variants) - set(training_factories))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("--seeds must contain unique integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = PPOTrainingConfig(total_timesteps=args.timesteps, validation_episodes=args.validation_episodes)
    models: dict[tuple[str, int], Path] = {}
    metadata: list[dict[str, object]] = []
    total_runs = len(requested_variants) * len(seeds)
    run_index = 0
    for variant in requested_variants:
        for seed in seeds:
            run_index += 1
            print(f"Training {variant} seed {seed} ({run_index}/{total_runs})...", flush=True)
            model_path, run_metadata = train_ppo_seed(
                args.calibration_artifact,
                args.output_dir / variant,
                seed,
                config,
                behavior_factory=training_factories[variant],
                experiment_label=variant,
                history_length=args.history_length,
            )
            models[(variant, seed)] = model_path
            metadata.append(run_metadata)
            print(
                f"  selected={model_path.name} validation_reward={run_metadata['best_validation_reward']}", flush=True
            )

    all_rows: list[dict[str, float | int | str]] = []
    for variant in requested_variants:
        for seed in seeds:
            controller = load_ppo_controller(models[(variant, seed)], history_length=args.history_length)
            for test_name, test_factory in test_factories.items():
                rows = evaluate(FireEvacuationEnv(test_factory()), [controller], args.eval_episodes, seed=40_000)
                for row in rows:
                    row["training_variant"] = variant
                    row["test_behavior"] = test_name
                    row["training_seed"] = seed
                all_rows.extend(rows)

    # Baselines are recorded separately so the learned-policy matrix stays rectangular.
    baseline_rows: list[dict[str, float | int | str]] = []
    for test_name, test_factory in test_factories.items():
        rows = evaluate(
            FireEvacuationEnv(test_factory()),
            [NeutralController(), StaticNorthController(), HazardAwareController()],
            args.eval_episodes,
            seed=40_000,
        )
        for row in rows:
            row["test_behavior"] = test_name
        baseline_rows.extend(rows)

    write_csv(all_rows, args.output_dir / "cross_episode_results.csv")
    write_csv(baseline_rows, args.output_dir / "cross_baseline_results.csv")
    per_seed, aggregate = summarize_cross(all_rows)
    write_csv(per_seed, args.output_dir / "cross_per_seed_summary.csv")
    write_csv(aggregate, args.output_dir / "cross_matrix.csv")
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "training_runs": metadata,
                "variants": requested_variants,
                "seeds": seeds,
                "history_length": args.history_length,
                "evaluation_episodes": args.eval_episodes,
                "warning": "The LLM surrogate and calibration data are synthetic engineering placeholders.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nCross-test incapacitation rates (mean across training seeds):")
    for row in aggregate:
        print(
            f"train={row['training_variant']:<9} test={row['test_behavior']:<28} "
            f"incap={row['incapacitation_rate_mean']:.4f}±{row['incapacitation_rate_between_seed_std']:.4f} "
            f"fed={row['mean_cumulative_fed_mean']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
