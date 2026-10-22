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
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate
from fire_evac.gym_env import GymFireEvacuationEnv, HistoryStackWrapper
from fire_evac.ppo import PPOController, load_ppo_controller
from fire_evac.robust_training import worst_case_sampling_weights, worst_case_score


METRICS = ("evacuation_rate", "incapacitation_rate", "mean_cumulative_fed", "total_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive worst-case behavior-ensemble PPO training.")
    parser.add_argument("--seeds", default="51,52,53")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--phases", type=int, default=6)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--exploration-mass", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "robust_ensemble")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--cross-per-seed",
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


def make_factories(calibration_artifact: Path) -> dict[str, Callable[[], BehaviorModel]]:
    return {
        "rule": RuleBehavior,
        "mixed_logit": MixedLogitBehavior,
        "calibrated_logit": lambda: CalibratedLogitBehavior(calibration_artifact),
        "llm_surrogate_unvalidated": LLMSurrogateBehavior,
    }


def make_training_env(
    factories: dict[str, Callable[[], BehaviorModel]], weights: np.ndarray, history_length: int
):
    base = GymFireEvacuationEnv(EpisodeBehaviorEnsemble(tuple(factories.values()), weights=weights))
    return HistoryStackWrapper(base, history_length=history_length)


def summarize_per_seed(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        groups.setdefault((str(row["test_behavior"]), int(row["training_seed"])), []).append(row)
    output: list[dict[str, float | int | str]] = []
    for (test_behavior, training_seed), items in sorted(groups.items()):
        record: dict[str, float | int | str] = {
            "training_method": "adaptive_worst_case",
            "test_behavior": test_behavior,
            "training_seed": training_seed,
            "episodes": len(items),
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in items]))
        output.append(record)
    return output


def method_ranking(per_seed_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str], list[float]] = {}
    fed_groups: dict[tuple[str, str], list[float]] = {}
    for row in per_seed_rows:
        key = (str(row["training_method"]), str(row["test_behavior"]))
        groups.setdefault(key, []).append(float(row["incapacitation_rate_mean"]))
        fed_groups.setdefault(key, []).append(float(row["mean_cumulative_fed_mean"]))
    methods = sorted({key[0] for key in groups})
    ranking: list[dict[str, float | int | str]] = []
    for method in methods:
        behavior_means = [np.mean(values) for (candidate, _), values in groups.items() if candidate == method]
        fed_means = [np.mean(values) for (candidate, _), values in fed_groups.items() if candidate == method]
        seed_stds = [
            np.std(values, ddof=1) if len(values) > 1 else 0.0
            for (candidate, _), values in groups.items()
            if candidate == method
        ]
        ranking.append(
            {
                "rank": 0,
                "training_method": method,
                "mean_incapacitation_rate": float(np.mean(behavior_means)),
                "worst_behavior_incapacitation_rate": float(np.max(behavior_means)),
                "mean_between_seed_std": float(np.mean(seed_stds)),
                "mean_cumulative_fed": float(np.mean(fed_means)),
            }
        )
    ranking.sort(key=lambda row: (float(row["worst_behavior_incapacitation_rate"]), float(row["mean_incapacitation_rate"])))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.timesteps % args.phases != 0:
        raise SystemExit("--timesteps must be divisible by --phases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factories = make_factories(args.calibration_artifact)
    behavior_order = list(factories)
    phase_timesteps = args.timesteps // args.phases
    selected_models: dict[int, Path] = {}
    curriculum_rows: list[dict[str, float | int | str]] = []
    training_metadata: list[dict[str, object]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"Adaptive robust training seed {seed} ({seed_index}/{len(seeds)})...", flush=True)
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        weights = np.full(len(behavior_order), 1.0 / len(behavior_order), dtype=float)

        def env_factory():
            return make_training_env(factories, weights, args.history_length)

        training_env = make_vec_env(env_factory, n_envs=4, seed=seed)
        model = PPO(
            "MlpPolicy",
            training_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": [128, 128]},
        )
        best_score = -np.inf
        best_path = seed_dir / "best_worst_case_model"
        phase_records = []
        for phase in range(1, args.phases + 1):
            if phase > 1:
                new_env = make_vec_env(env_factory, n_envs=4, seed=seed + phase * 1000)
                old_env = model.get_env()
                model.set_env(new_env, force_reset=True)
                if old_env is not None:
                    old_env.close()
            model.learn(total_timesteps=phase_timesteps, reset_num_timesteps=False, progress_bar=False)
            controller = PPOController(model, history_length=args.history_length)
            mean_rewards: dict[str, float] = {}
            mean_incapacitation: dict[str, float] = {}
            for behavior_name, factory in factories.items():
                validation = evaluate(
                    FireEvacuationEnv(factory()), [controller], args.validation_episodes, seed=45_000
                )
                mean_rewards[behavior_name] = float(np.mean([float(row["total_reward"]) for row in validation]))
                mean_incapacitation[behavior_name] = float(
                    np.mean([float(row["incapacitation_rate"]) for row in validation])
                )
            score = worst_case_score(mean_rewards)
            if score > best_score:
                best_score = score
                model.save(best_path)
            record: dict[str, float | int | str] = {
                "training_seed": seed,
                "phase": phase,
                "cumulative_timesteps": phase * phase_timesteps,
                "worst_case_reward": score,
            }
            for index, behavior_name in enumerate(behavior_order):
                record[f"weight_{behavior_name}"] = float(weights[index])
                record[f"reward_{behavior_name}"] = mean_rewards[behavior_name]
                record[f"incapacitation_{behavior_name}"] = mean_incapacitation[behavior_name]
            curriculum_rows.append(record)
            phase_records.append(record)
            weights = worst_case_sampling_weights(
                mean_rewards,
                behavior_order,
                temperature=args.temperature,
                exploration_mass=args.exploration_mass,
            )
            print(
                f"  phase {phase}/{args.phases} worst_reward={score:.4f} "
                + " ".join(f"{name}={weight:.2f}" for name, weight in zip(behavior_order, weights)),
                flush=True,
            )
        final_path = seed_dir / "final_model"
        model.save(final_path)
        model.get_env().close()
        selected_models[seed] = best_path.with_suffix(".zip")
        training_metadata.append(
            {
                "training_seed": seed,
                "selected_model": str(selected_models[seed].resolve()),
                "best_worst_case_reward": best_score,
                "final_sampling_weights": {name: float(value) for name, value in zip(behavior_order, weights)},
                "phases": phase_records,
            }
        )

    write_csv(curriculum_rows, args.output_dir / "curriculum_log.csv")
    episode_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        controller = load_ppo_controller(selected_models[seed], history_length=args.history_length)
        for test_behavior, factory in factories.items():
            rows = evaluate(FireEvacuationEnv(factory()), [controller], args.eval_episodes, seed=40_000)
            for row in rows:
                row["training_method"] = "adaptive_worst_case"
                row["test_behavior"] = test_behavior
                row["training_seed"] = seed
            episode_rows.extend(rows)
    write_csv(episode_rows, args.output_dir / "adaptive_episode_results.csv")
    adaptive_per_seed = summarize_per_seed(episode_rows)
    write_csv(adaptive_per_seed, args.output_dir / "adaptive_per_seed_summary.csv")

    with args.cross_per_seed.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    comparison_rows: list[dict[str, float | int | str]] = []
    for source_method, output_method in (("logit", "logit_only"), ("ensemble", "equal_ensemble")):
        for row in source_rows:
            if row["training_variant"] != source_method:
                continue
            comparison_rows.append(
                {
                    "training_method": output_method,
                    "test_behavior": row["test_behavior"],
                    "training_seed": int(row["training_seed"]),
                    "episodes": int(row["episodes"]),
                    **{f"{metric}_mean": float(row[f"{metric}_mean"]) for metric in METRICS},
                }
            )
    comparison_rows.extend(adaptive_per_seed)
    write_csv(comparison_rows, args.output_dir / "robust_methods_per_seed.csv")
    ranking = method_ranking(comparison_rows)
    write_csv(ranking, args.output_dir / "robust_methods_ranking.csv")
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "training_runs": training_metadata,
                "behavior_order": behavior_order,
                "temperature": args.temperature,
                "exploration_mass": args.exploration_mass,
                "timesteps": args.timesteps,
                "phases": args.phases,
                "validation_episodes": args.validation_episodes,
                "evaluation_episodes": args.eval_episodes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nRobust-method ranking:")
    for row in ranking:
        print(
            f"{row['rank']}. {row['training_method']:<20} mean={row['mean_incapacitation_rate']:.4f} "
            f"worst={row['worst_behavior_incapacitation_rate']:.4f} seed_std={row['mean_between_seed_std']:.4f}"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
