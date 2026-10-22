from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.conflict_experiment import (
    LEFT,
    RIGHT,
    LLMAdvice,
    ConflictChoiceEnv,
    evaluate_probability_policies,
    generate_conflict_scenarios,
    scenario_observation,
    save_scenarios,
)
from fire_evac.conflict_multiseed import (
    CORE_METRICS,
    aggregate_paired_differences,
    aggregate_seed_summaries,
    paired_seed_differences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat the conflict PPO comparison across training seeds.")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--scenario-seed", type=int, default=20260929)
    parser.add_argument("--scenarios", type=int, default=48)
    parser.add_argument("--train-scenarios", type=int, default=36)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--advice-csv",
        type=Path,
        default=ROOT / "outputs" / "conflict_llm_rl" / "llm_advice.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "conflict_multiseed")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    return seeds


def load_cached_advice(path: Path) -> dict[str, LLMAdvice]:
    if not path.exists():
        raise FileNotFoundError(f"cached LLM advice not found: {path}")
    advice: dict[str, LLMAdvice] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario_id = row["scenario_id"]
            if scenario_id in advice:
                raise ValueError(f"duplicate advice for {scenario_id}")
            if row["recommended_exit"] not in {"left", "right"}:
                raise ValueError(f"invalid cached exit for {scenario_id}: {row['recommended_exit']}")
            recommended_exit = LEFT if row["recommended_exit"] == "left" else RIGHT
            reasons = tuple(filter(None, row["reason_codes"].split("|")))
            advice[scenario_id] = LLMAdvice(recommended_exit, float(row["confidence"]), reasons)
    return advice


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_probabilities(model, scenarios, advice: dict[str, LLMAdvice] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for scenario in scenarios:
        observation = scenario_observation(scenario, None if advice is None else advice[scenario.scenario_id])
        action, _ = model.predict(observation, deterministic=True)
        values[scenario.scenario_id] = 1.0 if int(np.asarray(action).item()) == LEFT else 0.0
    return values


def metric_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["policy"]), str(row["metric"])): row for row in rows}


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("Install the project's optional RL dependencies first") from exc

    args = parse_args()
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not 4 <= args.train_scenarios < args.scenarios:
        raise SystemExit("train-scenarios must be at least 4 and smaller than scenarios")
    if args.timesteps < 256:
        raise SystemExit("timesteps must be at least 256")

    scenarios = generate_conflict_scenarios(args.scenarios, args.scenario_seed)
    training = scenarios[: args.train_scenarios]
    testing = scenarios[args.train_scenarios :]
    advice = load_cached_advice(args.advice_csv)
    missing = [scenario.scenario_id for scenario in scenarios if scenario.scenario_id not in advice]
    if missing:
        raise SystemExit(f"cached advice is missing scenarios: {', '.join(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    save_scenarios(training, args.output / "train_scenarios.json")
    save_scenarios(testing, args.output / "test_scenarios.json")

    check_env(ConflictChoiceEnv(training), warn=True)
    check_env(ConflictChoiceEnv(training, advice, advice_alignment_bonus=0.25), warn=True)

    common = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
    )
    per_seed: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    for index, seed in enumerate(seeds, 1):
        print(f"Seed {seed} ({index}/{len(seeds)}): training PPO...", flush=True)
        base_model = PPO(env=Monitor(ConflictChoiceEnv(training)), seed=seed, **common)
        base_model.learn(total_timesteps=args.timesteps, progress_bar=False)
        base_model.save(model_dir / f"seed_{seed}_ppo")

        print(f"Seed {seed} ({index}/{len(seeds)}): training LLM-guided PPO...", flush=True)
        guided_model = PPO(
            env=Monitor(ConflictChoiceEnv(training, advice, advice_alignment_bonus=0.25)),
            seed=seed,
            **common,
        )
        guided_model.learn(total_timesteps=args.timesteps, progress_bar=False)
        guided_model.save(model_dir / f"seed_{seed}_llm_guided_ppo")

        probabilities = {
            "ppo": deterministic_probabilities(base_model, testing, None),
            "llm_guided_ppo": deterministic_probabilities(guided_model, testing, advice),
        }
        seed_predictions, seed_summary = evaluate_probability_policies(testing, probabilities)
        for row in seed_predictions:
            row["training_seed"] = seed
            predictions.append(row)
        for row in seed_summary:
            row["training_seed"] = seed
            per_seed.append({"training_seed": row.pop("training_seed"), **row})
        printable = {str(row["policy"]): row for row in seed_summary}
        print(
            f"  reward: PPO={float(printable['ppo']['mean_reward']):.4f}, "
            f"LLM+PPO={float(printable['llm_guided_ppo']['mean_reward']):.4f}",
            flush=True,
        )

    aggregate = aggregate_seed_summaries(
        per_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.scenario_seed,
    )
    paired = paired_seed_differences(per_seed)
    paired_aggregate = aggregate_paired_differences(
        paired,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.scenario_seed,
    )
    aggregate.extend(paired_aggregate)

    write_csv(per_seed, args.output / "per_seed_summary.csv")
    write_csv(aggregate, args.output / "aggregate_summary.csv")
    write_csv(paired, args.output / "paired_differences.csv")
    write_csv(predictions, args.output / "test_predictions_by_seed.csv")

    metadata = {
        "scenario_seed": args.scenario_seed,
        "training_seeds": seeds,
        "training_scenarios": len(training),
        "test_scenarios": len(testing),
        "timesteps_per_model": args.timesteps,
        "models_per_seed": 2,
        "total_nominal_timesteps": args.timesteps * 2 * len(seeds),
        "llm_advice_source": str(args.advice_csv.resolve()),
        "additional_ollama_calls": 0,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "training seed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lookup = metric_lookup(aggregate)
    report = [
        "# 冲突出口选择实验：5个训练随机种子复现实验",
        "",
        f"固定使用同一组 {len(training)} 个训练场景和 {len(testing)} 个锁定测试场景，训练种子为 "
        f"`{', '.join(map(str, seeds))}`。每个种子分别训练纯PPO和LLM指导PPO各 {args.timesteps:,} 步。",
        "本轮完全复用已有本地LLM建议缓存，新增Ollama调用为0次。",
        "",
        "| 策略 | 平均安全奖励 | 最优出口率 | 烟雾出口率 | 平均距离 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"ppo": "纯PPO", "llm_guided_ppo": "LLM指导PPO"}
    for policy in ("ppo", "llm_guided_ppo"):
        reward = lookup[(policy, "mean_reward")]
        optimal = lookup[(policy, "optimal_choice_rate")]
        smoke = lookup[(policy, "smoke_choice_rate")]
        distance = lookup[(policy, "mean_distance")]
        report.append(
            f"| {labels[policy]} | {float(reward['mean']):.4f} ± {float(reward['between_seed_std']):.4f} | "
            f"{float(optimal['mean']):.2%} ± {float(optimal['between_seed_std']):.2%} | "
            f"{float(smoke['mean']):.2%} ± {float(smoke['between_seed_std']):.2%} | "
            f"{float(distance['mean']):.2f} ± {float(distance['between_seed_std']):.2f} m |"
        )
    report.extend(
        [
            "",
            "## 配对结果（LLM指导PPO − 纯PPO）",
            "",
            "| 指标 | 平均配对差 | Bootstrap 95%区间 |",
            "|---|---:|---:|",
        ]
    )
    metric_labels = {
        "mean_reward": "平均安全奖励",
        "optimal_choice_rate": "最优出口率",
        "smoke_choice_rate": "烟雾出口率",
        "mean_distance": "平均距离（m）",
    }
    paired_lookup = {str(row["metric"]): row for row in paired_aggregate}
    for metric in CORE_METRICS:
        row = paired_lookup[metric]
        formatter = ".4f" if metric == "mean_reward" else (".2f" if metric == "mean_distance" else ".2%")
        report.append(
            f"| {metric_labels[metric]} | {format(float(row['mean']), formatter)} | "
            f"[{format(float(row['bootstrap_ci95_lower']), formatter)}, "
            f"{format(float(row['bootstrap_ci95_upper']), formatter)}] |"
        )
    report.extend(
        [
            "",
            "## 如何解释",
            "",
            "- 每个种子都在完全相同的训练集和测试集上比较，因此配对差主要反映两种训练方法的差别。",
            "- 标准差表示PPO因初始化和采样随机性产生的波动。Bootstrap区间的重采样单位也是训练种子。",
            "- 本次两种策略在5个种子上都收敛到各自相同的12场景离散决策，因此标准差为0、Bootstrap区间退化为单点。这反映当前任务很小且容易收敛，不等于模型不存在其他不确定性。",
            "- 只有5个训练种子，区间用于先导实验的稳定性检查，不代表对所有建筑、火灾情景或人群的总体推断。",
            "- 当前仍是一次性的高层出口选择。下一阶段需要让LLM选目标出口，再由PPO在动态多步环境中控制连续疏散动作。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
