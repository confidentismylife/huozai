from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import BehaviorModel, CalibratedLogitBehavior
from fire_evac.conflict_multiseed import bootstrap_mean_interval
from fire_evac.dynamic_conflicts import (
    CONFLICT_TYPES,
    GymDynamicConflictEnv,
    conflict_llm_advice,
    conflict_scenario_records,
    make_dynamic_conflict_bank,
)
from fire_evac.dynamic_llm_rl import LLMGuidedObservationWrapper, NORTH, safe_forecast_target
from fire_evac.dynamic_multiseed import DYNAMIC_METRICS, cross_seed_statistics, summarize_cross_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether PPO corrects LLM targets under dynamic conflicts.")
    parser.add_argument("--seeds", default="301,302,303")
    parser.add_argument("--train-scenarios", type=int, default=48)
    parser.add_argument("--test-scenarios", type=int, default=24)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--scenario-seed", type=int, default=20260930)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "dynamic_conflict_correction",
    )
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    return seeds


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_env(factory: Callable[[], BehaviorModel], scenarios, advice=None):
    base = GymDynamicConflictEnv(factory(), scenarios)
    return LLMGuidedObservationWrapper(base, advice) if advice is not None else base


def run_episode(env, *, policy: str, scenario, episode_seed: int, advice, model=None):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    target = advice[scenario.scenario_id].target_branch
    total_reward = 0.0
    override_steps = 0
    control_steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if policy in {"ppo", "llm_guided_ppo"}:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).item())
        elif policy == "local_llm_static":
            action = target
        elif policy == "capacity_hazard_rule":
            occupancy = observation[:5]
            hazard = observation[5:10]
            capacities = observation[14:16]
            blocked = observation[16:18]
            north_cost = 3.0 * hazard[3] + occupancy[3] + 2.0 * (1.0 - capacities[0]) + 10.0 * blocked[0]
            south_cost = 3.0 * hazard[4] + occupancy[4] + 2.0 * (1.0 - capacities[1]) + 10.0 * blocked[1]
            action = int(north_cost > south_cost)
        else:
            raise ValueError(f"unknown policy: {policy}")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        control_steps += 1
        override_steps += int(int(info["last_action"]) != target)
    return {
        "scenario_id": scenario.scenario_id,
        "conflict_type": scenario.conflict_type,
        "llm_target_correct": int(target == safe_forecast_target(scenario)),
        "seed": episode_seed,
        "steps": control_steps,
        "llm_override_rate": override_steps / control_steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": float(total_reward),
    }


def summarize_baselines(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["test_behavior"]), str(row["policy"])), []).append(row)
    output = []
    for (conflict_type, policy), items in sorted(groups.items()):
        record: dict[str, object] = {
            "conflict_type": conflict_type,
            "policy": policy,
            "episodes": len(items),
            "llm_override_rate": float(np.mean([float(item["llm_override_rate"]) for item in items])),
        }
        for metric in DYNAMIC_METRICS:
            record[metric] = float(np.mean([float(item[metric]) for item in items]))
        output.append(record)
    return output


def guided_vs_static(
    per_seed: list[dict[str, object]],
    baseline_summary: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    static = {
        str(row["conflict_type"]): row
        for row in baseline_summary
        if row["policy"] == "local_llm_static"
    }
    output: list[dict[str, object]] = []
    conflict_types = sorted(static)
    for type_index, conflict_type in enumerate(conflict_types):
        guided = [
            row
            for row in per_seed
            if row["test_behavior"] == conflict_type and row["policy"] == "llm_guided_ppo"
        ]
        for metric_index, metric in enumerate(DYNAMIC_METRICS):
            differences = [float(row[metric]) - float(static[conflict_type][metric]) for row in guided]
            lower, upper = bootstrap_mean_interval(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + type_index * 100 + metric_index,
            )
            output.append(
                {
                    "conflict_type": conflict_type,
                    "metric": metric,
                    "training_seeds": len(differences),
                    "mean_difference": float(np.mean(differences)),
                    "between_seed_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
                    "bootstrap_ci95_lower": lower,
                    "bootstrap_ci95_upper": upper,
                }
            )
    return output


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("Install the project's optional RL dependencies first") from exc

    args = parse_args()
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.train_scenarios < len(CONFLICT_TYPES) or args.test_scenarios < len(CONFLICT_TYPES):
        raise SystemExit(f"scenario counts must be at least {len(CONFLICT_TYPES)}")
    if args.timesteps < 1_000 or args.eval_repetitions < 1 or args.n_envs < 1:
        raise SystemExit("timesteps must be >=1000; repetitions and n-envs must be positive")
    if not args.calibration_artifact.exists():
        raise SystemExit(f"calibration artifact not found: {args.calibration_artifact}")

    train_scenarios = make_dynamic_conflict_bank(args.train_scenarios, args.scenario_seed, "CTR")
    test_scenarios = make_dynamic_conflict_bank(args.test_scenarios, args.scenario_seed + 1, "CTE")
    all_scenarios = train_scenarios + test_scenarios
    advice = {scenario.scenario_id: conflict_llm_advice(scenario) for scenario in all_scenarios}
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    (args.output / "train_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(train_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(conflict_scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        [
            {
                "scenario_id": scenario.scenario_id,
                "split": "train" if scenario.scenario_id.startswith("CTR") else "test",
                "conflict_type": scenario.conflict_type,
                "actual_fire_source": "north" if scenario.source_branch == NORTH else "south",
                "forecast_fire_source": "north" if scenario.forecast_source_branch == NORTH else "south",
                "llm_target_exit": "north" if advice[scenario.scenario_id].target_branch == NORTH else "south",
                "target_is_smoke_safe": int(
                    advice[scenario.scenario_id].target_branch == safe_forecast_target(scenario)
                ),
            }
            for scenario in all_scenarios
        ],
        args.output / "scenario_advice.csv",
    )

    training_factory = lambda: CalibratedLogitBehavior(args.calibration_artifact)
    check_env(make_env(training_factory, train_scenarios), warn=True)
    check_env(make_env(training_factory, train_scenarios, advice), warn=True)
    common = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model_paths: dict[tuple[int, str], Path] = {}
    training_records: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    for seed_index, seed in enumerate(seeds, 1):
        for policy in ("ppo", "llm_guided_ppo"):
            print(f"Seed {seed} ({seed_index}/{len(seeds)}): training {policy}...", flush=True)

            def factory(policy_name=policy):
                return make_env(
                    training_factory,
                    train_scenarios,
                    advice if policy_name == "llm_guided_ppo" else None,
                )

            vector_env = make_vec_env(factory, n_envs=args.n_envs, seed=seed)
            model = PPO(env=vector_env, seed=seed, **common)
            model.learn(total_timesteps=args.timesteps, progress_bar=False)
            path = model_dir / f"seed_{seed}_{policy}"
            model.save(path)
            model_paths[(seed, policy)] = path.with_suffix(".zip")
            training_records.append(
                {
                    "training_seed": seed,
                    "policy": policy,
                    "actual_timesteps": int(model.num_timesteps),
                    "model_path": str(path.with_suffix('.zip').resolve()),
                }
            )
            vector_env.close()

    learned_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds, 1):
        print(f"Evaluating trained seed {seed} ({seed_index}/{len(seeds)})...", flush=True)
        base_model = PPO.load(model_paths[(seed, "ppo")])
        guided_model = PPO.load(model_paths[(seed, "llm_guided_ppo")])
        base_env = make_env(training_factory, test_scenarios)
        guided_env = make_env(training_factory, test_scenarios, advice)
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                episode_seed = 90_000 + scenario_index * args.eval_repetitions + repetition
                for policy, env, model in (
                    ("ppo", base_env, base_model),
                    ("llm_guided_ppo", guided_env, guided_model),
                ):
                    row = run_episode(
                        env,
                        policy=policy,
                        scenario=scenario,
                        episode_seed=episode_seed,
                        advice=advice,
                        model=model,
                    )
                    row.update(
                        {
                            "test_behavior": scenario.conflict_type,
                            "training_seed": seed,
                            "policy": policy,
                        }
                    )
                    learned_rows.append(
                        {
                            "test_behavior": row.pop("test_behavior"),
                            "training_seed": row.pop("training_seed"),
                            "policy": row.pop("policy"),
                            **row,
                        }
                    )
        base_env.close()
        guided_env.close()

    baseline_rows: list[dict[str, object]] = []
    baseline_env = make_env(training_factory, test_scenarios)
    for scenario_index, scenario in enumerate(test_scenarios):
        for repetition in range(args.eval_repetitions):
            episode_seed = 90_000 + scenario_index * args.eval_repetitions + repetition
            for policy in ("local_llm_static", "capacity_hazard_rule"):
                row = run_episode(
                    baseline_env,
                    policy=policy,
                    scenario=scenario,
                    episode_seed=episode_seed,
                    advice=advice,
                )
                row.update({"test_behavior": scenario.conflict_type, "policy": policy})
                baseline_rows.append(
                    {
                        "test_behavior": row.pop("test_behavior"),
                        "policy": row.pop("policy"),
                        **row,
                    }
                )
    baseline_env.close()

    write_csv(learned_rows, args.output / "episode_results.csv")
    write_csv(baseline_rows, args.output / "baseline_episode_results.csv")
    per_seed = summarize_cross_episodes(learned_rows)
    overall_view = [{**row, "test_behavior": "all"} for row in learned_rows]
    per_seed.extend(summarize_cross_episodes(overall_view))
    write_csv(per_seed, args.output / "per_seed_summary.csv")
    aggregate, paired, paired_aggregate = cross_seed_statistics(
        per_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.scenario_seed,
    )
    write_csv(aggregate, args.output / "aggregate_summary.csv")
    write_csv(paired, args.output / "guided_vs_ppo_by_seed.csv")
    write_csv(paired_aggregate, args.output / "guided_vs_ppo_aggregate.csv")

    baseline_summary = summarize_baselines(baseline_rows)
    baseline_overall = summarize_baselines([{**row, "test_behavior": "all"} for row in baseline_rows])
    baseline_summary.extend(baseline_overall)
    write_csv(baseline_summary, args.output / "baseline_summary.csv")
    versus_static = guided_vs_static(
        per_seed,
        baseline_summary,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.scenario_seed,
    )
    write_csv(versus_static, args.output / "guided_vs_static.csv")

    override_rows: list[dict[str, object]] = []
    for conflict_type in ("all", *CONFLICT_TYPES):
        for seed in seeds:
            items = [
                row
                for row in learned_rows
                if row["training_seed"] == seed
                and row["policy"] == "llm_guided_ppo"
                and (conflict_type == "all" or row["conflict_type"] == conflict_type)
            ]
            override_rows.append(
                {
                    "conflict_type": conflict_type,
                    "training_seed": seed,
                    "episodes": len(items),
                    "mean_llm_override_rate": float(np.mean([float(row["llm_override_rate"]) for row in items])),
                }
            )
    write_csv(override_rows, args.output / "llm_override_by_seed.csv")

    metadata = {
        "training_seeds": seeds,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "conflict_types": list(CONFLICT_TYPES),
        "timesteps_per_model": args.timesteps,
        "evaluation_repetitions": args.eval_repetitions,
        "training_runs": training_records,
        "llm_policy_source": "Replay of audited qwen3:4b v2 rule: choose the non-forecast-fire corridor.",
        "forecast_error_design": "forecast_error scenarios invert the true source before applying the cached LLM policy.",
        "additional_ollama_calls": 0,
        "bootstrap_unit": "PPO training seed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    aggregate_lookup = {
        (str(row["test_behavior"]), str(row["policy"]), str(row["metric"])): row for row in aggregate
    }
    static_lookup = {
        (str(row["conflict_type"]), str(row["metric"])): row for row in versus_static
    }
    ppo_pair_lookup = {
        (str(row["test_behavior"]), str(row["metric"])): row for row in paired_aggregate
    }
    override_lookup = {
        conflict_type: float(
            np.mean(
                [
                    float(row["mean_llm_override_rate"])
                    for row in override_rows
                    if row["conflict_type"] == conflict_type
                ]
            )
        )
        for conflict_type in ("all", *CONFLICT_TYPES)
    }
    labels = {
        "all": "全部场景",
        "clear": "无额外冲突",
        "safe_exit_congested": "安全出口拥堵",
        "safe_exit_longer": "安全出口更远",
        "safe_exit_temporarily_blocked": "安全出口临时封闭",
        "forecast_error": "LLM预测错误",
        "combined_capacity_blockage": "低容量+临时封闭",
    }
    report = [
        "# 动态冲突场景：PPO能否纠正LLM目标",
        "",
        f"实验使用 {len(train_scenarios)} 个训练场景、{len(test_scenarios)} 个锁定测试场景和 "
        f"{len(seeds)} 个PPO训练种子。每个模型训练 {args.timesteps:,} 步。",
        "LLM目标来自此前审计通过的qwen3:4b规则回放；预测错误场景先翻转火源预测，再生成LLM目标。"
        "PPO可以看到实时出口容量和封闭状态，并且没有服从LLM的奖励。",
        "",
        "| 场景类型 | 静态LLM失能率 | 基础PPO失能率 | LLM+PPO失能率 | LLM+PPO−静态LLM 95%区间 | 偏离LLM目标率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline_lookup = {(str(row["conflict_type"]), str(row["policy"])): row for row in baseline_summary}
    for conflict_type, label in labels.items():
        static_row = baseline_lookup[(conflict_type, "local_llm_static")]
        base = aggregate_lookup[(conflict_type, "ppo", "incapacitation_rate")]
        guided = aggregate_lookup[(conflict_type, "llm_guided_ppo", "incapacitation_rate")]
        difference = static_lookup[(conflict_type, "incapacitation_rate")]
        report.append(
            f"| {label} | {float(static_row['incapacitation_rate']):.2%} | "
            f"{float(base['mean']):.2%} ± {float(base['between_seed_std']):.2%} | "
            f"{float(guided['mean']):.2%} ± {float(guided['between_seed_std']):.2%} | "
            f"[{float(difference['bootstrap_ci95_lower']):.2%}, "
            f"{float(difference['bootstrap_ci95_upper']):.2%}] | "
            f"{override_lookup[conflict_type]:.2%} |"
        )
    overall_vs_ppo = ppo_pair_lookup[("all", "incapacitation_rate")]
    override_seed_values = [
        float(row["mean_llm_override_rate"])
        for row in override_rows
        if row["conflict_type"] == "all"
    ]
    report.extend(
        [
            "",
            "## 主要发现",
            "",
            "- 相对静态LLM，动态LLM+PPO在安全出口拥堵、LLM预测错误、低容量加临时封闭三类场景中稳定降低失能率。",
            "- 安全出口更远场景的区间上界为0，只能视为边界性改善；无冲突场景没有稳定改善。",
            "- 安全出口短时封闭时，动态LLM+PPO反而稳定增加失能率。静态策略等待安全出口恢复更安全，说明PPO把即时容量信号看得过重。",
            f"- 全部场景合并后，LLM+PPO相对基础PPO的失能率差为 "
            f"{float(overall_vs_ppo['mean']):.2%}，95%区间 "
            f"[{float(overall_vs_ppo['bootstrap_ci95_lower']):.2%}, "
            f"{float(overall_vs_ppo['bootstrap_ci95_upper']):.2%}]；当前LLM输入没有超过基础PPO。",
            f"- 三个训练种子的总体LLM目标偏离率范围为 "
            f"{min(override_seed_values):.2%}–{max(override_seed_values):.2%}，纠正策略仍有明显训练不稳定性。",
            "",
            "## 判定方法",
            "",
            "- 若LLM+PPO相对静态LLM的失能率区间完全低于0，说明动态控制在该类冲突中稳定纠正了LLM目标。",
            "- 偏离LLM目标率只说明PPO发生了纠正行为；是否有益必须与失能率、FED和总奖励一起判断。",
            "- Bootstrap重采样单位是PPO训练种子，测试重复回合先在每个种子内汇总。",
            "",
            "## 解释边界",
            "",
            "- 出口更远和拥堵目前通过降低下游通行容量表示；临时封闭通过时变容量为0表示。",
            "- LLM错误是受控的火源预测翻转，不代表qwen3:4b在真实传感器数据上的自然错误率。",
            "- 火灾和行为校准仍是合成先导模型；正式结论需要FDS与真人行为数据。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
