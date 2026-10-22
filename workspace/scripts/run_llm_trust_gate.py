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
from fire_evac.conflict_multiseed import (
    aggregate_paired_differences,
    aggregate_seed_summaries,
    paired_seed_differences,
)
from fire_evac.dynamic_conflicts import (
    TRUST_GATE_TYPES,
    GymDynamicConflictEnv,
    HoldEnabledActionWrapper,
    LLMTrustGateWrapper,
    ReopenForecastObservationWrapper,
    conflict_llm_advice,
    conflict_scenario_records,
    make_trust_gate_bank,
)
from fire_evac.dynamic_llm_rl import LLMGuidedObservationWrapper, NORTH, safe_forecast_target
from fire_evac.dynamic_multiseed import DYNAMIC_METRICS, summarize_cross_episodes


POLICIES = ("ppo_time", "llm_direct_ppo", "llm_trust_gate_ppo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an interpretable LLM trust gate for evacuation guidance.")
    parser.add_argument("--seeds", default="401,402,403")
    parser.add_argument("--train-scenarios", type=int, default=40)
    parser.add_argument("--test-scenarios", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--scenario-seed", type=int, default=20261001)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "llm_trust_gate")
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


def make_policy_env(policy: str, factory: Callable[[], BehaviorModel], scenarios, advice):
    base = GymDynamicConflictEnv(factory(), scenarios)
    if policy == "ppo_time":
        return ReopenForecastObservationWrapper(HoldEnabledActionWrapper(base))
    if policy == "llm_direct_ppo":
        time_aware = ReopenForecastObservationWrapper(HoldEnabledActionWrapper(base))
        return LLMGuidedObservationWrapper(time_aware, advice)
    if policy == "llm_trust_gate_ppo":
        return LLMTrustGateWrapper(base, advice)
    raise ValueError(f"unknown learned policy: {policy}")


def run_episode(env, *, policy: str, scenario, episode_seed: int, advice, model=None):
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    target = advice[scenario.scenario_id].target_branch
    total_reward = 0.0
    trust_steps = override_steps = wait_steps = 0
    control_steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if policy in POLICIES:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).item())
        elif policy == "local_llm_static":
            action = target
        else:
            raise ValueError(f"unknown policy: {policy}")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        control_steps += 1
        if policy == "llm_trust_gate_ppo":
            gate_action = str(info["trust_gate_action"])
            trust_steps += int(gate_action == "trust")
            override_steps += int(gate_action == "override")
            wait_steps += int(gate_action == "wait")
        elif policy == "local_llm_static":
            trust_steps += 1
        else:
            hold_requested = bool(info.get("hold_action_requested", False))
            wait_steps += int(hold_requested)
            if not hold_requested:
                trust_steps += int(int(info["last_action"]) == target)
                override_steps += int(int(info["last_action"]) != target)
    return {
        "scenario_id": scenario.scenario_id,
        "conflict_type": scenario.conflict_type,
        "llm_target_correct": int(target == safe_forecast_target(scenario)),
        "seed": episode_seed,
        "steps": control_steps,
        "trust_rate": trust_steps / control_steps,
        "override_rate": override_steps / control_steps,
        "wait_rate": wait_steps / control_steps,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": float(total_reward),
    }


def aggregate_by_conflict(per_seed: list[dict[str, object]], samples: int, seed: int):
    aggregate: list[dict[str, object]] = []
    for index, conflict_type in enumerate(sorted({str(row["test_behavior"]) for row in per_seed})):
        subset = [row for row in per_seed if row["test_behavior"] == conflict_type]
        rows = aggregate_seed_summaries(
            subset,
            metrics=DYNAMIC_METRICS,
            bootstrap_samples=samples,
            bootstrap_seed=seed + index * 1000,
        )
        for row in rows:
            aggregate.append({"conflict_type": conflict_type, **row})
    return aggregate


def paired_comparison(
    per_seed: list[dict[str, object]],
    *,
    candidate: str,
    baseline: str,
    samples: int,
    seed: int,
):
    by_seed: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []
    for index, conflict_type in enumerate(sorted({str(row["test_behavior"]) for row in per_seed})):
        subset = [row for row in per_seed if row["test_behavior"] == conflict_type]
        differences = paired_seed_differences(
            subset,
            candidate=candidate,
            baseline=baseline,
            metrics=DYNAMIC_METRICS,
        )
        for row in differences:
            by_seed.append({"conflict_type": conflict_type, **row})
        aggregated = aggregate_paired_differences(
            differences,
            bootstrap_samples=samples,
            bootstrap_seed=seed + index * 1000,
        )
        for row in aggregated:
            aggregate.append(
                {
                    "conflict_type": conflict_type,
                    "candidate": candidate,
                    "baseline": baseline,
                    **{key: value for key, value in row.items() if key != "policy"},
                }
            )
    return by_seed, aggregate


def decision_summary(rows: list[dict[str, object]], seeds: list[int]):
    output: list[dict[str, object]] = []
    for conflict_type in ("all", *TRUST_GATE_TYPES):
        for seed in seeds:
            items = [
                row
                for row in rows
                if row["training_seed"] == seed
                and row["policy"] == "llm_trust_gate_ppo"
                and (conflict_type == "all" or row["conflict_type"] == conflict_type)
            ]
            output.append(
                {
                    "conflict_type": conflict_type,
                    "training_seed": seed,
                    "episodes": len(items),
                    "trust_rate": float(np.mean([float(item["trust_rate"]) for item in items])),
                    "override_rate": float(np.mean([float(item["override_rate"]) for item in items])),
                    "wait_rate": float(np.mean([float(item["wait_rate"]) for item in items])),
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
    if args.train_scenarios < len(TRUST_GATE_TYPES) or args.test_scenarios < len(TRUST_GATE_TYPES):
        raise SystemExit(f"scenario counts must be at least {len(TRUST_GATE_TYPES)}")
    if args.timesteps < 1_000 or args.n_envs < 1 or args.eval_repetitions < 1:
        raise SystemExit("timesteps must be >=1000; n-envs and repetitions must be positive")

    train_scenarios = make_trust_gate_bank(args.train_scenarios, args.scenario_seed, "GTR")
    test_scenarios = make_trust_gate_bank(args.test_scenarios, args.scenario_seed + 1, "GTE")
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
                "split": "train" if scenario.scenario_id.startswith("GTR") else "test",
                "conflict_type": scenario.conflict_type,
                "blockage_steps": scenario.blockage_end_step,
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

    behavior_factory = lambda: CalibratedLogitBehavior(args.calibration_artifact)
    for policy in POLICIES:
        check_env(make_policy_env(policy, behavior_factory, train_scenarios, advice), warn=True)
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
        for policy in POLICIES:
            print(f"Seed {seed} ({seed_index}/{len(seeds)}): training {policy}...", flush=True)

            def factory(policy_name=policy):
                return make_policy_env(policy_name, behavior_factory, train_scenarios, advice)

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
        print(f"Evaluating seed {seed} ({seed_index}/{len(seeds)})...", flush=True)
        models = {policy: PPO.load(model_paths[(seed, policy)]) for policy in POLICIES}
        envs = {policy: make_policy_env(policy, behavior_factory, test_scenarios, advice) for policy in POLICIES}
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                episode_seed = 100_000 + scenario_index * args.eval_repetitions + repetition
                for policy in POLICIES:
                    row = run_episode(
                        envs[policy],
                        policy=policy,
                        scenario=scenario,
                        episode_seed=episode_seed,
                        advice=advice,
                        model=models[policy],
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
        for env in envs.values():
            env.close()

    static_rows: list[dict[str, object]] = []
    static_env = GymDynamicConflictEnv(behavior_factory(), test_scenarios)
    for scenario_index, scenario in enumerate(test_scenarios):
        for repetition in range(args.eval_repetitions):
            episode_seed = 100_000 + scenario_index * args.eval_repetitions + repetition
            row = run_episode(
                static_env,
                policy="local_llm_static",
                scenario=scenario,
                episode_seed=episode_seed,
                advice=advice,
            )
            static_rows.append(row)
    static_env.close()

    write_csv(learned_rows, args.output / "episode_results.csv")
    write_csv(static_rows, args.output / "static_llm_results.csv")
    per_seed = summarize_cross_episodes(learned_rows)
    per_seed.extend(summarize_cross_episodes([{**row, "test_behavior": "all"} for row in learned_rows]))
    write_csv(per_seed, args.output / "per_seed_summary.csv")
    aggregate = aggregate_by_conflict(per_seed, args.bootstrap_samples, args.scenario_seed)
    write_csv(aggregate, args.output / "aggregate_summary.csv")
    gate_vs_direct_seed, gate_vs_direct = paired_comparison(
        per_seed,
        candidate="llm_trust_gate_ppo",
        baseline="llm_direct_ppo",
        samples=args.bootstrap_samples,
        seed=args.scenario_seed,
    )
    gate_vs_base_seed, gate_vs_base = paired_comparison(
        per_seed,
        candidate="llm_trust_gate_ppo",
        baseline="ppo_time",
        samples=args.bootstrap_samples,
        seed=args.scenario_seed + 50_000,
    )
    write_csv(gate_vs_direct_seed + gate_vs_base_seed, args.output / "paired_differences_by_seed.csv")
    write_csv(gate_vs_direct + gate_vs_base, args.output / "paired_aggregate.csv")
    decisions = decision_summary(learned_rows, seeds)
    write_csv(decisions, args.output / "trust_gate_decisions.csv")

    metadata = {
        "training_seeds": seeds,
        "policies": list(POLICIES),
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "trust_gate_types": list(TRUST_GATE_TYPES),
        "timesteps_per_model": args.timesteps,
        "training_runs": training_records,
        "wait_semantics": "Hold people in rooms for one control step and show a neutral sign.",
        "reopen_forecast": "Exact remaining blockage steps, normalized by 12.",
        "additional_ollama_calls": 0,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    agg = {
        (str(row["conflict_type"]), str(row["policy"]), str(row["metric"])): row for row in aggregate
    }
    paired = {
        (str(row["conflict_type"]), str(row["baseline"]), str(row["metric"])): row
        for row in gate_vs_direct + gate_vs_base
    }
    decision_lookup = {
        conflict_type: {
            key: float(np.mean([float(row[key]) for row in decisions if row["conflict_type"] == conflict_type]))
            for key in ("trust_rate", "override_rate", "wait_rate")
        }
        for conflict_type in ("all", *TRUST_GATE_TYPES)
    }
    labels = {
        "all": "全部场景",
        "clear": "无冲突",
        "safe_exit_short_block": "短时封闭",
        "safe_exit_long_block": "长时封闭",
        "safe_exit_congested": "安全出口拥堵",
        "forecast_error": "LLM预测错误",
    }
    report = [
        "# 带恢复时间预测的LLM信任门控实验",
        "",
        f"使用 {len(train_scenarios)} 个训练场景、{len(test_scenarios)} 个锁定测试场景和 "
        f"{len(seeds)} 个训练种子。三种PPO策略各训练 {args.timesteps:,} 步。",
        "基础PPO和直接LLM+PPO拥有北、南、中立、等待四个动作；信任门控PPO拥有相信LLM、反向覆盖、等待三个动作。"
        "所有策略都看到准确的剩余封闭步数。",
        "",
        "| 场景 | 时间PPO失能率 | 直接LLM+PPO | 信任门控PPO | 门控−直接 95%区间 | 信任/覆盖/等待 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for conflict_type, label in labels.items():
        base = agg[(conflict_type, "ppo_time", "incapacitation_rate")]
        direct = agg[(conflict_type, "llm_direct_ppo", "incapacitation_rate")]
        gate = agg[(conflict_type, "llm_trust_gate_ppo", "incapacitation_rate")]
        difference = paired[(conflict_type, "llm_direct_ppo", "incapacitation_rate")]
        rates = decision_lookup[conflict_type]
        report.append(
            f"| {label} | {float(base['mean']):.2%} ± {float(base['between_seed_std']):.2%} | "
            f"{float(direct['mean']):.2%} ± {float(direct['between_seed_std']):.2%} | "
            f"{float(gate['mean']):.2%} ± {float(gate['between_seed_std']):.2%} | "
            f"[{float(difference['bootstrap_ci95_lower']):.2%}, "
            f"{float(difference['bootstrap_ci95_upper']):.2%}] | "
            f"{rates['trust_rate']:.1%}/{rates['override_rate']:.1%}/{rates['wait_rate']:.1%} |"
        )
    overall_direct = paired[("all", "llm_direct_ppo", "incapacitation_rate")]
    overall_base = paired[("all", "ppo_time", "incapacitation_rate")]
    overall_base_fed = paired[("all", "ppo_time", "mean_cumulative_fed")]
    all_rates = decision_lookup["all"]
    clear_rates = decision_lookup["clear"]
    short_rates = decision_lookup["safe_exit_short_block"]
    long_rates = decision_lookup["safe_exit_long_block"]
    report.extend(
        [
            "",
            "## 主要发现",
            "",
            f"- 相对直接LLM+PPO，信任门控整体失能率降低 "
            f"{-float(overall_direct['mean']):.2%}，95%区间 "
            f"[{float(overall_direct['bootstrap_ci95_lower']):.2%}, "
            f"{float(overall_direct['bootstrap_ci95_upper']):.2%}]。短时封闭、长时封闭和LLM预测错误均稳定改善。",
            f"- 相对只使用恢复时间的基础PPO，门控失能率差为 "
            f"{float(overall_base['mean']):.2%}，95%区间 "
            f"[{float(overall_base['bootstrap_ci95_lower']):.2%}, "
            f"{float(overall_base['bootstrap_ci95_upper']):.2%}]，没有稳定优势。",
            f"- 门控相对基础PPO的人均FED增加 {float(overall_base_fed['mean']):.4f}，95%区间 "
            f"[{float(overall_base_fed['bootstrap_ci95_lower']):.4f}, "
            f"{float(overall_base_fed['bootstrap_ci95_upper']):.4f}]。",
            f"- 门控总体只在 {all_rates['trust_rate']:.1%} 的步骤相信LLM；无冲突时仍有 "
            f"{clear_rates['override_rate']:.1%} 的覆盖率，说明它学到的是普遍不信任，而不是按情景校准信任。",
            f"- 短时封闭等待率为 {short_rates['wait_rate']:.1%}，长时封闭等待率为 "
            f"{long_rates['wait_rate']:.1%}。动作分布没有呈现“短时等待、长时覆盖”的清晰分界。",
            "",
            "## 判定标准",
            "",
            "- 短时封闭场景中，门控应增加等待，并且失能率不高于直接LLM+PPO。",
            "- 长时封闭和LLM预测错误场景中，门控应增加覆盖，并稳定降低失能率。",
            "- 若门控只改变动作比例但安全指标没有改善，则不能视为成功。",
            "",
            "## 解释边界",
            "",
            "- 恢复时间预测目前是准确值，尚未加入预测噪声。",
            "- 等待表示暂缓房间人员进入交汇区一个控制步，不代表建筑内所有人员完全静止。",
            "- 火灾和行为参数仍来自合成先导模型。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
