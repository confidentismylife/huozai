from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.comparison import paired_bootstrap_comparisons
from fire_evac.dynamic_llm_rl import (
    NORTH,
    SOUTH,
    DynamicLLMAdvice,
    LLMGuidedObservationWrapper,
    ScenarioBankHazardProvider,
    dynamic_advice_prompt,
    make_scenario_bank,
    parse_dynamic_advice,
    safe_forecast_target,
    scenario_records,
)
from fire_evac.gym_env import GymFireEvacuationEnv


ADVICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_exit": {"type": "string", "enum": ["north", "south"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "string",
                "enum": [
                    "avoid_predicted_smoke",
                    "early_guidance",
                    "reduce_exposure",
                    "balance_congestion",
                ],
            },
        },
    },
    "required": ["target_exit", "confidence", "reason_codes"],
}


SUMMARY_METRICS = (
    "evacuation_rate",
    "incapacitation_rate",
    "mean_cumulative_fed",
    "cumulative_congestion",
    "sign_switches",
    "total_reward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with cached episode-level LLM fire forecasts.")
    parser.add_argument("--train-scenarios", type=int, default=24)
    parser.add_argument("--test-scenarios", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dynamic_llm_rl_v2")
    return parser.parse_args()


def api_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_advice(
    scenarios,
    *,
    output: Path,
    base_url: str,
    model: str,
    model_digest: str,
    timeout: int,
    seed: int,
) -> dict[str, DynamicLLMAdvice]:
    cache = output / "llm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    advice: dict[str, DynamicLLMAdvice] = {}
    for index, scenario in enumerate(scenarios, 1):
        prompt = dynamic_advice_prompt(scenario)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache / f"{scenario.scenario_id}.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if stored and stored.get("model_digest") == model_digest and stored.get("prompt_hash") == prompt_hash:
            advice[scenario.scenario_id] = parse_dynamic_advice(stored["parsed"])
            continue
        response = api_json(
            f"{base_url}/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": ADVICE_SCHEMA,
                "options": {"temperature": 0, "seed": seed, "num_predict": 80},
                "think": False,
                "stream": False,
            },
            timeout,
        )
        raw = response["message"]["content"]
        parsed_payload = json.loads(raw)
        parsed = parse_dynamic_advice(parsed_payload)
        advice[scenario.scenario_id] = parsed
        path.write_text(
            json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "model": model,
                    "model_digest": model_digest,
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed": parsed_payload,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_duration_ns": response.get("total_duration"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        target = "north" if parsed.target_branch == NORTH else "south"
        print(f"LLM {index}/{len(scenarios)} {scenario.scenario_id}: {target}", flush=True)
    return advice


def make_base_env(artifact: Path, scenarios):
    return GymFireEvacuationEnv(
        CalibratedLogitBehavior(artifact),
        hazard_provider=ScenarioBankHazardProvider(scenarios),
    )


def make_guided_env(artifact: Path, scenarios, advice):
    return LLMGuidedObservationWrapper(make_base_env(artifact, scenarios), advice)


def rollout(
    *,
    controller_name: str,
    scenario,
    episode_seed: int,
    artifact: Path,
    test_scenarios,
    advice: dict[str, DynamicLLMAdvice],
    model=None,
) -> dict[str, object]:
    guided = controller_name == "llm_guided_ppo"
    env = (
        make_guided_env(artifact, test_scenarios, advice)
        if guided
        else make_base_env(artifact, test_scenarios)
    )
    observation, info = env.reset(seed=episode_seed, options={"scenario": scenario})
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        if controller_name in {"ppo", "llm_guided_ppo"}:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).item())
        elif controller_name == "neutral":
            action = 2
        elif controller_name == "hazard_aware_rule":
            occupancy = observation[:5]
            hazard = observation[5:10]
            north_cost = 3.0 * hazard[3] + occupancy[3]
            south_cost = 3.0 * hazard[4] + occupancy[4]
            action = int(north_cost > south_cost)
        elif controller_name == "local_llm_static":
            action = advice[scenario.scenario_id].target_branch
        else:
            raise ValueError(f"unknown controller: {controller_name}")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    env.close()
    return {
        "controller": controller_name,
        "behavior_model": str(info["behavior_model"]),
        "scenario_id": scenario.scenario_id,
        "source_branch": "north" if scenario.source_branch == NORTH else "south",
        "llm_target_exit": "north" if advice[scenario.scenario_id].target_branch == NORTH else "south",
        "seed": episode_seed,
        "steps": int(info["step"]),
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "remaining_rate": float(info["remaining"]) / 240.0,
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "total_reward": float(total_reward),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["controller"]), []).append(row)
    output: list[dict[str, object]] = []
    order = ["neutral", "hazard_aware_rule", "local_llm_static", "ppo", "llm_guided_ppo"]
    for controller in order:
        items = groups[controller]
        record: dict[str, object] = {"controller": controller, "episodes": len(items)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(item[metric]) for item in items], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(record)
    return output


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise SystemExit("Install the project's optional RL dependencies first") from exc

    args = parse_args()
    if args.timesteps < 1_000:
        raise SystemExit("timesteps must be at least 1000")
    if args.n_envs < 1 or args.eval_repetitions < 1:
        raise SystemExit("n-envs and eval-repetitions must be positive")
    if not args.calibration_artifact.exists():
        raise SystemExit(f"calibration artifact not found: {args.calibration_artifact}")
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)

    train_scenarios = make_scenario_bank(args.train_scenarios, args.seed, "DTR")
    test_scenarios = make_scenario_bank(args.test_scenarios, args.seed + 1, "DTE")
    all_scenarios = train_scenarios + test_scenarios
    (args.output / "train_scenarios.json").write_text(
        json.dumps(scenario_records(train_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "test_scenarios.json").write_text(
        json.dumps(scenario_records(test_scenarios), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tags = api_json(f"{args.base_url}/api/tags")
    matching = [item for item in tags.get("models", []) if item.get("name") == args.model]
    if not matching:
        raise SystemExit(f"Ollama model {args.model!r} is not installed")
    model_digest = matching[0].get("digest", "")
    advice = collect_advice(
        all_scenarios,
        output=args.output,
        base_url=args.base_url,
        model=args.model,
        model_digest=model_digest,
        timeout=args.timeout,
        seed=args.seed,
    )
    write_csv(
        [
            {
                "scenario_id": scenario.scenario_id,
                "split": "train" if scenario.scenario_id.startswith("DTR") else "test",
                "source_branch": "north" if scenario.source_branch == NORTH else "south",
                "safe_forecast_exit": "north" if safe_forecast_target(scenario) == NORTH else "south",
                "llm_target_exit": "north" if advice[scenario.scenario_id].target_branch == NORTH else "south",
                "confidence": advice[scenario.scenario_id].confidence,
                "reason_codes": "|".join(advice[scenario.scenario_id].reason_codes),
                "matches_safe_forecast": int(advice[scenario.scenario_id].target_branch == safe_forecast_target(scenario)),
            }
            for scenario in all_scenarios
        ],
        args.output / "scenario_advice.csv",
    )

    check_env(make_base_env(args.calibration_artifact, train_scenarios), warn=True)
    check_env(make_guided_env(args.calibration_artifact, train_scenarios, advice), warn=True)

    def base_factory():
        return make_base_env(args.calibration_artifact, train_scenarios)

    def guided_factory():
        return make_guided_env(args.calibration_artifact, train_scenarios, advice)

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
        seed=args.seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    started = datetime.now(timezone.utc)
    print("Training dynamic PPO without LLM forecast...", flush=True)
    base_vec = make_vec_env(base_factory, n_envs=args.n_envs, seed=args.seed)
    base_model = PPO(env=base_vec, **common)
    base_model.learn(total_timesteps=args.timesteps, progress_bar=False)
    base_model.save(model_dir / "ppo")
    base_vec.close()

    print("Training dynamic PPO with cached LLM forecast...", flush=True)
    guided_vec = make_vec_env(guided_factory, n_envs=args.n_envs, seed=args.seed)
    guided_model = PPO(env=guided_vec, **common)
    guided_model.learn(total_timesteps=args.timesteps, progress_bar=False)
    guided_model.save(model_dir / "llm_guided_ppo")
    guided_vec.close()

    print("Evaluating on locked test scenarios...", flush=True)
    controllers = {
        "neutral": None,
        "hazard_aware_rule": None,
        "local_llm_static": None,
        "ppo": base_model,
        "llm_guided_ppo": guided_model,
    }
    episode_rows: list[dict[str, object]] = []
    for controller_name, controller_model in controllers.items():
        for scenario_index, scenario in enumerate(test_scenarios):
            for repetition in range(args.eval_repetitions):
                episode_seed = 70_000 + scenario_index * args.eval_repetitions + repetition
                episode_rows.append(
                    rollout(
                        controller_name=controller_name,
                        scenario=scenario,
                        episode_seed=episode_seed,
                        artifact=args.calibration_artifact,
                        test_scenarios=test_scenarios,
                        advice=advice,
                        model=controller_model,
                    )
                )
    write_csv(episode_rows, args.output / "episode_results.csv")
    summary = summarize(episode_rows)
    write_csv(summary, args.output / "summary.csv")
    paired = paired_bootstrap_comparisons(
        episode_rows,
        candidate="llm_guided_ppo",
        baselines=("ppo", "local_llm_static", "hazard_aware_rule"),
        metrics=("incapacitation_rate", "mean_cumulative_fed", "cumulative_congestion", "total_reward"),
        bootstrap_samples=5_000,
        seed=args.seed,
    )
    write_csv(paired, args.output / "paired_comparisons.csv")

    test_advice_accuracy = float(
        np.mean(
            [advice[scenario.scenario_id].target_branch == safe_forecast_target(scenario) for scenario in test_scenarios]
        )
    )
    metadata = {
        "seed": args.seed,
        "model": args.model,
        "model_digest": model_digest,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "evaluation_repetitions_per_scenario": args.eval_repetitions,
        "evaluation_episodes_per_controller": len(test_scenarios) * args.eval_repetitions,
        "requested_timesteps_per_ppo": args.timesteps,
        "actual_timesteps_base": int(base_model.num_timesteps),
        "actual_timesteps_guided": int(guided_model.num_timesteps),
        "llm_calls_maximum": len(all_scenarios),
        "llm_test_safe_forecast_accuracy": test_advice_accuracy,
        "reward_shaping_for_llm": False,
        "prompt_version": "v2_explicit_fire_source_semantics",
        "calibration_artifact": str(args.calibration_artifact.resolve()),
        "behavior_evidence_warning": "The current calibrated behavior artifact was fitted to synthetic demo data.",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_by_name = {str(row["controller"]): row for row in summary}
    labels = {
        "neutral": "中立标志",
        "hazard_aware_rule": "实时烟气规则",
        "local_llm_static": "LLM目标静态标志",
        "ppo": "动态PPO",
        "llm_guided_ppo": "LLM指导动态PPO",
    }
    report = [
        "# LLM高层出口规划 + PPO多步动态疏散引导实验",
        "",
        f"本实验使用 {len(train_scenarios)} 个训练火灾场景和 {len(test_scenarios)} 个锁定测试场景。"
        f"每个测试场景重复 {args.eval_repetitions} 个人群随机种子，因此每个控制器评估 "
        f"{len(test_scenarios) * args.eval_repetitions} 个回合。",
        "",
        "LLM只在回合开始时读取火源走廊、烟气增长率和起烟时刻，给出北/南目标出口与置信度。"
        "PPO随后每5秒选择北向标志、南向标志或中立标志。LLM建议只加入观测，没有额外奖励塑形。",
        "",
        f"锁定测试集上，LLM选择非火源走廊的比例为 **{test_advice_accuracy:.2%}**。",
        "",
        "| 控制器 | 疏散率 | 失能率 | 人均累计FED | 累计拥堵 | 标志切换 | 总奖励 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in labels:
        row = summary_by_name[name]
        report.append(
            f"| {labels[name]} | {float(row['evacuation_rate_mean']):.2%} | "
            f"{float(row['incapacitation_rate_mean']):.2%} | "
            f"{float(row['mean_cumulative_fed_mean']):.4f} | "
            f"{float(row['cumulative_congestion_mean']):.2f} | "
            f"{float(row['sign_switches_mean']):.2f} | {float(row['total_reward_mean']):.4f} |"
        )
    paired_lookup = {(str(row["baseline"]), str(row["metric"])): row for row in paired}
    report.extend(
        [
            "",
            "## 配对比较（LLM指导动态PPO − 对照）",
            "",
            "| 对照 | 指标 | 平均差 | Bootstrap 95%区间 |",
            "|---|---|---:|---:|",
        ]
    )
    paired_specs = (
        ("ppo", "动态PPO", "incapacitation_rate", "失能率", ".2%"),
        ("ppo", "动态PPO", "mean_cumulative_fed", "人均累计FED", ".4f"),
        ("local_llm_static", "LLM静态标志", "mean_cumulative_fed", "人均累计FED", ".4f"),
        ("local_llm_static", "LLM静态标志", "cumulative_congestion", "累计拥堵", ".4f"),
        ("hazard_aware_rule", "实时烟气规则", "mean_cumulative_fed", "人均累计FED", ".4f"),
    )
    for baseline, baseline_label, metric, metric_label, formatter in paired_specs:
        row = paired_lookup[(baseline, metric)]
        report.append(
            f"| {baseline_label} | {metric_label} | {format(float(row['mean_difference']), formatter)} | "
            f"[{format(float(row['ci_2_5']), formatter)}, {format(float(row['ci_97_5']), formatter)}] |"
        )
    report.extend(
        [
            "",
            "## 提示词失败审计",
            "",
            "第一版提示词没有解释 `fire source corridor` 只表示烟气来源，qwen3:4b在36/36个场景中都把火源走廊选成目标出口。"
            "该版本完整保存在 `outputs/dynamic_llm_rl_prompt_v1_failure`。第二版明确火源走廊会变得危险后，36/36个场景均选择非火源走廊。"
            "这说明小模型输出高度依赖术语定义，部署前必须做语义单元测试和建议正确性审计。",
            "",
            "## 解释边界",
            "",
            "- 这是图结构、多时间步的动态标志控制实验。它还不是二维连续坐标下的单个行人导航。",
            "- 火灾采用快速合成FED模型；正式论文需要用FDS导出的烟气/FED时间序列替换。",
            "- 当前行为校准文件来自合成演示数据，因此本轮验证的是系统链路和可检验假设，不是最终人因结论。",
            "- 当前只训练一个PPO随机种子。若动态实验显示有价值，下一步再做多训练种子和跨行为模型检验。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()
