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

from fire_evac.conflict_experiment import (
    LEFT,
    LLMAdvice,
    ConflictChoiceEnv,
    conflict_prompt,
    evaluate_probability_policies,
    generate_conflict_scenarios,
    paper_derived_logit_probability_left,
    parse_llm_advice,
    save_scenarios,
    scenario_observation,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recommended_exit": {"type": "string", "enum": ["left", "right"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "string",
                "enum": [
                    "avoid_smoke",
                    "shorter_distance",
                    "avoid_congestion",
                    "follow_sign",
                    "follow_crowd",
                    "follow_warden",
                    "resolve_conflict",
                ],
            },
        },
    },
    "required": ["recommended_exit", "confidence", "reason_codes"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paper-logit, PPO and cached local-LLM+PPO on cue conflicts.")
    parser.add_argument("--scenarios", type=int, default=48)
    parser.add_argument("--train-scenarios", type=int, default=36)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "conflict_llm_rl")
    return parser.parse_args()


def api_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_advice(args: argparse.Namespace, scenarios, model_digest: str) -> dict[str, LLMAdvice]:
    cache = args.output / "llm_cache"
    cache.mkdir(exist_ok=True)
    advice: dict[str, LLMAdvice] = {}
    for index, scenario in enumerate(scenarios, 1):
        prompt = conflict_prompt(scenario)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache / f"{scenario.scenario_id}.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if stored and stored.get("model_digest") == model_digest and stored.get("prompt_hash") == prompt_hash:
            parsed = parse_llm_advice(stored["parsed"])
            advice[scenario.scenario_id] = parsed
            continue
        response = api_json(
            f"{args.base_url}/api/chat",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": SCHEMA,
                "options": {"temperature": 0, "seed": args.seed, "num_predict": 100},
                "think": False,
                "stream": False,
            },
            args.timeout,
        )
        raw = response["message"]["content"]
        parsed = parse_llm_advice(raw)
        advice[scenario.scenario_id] = parsed
        path.write_text(
            json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "model": args.model,
                    "model_digest": model_digest,
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed": json.loads(raw),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "total_duration_ns": response.get("total_duration"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"LLM {index}/{len(scenarios)} {scenario.scenario_id}: {json.loads(raw)['recommended_exit']}", flush=True)
    return advice


def deterministic_probabilities(model, scenarios, advice: dict[str, LLMAdvice] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for scenario in scenarios:
        observation = scenario_observation(scenario, None if advice is None else advice[scenario.scenario_id])
        action, _ = model.predict(observation, deterministic=True)
        values[scenario.scenario_id] = 1.0 if int(np.asarray(action).item()) == LEFT else 0.0
    return values


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("Install the project's optional RL dependencies first") from exc

    args = parse_args()
    if not 4 <= args.train_scenarios < args.scenarios:
        raise SystemExit("train-scenarios must be at least 4 and smaller than scenarios")
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = generate_conflict_scenarios(args.scenarios, args.seed)
    training = scenarios[: args.train_scenarios]
    testing = scenarios[args.train_scenarios :]
    save_scenarios(training, args.output / "train_scenarios.json")
    save_scenarios(testing, args.output / "test_scenarios.json")

    tags = api_json(f"{args.base_url}/api/tags")
    matching = [item for item in tags.get("models", []) if item.get("name") == args.model]
    if not matching:
        raise SystemExit(f"Ollama model {args.model!r} is not installed")
    model_digest = matching[0].get("digest", "")
    advice = collect_advice(args, scenarios, model_digest)

    base_env = ConflictChoiceEnv(training)
    llm_env = ConflictChoiceEnv(training, advice, advice_alignment_bonus=0.25)
    check_env(base_env, warn=True)
    check_env(llm_env, warn=True)

    common = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=args.seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
    )
    ppo = PPO(env=Monitor(base_env), **common)
    ppo.learn(total_timesteps=args.timesteps, progress_bar=False)
    ppo.save(args.output / "ppo_base")
    llm_ppo = PPO(env=Monitor(llm_env), **common)
    llm_ppo.learn(total_timesteps=args.timesteps, progress_bar=False)
    llm_ppo.save(args.output / "ppo_llm_guided")

    probabilities = {
        "paper_derived_fused_logit": {s.scenario_id: paper_derived_logit_probability_left(s) for s in testing},
        "ppo": deterministic_probabilities(ppo, testing, None),
        "local_llm": {s.scenario_id: float(advice[s.scenario_id].recommended_exit == LEFT) for s in testing},
        "llm_guided_ppo": deterministic_probabilities(llm_ppo, testing, advice),
    }
    rows, summary = evaluate_probability_policies(testing, probabilities)
    write_csv(rows, args.output / "test_predictions.csv")
    write_csv(summary, args.output / "summary.csv")
    write_csv(
        [
            {
                "scenario_id": s.scenario_id,
                "recommended_exit": "left" if advice[s.scenario_id].recommended_exit == LEFT else "right",
                "confidence": advice[s.scenario_id].confidence,
                "reason_codes": "|".join(advice[s.scenario_id].reason_codes),
            }
            for s in scenarios
        ],
        args.output / "llm_advice.csv",
    )
    metadata = {
        "seed": args.seed,
        "model": args.model,
        "model_digest": model_digest,
        "training_scenarios": len(training),
        "test_scenarios": len(testing),
        "timesteps_per_ppo": args.timesteps,
        "observation": "14 normalized features; final 3 are cached LLM advice for llm_plus_ppo only",
        "reward": "1 - 4*smoke - 0.08*distance - 0.025*crowd",
        "llm_guidance_training": (
            "LLM advice is appended to the observation and a 0.25*confidence agreement bonus is used during training. "
            "Reported evaluation reward excludes this shaping bonus."
        ),
        "paper_logit_warning": (
            "Transparent fusion of coefficients from Lu et al. 2025 and 2026; not a single published fitted model. "
            "No signage coefficient is included because Tong et al. exact estimates are unavailable."
        ),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 烟雾—群众—标志—引导员冲突实验：LLM+RL 最小成本验证",
        "",
        f"本次使用 {len(training)} 个训练场景和 {len(testing)} 个锁定测试场景。每个唯一场景只调用一次本地 `{args.model}`，随后从缓存读取。纯PPO与LLM指导PPO各训练 {args.timesteps} 步。",
        "",
        "| 策略 | 平均安全奖励 | 最优出口率 | 选择烟雾出口率 | 平均距离 | 跟随标志 | 跟随群众 | 跟随引导员 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['policy']} | {float(row['mean_reward']):.4f} | {float(row['optimal_choice_rate']):.2%} | "
            f"{float(row['smoke_choice_rate']):.2%} | {float(row['mean_distance']):.2f} m | "
            f"{float(row['follow_sign_rate']):.2%} | {float(row['follow_crowd_rate']):.2%} | "
            f"{float(row['follow_warden_rate']):.2%} |"
        )
    report.extend(
        [
            "",
            "## 实现说明",
            "",
            "- `paper_derived_fused_logit`透明组合两篇已下载论文的公开系数；它不是任何一篇论文直接发表的联合模型。由于童云鹤论文的完整回归表尚未取得，该基线不含标志系数。",
            "- `ppo`只读取环境数值状态，训练奖励为 `1 - 4×烟雾 - 0.08×距离 - 0.025×人群数量`。",
            "- `local_llm`直接使用缓存的高层出口建议。",
            "- `llm_guided_ppo`将LLM建议和置信度加入观测，并在训练期给予 `0.25×置信度` 的一致性奖励；表中最终安全奖励不包含这项塑形奖励。",
            "",
            "## 解释边界",
            "",
            "这是单次关键出口选择的最小可行实验，不包含连续行走、动态烟气传播或多智能体物理交互。锁定测试集只有12个场景且当前只有一个训练随机种子，因此结果用于验证代码链路和提出下一轮假设，不能作为论文最终统计结论。",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

