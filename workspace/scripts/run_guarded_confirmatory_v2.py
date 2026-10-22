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

from fire_evac.advice_protocol_v2 import (
    PROMPT_VERSION,
    advice_prompt_v2,
    expected_semantic_target,
    safety_validate_advice,
)
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper, NOISE_LEVELS
from fire_evac.dynamic_conflicts import (
    DynamicConflictScenario,
    GymDynamicConflictEnv,
    conflict_scenario_records,
    make_domain_randomized_trust_bank,
)
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice, NORTH, parse_dynamic_advice
from run_confirmatory_joint_control import (
    SCHEMA,
    behavior_factory,
    parse_seeds,
    rollout,
    summarize_by_seed,
    write_csv,
)


ARMS = ("pure_ppo", "rule_advice_ppo", "guarded_llm_ppo", "raw_llm_ppo")
CONFIRMATORY_ARMS = ("pure_ppo", "rule_advice_ppo", "guarded_llm_ppo")
ARM_MODES = {
    "pure_ppo": "none",
    "rule_advice_ppo": "rule",
    "guarded_llm_ppo": "llm",
    "raw_llm_ppo": "llm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded LLM advice confirmatory study v2.")
    parser.add_argument("--seeds", default="601,602,603,604,605")
    parser.add_argument("--train-scenarios", type=int, default=120)
    parser.add_argument("--final-scenarios", type=int, default=60)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-repetitions", type=int, default=5)
    parser.add_argument("--scenario-seed", type=int, default=20261211)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "guarded_confirmatory_v2"
    )
    return parser.parse_args()


def api_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def collect_advice(
    scenarios: tuple[DynamicConflictScenario, ...], args: argparse.Namespace
) -> tuple[dict[str, DynamicLLMAdvice], dict[str, DynamicLLMAdvice], str, list[dict[str, object]]]:
    tags = api_json(f"{args.base_url}/api/tags", timeout=10)
    matches = [x for x in tags.get("models", []) if x.get("name") == args.model]
    if not matches:
        raise RuntimeError(f"missing Ollama model {args.model}")
    digest = str(matches[0].get("digest", ""))
    cache = args.output / "llm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    raw_advice: dict[str, DynamicLLMAdvice] = {}
    guarded_advice: dict[str, DynamicLLMAdvice] = {}
    audit = []
    for index, scenario in enumerate(scenarios, 1):
        prompt = advice_prompt_v2(scenario)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        path = cache / f"{scenario.scenario_id}.json"
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if stored and stored.get("model_digest") == digest and stored.get("prompt_hash") == prompt_hash:
            parsed_payload = stored["parsed"]
        else:
            response = api_json(
                f"{args.base_url}/api/chat",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": SCHEMA,
                    "options": {"temperature": 0, "seed": args.scenario_seed, "num_predict": 96},
                    "think": False,
                    "stream": False,
                },
                args.timeout,
            )
            parsed_payload = json.loads(response["message"]["content"])
            path.write_text(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "model": args.model,
                        "model_digest": digest,
                        "prompt_version": PROMPT_VERSION,
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "parsed": parsed_payload,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raw = parse_dynamic_advice(parsed_payload)
        guarded, corrected = safety_validate_advice(scenario, raw)
        raw_advice[scenario.scenario_id] = raw
        guarded_advice[scenario.scenario_id] = guarded
        audit.append(
            {
                "scenario_id": scenario.scenario_id,
                "conflict_type": scenario.conflict_type,
                "raw_target": "north" if raw.target_branch == NORTH else "south",
                "expected_target": "north" if expected_semantic_target(scenario) == NORTH else "south",
                "guarded_target": "north" if guarded.target_branch == NORTH else "south",
                "raw_confidence": raw.confidence,
                "guarded_confidence": guarded.confidence,
                "corrected": int(corrected),
                "reason_codes": "|".join(raw.reason_codes),
            }
        )
        print(f"LLM advice {index}/{len(scenarios)} {scenario.scenario_id}", flush=True)
    return raw_advice, guarded_advice, digest, audit


def make_env(
    behavior_name: str,
    artifact: Path,
    scenarios: tuple[DynamicConflictScenario, ...],
    arm: str,
    raw_advice: dict[str, DynamicLLMAdvice],
    guarded_advice: dict[str, DynamicLLMAdvice],
    noise_name: str = "none",
):
    advice = guarded_advice if arm == "guarded_llm_ppo" else raw_advice
    return ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(behavior_factory(behavior_name, artifact)(), scenarios),
        advice_mode=ARM_MODES[arm],
        advice_by_scenario=advice,
        noise=NOISE_LEVELS[noise_name],
    )


def paired_effects(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ("incapacitation_rate_mean", "mean_cumulative_fed_mean")
    comparisons = (
        ("rule_advice_ppo", "pure_ppo", "confirmatory"),
        ("guarded_llm_ppo", "pure_ppo", "confirmatory"),
        ("guarded_llm_ppo", "rule_advice_ppo", "confirmatory"),
        ("raw_llm_ppo", "pure_ppo", "exploratory"),
        ("guarded_llm_ppo", "raw_llm_ppo", "exploratory"),
    )
    index = {
        (r["arm"], int(r["training_seed"]), r["test_behavior"], r["noise_level"], r["conflict_type"]): r
        for r in seed_rows
    }
    output = []
    factors = sorted({(r["test_behavior"], r["noise_level"], r["conflict_type"]) for r in seed_rows})
    for candidate, baseline, status in comparisons:
        for behavior, noise, conflict in factors:
            for metric in metrics:
                differences = []
                for seed in sorted({int(r["training_seed"]) for r in seed_rows}):
                    a = index.get((candidate, seed, behavior, noise, conflict))
                    b = index.get((baseline, seed, behavior, noise, conflict))
                    if a is not None and b is not None:
                        differences.append(float(a[metric]) - float(b[metric]))
                if not differences:
                    continue
                values = np.asarray(differences)
                rng = np.random.default_rng(712_011 + len(output))
                bootstrap = np.asarray(
                    [rng.choice(values, size=len(values), replace=True).mean() for _ in range(10_000)]
                )
                output.append(
                    {
                        "analysis_status": status,
                        "candidate": candidate,
                        "baseline": baseline,
                        "test_behavior": behavior,
                        "noise_level": noise,
                        "conflict_type": conflict,
                        "metric": metric.removesuffix("_mean"),
                        "training_seeds": len(values),
                        "mean_seed_difference": float(values.mean()),
                        "seed_sd": float(values.std(ddof=1)),
                        "seed_bootstrap_ci_2_5": float(np.quantile(bootstrap, 0.025)),
                        "seed_bootstrap_ci_97_5": float(np.quantile(bootstrap, 0.975)),
                    }
                )
    return output


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"; model_dir.mkdir(exist_ok=True)
    train = make_domain_randomized_trust_bank(args.train_scenarios, args.scenario_seed, "V2TR")
    final = make_domain_randomized_trust_bank(args.final_scenarios, args.scenario_seed + 1, "V2FE")
    (args.output / "train_scenarios.json").write_text(json.dumps(conflict_scenario_records(train), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "final_scenarios.json").write_text(json.dumps(conflict_scenario_records(final), ensure_ascii=False, indent=2), encoding="utf-8")
    raw, guarded, digest, audit = collect_advice(train + final, args)
    train_ids = {s.scenario_id for s in train}
    for row in audit:
        row["split"] = "train" if row["scenario_id"] in train_ids else "final"
    write_csv(audit, args.output / "advice_guard_audit.csv")
    for arm in ARMS:
        env = make_env("calibrated_logit", args.calibration_artifact, train, arm, raw, guarded)
        check_env(env, warn=True); env.close()
    common = dict(
        policy="MlpPolicy", learning_rate=3e-4, n_steps=256, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
        device="cpu", verbose=0, policy_kwargs={"net_arch": [128, 128]},
    )
    models = {}; training_rows = []
    for seed in seeds:
        for arm in ARMS:
            path = model_dir / f"seed_{seed}_{arm}.zip"
            if args.reuse_models and path.exists():
                model = PPO.load(path, device="cpu")
            else:
                print(f"Training seed={seed}, arm={arm}", flush=True)
                def factory(arm_name=arm):
                    return make_env("calibrated_logit", args.calibration_artifact, train, arm_name, raw, guarded)
                vec = make_vec_env(factory, n_envs=args.n_envs, seed=seed)
                model = PPO(env=vec, seed=seed, **common)
                model.learn(total_timesteps=args.timesteps, progress_bar=False)
                model.save(path.with_suffix("")); vec.close()
            models[(seed, arm)] = model
            training_rows.append({"training_seed":seed,"arm":arm,"requested_timesteps":args.timesteps,"actual_timesteps":int(model.num_timesteps),"model_path":str(path.resolve())})
    write_csv(training_rows, args.output / "training_summary.csv")
    factors = [("calibrated_logit", "none")]
    if not args.skip_robustness:
        factors += [
            (b, n) for b in ("calibrated_logit","mixed_logit","rule","episode_ensemble")
            for n in NOISE_LEVELS if (b, n) != ("calibrated_logit", "none")
        ]
    result_path = args.output / "final_episode_results.csv"
    rows = []
    if args.reuse_models and result_path.exists():
        with result_path.open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
        print(f"Loaded {len(rows)} checkpoint rows", flush=True)
    expected = len(final) * args.eval_repetitions * len(seeds) * len(ARMS)
    for behavior, noise in factors:
        done = sum(r["test_behavior"] == behavior and r["noise_level"] == noise for r in rows)
        if done == expected:
            print(f"Skipping completed {behavior}/{noise}", flush=True); continue
        if done:
            rows = [r for r in rows if not (r["test_behavior"] == behavior and r["noise_level"] == noise)]
        print(f"Evaluating {behavior}/{noise}", flush=True)
        envs = {a:make_env(behavior,args.calibration_artifact,final,a,raw,guarded,noise) for a in ARMS}
        for si, scenario in enumerate(final):
            for rep in range(args.eval_repetitions):
                population_seed = 900_000 + si * args.eval_repetitions + rep
                for seed in seeds:
                    for arm in ARMS:
                        rows.append({"arm":arm,"training_seed":seed,"test_behavior":behavior,"noise_level":noise,**rollout(envs[arm],models[(seed,arm)],scenario,population_seed)})
        for env in envs.values(): env.close()
        write_csv(rows, result_path); print(f"Checkpointed {behavior}/{noise}", flush=True)
    per_seed = summarize_by_seed(rows)
    per_seed.extend(summarize_by_seed([{**r,"conflict_type":"all"} for r in rows]))
    write_csv(per_seed, args.output / "final_per_seed_summary.csv")
    write_csv(paired_effects(per_seed), args.output / "paired_training_seed_effects.csv")
    metadata = {
        "status":"completed","prompt_version":PROMPT_VERSION,"model":args.model,
        "model_digest":digest,"training_seeds":seeds,"arms":ARMS,
        "confirmatory_arms":CONFIRMATORY_ARMS,"raw_llm_analysis_status":"exploratory",
        "train_scenario_seed":args.scenario_seed,"final_scenario_seed":args.scenario_seed+1,
        "timesteps_per_arm_seed":args.timesteps,"finished_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
