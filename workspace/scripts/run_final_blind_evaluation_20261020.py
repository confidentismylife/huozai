from __future__ import annotations

"""Frozen, paired evaluation of advice inputs for one shared PPO controller.

The script deliberately separates ``prepare`` from ``run``.  ``prepare`` writes
the protocol, hashes and complete randomized schedule without loading the 48
held-out scenario records.  ``run`` verifies those frozen files, loads the
scenario records once, freezes all LLM responses, and only then executes any
evacuation outcome rollout.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.advice_protocol_v2 import PROMPT_VERSION, advice_prompt_v2
from fire_evac.confirmatory import (
    NOISE_LEVELS,
    ConfirmatoryJointControlWrapper,
    decode_absolute_action,
    frozen_rule_advice,
)
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice, NORTH, parse_dynamic_advice
from fire_evac.behavior import CalibratedLogitBehavior, MixedLogitBehavior


OUT = ROOT / "outputs" / "final_blind_evaluation_20261020"
REPORT = ROOT / "reports" / "LLM_PPO最终盲测报告_2026-10-20.md"
SOURCE_MANIFEST = ROOT / "outputs" / "llm_semantic_channel_diagnostic_20261005" / "manifest.json"
BEHAVIOR_ARTIFACT = ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json"
MODEL_DIR = ROOT / "outputs" / "guarded_confirmatory_v2" / "models"
MODEL_SEEDS = tuple(range(601, 606))
METHODS = ("pure_ppo", "rule_advice_ppo", "llm_advice_ppo")
BEHAVIORS = ("calibrated_logit", "mixed_logit")
NOISES = ("none", "medium")
SCENARIO_IDS = tuple(f"SCF{i:04d}" for i in range(1, 49))
SCHEDULE_SEED = 2026102001
BOOTSTRAP_SEED = 2026102002
PERMUTATION_SEED = 2026102003
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SAMPLES = 20_000
MODEL_NAME = "qwen3:4b"
MODEL_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
PRIMARY_BEHAVIOR = "calibrated_logit"
PRIMARY_NOISE = "none"
PRIMARY_METRICS = ("incapacitation_rate", "mean_cumulative_fed")
SECONDARY_METRICS = ("evacuation_time_seconds", "max_congestion", "cumulative_congestion")
COMPARISONS = (
    ("llm_advice_ppo", "pure_ppo"),
    ("llm_advice_ppo", "rule_advice_ppo"),
    ("rule_advice_ppo", "pure_ppo"),
)

SCHEMA = {
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


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def model_path(seed: int) -> Path:
    return MODEL_DIR / f"seed_{seed}_raw_llm_ppo.zip"


def protected_paths() -> list[Path]:
    paths = [
        SOURCE_MANIFEST,
        BEHAVIOR_ARTIFACT,
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_guarded_confirmatory_v2.py",
        ROOT / "scripts" / "run_confirmatory_joint_control.py",
    ]
    paths.extend(sorted((SRC / "fire_evac").glob("*.py")))
    paths.extend(model_path(seed) for seed in MODEL_SEEDS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen input(s): " + ", ".join(missing))
    return paths


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def build_schedule() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_id in SCENARIO_IDS:
        scenario_number = int(scenario_id[-4:])
        population_seed = 2_086_102_000 + scenario_number
        for training_seed in MODEL_SEEDS:
            for behavior in BEHAVIORS:
                for noise in NOISES:
                    for method in METHODS:
                        rows.append(
                            {
                                "run_id": f"{scenario_id}_{training_seed}_{behavior}_{noise}_{method}",
                                "scenario_id": scenario_id,
                                "training_seed": training_seed,
                                "behavior_model": behavior,
                                "noise_level": noise,
                                "method": method,
                                "population_seed": population_seed,
                            }
                        )
    rng = np.random.default_rng(SCHEDULE_SEED)
    order = rng.permutation(len(rows))
    shuffled = []
    for execution_order, index in enumerate(order, 1):
        row = dict(rows[int(index)])
        row["execution_order"] = execution_order
        shuffled.append(row)
    assert len(shuffled) == 48 * 5 * 2 * 2 * 3 == 2_880
    assert len({row["run_id"] for row in shuffled}) == len(shuffled)
    return shuffled


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen directory: {OUT}")
    OUT.mkdir(parents=True)
    parents = {relative(path): sha256(path) for path in protected_paths()}
    schedule = build_schedule()
    protocol = {
        "protocol_version": "final_blind_shared_controller_v1",
        "frozen_at_utc": stamp(),
        "research_question": "在同一PPO控制器、同一观测和安全约束下，规则或LLM建议是否改善疏散结果？",
        "scenario_source": relative(SOURCE_MANIFEST),
        "scenario_source_sha256": parents[relative(SOURCE_MANIFEST)],
        "scenario_ids": list(SCENARIO_IDS),
        "scenario_count": 48,
        "scenario_history": {
            "original_role": "2026-10-05 generated, disjoint, unscored development holdout",
            "current_role": "frozen final evaluation bank",
            "outcomes_previously_generated": False,
            "limitation": "parameters were serialized previously; this is outcome-blind rather than cryptographically hidden",
        },
        "methods": list(METHODS),
        "training_seeds": list(MODEL_SEEDS),
        "behavior_models": list(BEHAVIORS),
        "noise_levels": list(NOISES),
        "episodes": len(schedule),
        "same_population_seed_within_scenario": True,
        "shared_controller_weights_across_methods": True,
        "new_ppo_training": 0,
        "local_llm": {
            "model": MODEL_NAME,
            "digest": MODEL_DIGEST,
            "prompt_version": PROMPT_VERSION,
            "temperature": 0,
            "seed": 2026100551,
            "num_predict": 96,
            "think": False,
            "requests_planned": 48,
            "paid_api_requests": 0,
            "all_advice_frozen_before_outcome_rollouts": True,
        },
        "deviations_forbidden_after_unblinding": [
            "reward change",
            "threshold change",
            "prompt change",
            "method selection",
            "model selection",
            "scenario exclusion except documented execution corruption",
        ],
        "existing_hard_gate_qualification_passed": False,
        "real_building_claim_allowed": False,
    }
    methods = {
        "controller": {
            "per_training_seed": {str(seed): relative(model_path(seed)) for seed in MODEL_SEEDS},
            "observation_dimensions": 34,
            "actions": {
                "0-3": "north exit x 25/50/75/100% release",
                "4-7": "south exit x 25/50/75/100% release",
                "8": "wait / zero room release",
            },
            "physical_and_forecast_fields_identical": True,
            "sign_hold_and_capacity_constraints_identical": True,
        },
        "pure_ppo": {"advice_channel": "five zeros"},
        "rule_advice_ppo": {"advice_channel": "frozen_rule_advice(scenario)"},
        "llm_advice_ppo": {
            "advice_channel": "qwen3:4b response to advice_prompt_v2",
            "safety_rewrite": "none",
            "reason": "preserve identifiable LLM contribution; all methods still share physical constraints",
        },
    }
    analysis = {
        "analysis_version": "crossed_paired_v1",
        "unit_of_generalization": "scenario; PPO training seed is a crossed replication block",
        "primary_condition": {"behavior_model": PRIMARY_BEHAVIOR, "noise_level": PRIMARY_NOISE},
        "primary_outcomes": list(PRIMARY_METRICS),
        "secondary_outcomes": list(SECONDARY_METRICS),
        "primary_comparisons": [
            ["llm_advice_ppo", "pure_ppo"],
            ["llm_advice_ppo", "rule_advice_ppo"],
        ],
        "secondary_comparison": ["rule_advice_ppo", "pure_ppo"],
        "effect_direction": "candidate minus baseline; negative is better for every listed outcome",
        "interval": {
            "method": "two-way crossed bootstrap resampling scenarios and training seeds",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": 0.95,
        },
        "primary_test": {
            "method": "two-sided scenario-level sign-flip randomization test after averaging training seeds",
            "samples": PERMUTATION_SAMPLES,
            "seed": PERMUTATION_SEED,
            "multiplicity": "Holm correction across 2 primary comparisons x 2 primary outcomes",
            "alpha": 0.05,
        },
        "effect_sizes": ["raw paired mean difference", "relative mean change", "paired Cohen dz"],
        "assumption_checks": ["Q-Q plot", "Shapiro-Wilk on 48 scenario-level paired differences"],
        "robustness": "all other behavior/noise cells are preplanned descriptive robustness strata",
        "missing_data": "no silent deletion; incomplete cells stop analysis",
        "equivalence_claim": False,
    }
    write_json(OUT / "protocol_frozen.json", protocol)
    write_json(OUT / "methods_frozen.json", methods)
    write_json(
        OUT / "model_hashes.json",
        {str(seed): {"path": relative(model_path(seed)), "sha256": sha256(model_path(seed))} for seed in MODEL_SEEDS},
    )
    write_csv(OUT / "run_schedule_frozen.csv", schedule)
    write_json(OUT / "analysis_plan_frozen.json", analysis)
    write_json(OUT / "protected_parent_hashes.json", parents)
    freeze_files = [
        OUT / "protocol_frozen.json",
        OUT / "methods_frozen.json",
        OUT / "model_hashes.json",
        OUT / "run_schedule_frozen.csv",
        OUT / "analysis_plan_frozen.json",
        OUT / "protected_parent_hashes.json",
    ]
    write_json(OUT / "freeze_manifest.json", {relative(path): sha256(path) for path in freeze_files})
    write_json(
        OUT / "run_status.json",
        {
            "state": "frozen_ready",
            "scenario_contents_loaded_by_this_runner": False,
            "outcome_rollouts_completed": 0,
            "llm_requests_completed": 0,
            "updated_at_utc": stamp(),
        },
    )
    print(f"Frozen protocol and 2,880-row schedule at {OUT}")


def verify_frozen_inputs() -> None:
    freeze_manifest = read_json(OUT / "freeze_manifest.json")
    for name, expected in freeze_manifest.items():
        path = ROOT / str(name)
        if sha256(path) != expected:
            raise RuntimeError(f"frozen protocol file changed: {name}")
    parents = read_json(OUT / "protected_parent_hashes.json")
    for name, expected in parents.items():
        path = ROOT / str(name)
        if sha256(path) != expected:
            raise RuntimeError(f"protected input changed: {name}")


def behavior_factory(name: str):
    if name == "calibrated_logit":
        return CalibratedLogitBehavior(BEHAVIOR_ARTIFACT)
    if name == "mixed_logit":
        return MixedLogitBehavior()
    raise ValueError(name)


def make_env(
    scenarios: tuple[DynamicConflictScenario, ...],
    behavior: str,
    noise: str,
    method: str,
    advice: dict[str, DynamicLLMAdvice],
):
    mode = {"pure_ppo": "none", "rule_advice_ppo": "rule", "llm_advice_ppo": "llm"}[method]
    return ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(behavior_factory(behavior), scenarios),
        advice_mode=mode,
        advice_by_scenario=advice,
        noise=NOISE_LEVELS[noise],
    )


def api_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def collect_llm_advice(
    scenarios: tuple[DynamicConflictScenario, ...], base_url: str
) -> dict[str, DynamicLLMAdvice]:
    tags = api_json(f"{base_url}/api/tags", timeout=10)
    matches = [item for item in tags.get("models", []) if item.get("name") == MODEL_NAME]
    if len(matches) != 1 or matches[0].get("digest") != MODEL_DIGEST:
        raise RuntimeError(f"exact frozen {MODEL_NAME} digest is not available at {base_url}")
    cache_dir = OUT / "llm_cache"
    cache_dir.mkdir(exist_ok=True)
    order = list(SCENARIO_IDS)
    np.random.default_rng(2026102004).shuffle(order)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    advice: dict[str, DynamicLLMAdvice] = {}
    audit: list[dict[str, object]] = []
    new_requests = 0
    for index, scenario_id in enumerate(order, 1):
        scenario = by_id[scenario_id]
        prompt = advice_prompt_v2(scenario)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache_dir / f"{scenario_id}.json"
        cached = read_json(path) if path.exists() else None
        if cached is not None:
            if cached["model_digest"] != MODEL_DIGEST or cached["prompt_hash"] != prompt_hash:
                raise RuntimeError(f"incompatible cached response: {path}")
            parsed = cached["parsed"]
        else:
            started = time.monotonic()
            response = api_json(
                f"{base_url}/api/chat",
                {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": SCHEMA,
                    "options": {"temperature": 0, "seed": 2026100551, "num_predict": 96},
                    "think": False,
                    "stream": False,
                },
                timeout=180,
            )
            parsed = json.loads(str(response["message"]["content"]))
            parsed_advice = parse_dynamic_advice(parsed)
            write_json(
                path,
                {
                    "scenario_id": scenario_id,
                    "model": MODEL_NAME,
                    "model_digest": MODEL_DIGEST,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "parsed": parsed,
                    "raw_response": response,
                    "latency_seconds": time.monotonic() - started,
                    "created_at_utc": stamp(),
                    "target_branch": parsed_advice.target_branch,
                },
            )
            new_requests += 1
        item = parse_dynamic_advice(parsed)
        advice[scenario_id] = item
        rule = frozen_rule_advice(scenario)
        audit.append(
            {
                "scenario_id": scenario_id,
                "llm_target": "north" if item.target_branch == NORTH else "south",
                "llm_confidence": item.confidence,
                "rule_target": "north" if rule.target_branch == NORTH else "south",
                "rule_confidence": rule.confidence,
                "targets_agree": int(item.target_branch == rule.target_branch),
                "prompt_sha256": prompt_hash,
                "cache_sha256": sha256(path),
            }
        )
        print(f"LLM advice frozen {index}/48: {scenario_id}", flush=True)
    write_csv(OUT / "llm_advice_audit.csv", sorted(audit, key=lambda row: str(row["scenario_id"])))
    payload = {
        "model": MODEL_NAME,
        "model_digest": MODEL_DIGEST,
        "prompt_version": PROMPT_VERSION,
        "advice": {
            key: {
                "target_branch": value.target_branch,
                "confidence": value.confidence,
                "reason_codes": list(value.reason_codes),
            }
            for key, value in sorted(advice.items())
        },
        "frozen_at_utc": stamp(),
        "new_requests_this_run": new_requests,
        "outcome_rollouts_completed_before_freeze": 0,
    }
    write_json(OUT / "llm_advice_frozen.json", payload)
    write_json(OUT / "llm_advice_freeze_hash.json", {"sha256": sha256(OUT / "llm_advice_frozen.json")})
    return advice


def load_scenarios_after_freeze() -> tuple[DynamicConflictScenario, ...]:
    source = read_json(SOURCE_MANIFEST)
    records = source["fresh"]
    scenarios = tuple(DynamicConflictScenario(**record) for record in records)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if set(by_id) != set(SCENARIO_IDS) or len(by_id) != 48:
        raise RuntimeError("held-out scenario identity/count mismatch")
    return tuple(by_id[scenario_id] for scenario_id in SCENARIO_IDS)


def load_models():
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    models = {}
    for seed in MODEL_SEEDS:
        model = PPO.load(model_path(seed), device="cpu")
        if model.action_space.n != 9 or model.observation_space.shape != (34,):
            raise RuntimeError(f"unexpected model interface for seed {seed}")
        model.policy.set_training_mode(False)
        models[seed] = model
    return models


def rollout(env, model, scenario: DynamicConflictScenario, population_seed: int, method: str) -> dict[str, object]:
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    terminated = truncated = False
    steps = 0
    total_reward = 0.0
    max_congestion = 0.0
    wait_steps = 0
    sign_hold_blocks = 0
    requested_actual_matches = 0
    llm_branch_steps = 0
    llm_override_steps = 0
    first_requested_action = -1
    first_actual_action = -1
    first_nonwait_branch = -1
    density_scale = float(env.unwrapped.domain_env.config.density_scale)
    dt_seconds = float(env.unwrapped.domain_env.config.dt_seconds)
    while not (terminated or truncated):
        action, _ = model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
        requested = int(np.asarray(action).item())
        if steps == 0:
            first_requested_action = requested
        branch, _ = decode_absolute_action(requested)
        if branch is None:
            wait_steps += 1
        else:
            if first_nonwait_branch < 0:
                first_nonwait_branch = branch
            if method == "llm_advice_ppo":
                llm_branch_steps += 1
                llm_override_steps += int(branch != env.current_advice.target_branch)
        observation, reward, terminated, truncated, info = env.step(requested)
        if steps == 0:
            first_actual_action = int(info["actual_joint_action"])
        total_reward += float(reward)
        requested_actual_matches += int(int(info["requested_joint_action"]) == int(info["actual_joint_action"]))
        sign_hold_blocks += int(bool(info["action_blocked_by_sign_hold"]))
        congestion = sum(max(0.0, float(count) / density_scale - 1.0) for count in info["zone_counts"])
        max_congestion = max(max_congestion, congestion)
        steps += 1
        if steps > int(env.unwrapped.domain_env.config.max_steps):
            raise RuntimeError("rollout exceeded environment limit")
    return {
        "steps": steps,
        "evacuation_time_seconds": steps * dt_seconds,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "remaining_rate": float(info["remaining"]) / float(env.unwrapped.domain_env.config.population),
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "max_congestion": max_congestion,
        "sign_switches": int(info["sign_switches"]),
        "total_reward": total_reward,
        "wait_fraction": wait_steps / steps,
        "sign_hold_block_rate": sign_hold_blocks / steps,
        "requested_actual_joint_match_rate": requested_actual_matches / steps,
        "first_requested_action": first_requested_action,
        "first_actual_action": first_actual_action,
        "first_nonwait_branch": first_nonwait_branch,
        "llm_requested_override_rate": (llm_override_steps / llm_branch_steps) if llm_branch_steps else math.nan,
        "truncated": int(truncated),
    }


def run_rollouts(scenarios: tuple[DynamicConflictScenario, ...], advice: dict[str, DynamicLLMAdvice]) -> pd.DataFrame:
    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    results_path = OUT / "episode_results.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        rows = existing.to_dict("records")
        done = set(existing["run_id"].astype(str))
    else:
        rows, done = [], set()
    models = load_models()
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    envs = {
        (behavior, noise, method): make_env(scenarios, behavior, noise, method, advice)
        for behavior in BEHAVIORS
        for noise in NOISES
        for method in METHODS
    }
    try:
        for position, scheduled in schedule.sort_values("execution_order").iterrows():
            run_id = str(scheduled["run_id"])
            if run_id in done:
                continue
            method = str(scheduled["method"])
            behavior = str(scheduled["behavior_model"])
            noise = str(scheduled["noise_level"])
            seed = int(scheduled["training_seed"])
            scenario = scenario_map[str(scheduled["scenario_id"])]
            result = rollout(
                envs[(behavior, noise, method)],
                models[seed],
                scenario,
                int(scheduled["population_seed"]),
                method,
            )
            rows.append({**scheduled.to_dict(), "conflict_type": scenario.conflict_type, **result})
            done.add(run_id)
            if len(rows) % 48 == 0:
                write_csv(results_path, rows)
                write_json(
                    OUT / "run_status.json",
                    {
                        "state": "running_outcomes",
                        "scenario_contents_loaded_by_this_runner": True,
                        "outcome_rollouts_completed": len(rows),
                        "llm_requests_completed": 48,
                        "updated_at_utc": stamp(),
                    },
                )
                print(f"Outcome rollouts {len(rows)}/2880", flush=True)
        write_csv(results_path, rows)
    finally:
        for env in envs.values():
            env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 2_880 or frame["run_id"].nunique() != 2_880:
        raise RuntimeError("evaluation matrix incomplete or duplicated")
    return frame


def two_way_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    if matrix.shape != (48, 5):
        raise ValueError(f"expected crossed 48x5 matrix, got {matrix.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenarios = rng.integers(0, 48, 48)
        seeds = rng.integers(0, 5, 5)
        estimates[index] = matrix[np.ix_(scenarios, seeds)].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def sign_flip_p(values: np.ndarray, seed: int) -> float:
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATION_SAMPLES):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        exceed += abs(float(np.mean(values * signs))) >= observed - 1e-15
    return (exceed + 1) / (PERMUTATION_SAMPLES + 1)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def paired_matrix(frame: pd.DataFrame, behavior: str, noise: str, candidate: str, baseline: str, metric: str):
    subset = frame[(frame.behavior_model == behavior) & (frame.noise_level == noise)]
    pivot = subset.pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    difference = (pivot[candidate] - pivot[baseline]).unstack("training_seed")
    difference = difference.reindex(index=SCENARIO_IDS, columns=MODEL_SEEDS)
    if difference.isna().any().any():
        raise RuntimeError("missing paired cells")
    return difference.to_numpy(float)


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if frame.isna().any().any():
        allowed = frame.columns.isin(["llm_requested_override_rate"])
        bad = frame.columns[frame.isna().any() & ~allowed]
        if len(bad):
            raise RuntimeError(f"unexpected missing values: {list(bad)}")
    conservation = frame.evacuation_rate + frame.incapacitation_rate + frame.remaining_rate
    if not np.allclose(conservation, 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("population conservation failed")
    grouped = (
        frame.groupby(["behavior_model", "noise_level", "method"], as_index=False)
        .agg(
            episodes=("run_id", "count"),
            scenarios=("scenario_id", "nunique"),
            training_seeds=("training_seed", "nunique"),
            incapacitation_rate_mean=("incapacitation_rate", "mean"),
            incapacitation_rate_sd=("incapacitation_rate", "std"),
            mean_cumulative_fed_mean=("mean_cumulative_fed", "mean"),
            mean_cumulative_fed_sd=("mean_cumulative_fed", "std"),
            evacuation_time_seconds_mean=("evacuation_time_seconds", "mean"),
            max_congestion_mean=("max_congestion", "mean"),
            cumulative_congestion_mean=("cumulative_congestion", "mean"),
            truncated_episodes=("truncated", "sum"),
        )
        .sort_values(["behavior_model", "noise_level", "method"])
    )
    effects: list[dict[str, object]] = []
    primary_indices: list[int] = []
    primary_p: list[float] = []
    metrics = PRIMARY_METRICS + SECONDARY_METRICS
    counter = 0
    for behavior in BEHAVIORS:
        for noise in NOISES:
            for candidate, baseline in COMPARISONS:
                for metric in metrics:
                    matrix = paired_matrix(frame, behavior, noise, candidate, baseline, metric)
                    scenario_values = matrix.mean(axis=1)
                    mean_difference = float(matrix.mean())
                    ci_low, ci_high = two_way_bootstrap(matrix, BOOTSTRAP_SEED + counter)
                    baseline_mean = float(
                        frame[
                            (frame.behavior_model == behavior)
                            & (frame.noise_level == noise)
                            & (frame.method == baseline)
                        ][metric].mean()
                    )
                    sd = float(scenario_values.std(ddof=1))
                    dz = mean_difference / sd if sd > 0 else (0.0 if mean_difference == 0 else math.copysign(math.inf, mean_difference))
                    shapiro_w, shapiro_p = stats.shapiro(scenario_values)
                    nonzero = scenario_values[np.abs(scenario_values) > 1e-15]
                    if len(nonzero):
                        wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
                        wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                    else:
                        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                    record = {
                        "behavior_model": behavior,
                        "noise_level": noise,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_difference": mean_difference,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "baseline_mean": baseline_mean,
                        "relative_change_percent": 100 * mean_difference / baseline_mean if baseline_mean else math.nan,
                        "paired_cohen_dz": dz,
                        "scenario_wins": int((scenario_values < -1e-15).sum()),
                        "scenario_ties": int((np.abs(scenario_values) <= 1e-15).sum()),
                        "scenario_harms": int((scenario_values > 1e-15).sum()),
                        "shapiro_w": float(shapiro_w),
                        "shapiro_p": float(shapiro_p),
                        "wilcoxon_statistic": wilcoxon_stat,
                        "wilcoxon_p": wilcoxon_p,
                        "sign_flip_p": math.nan,
                        "holm_adjusted_p": math.nan,
                        "analysis_role": "primary" if behavior == PRIMARY_BEHAVIOR and noise == PRIMARY_NOISE and metric in PRIMARY_METRICS and (candidate, baseline) in COMPARISONS[:2] else "preplanned_secondary_or_robustness",
                    }
                    if record["analysis_role"] == "primary":
                        p_value = sign_flip_p(scenario_values, PERMUTATION_SEED + len(primary_p))
                        record["sign_flip_p"] = p_value
                        primary_indices.append(len(effects))
                        primary_p.append(p_value)
                    effects.append(record)
                    counter += 1
    adjusted = holm(primary_p)
    for index, value in zip(primary_indices, adjusted):
        effects[index]["holm_adjusted_p"] = value
    effect_frame = pd.DataFrame(effects)

    advice = pd.read_csv(OUT / "llm_advice_audit.csv")
    llm_primary = frame[
        (frame.behavior_model == PRIMARY_BEHAVIOR)
        & (frame.noise_level == PRIMARY_NOISE)
        & (frame.method == "llm_advice_ppo")
    ].copy()
    rule_primary = frame[
        (frame.behavior_model == PRIMARY_BEHAVIOR)
        & (frame.noise_level == PRIMARY_NOISE)
        & (frame.method == "rule_advice_ppo")
    ][["scenario_id", "training_seed", "incapacitation_rate", "mean_cumulative_fed"]].rename(
        columns={
            "incapacitation_rate": "rule_incapacitation_rate",
            "mean_cumulative_fed": "rule_mean_cumulative_fed",
        }
    )
    diagnostic = llm_primary.merge(rule_primary, on=["scenario_id", "training_seed"], validate="one_to_one").merge(
        advice[["scenario_id", "targets_agree"]], on="scenario_id", validate="many_to_one"
    )
    diagnostic["llm_minus_rule_incapacitation"] = diagnostic.incapacitation_rate - diagnostic.rule_incapacitation_rate
    diagnostic["llm_minus_rule_fed"] = diagnostic.mean_cumulative_fed - diagnostic.rule_mean_cumulative_fed
    diagnostic["harmful"] = (diagnostic.llm_minus_rule_incapacitation > 1e-15) | (
        (diagnostic.llm_minus_rule_incapacitation.abs() <= 1e-15) & (diagnostic.llm_minus_rule_fed > 1e-15)
    )
    diagnostic["beneficial"] = (diagnostic.llm_minus_rule_incapacitation < -1e-15) | (
        (diagnostic.llm_minus_rule_incapacitation.abs() <= 1e-15) & (diagnostic.llm_minus_rule_fed < -1e-15)
    )
    diagnostic["first_branch_overrode_llm"] = diagnostic.apply(
        lambda row: int(
            row.first_nonwait_branch >= 0
            and row.first_nonwait_branch
            != (NORTH if advice.set_index("scenario_id").loc[row.scenario_id, "llm_target"] == "north" else 1)
        ),
        axis=1,
    )
    write_csv(OUT / "llm_override_diagnostics.csv", diagnostic.to_dict("records"))
    special = {
        "llm_rule_target_agreement_scenarios": int(advice.targets_agree.sum()),
        "llm_rule_target_disagreement_scenarios": int((1 - advice.targets_agree).sum()),
        "primary_scenario_seed_units": len(diagnostic),
        "harmful_units_lexicographic": int(diagnostic.harmful.sum()),
        "beneficial_units_lexicographic": int(diagnostic.beneficial.sum()),
        "neutral_units_lexicographic": int((~diagnostic.harmful & ~diagnostic.beneficial).sum()),
        "mean_llm_requested_override_rate": float(diagnostic.llm_requested_override_rate.mean()),
        "first_branch_override_rate": float(diagnostic.first_branch_overrode_llm.mean()),
        "first_branch_override_rate_in_harmful_units": float(diagnostic.loc[diagnostic.harmful, "first_branch_overrode_llm"].mean()) if diagnostic.harmful.any() else None,
        "first_branch_override_rate_in_beneficial_units": float(diagnostic.loc[diagnostic.beneficial, "first_branch_overrode_llm"].mean()) if diagnostic.beneficial.any() else None,
        "interpretation": "These are spontaneous PPO response diagnostics, not a separately trained trust-gate interception rate.",
    }
    return grouped, effect_frame, special


def make_figure(effect_frame: pd.DataFrame) -> None:
    subset = effect_frame[
        effect_frame.metric.isin(PRIMARY_METRICS)
        & effect_frame.apply(lambda row: (row.candidate, row.baseline) in COMPARISONS[:2], axis=1)
    ].copy()
    labels = {
        ("calibrated_logit", "none"): "校准Logit / 无噪声",
        ("calibrated_logit", "medium"): "校准Logit / 中噪声",
        ("mixed_logit", "none"): "Mixed Logit / 无噪声",
        ("mixed_logit", "medium"): "Mixed Logit / 中噪声",
    }
    colors = {("llm_advice_ppo", "pure_ppo"): "#0072B2", ("llm_advice_ppo", "rule_advice_ppo"): "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for axis, metric, title in zip(axes, PRIMARY_METRICS, ("失能率差（百分点）", "人均累计FED差")):
        current = subset[subset.metric == metric]
        y = 0
        ticks, ticklabels = [], []
        for condition in labels:
            for comparison in COMPARISONS[:2]:
                row = current[
                    (current.behavior_model == condition[0])
                    & (current.noise_level == condition[1])
                    & (current.candidate == comparison[0])
                    & (current.baseline == comparison[1])
                ].iloc[0]
                scale = 100 if metric == "incapacitation_rate" else 1
                mean = row.mean_difference * scale
                low = row.ci95_low * scale
                high = row.ci95_high * scale
                axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=colors[comparison], capsize=3)
                ticks.append(y)
                comp_label = "LLM−纯PPO" if comparison[1] == "pure_ppo" else "LLM−规则"
                ticklabels.append(f"{labels[condition]} | {comp_label}")
                y += 1
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks(ticks)
        axis.set_yticklabels(ticklabels, fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("负值表示LLM方案更好；误差线为95%交叉bootstrap区间")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "primary_effects.png", dpi=220)
    plt.close(fig)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(descriptive: pd.DataFrame, effects: pd.DataFrame, special: dict[str, object]) -> None:
    primary = effects[effects.analysis_role == "primary"].copy()
    lines = [
        "# LLM＋PPO火灾疏散最终盲测报告",
        "",
        f"日期：2026-10-20。状态：48个冻结留出场景、5个PPO训练种子、3种建议方式、2种人群行为和2档观测噪声全部运行完成，共{len(pd.read_csv(OUT / 'episode_results.csv')):,}个完整回合。",
        "",
        "## 1. 这次到底比较了什么",
        "",
        "三种方法逐个训练种子使用完全相同的`raw_llm PPO`权重。纯PPO把5维建议通道置零；规则组写入冻结规则建议；LLM组写入本地Qwen3 4B在任何疏散结局运行前一次性冻结的场景建议。三组共享34维输入结构、9动作、人口种子、标志保持、出口容量和封闭约束。因此主要差异可以归因于建议输入，而不是三套不同PPO权重。",
        "",
        "48场景是2026-10-05生成、与历史场景签名分离且从未评分的SCF银行。当时协议把它叫作开发留出并因语义前置检查失败而没有运行。本轮在读取结局前将其重新登记为最终评价银行。它属于结局盲测，但场景参数曾被序列化，不能描述成密码学意义上从未可见。",
        "",
        "主要条件为校准Logit人群、无观测噪声。主要终点为失能率和人均累计FED；主要比较为LLM−纯PPO和LLM−规则。负值代表LLM更好。置信区间同时重采样48场景与5个训练种子；显著性辅助检验先在每场景平均5个种子，再做双侧符号翻转，并对4项主要检验用Holm法校正。",
        "",
        "## 2. 主要结果",
        "",
        "|比较|指标|均值差|95% CI|相对变化|Cohen dz|场景 改善/持平/变差|Holm校正p|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        scale = 100 if row.metric == "incapacitation_rate" else 1
        unit = "百分点" if row.metric == "incapacitation_rate" else ""
        comparison = "LLM−纯PPO" if row.baseline == "pure_ppo" else "LLM−规则"
        metric = "失能率" if row.metric == "incapacitation_rate" else "人均累计FED"
        lines.append(
            f"|{comparison}|{metric}|{fmt(row.mean_difference * scale)}{unit}|[{fmt(row.ci95_low * scale)}, {fmt(row.ci95_high * scale)}]|{fmt(row.relative_change_percent, 2)}%|{fmt(row.paired_cohen_dz, 3)}|{int(row.scenario_wins)}/{int(row.scenario_ties)}/{int(row.scenario_harms)}|{fmt(row.holm_adjusted_p, 4)}|"
        )
    lines.extend(
        [
            "",
            "判定以效应量和区间为主。区间跨0表示当前证据不能排除改善或变差；不把非显著结果称为等效。失能率存在大量场景内并列和地板效应，因此参数正态检验只作诊断，主区间使用预先规定的交叉bootstrap。",
            "",
            "## 3. 四种行为与噪声条件",
            "",
            "|人群|噪声|方法|回合|失能率%|人均累计FED|疏散时间s|最大拥堵|",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in descriptive.iterrows():
        lines.append(
            f"|{row.behavior_model}|{row.noise_level}|{row.method}|{int(row.episodes)}|{fmt(row.incapacitation_rate_mean * 100)}|{fmt(row.mean_cumulative_fed_mean)}|{fmt(row.evacuation_time_seconds_mean, 2)}|{fmt(row.max_congestion_mean, 3)}|"
        )
    lines.extend(
        [
            "",
            "各鲁棒性分层的完整配对差、区间、相对变化、正态性检查和Wilcoxon敏感性结果保存在`paired_effects.csv`。这些分层在解封前已列入方案，但不从其中挑一个最有利子组替代主要结论。",
            "",
            "## 4. LLM建议与PPO覆盖诊断",
            "",
            f"LLM与规则目标在{special['llm_rule_target_agreement_scenarios']}/48个场景一致，在{special['llm_rule_target_disagreement_scenarios']}/48个场景不同。主要条件下共有{special['primary_scenario_seed_units']}个场景×种子配对单元；按失能率优先、FED次优的冻结字典序，LLM输入相对规则输入有害{special['harmful_units_lexicographic']}个、有益{special['beneficial_units_lexicographic']}个、中性{special['neutral_units_lexicographic']}个。",
            "",
            f"LLM组在非等待步骤中请求与LLM目标相反方向的平均比例为{100 * special['mean_llm_requested_override_rate']:.2f}%，首次非等待方向覆盖LLM的比例为{100 * special['first_branch_override_rate']:.2f}%。这些是PPO自然响应，不是一个单独训练并通过资格认证的信任门控，因此不能把它直接叫作‘有害建议拦截率’。有害与有益单元中的首次覆盖比例分别记录在`llm_override_diagnostics.csv`。",
            "",
            "## 5. 完整性与解释边界",
            "",
            "所有2,880个计划单元均完成，没有静默删除；每个单元满足撤离率＋失能率＋剩余率=1。五套模型均为34维输入、9动作，模型和模拟器哈希在解封前冻结且运行后复核。LLM建议在第一个结局回合前全部缓存，温度0、固定种子、关闭思考输出，付费API调用为0。",
            "",
            "本实验检验合成五分区、两出口环境中的算法内部增量价值。既有硬护栏95%/10%/5%/95%资格没有通过，本轮也没有重新认证；SCF场景不是FDS真实建筑，也没有真人动态轨迹校准。因此即使某项平均值改善，也不能写成真实火灾伤亡已经降低或系统可部署。",
            "",
            "![主要效应图](../outputs/final_blind_evaluation_20261020/primary_effects.png)",
            "",
            "## 6. 可复核文件",
            "",
            "- 冻结协议：`outputs/final_blind_evaluation_20261020/protocol_frozen.json`",
            "- 冻结运行表：`outputs/final_blind_evaluation_20261020/run_schedule_frozen.csv`",
            "- 逐回合结果：`outputs/final_blind_evaluation_20261020/episode_results.csv`",
            "- 描述统计：`outputs/final_blind_evaluation_20261020/descriptive_summary.csv`",
            "- 配对效应：`outputs/final_blind_evaluation_20261020/paired_effects.csv`",
            "- LLM覆盖诊断：`outputs/final_blind_evaluation_20261020/llm_override_diagnostics.csv`",
            "- 独立核查摘要：`outputs/final_blind_evaluation_20261020/verification.json`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(frame: pd.DataFrame) -> None:
    descriptive, effects, special = analyze(frame)
    write_csv(OUT / "descriptive_summary.csv", descriptive.to_dict("records"))
    write_csv(OUT / "paired_effects.csv", effects.to_dict("records"))
    write_json(OUT / "llm_special_diagnostics.json", special)
    make_figure(effects)
    write_report(descriptive, effects, special)
    verify_frozen_inputs()
    verification = {
        "status": "passed",
        "episodes": len(frame),
        "unique_run_ids": int(frame.run_id.nunique()),
        "scenarios": int(frame.scenario_id.nunique()),
        "training_seeds": sorted(map(int, frame.training_seed.unique())),
        "methods": sorted(frame.method.unique().tolist()),
        "behaviors": sorted(frame.behavior_model.unique().tolist()),
        "noise_levels": sorted(frame.noise_level.unique().tolist()),
        "population_conservation_max_abs_error": float(
            np.max(np.abs(frame.evacuation_rate + frame.incapacitation_rate + frame.remaining_rate - 1))
        ),
        "all_frozen_parent_hashes_match": True,
        "llm_advice_frozen_before_outcomes": True,
        "paid_api_requests": 0,
        "new_ppo_updates": 0,
        "completed_at_utc": stamp(),
    }
    write_json(OUT / "verification.json", verification)
    completion_paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest.json"]
    completion_paths.append(REPORT)
    write_json(
        OUT / "completion_manifest.json",
        {"status": "complete", "files": {relative(path): sha256(path) for path in sorted(completion_paths)}, "created_at_utc": stamp()},
    )
    write_json(
        OUT / "run_status.json",
        {
            "state": "complete",
            "scenario_contents_loaded_by_this_runner": True,
            "outcome_rollouts_completed": 2_880,
            "llm_requests_completed": 48,
            "report": relative(REPORT),
            "updated_at_utc": stamp(),
        },
    )


def self_check() -> None:
    from fire_evac.dynamic_conflicts import make_domain_randomized_trust_bank

    scenarios = make_domain_randomized_trust_bank(6, 2026101999, "SELFCHECK")
    dummy = {
        scenario.scenario_id: DynamicLLMAdvice(NORTH, 0.75, ("avoid_predicted_smoke",))
        for scenario in scenarios
    }
    models = load_models()
    for method in METHODS:
        env = make_env(scenarios, "calibrated_logit", "none", method, dummy)
        try:
            row = rollout(env, models[601], scenarios[0], 2_080_000_001, method)
            assert row["steps"] > 0
            assert abs(row["evacuation_rate"] + row["incapacitation_rate"] + row["remaining_rate"] - 1) < 1e-12
        finally:
            env.close()
    assert len(build_schedule()) == 2_880
    print("Self-check passed without reading SCF held-out scenario records.")


def run(base_url: str) -> None:
    verify_frozen_inputs()
    status = read_json(OUT / "run_status.json")
    if status.get("state") == "complete":
        print("Evaluation already complete; verifying outputs.")
        frame = pd.read_csv(OUT / "episode_results.csv")
        finalize(frame)
        return
    write_json(
        OUT / "unblinding_event.json",
        {
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
            "protocol_freeze_manifest_sha256": sha256(OUT / "freeze_manifest.json"),
            "unblinded_at_utc": stamp(),
            "protocol_frozen_before_unblinding": True,
        },
    )
    scenarios = load_scenarios_after_freeze()
    write_json(OUT / "final_scenarios_opened_once.json", [asdict(scenario) for scenario in scenarios])
    advice = collect_llm_advice(scenarios, base_url)
    if len(advice) != 48:
        raise RuntimeError("LLM advice bank incomplete")
    frame = run_rollouts(scenarios, advice)
    finalize(frame)
    print(f"Complete: {REPORT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-check", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check()
    elif arguments.prepare:
        prepare()
    else:
        run(arguments.base_url.rstrip("/"))
