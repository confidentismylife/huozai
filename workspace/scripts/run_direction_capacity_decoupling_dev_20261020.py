from __future__ import annotations

"""Development-only test of separating route direction from release capacity.

The 60 V2FE scenarios were used in earlier development work and are not a new
test set.  Existing pure-PPO and joint LLM-PPO outcomes are reused.  New hybrid
rollouts take release capacity (including wait) from the independently trained
pure PPO and direction from either the raw-LLM PPO, frozen LLM advice, or the
frozen rule.  No model is retrained and no LLM request is made.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_final_pipeline_context_20261020 as pipeline
from fire_evac.confirmatory import (
    NOISE_LEVELS,
    ConfirmatoryJointControlWrapper,
    decode_absolute_action,
    encode_absolute_action,
    frozen_rule_advice,
)
from fire_evac.config import EnvConfig
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice, parse_dynamic_advice


OUT = ROOT / "outputs" / "direction_capacity_decoupling_dev_20261020_v2"
REPORT = ROOT / "reports" / "方向容量解耦开发实验报告_2026-10-20.md"
SOURCE_DIR = ROOT / "outputs" / "guarded_confirmatory_v2"
SCENARIO_FILE = SOURCE_DIR / "final_scenarios.json"
SOURCE_RESULTS = SOURCE_DIR / "final_episode_results.csv"
LLM_CACHE = SOURCE_DIR / "llm_cache"
BEHAVIOR_ARTIFACT = ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json"
SEEDS = tuple(range(601, 606))
HYBRIDS = (
    "llm_policy_direction_pure_capacity",
    "llm_advice_direction_pure_capacity",
    "rule_direction_pure_capacity",
)
BASELINES = ("trained_pure_ppo", "trained_llm_joint")
ALL_METHODS = BASELINES + HYBRIDS
COMPARISONS = (
    ("llm_policy_direction_pure_capacity", "trained_llm_joint"),
    ("llm_advice_direction_pure_capacity", "trained_llm_joint"),
    ("llm_policy_direction_pure_capacity", "trained_pure_ppo"),
    ("llm_advice_direction_pure_capacity", "trained_pure_ppo"),
    ("rule_direction_pure_capacity", "trained_pure_ppo"),
)
METRICS = (
    "incapacitation_rate",
    "mean_cumulative_fed",
    "evacuation_time_seconds",
    "cumulative_congestion",
    "mean_release_fraction",
    "wait_fraction",
    "safe_branch_fraction_nonwait",
)
SCHEDULE_SEED = 2026102031
BOOTSTRAP_SEED = 2026102032
BOOTSTRAP_SAMPLES = 10_000
DT_SECONDS = float(EnvConfig().dt_seconds)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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


def load_scenarios() -> tuple[DynamicConflictScenario, ...]:
    return tuple(DynamicConflictScenario(**record) for record in read_json(SCENARIO_FILE))


def load_advice(scenarios) -> dict[str, DynamicLLMAdvice]:
    result = {}
    for scenario in scenarios:
        path = LLM_CACHE / f"{scenario.scenario_id}.json"
        payload = read_json(path)
        if payload["scenario_id"] != scenario.scenario_id:
            raise RuntimeError(f"cache identity mismatch: {path}")
        result[scenario.scenario_id] = parse_dynamic_advice(payload["parsed"])
    return result


def protected_paths() -> list[Path]:
    scenarios = load_scenarios()
    paths = [
        Path(__file__).resolve(),
        SCENARIO_FILE,
        SOURCE_RESULTS,
        BEHAVIOR_ARTIFACT,
        SCRIPTS / "run_guarded_confirmatory_v2.py",
        SCRIPTS / "run_final_pipeline_context_20261020.py",
    ]
    paths.extend(LLM_CACHE / f"{scenario.scenario_id}.json" for scenario in scenarios)
    paths.extend(sorted((SRC / "fire_evac").glob("*.py")))
    for seed in SEEDS:
        paths.append(pipeline.model_path(seed, "trained_pure_ppo"))
        paths.append(pipeline.model_path(seed, "trained_llm_advice_ppo"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing development input(s): " + ", ".join(missing))
    return paths


def build_schedule() -> list[dict[str, object]]:
    scenarios = load_scenarios()
    rows = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition in range(5):
            population_seed = 900_000 + scenario_index * 5 + repetition
            for seed in SEEDS:
                for method in HYBRIDS:
                    rows.append(
                        {
                            "run_id": f"{scenario.scenario_id}_{population_seed}_{seed}_{method}",
                            "scenario_id": scenario.scenario_id,
                            "conflict_type": scenario.conflict_type,
                            "population_repetition": repetition,
                            "population_seed": population_seed,
                            "training_seed": seed,
                            "method": method,
                        }
                    )
    rng = np.random.default_rng(SCHEDULE_SEED)
    shuffled = []
    for order, index in enumerate(rng.permutation(len(rows)), 1):
        row = dict(rows[int(index)])
        row["execution_order"] = order
        shuffled.append(row)
    assert len(shuffled) == 4_500
    return shuffled


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    parents = {relative(path): sha256(path) for path in protected_paths()}
    write_json(
        OUT / "protocol_frozen.json",
        {
            "protocol_version": "direction_capacity_decoupling_development_v2",
            "frozen_at_utc": stamp(),
            "analysis_status": "development_hypothesis_test",
            "supersedes": "outputs/direction_capacity_decoupling_dev_20261020",
            "pre_outcome_deviation": "v1 stopped before environment creation because behavior_factory was referenced from the wrong module; v2 uses CalibratedLogitBehavior directly; v1 completed zero rollouts",
            "scenario_bank": "60 previously used V2FE development scenarios",
            "conflict_types": 6,
            "scenarios_per_conflict": 10,
            "population_repetitions": 5,
            "training_seeds": list(SEEDS),
            "new_hybrid_rollouts": 4_500,
            "reused_baseline_rollouts": 3_000,
            "new_ppo_training": 0,
            "new_llm_requests": 0,
            "hybrid_semantics": {
                "capacity_and_wait": "action from independently trained pure PPO with advice indices 28:33 set to zero",
                "dynamic_llm_direction": "branch from raw-LLM PPO; if it requests wait, fall back to frozen LLM target",
                "static_llm_direction": "frozen LLM target",
                "rule_direction": "frozen_rule_advice target",
                "if_pure_capacity_requests_wait": "execute wait regardless of direction source",
            },
            "primary_comparison": ["llm_policy_direction_pure_capacity", "trained_llm_joint"],
            "development_success_rule": {
                "evacuation_time": "95% CI upper bound for hybrid minus joint LLM is below 0 seconds",
                "incapacitation_noninferiority": "95% CI upper bound is below +0.005 absolute rate",
                "fed_noninferiority": "95% CI upper bound is below +0.01",
                "all_required": True,
            },
            "claim_boundary": "candidate selection on old development scenes only; cannot establish final generalization",
        },
    )
    write_json(
        OUT / "analysis_plan_frozen.json",
        {
            "comparisons": [list(value) for value in COMPARISONS],
            "metrics": list(METRICS),
            "pairing": "scenario x population seed x training seed",
            "uncertainty": "average five population repetitions, then crossed 60-scenario x 5-training-seed bootstrap, 10000 draws",
            "all_six_conflict_types_reported": True,
        },
    )
    write_csv(OUT / "run_schedule_frozen.csv", build_schedule())
    write_json(OUT / "protected_parent_hashes.json", parents)
    frozen = [
        OUT / "protocol_frozen.json",
        OUT / "analysis_plan_frozen.json",
        OUT / "run_schedule_frozen.csv",
        OUT / "protected_parent_hashes.json",
    ]
    write_json(OUT / "freeze_manifest.json", {relative(path): sha256(path) for path in frozen})
    write_json(
        OUT / "run_status.json",
        {"state": "frozen_ready", "new_rollouts_completed": 0, "new_ppo_training": 0, "new_llm_requests": 0},
    )
    print("Frozen 4,500-rollout development decoupling protocol.")


def verify_frozen() -> None:
    for name, digest in read_json(OUT / "freeze_manifest.json").items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"frozen file changed: {name}")
    for name, digest in read_json(OUT / "protected_parent_hashes.json").items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"protected development input changed: {name}")


def load_models():
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    result = {}
    for seed in SEEDS:
        for role, method in (
            ("pure", "trained_pure_ppo"),
            ("llm", "trained_llm_advice_ppo"),
        ):
            model = PPO.load(pipeline.model_path(seed, method), device="cpu")
            if model.action_space.n != 9 or model.observation_space.shape != (34,):
                raise RuntimeError(f"model interface mismatch: {seed}/{role}")
            model.policy.set_training_mode(False)
            result[(seed, role)] = model
    return result


def make_env(scenarios, advice):
    from fire_evac.behavior import CalibratedLogitBehavior

    return ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(CalibratedLogitBehavior(BEHAVIOR_ARTIFACT), scenarios),
        advice_mode="llm",
        advice_by_scenario=advice,
        noise=NOISE_LEVELS["none"],
    )


def hybrid_rollout(env, pure_model, llm_model, scenario, population_seed: int, method: str, llm_advice):
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    terminated = truncated = False
    steps = 0
    release_sum = 0.0
    wait_steps = 0
    safe_steps = 0
    source_steps = 0
    nonwait = 0
    total_reward = 0.0
    safe_branch = 1 - int(scenario.source_branch)
    rule_target = frozen_rule_advice(scenario).target_branch
    while not (terminated or truncated):
        full_observation = np.asarray(observation, dtype=np.float32)
        pure_observation = full_observation.copy()
        pure_observation[28:33] = 0.0
        pure_action, _ = pure_model.predict(pure_observation, deterministic=True)
        pure_branch, pure_release = decode_absolute_action(int(np.asarray(pure_action).item()))
        if pure_branch is None:
            requested = 8
            wait_steps += 1
        else:
            if method == "llm_policy_direction_pure_capacity":
                llm_action, _ = llm_model.predict(full_observation, deterministic=True)
                direction, _ = decode_absolute_action(int(np.asarray(llm_action).item()))
                if direction is None:
                    direction = llm_advice.target_branch
            elif method == "llm_advice_direction_pure_capacity":
                direction = llm_advice.target_branch
            elif method == "rule_direction_pure_capacity":
                direction = rule_target
            else:
                raise ValueError(method)
            requested = encode_absolute_action(int(direction), float(pure_release))
            nonwait += 1
            release_sum += float(pure_release)
            safe_steps += int(direction == safe_branch)
            source_steps += int(direction == scenario.source_branch)
        observation, reward, terminated, truncated, info = env.step(requested)
        total_reward += float(reward)
        steps += 1
    denominator = max(1, nonwait)
    dt = float(env.unwrapped.domain_env.config.dt_seconds)
    population = float(env.unwrapped.domain_env.config.population)
    return {
        "steps": steps,
        "evacuation_time_seconds": steps * dt,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "remaining_rate": float(info["remaining"]) / population,
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "mean_release_fraction": release_sum / denominator,
        "wait_fraction": wait_steps / steps,
        "safe_branch_fraction_nonwait": safe_steps / denominator,
        "source_branch_fraction_nonwait": source_steps / denominator,
        "total_reward": total_reward,
        "truncated": int(truncated),
    }


def run_hybrids() -> pd.DataFrame:
    verify_frozen()
    scenarios = load_scenarios()
    advice = load_advice(scenarios)
    scenario_map = {item.scenario_id: item for item in scenarios}
    models = load_models()
    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    path = OUT / "hybrid_episode_results.csv"
    if path.exists():
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        done = set(existing.run_id.astype(str))
    else:
        rows, done = [], set()
    envs = {method: make_env(scenarios, advice) for method in HYBRIDS}
    try:
        for _, planned in schedule.sort_values("execution_order").iterrows():
            run_id = str(planned.run_id)
            if run_id in done:
                continue
            method = str(planned.method)
            seed = int(planned.training_seed)
            scenario = scenario_map[str(planned.scenario_id)]
            result = hybrid_rollout(
                envs[method],
                models[(seed, "pure")],
                models[(seed, "llm")],
                scenario,
                int(planned.population_seed),
                method,
                advice[scenario.scenario_id],
            )
            rows.append({**planned.to_dict(), **result})
            done.add(run_id)
            if len(rows) % 75 == 0:
                write_csv(path, rows)
                write_json(
                    OUT / "run_status.json",
                    {
                        "state": "running_hybrids",
                        "new_rollouts_completed": len(rows),
                        "new_ppo_training": 0,
                        "new_llm_requests": 0,
                    },
                )
                print(f"Hybrid development rollouts {len(rows)}/4500", flush=True)
        write_csv(path, rows)
    finally:
        for env in envs.values():
            env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 4_500 or frame.run_id.nunique() != 4_500:
        raise RuntimeError("hybrid matrix incomplete")
    return frame


def reuse_baselines() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_RESULTS)
    source = source[(source.test_behavior == "calibrated_logit") & (source.noise_level == "none")]
    source = source[source.arm.isin(("pure_ppo", "raw_llm_ppo"))].copy()
    names = {"pure_ppo": "trained_pure_ppo", "raw_llm_ppo": "trained_llm_joint"}
    source["method"] = source.arm.map(names)
    source["evacuation_time_seconds"] = source.steps.astype(float) * DT_SECONDS
    source["remaining_rate"] = 1.0 - source.evacuation_rate - source.incapacitation_rate
    source["mean_release_fraction"] = source.requested_mean_release_fraction.astype(float)
    source["wait_fraction"] = np.nan
    source["safe_branch_fraction_nonwait"] = np.nan
    source["source_branch_fraction_nonwait"] = np.nan
    source["truncated"] = 0
    source["population_repetition"] = source.groupby(["scenario_id", "training_seed", "arm"]).cumcount()
    source["run_id"] = source.apply(
        lambda row: f"{row.scenario_id}_{int(row.population_seed)}_{int(row.training_seed)}_{row.method}", axis=1
    )
    columns = [
        "run_id",
        "scenario_id",
        "conflict_type",
        "population_repetition",
        "population_seed",
        "training_seed",
        "method",
        "steps",
        "evacuation_time_seconds",
        "evacuation_rate",
        "incapacitation_rate",
        "remaining_rate",
        "mean_cumulative_fed",
        "cumulative_congestion",
        "sign_switches",
        "mean_release_fraction",
        "wait_fraction",
        "safe_branch_fraction_nonwait",
        "source_branch_fraction_nonwait",
        "total_reward",
        "truncated",
    ]
    result = source[columns]
    if len(result) != 3_000 or result.run_id.nunique() != 3_000:
        raise RuntimeError("baseline reuse matrix mismatch")
    return result


def paired_matrix(frame, candidate, baseline, metric):
    averaged = (
        frame.groupby(["scenario_id", "training_seed", "method"], as_index=False)[metric]
        .mean()
        .pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    )
    scenario_order = [item.scenario_id for item in load_scenarios()]
    values = (averaged[candidate] - averaged[baseline]).unstack("training_seed")
    return values.reindex(index=scenario_order, columns=SEEDS).to_numpy(float)


def crossed_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    if matrix.shape != (60, 5):
        raise ValueError(matrix.shape)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenarios = rng.integers(0, 60, 60)
        seeds = rng.integers(0, 5, 5)
        estimates[index] = matrix[np.ix_(scenarios, seeds)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if len(frame) != 7_500 or frame.run_id.nunique() != 7_500:
        raise RuntimeError("combined development matrix incomplete")
    conservation = frame.evacuation_rate + frame.incapacitation_rate + frame.remaining_rate
    if not np.allclose(conservation, 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("population conservation failed")
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            episodes=("run_id", "count"),
            incapacitation_rate_mean=("incapacitation_rate", "mean"),
            mean_cumulative_fed_mean=("mean_cumulative_fed", "mean"),
            evacuation_time_seconds_mean=("evacuation_time_seconds", "mean"),
            cumulative_congestion_mean=("cumulative_congestion", "mean"),
            mean_release_fraction_mean=("mean_release_fraction", "mean"),
            truncated=("truncated", "sum"),
        )
        .sort_values("method")
    )
    effects = []
    counter = 0
    for candidate, baseline in COMPARISONS:
        for metric in METRICS[:5]:
            values = paired_matrix(frame, candidate, baseline, metric)
            scenario_values = values.mean(axis=1)
            low, high = crossed_bootstrap(values, BOOTSTRAP_SEED + counter)
            effects.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "mean_difference": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "scenario_wins_lower": int((scenario_values < -1e-15).sum()),
                    "scenario_ties": int((np.abs(scenario_values) <= 1e-15).sum()),
                    "scenario_harms_higher": int((scenario_values > 1e-15).sum()),
                }
            )
            counter += 1
    effects = pd.DataFrame(effects)
    conflict_rows = []
    for conflict, current in frame.groupby("conflict_type"):
        for candidate, baseline in COMPARISONS[:2]:
            for metric in ("incapacitation_rate", "mean_cumulative_fed", "evacuation_time_seconds"):
                pivot = current.pivot(
                    index=["scenario_id", "population_seed", "training_seed"], columns="method", values=metric
                )
                conflict_rows.append(
                    {
                        "conflict_type": conflict,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "scenarios": current.scenario_id.nunique(),
                        "mean_difference": float((pivot[candidate] - pivot[baseline]).mean()),
                    }
                )
    conflicts = pd.DataFrame(conflict_rows)
    primary = effects[
        (effects.candidate == "llm_policy_direction_pure_capacity")
        & (effects.baseline == "trained_llm_joint")
    ].set_index("metric")
    gate = {
        "time_ci_upper_below_zero": bool(primary.loc["evacuation_time_seconds", "ci95_high"] < 0),
        "incapacitation_ci_upper_below_0_005": bool(primary.loc["incapacitation_rate", "ci95_high"] < 0.005),
        "fed_ci_upper_below_0_01": bool(primary.loc["mean_cumulative_fed", "ci95_high"] < 0.01),
    }
    gate["all_pass"] = all(gate.values())
    return summary, effects, conflicts, gate


def make_figure(effects: pd.DataFrame) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    methods = (
        "llm_policy_direction_pure_capacity",
        "llm_advice_direction_pure_capacity",
    )
    labels = ("LLM策略方向＋纯PPO容量", "LLM建议方向＋纯PPO容量")
    metrics = (
        ("evacuation_time_seconds", "疏散时间差（秒）", 1),
        ("incapacitation_rate", "失能率差（百分点）", 100),
        ("mean_cumulative_fed", "人均FED差", 1),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for axis, (metric, title, scale) in zip(axes, metrics):
        current = effects[(effects.baseline == "trained_llm_joint") & (effects.metric == metric)]
        for y, (method, label) in enumerate(zip(methods, labels)):
            row = current[current.candidate == method].iloc[0]
            mean, low, high = row.mean_difference * scale, row.ci95_low * scale, row.ci95_high * scale
            axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", capsize=4)
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks((0, 1))
        axis.set_yticklabels(labels, fontproperties=font)
        axis.set_title(title, fontproperties=font)
        axis.set_xlabel("混合控制−原LLM联合控制", fontproperties=font)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("历史开发场景中的方向与容量解耦", fontproperties=font)
    fig.tight_layout()
    fig.savefig(OUT / "decoupling_primary_effects_zh.png", dpi=220)
    plt.close(fig)


def write_report(summary, effects, conflicts, gate) -> None:
    names = {
        "trained_pure_ppo": "独立纯PPO",
        "trained_llm_joint": "原LLM方向＋容量联合控制",
        "llm_policy_direction_pure_capacity": "LLM策略方向＋纯PPO容量",
        "llm_advice_direction_pure_capacity": "LLM静态建议方向＋纯PPO容量",
        "rule_direction_pure_capacity": "规则方向＋纯PPO容量",
    }
    primary = effects[
        (effects.candidate == "llm_policy_direction_pure_capacity")
        & (effects.baseline == "trained_llm_joint")
    ].set_index("metric")
    lines = [
        "# 方向与容量解耦开发实验报告",
        "",
        "日期：2026-10-20。状态：在60个历史V2FE开发场景中运行4,500个新混合控制回合，并复用3,000个原基线回合。没有重新训练PPO，没有新增LLM请求。",
        "",
        "> **证据等级：开发实验。** V2FE场景以前已经用于模型开发和评价。本结果只能用于选择下一轮候选结构，不能作为新的最终泛化证据。48个SCF最终场景未参与本实验。",
        "",
        "## 1. 方法",
        "",
        "混合控制把一个9动作拆成“出口方向”和“放行容量”。纯PPO决定25%/50%/75%/100%容量以及是否等待；方向分别来自LLM动态策略、冻结LLM建议或冻结规则。这样可以直接检验此前观察到的低容量尾部是否来自LLM联合控制。",
        "",
        "## 2. 总体结果",
        "",
        "|方法|回合|失能率%|人均FED|疏散时间s|累计拥堵|平均放行档位|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"|{names[row.method]}|{int(row.episodes)}|{100 * row.incapacitation_rate_mean:.4f}|{row.mean_cumulative_fed_mean:.4f}|{row.evacuation_time_seconds_mean:.2f}|{row.cumulative_congestion_mean:.3f}|{row.mean_release_fraction_mean:.3f}|"
        )
    lines.extend(
        [
            "",
            "## 3. 主要候选：LLM策略方向＋纯PPO容量",
            "",
            "与原LLM方向＋容量联合控制相比：",
            "",
            "|指标|均值差|95% CI|",
            "|---|---:|---:|",
        ]
    )
    for metric, label, scale, suffix in (
        ("evacuation_time_seconds", "疏散时间", 1, "秒"),
        ("incapacitation_rate", "失能率", 100, "百分点"),
        ("mean_cumulative_fed", "人均FED", 1, ""),
        ("cumulative_congestion", "累计拥堵", 1, ""),
        ("mean_release_fraction", "平均放行档位", 1, ""),
    ):
        row = primary.loc[metric]
        lines.append(
            f"|{label}|{row.mean_difference * scale:.4f}{suffix}|[{row.ci95_low * scale:.4f}, {row.ci95_high * scale:.4f}]|"
        )
    lines.extend(
        [
            "",
            f"冻结开发门槛：时间改善={'通过' if gate['time_ci_upper_below_zero'] else '未通过'}；失能率非劣={'通过' if gate['incapacitation_ci_upper_below_0_005'] else '未通过'}；FED非劣={'通过' if gate['fed_ci_upper_below_0_01'] else '未通过'}；综合={'通过' if gate['all_pass'] else '未通过'}。",
            "",
            "![方向容量解耦效应](../outputs/direction_capacity_decoupling_dev_20261020_v2/decoupling_primary_effects_zh.png)",
            "",
            "## 4. 六类冲突场景",
            "",
            "|候选|冲突类型|场景数|失能率差（百分点）|FED差|时间差（秒）|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in COMPARISONS[:2]:
        current = conflicts[(conflicts.candidate == candidate[0]) & (conflicts.baseline == candidate[1])]
        for conflict in sorted(current.conflict_type.unique()):
            rows = current[current.conflict_type == conflict].set_index("metric")
            lines.append(
                f"|{names[candidate[0]]}|{conflict}|{int(rows.iloc[0].scenarios)}|{100 * rows.loc['incapacitation_rate', 'mean_difference']:.4f}|{rows.loc['mean_cumulative_fed', 'mean_difference']:.4f}|{rows.loc['evacuation_time_seconds', 'mean_difference']:.2f}|"
            )
    lines.extend(
        [
            "",
            "## 5. 解释边界与下一步",
            "",
            "只有综合开发门槛通过，才把该结构作为唯一候选冻结，并在全新场景银行上做确认性评价。若门槛未通过，则不再扩大这一混合结构。开发场景子组只用于检查失败模式，不按最有利子组选择结论。",
            "",
            "## 6. 可复核文件",
            "",
            "- 冻结协议：`outputs/direction_capacity_decoupling_dev_20261020_v2/protocol_frozen.json`",
            "- 新混合回合：`outputs/direction_capacity_decoupling_dev_20261020_v2/hybrid_episode_results.csv`",
            "- 合并开发矩阵：`outputs/direction_capacity_decoupling_dev_20261020_v2/combined_episode_results.csv`",
            "- 配对效应：`outputs/direction_capacity_decoupling_dev_20261020_v2/paired_effects.csv`",
            "- 冲突类型分析：`outputs/direction_capacity_decoupling_dev_20261020_v2/conflict_type_effects.csv`",
            "- 完整性核查：`outputs/direction_capacity_decoupling_dev_20261020_v2/verification.json`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    hybrids = run_hybrids()
    baselines = reuse_baselines()
    common = [column for column in hybrids.columns if column in baselines.columns]
    combined = pd.concat([hybrids[common], baselines[common]], ignore_index=True)
    write_csv(OUT / "combined_episode_results.csv", combined.to_dict("records"))
    summary, effects, conflicts, gate = analyze(combined)
    write_csv(OUT / "descriptive_summary.csv", summary.to_dict("records"))
    write_csv(OUT / "paired_effects.csv", effects.to_dict("records"))
    write_csv(OUT / "conflict_type_effects.csv", conflicts.to_dict("records"))
    write_json(OUT / "development_gate.json", gate)
    make_figure(effects)
    write_report(summary, effects, conflicts, gate)
    verify_frozen()
    verification = {
        "status": "passed",
        "analysis_status": "development_hypothesis_test",
        "new_hybrid_rollouts": len(hybrids),
        "reused_baseline_rollouts": len(baselines),
        "combined_rollouts": len(combined),
        "scenarios": int(combined.scenario_id.nunique()),
        "training_seeds": int(combined.training_seed.nunique()),
        "population_seeds": int(combined.population_seed.nunique()),
        "truncated_episodes": int(combined.truncated.sum()),
        "population_conservation_max_abs_error": float(
            np.abs(combined.evacuation_rate + combined.incapacitation_rate + combined.remaining_rate - 1).max()
        ),
        "development_gate": gate,
        "scf_final_scenarios_used": 0,
        "new_ppo_training": 0,
        "new_llm_requests": 0,
        "verified_at_utc": stamp(),
    }
    write_json(OUT / "verification.json", verification)
    write_json(
        OUT / "run_status.json",
        {
            "state": "complete",
            "new_rollouts_completed": len(hybrids),
            "combined_rollouts": len(combined),
            "report": relative(REPORT),
        },
    )
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest.json"] + [REPORT]
    write_json(
        OUT / "completion_manifest.json",
        {"status": "complete", "files": {relative(path): sha256(path) for path in sorted(files)}},
    )
    print("Direction-capacity development experiment complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
