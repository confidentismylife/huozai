from __future__ import annotations

"""Post-unblinding, exploratory whole-pipeline comparison.

This runner complements, but never replaces, the frozen shared-controller
evaluation.  It evaluates the separately trained no-advice and rule-advice PPO
models on exactly the same 48 x 5 x 2 x 2 cells, then reuses the already frozen
raw-LLM pipeline outcomes.  No model is retrained and no LLM is called.
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_final_blind_evaluation_20261020 as shared
from fire_evac.confirmatory import NOISE_LEVELS, ConfirmatoryJointControlWrapper
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice


OUT = ROOT / "outputs" / "final_pipeline_context_20261020"
REPORT = ROOT / "reports" / "LLM_PPO完整管线补充对照_2026-10-20.md"
BLIND_OUT = ROOT / "outputs" / "final_blind_evaluation_20261020"
MODEL_DIR = ROOT / "outputs" / "guarded_confirmatory_v2" / "models"
BEHAVIOR_ARTIFACT = ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json"
MODEL_SEEDS = tuple(range(601, 606))
SCENARIO_IDS = tuple(f"SCF{i:04d}" for i in range(1, 49))
BEHAVIORS = ("calibrated_logit", "mixed_logit")
NOISES = ("none", "medium")
NEW_METHODS = ("trained_pure_ppo", "trained_rule_advice_ppo")
ALL_METHODS = NEW_METHODS + ("trained_llm_advice_ppo",)
MODEL_STEMS = {
    "trained_pure_ppo": "pure_ppo",
    "trained_rule_advice_ppo": "rule_advice_ppo",
    "trained_llm_advice_ppo": "raw_llm_ppo",
}
ADVICE_MODES = {
    "trained_pure_ppo": "none",
    "trained_rule_advice_ppo": "rule",
    "trained_llm_advice_ppo": "llm",
}
PRIMARY_BEHAVIOR = "calibrated_logit"
PRIMARY_NOISE = "none"
PRIMARY_METRICS = ("incapacitation_rate", "mean_cumulative_fed")
SECONDARY_METRICS = ("evacuation_time_seconds", "max_congestion", "cumulative_congestion")
COMPARISONS = (
    ("trained_llm_advice_ppo", "trained_pure_ppo"),
    ("trained_llm_advice_ppo", "trained_rule_advice_ppo"),
    ("trained_rule_advice_ppo", "trained_pure_ppo"),
)
SCHEDULE_SEED = 2026102011
BOOTSTRAP_SEED = 2026102012
BOOTSTRAP_SAMPLES = 10_000


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


def model_path(seed: int, method: str) -> Path:
    return MODEL_DIR / f"seed_{seed}_{MODEL_STEMS[method]}.zip"


def build_schedule() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_id in SCENARIO_IDS:
        scenario_number = int(scenario_id[-4:])
        population_seed = 2_086_102_000 + scenario_number
        for training_seed in MODEL_SEEDS:
            for behavior in BEHAVIORS:
                for noise in NOISES:
                    for method in NEW_METHODS:
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
    shuffled: list[dict[str, object]] = []
    for execution_order, index in enumerate(rng.permutation(len(rows)), 1):
        row = dict(rows[int(index)])
        row["execution_order"] = execution_order
        shuffled.append(row)
    assert len(shuffled) == 1_920
    assert len({row["run_id"] for row in shuffled}) == 1_920
    return shuffled


def protected_paths() -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        BLIND_OUT / "final_scenarios_opened_once.json",
        BLIND_OUT / "llm_advice_frozen.json",
        BLIND_OUT / "episode_results.csv",
        BLIND_OUT / "independent_verification.json",
        BEHAVIOR_ARTIFACT,
        SCRIPTS / "run_final_blind_evaluation_20261020.py",
        SCRIPTS / "run_guarded_confirmatory_v2.py",
    ]
    paths.extend(sorted((SRC / "fire_evac").glob("*.py")))
    for seed in MODEL_SEEDS:
        paths.extend(model_path(seed, method) for method in ALL_METHODS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen input(s): " + ", ".join(missing))
    return paths


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol directory: {OUT}")
    OUT.mkdir(parents=True)
    parents = {relative(path): sha256(path) for path in protected_paths()}
    schedule = build_schedule()
    write_json(
        OUT / "protocol_frozen.json",
        {
            "protocol_version": "post_unblinding_whole_pipeline_context_v1",
            "frozen_at_utc": stamp(),
            "analysis_status": "exploratory_post_unblinding",
            "reason": "The shared-controller outcome was inspected before this protocol was frozen.",
            "question": "How do the three separately trained PPO pipelines compare on the same final cells?",
            "scenarios": 48,
            "training_seeds": list(MODEL_SEEDS),
            "behavior_models": list(BEHAVIORS),
            "noise_levels": list(NOISES),
            "new_methods": list(NEW_METHODS),
            "reused_method": "trained_llm_advice_ppo",
            "new_rollouts": 1_920,
            "reused_rollouts": 960,
            "combined_rollouts": 2_880,
            "new_ppo_training": 0,
            "new_llm_requests": 0,
            "selection_rule": "all five pre-existing seeds; no tuning or seed exclusion",
            "claim_boundary": "contextual whole-pipeline comparison; not a preregistered confirmatory result",
        },
    )
    write_json(
        OUT / "methods_frozen.json",
        {
            method: {
                "controller": f"seed_601-605_{MODEL_STEMS[method]}",
                "advice_mode": ADVICE_MODES[method],
                "trained_with_same_advice_mode": True,
            }
            for method in ALL_METHODS
        },
    )
    write_json(
        OUT / "model_hashes.json",
        {
            method: {
                str(seed): {"path": relative(model_path(seed, method)), "sha256": sha256(model_path(seed, method))}
                for seed in MODEL_SEEDS
            }
            for method in ALL_METHODS
        },
    )
    write_csv(OUT / "run_schedule_frozen.csv", schedule)
    write_json(
        OUT / "analysis_plan_frozen.json",
        {
            "primary_condition": {"behavior_model": PRIMARY_BEHAVIOR, "noise_level": PRIMARY_NOISE},
            "primary_metrics": list(PRIMARY_METRICS),
            "secondary_metrics": list(SECONDARY_METRICS),
            "comparisons": [list(item) for item in COMPARISONS],
            "pairing_unit": "scenario x training seed",
            "uncertainty": "crossed scenario-by-training-seed bootstrap, 10000 draws",
            "inference_status": "descriptive exploratory intervals; no confirmatory p-value",
            "all_behavior_noise_strata_reported": True,
        },
    )
    write_json(OUT / "protected_parent_hashes.json", parents)
    frozen = [
        OUT / "protocol_frozen.json",
        OUT / "methods_frozen.json",
        OUT / "model_hashes.json",
        OUT / "run_schedule_frozen.csv",
        OUT / "analysis_plan_frozen.json",
        OUT / "protected_parent_hashes.json",
    ]
    write_json(OUT / "freeze_manifest.json", {relative(path): sha256(path) for path in frozen})
    write_json(
        OUT / "run_status.json",
        {"state": "frozen_ready", "new_rollouts_completed": 0, "new_llm_requests": 0, "updated_at_utc": stamp()},
    )
    print(f"Frozen exploratory protocol and 1,920-row schedule at {OUT}")


def verify_frozen_inputs() -> None:
    for name, expected in read_json(OUT / "freeze_manifest.json").items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"frozen protocol file changed: {name}")
    for name, expected in read_json(OUT / "protected_parent_hashes.json").items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"protected input changed: {name}")


def load_scenarios() -> tuple[DynamicConflictScenario, ...]:
    records = read_json(BLIND_OUT / "final_scenarios_opened_once.json")
    scenarios = tuple(DynamicConflictScenario(**record) for record in records)
    by_id = {item.scenario_id: item for item in scenarios}
    if set(by_id) != set(SCENARIO_IDS):
        raise RuntimeError("scenario identity mismatch")
    return tuple(by_id[item] for item in SCENARIO_IDS)


def load_advice() -> dict[str, DynamicLLMAdvice]:
    payload = read_json(BLIND_OUT / "llm_advice_frozen.json")
    result = {}
    for scenario_id, item in payload["advice"].items():
        result[scenario_id] = DynamicLLMAdvice(
            target_branch=int(item["target_branch"]),
            confidence=float(item["confidence"]),
            reason_codes=tuple(item["reason_codes"]),
        )
    if set(result) != set(SCENARIO_IDS):
        raise RuntimeError("advice identity mismatch")
    return result


def make_env(scenarios, behavior: str, noise: str, method: str, advice):
    return ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(shared.behavior_factory(behavior), scenarios),
        advice_mode=ADVICE_MODES[method],
        advice_by_scenario=advice,
        noise=NOISE_LEVELS[noise],
    )


def load_models():
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    models = {}
    for seed in MODEL_SEEDS:
        for method in NEW_METHODS:
            model = PPO.load(model_path(seed, method), device="cpu")
            if model.action_space.n != 9 or model.observation_space.shape != (34,):
                raise RuntimeError(f"unexpected model interface: {seed}/{method}")
            model.policy.set_training_mode(False)
            models[(seed, method)] = model
    return models


def run_new_rollouts() -> pd.DataFrame:
    verify_frozen_inputs()
    scenarios = load_scenarios()
    advice = load_advice()
    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    path = OUT / "new_episode_results.csv"
    if path.exists():
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        done = set(existing.run_id.astype(str))
    else:
        rows, done = [], set()
    models = load_models()
    by_id = {item.scenario_id: item for item in scenarios}
    envs = {
        (behavior, noise, method): make_env(scenarios, behavior, noise, method, advice)
        for behavior in BEHAVIORS
        for noise in NOISES
        for method in NEW_METHODS
    }
    try:
        for _, scheduled in schedule.sort_values("execution_order").iterrows():
            run_id = str(scheduled.run_id)
            if run_id in done:
                continue
            method = str(scheduled.method)
            scenario = by_id[str(scheduled.scenario_id)]
            original_label = MODEL_STEMS[method]
            result = shared.rollout(
                envs[(str(scheduled.behavior_model), str(scheduled.noise_level), method)],
                models[(int(scheduled.training_seed), method)],
                scenario,
                int(scheduled.population_seed),
                original_label,
            )
            rows.append({**scheduled.to_dict(), "conflict_type": scenario.conflict_type, **result})
            done.add(run_id)
            if len(rows) % 48 == 0:
                write_csv(path, rows)
                write_json(
                    OUT / "run_status.json",
                    {
                        "state": "running",
                        "new_rollouts_completed": len(rows),
                        "new_llm_requests": 0,
                        "updated_at_utc": stamp(),
                    },
                )
                print(f"Whole-pipeline new rollouts {len(rows)}/1920", flush=True)
        write_csv(path, rows)
    finally:
        for env in envs.values():
            env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 1_920 or frame.run_id.nunique() != 1_920:
        raise RuntimeError("new rollout matrix incomplete or duplicated")
    return frame


def reuse_llm_results() -> pd.DataFrame:
    source = pd.read_csv(BLIND_OUT / "episode_results.csv")
    source = source[source.method == "llm_advice_ppo"].copy()
    if len(source) != 960:
        raise RuntimeError("expected 960 existing LLM pipeline cells")
    source["source_run_id"] = source.run_id
    source["method"] = "trained_llm_advice_ppo"
    source["run_id"] = source.apply(
        lambda row: f"{row.scenario_id}_{int(row.training_seed)}_{row.behavior_model}_{row.noise_level}_trained_llm_advice_ppo",
        axis=1,
    )
    return source


def crossed_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    if matrix.shape != (48, 5):
        raise ValueError(matrix.shape)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenario_index = rng.integers(0, 48, 48)
        seed_index = rng.integers(0, 5, 5)
        estimates[index] = matrix[np.ix_(scenario_index, seed_index)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def paired_matrix(frame, behavior, noise, candidate, baseline, metric):
    subset = frame[(frame.behavior_model == behavior) & (frame.noise_level == noise)]
    pivot = subset.pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    difference = (pivot[candidate] - pivot[baseline]).unstack("training_seed")
    difference = difference.reindex(index=SCENARIO_IDS, columns=MODEL_SEEDS)
    if difference.isna().any().any():
        raise RuntimeError("missing paired cell")
    return difference.to_numpy(float)


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) != 2_880 or frame.run_id.nunique() != 2_880:
        raise RuntimeError("combined matrix incomplete")
    counts = frame.groupby(["scenario_id", "training_seed", "behavior_model", "noise_level"]).method.nunique()
    if len(counts) != 960 or not (counts == 3).all():
        raise RuntimeError("pairing structure invalid")
    conservation = frame.evacuation_rate + frame.incapacitation_rate + frame.remaining_rate
    if not np.allclose(conservation, 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("population conservation failed")
    if int(frame.truncated.sum()) != 0:
        raise RuntimeError("truncated episode detected")
    summary = (
        frame.groupby(["behavior_model", "noise_level", "method"], as_index=False)
        .agg(
            episodes=("run_id", "count"),
            incapacitation_rate_mean=("incapacitation_rate", "mean"),
            mean_cumulative_fed_mean=("mean_cumulative_fed", "mean"),
            evacuation_time_seconds_mean=("evacuation_time_seconds", "mean"),
            max_congestion_mean=("max_congestion", "mean"),
            cumulative_congestion_mean=("cumulative_congestion", "mean"),
        )
        .sort_values(["behavior_model", "noise_level", "method"])
    )
    effects = []
    counter = 0
    for behavior in BEHAVIORS:
        for noise in NOISES:
            for candidate, baseline in COMPARISONS:
                for metric in PRIMARY_METRICS + SECONDARY_METRICS:
                    matrix = paired_matrix(frame, behavior, noise, candidate, baseline, metric)
                    scenario_values = matrix.mean(axis=1)
                    mean = float(matrix.mean())
                    low, high = crossed_bootstrap(matrix, BOOTSTRAP_SEED + counter)
                    baseline_mean = float(
                        frame[
                            (frame.behavior_model == behavior)
                            & (frame.noise_level == noise)
                            & (frame.method == baseline)
                        ][metric].mean()
                    )
                    effects.append(
                        {
                            "behavior_model": behavior,
                            "noise_level": noise,
                            "candidate": candidate,
                            "baseline": baseline,
                            "metric": metric,
                            "mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "baseline_mean": baseline_mean,
                            "relative_change_percent": 100 * mean / baseline_mean if baseline_mean else math.nan,
                            "scenario_wins": int((scenario_values < -1e-15).sum()),
                            "scenario_ties": int((np.abs(scenario_values) <= 1e-15).sum()),
                            "scenario_harms": int((scenario_values > 1e-15).sum()),
                            "analysis_status": "exploratory_post_unblinding",
                        }
                    )
                    counter += 1
    return summary, pd.DataFrame(effects)


def make_figure(effects: pd.DataFrame) -> None:
    current = effects[
        (effects.behavior_model == PRIMARY_BEHAVIOR)
        & (effects.noise_level == PRIMARY_NOISE)
        & effects.metric.isin(PRIMARY_METRICS)
    ]
    labels = {
        ("trained_llm_advice_ppo", "trained_pure_ppo"): "LLM管线−纯PPO管线",
        ("trained_llm_advice_ppo", "trained_rule_advice_ppo"): "LLM管线−规则管线",
        ("trained_rule_advice_ppo", "trained_pure_ppo"): "规则管线−纯PPO管线",
    }
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, metric, title in zip(axes, PRIMARY_METRICS, ("失能率差（百分点）", "人均累计FED差")):
        rows = current[current.metric == metric]
        for y, ((candidate, baseline), color) in enumerate(zip(COMPARISONS, colors)):
            row = rows[(rows.candidate == candidate) & (rows.baseline == baseline)].iloc[0]
            scale = 100 if metric == "incapacitation_rate" else 1
            mean, low, high = row.mean_difference * scale, row.ci95_low * scale, row.ci95_high * scale
            axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=4)
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(range(3), [labels[item] for item in COMPARISONS])
        axis.set_title(title)
        axis.set_xlabel("负值表示前一管线指标更低；95%交叉bootstrap区间")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("完整训练管线补充对照（解封后探索性）")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_primary_effects_zh.png", dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    primary = effects[
        (effects.behavior_model == PRIMARY_BEHAVIOR)
        & (effects.noise_level == PRIMARY_NOISE)
        & effects.metric.isin(PRIMARY_METRICS)
    ]
    names = {
        "trained_pure_ppo": "独立训练纯PPO",
        "trained_rule_advice_ppo": "独立训练规则建议PPO",
        "trained_llm_advice_ppo": "独立训练LLM建议PPO",
    }
    comparison_names = {
        ("trained_llm_advice_ppo", "trained_pure_ppo"): "LLM−纯PPO",
        ("trained_llm_advice_ppo", "trained_rule_advice_ppo"): "LLM−规则",
        ("trained_rule_advice_ppo", "trained_pure_ppo"): "规则−纯PPO",
    }
    lines = [
        "# LLM＋PPO完整训练管线补充对照",
        "",
        "日期：2026-10-20。状态：1,920个新回合完成，并与最终盲测中960个LLM管线回合合并，共2,880个配对回合。没有重新训练PPO，没有新增LLM请求。",
        "",
        "> **证据等级：解封后探索性。** 本协议在共享控制器盲测结果已经可见后冻结，用来回答不同训练管线的实际表现。它能修正‘零建议通道等于纯PPO’的解释问题，但不能升级为预注册确认性结论。所有五个既有训练种子均纳入，没有调参或按结果筛选。",
        "",
        "## 1. 比较设计",
        "",
        "三种PPO分别在与部署一致的建议模式下训练：纯PPO始终没有建议输入；规则PPO训练和测试都接收冻结规则建议；LLM建议PPO训练和测试都接收原始LLM建议。三组使用相同34维接口、9动作、训练预算、场景、人群种子和物理约束。",
        "",
        "主要观察条件仍为校准Logit人群、无观测噪声。负差表示表中前一种管线的指标更低。区间同时重采样48个场景和5个训练种子；由于本分析是结果解封后补做，只报告探索性区间，不报告确认性p值。",
        "",
        "## 2. 主要条件结果",
        "",
        "|比较|指标|均值差|95% CI|相对基线变化|场景 改善/持平/变差|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        key = (row.candidate, row.baseline)
        scale = 100 if row.metric == "incapacitation_rate" else 1
        unit = "百分点" if row.metric == "incapacitation_rate" else ""
        metric_name = "失能率" if row.metric == "incapacitation_rate" else "人均累计FED"
        lines.append(
            f"|{comparison_names[key]}|{metric_name}|{row.mean_difference * scale:.4f}{unit}|[{row.ci95_low * scale:.4f}, {row.ci95_high * scale:.4f}]|{row.relative_change_percent:.2f}%|{int(row.scenario_wins)}/{int(row.scenario_ties)}/{int(row.scenario_harms)}|"
        )
    lines.extend(
        [
            "",
            "## 3. 所有行为与噪声条件",
            "",
            "|人群|噪声|方法|回合|失能率%|人均累计FED|疏散时间s|最大拥堵|累计拥堵|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"|{row.behavior_model}|{row.noise_level}|{names[row.method]}|{int(row.episodes)}|{row.incapacitation_rate_mean * 100:.4f}|{row.mean_cumulative_fed_mean:.4f}|{row.evacuation_time_seconds_mean:.2f}|{row.max_congestion_mean:.3f}|{row.cumulative_congestion_mean:.3f}|"
        )
    lines.extend(
        [
            "",
            "![完整管线主要效应](../outputs/final_pipeline_context_20261020/pipeline_primary_effects_zh.png)",
            "",
            "## 4. 与共享控制器盲测的关系",
            "",
            "共享控制器盲测回答的是：固定同一套raw-LLM PPO权重，只改变建议输入会怎样。本文回答的是：训练和部署都采用各自建议模式时，三条完整管线会怎样。前者隔离输入通道，后者反映系统级表现；两者的问题不同，应并列解释。",
            "",
            "结论只适用于当前合成五分区、两出口环境。既有安全资格门槛仍未通过，没有真人动态轨迹或真实建筑验证。",
            "",
            "## 5. 可复核文件",
            "",
            "- 冻结协议：`outputs/final_pipeline_context_20261020/protocol_frozen.json`",
            "- 冻结运行表：`outputs/final_pipeline_context_20261020/run_schedule_frozen.csv`",
            "- 新运行结果：`outputs/final_pipeline_context_20261020/new_episode_results.csv`",
            "- 三管线合并结果：`outputs/final_pipeline_context_20261020/combined_episode_results.csv`",
            "- 描述统计：`outputs/final_pipeline_context_20261020/descriptive_summary.csv`",
            "- 配对效应：`outputs/final_pipeline_context_20261020/paired_effects.csv`",
            "- 完整性核查：`outputs/final_pipeline_context_20261020/verification.json`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    new = run_new_rollouts()
    reused = reuse_llm_results()
    common_columns = [column for column in new.columns if column in reused.columns]
    combined = pd.concat([new[common_columns], reused[common_columns]], ignore_index=True)
    write_csv(OUT / "combined_episode_results.csv", combined.to_dict("records"))
    summary, effects = analyze(combined)
    write_csv(OUT / "descriptive_summary.csv", summary.to_dict("records"))
    write_csv(OUT / "paired_effects.csv", effects.to_dict("records"))
    make_figure(effects)
    write_report(summary, effects)
    verify_frozen_inputs()
    verification = {
        "status": "passed",
        "analysis_status": "exploratory_post_unblinding",
        "new_rollouts": len(new),
        "reused_llm_rollouts": len(reused),
        "combined_rollouts": len(combined),
        "paired_blocks": int(combined.groupby(["scenario_id", "training_seed", "behavior_model", "noise_level"]).ngroups),
        "scenarios": int(combined.scenario_id.nunique()),
        "training_seeds": int(combined.training_seed.nunique()),
        "truncated_episodes": int(combined.truncated.sum()),
        "population_conservation_max_abs_error": float(
            np.abs(combined.evacuation_rate + combined.incapacitation_rate + combined.remaining_rate - 1).max()
        ),
        "new_llm_requests": 0,
        "new_ppo_training": 0,
        "verified_at_utc": stamp(),
    }
    write_json(OUT / "verification.json", verification)
    write_json(
        OUT / "run_status.json",
        {
            "state": "complete",
            "new_rollouts_completed": len(new),
            "combined_rollouts": len(combined),
            "new_llm_requests": 0,
            "report": relative(REPORT),
            "updated_at_utc": stamp(),
        },
    )
    products = [
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name not in {"completion_manifest.json"}
    ] + [REPORT]
    write_json(OUT / "completion_manifest.json", {relative(path): sha256(path) for path in sorted(products)})
    print("Whole-pipeline exploratory comparison complete.")


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
