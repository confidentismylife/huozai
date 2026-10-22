from __future__ import annotations

"""Exploratory mechanism attribution and scenario-level predictability audit.

All models, advice, scenarios, population seeds, and outcomes predate this
analysis.  The script replays the 2,880 whole-pipeline cells only to capture
step-level diagnostics that were not stored previously.  It then performs
grouped cross-validation by scenario; no model is retrained or selected using
the final outcome data.
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
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_final_blind_evaluation_20261020 as shared
import run_final_pipeline_context_20261020 as pipeline
from fire_evac.confirmatory import decode_absolute_action
from fire_evac.dynamic_llm_rl import NORTH, SOUTH


OUT = ROOT / "outputs" / "final_mechanism_attribution_20261020"
REPORT = ROOT / "reports" / "LLM_PPO机制归因与可预测性报告_2026-10-20.md"
PIPELINE_OUT = ROOT / "outputs" / "final_pipeline_context_20261020"
BLIND_OUT = ROOT / "outputs" / "final_blind_evaluation_20261020"
METHODS = pipeline.ALL_METHODS
BEHAVIORS = pipeline.BEHAVIORS
NOISES = pipeline.NOISES
SEEDS = pipeline.MODEL_SEEDS
SCENARIOS = pipeline.SCENARIO_IDS
COMPARISONS = (
    ("trained_llm_advice_ppo", "trained_pure_ppo"),
    ("trained_llm_advice_ppo", "trained_rule_advice_ppo"),
    ("trained_rule_advice_ppo", "trained_pure_ppo"),
)
SCHEDULE_SEED = 2026102021
BOOTSTRAP_SEED = 2026102022
CV_SEED = 2026102023
BOOTSTRAP_SAMPLES = 5_000
PRIMARY_BEHAVIOR = "calibrated_logit"
PRIMARY_NOISE = "none"
OBSERVATION_NAMES = (
    "occ_room_0",
    "occ_room_1",
    "occ_junction",
    "occ_north_corridor",
    "occ_south_corridor",
    "smoke_room_0",
    "smoke_room_1",
    "smoke_junction",
    "smoke_north_corridor",
    "smoke_south_corridor",
    "last_sign_north",
    "last_sign_south",
    "last_sign_neutral",
    "remaining_fraction",
    "current_north_capacity_scaled",
    "current_south_capacity_scaled",
    "current_north_blocked",
    "current_south_blocked",
    "predicted_reopen_steps_scaled",
    "forecast_source_north",
    "forecast_source_south",
    "forecast_growth_scaled",
    "forecast_onset_scaled",
    "forecast_north_capacity_scaled",
    "forecast_south_capacity_scaled",
    "forecast_north_blocked",
    "forecast_south_blocked",
    "forecast_blockage_end_scaled",
    "llm_target_north",
    "llm_target_south",
    "llm_confidence",
    "llm_advice_valid",
    "llm_smoke_contradiction",
    "previous_release_fraction",
)
DIAGNOSTIC_METRICS = (
    "evacuation_time_seconds",
    "incapacitation_rate",
    "mean_cumulative_fed",
    "cumulative_congestion",
    "wait_fraction_replay",
    "mean_requested_release_fraction_nonwait",
    "mean_actual_release_fraction",
    "full_release_fraction_nonwait",
    "low_release_fraction_nonwait",
    "safe_branch_fraction_nonwait",
    "source_branch_fraction_nonwait",
    "llm_target_fraction_nonwait",
    "first_people_release_seconds",
    "rooms_clear_seconds",
    "evacuated_50_seconds",
    "evacuated_90_seconds",
    "mean_active_people",
    "room_person_seconds",
    "junction_person_seconds",
    "corridor_person_seconds",
    "total_people_released",
    "sign_switches",
)
PREDICTION_BASELINES = ("trained_pure_ppo", "trained_rule_advice_ppo")


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


def build_schedule() -> list[dict[str, object]]:
    frame = pd.read_csv(PIPELINE_OUT / "combined_episode_results.csv")
    columns = [
        "run_id",
        "scenario_id",
        "training_seed",
        "behavior_model",
        "noise_level",
        "method",
        "population_seed",
    ]
    rows = frame[columns].to_dict("records")
    if len(rows) != 2_880 or len({str(row["run_id"]) for row in rows}) != 2_880:
        raise RuntimeError("source matrix is incomplete")
    rng = np.random.default_rng(SCHEDULE_SEED)
    shuffled = []
    for order, index in enumerate(rng.permutation(len(rows)), 1):
        row = dict(rows[int(index)])
        row["diagnostic_execution_order"] = order
        shuffled.append(row)
    return shuffled


def protected_paths() -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        PIPELINE_OUT / "combined_episode_results.csv",
        PIPELINE_OUT / "independent_verification.json",
        BLIND_OUT / "final_scenarios_opened_once.json",
        BLIND_OUT / "llm_advice_frozen.json",
        SCRIPTS / "run_final_pipeline_context_20261020.py",
        SCRIPTS / "run_final_blind_evaluation_20261020.py",
    ]
    paths.extend(sorted((SRC / "fire_evac").glob("*.py")))
    for seed in SEEDS:
        paths.extend(pipeline.model_path(seed, method) for method in METHODS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input: " + ", ".join(missing))
    return paths


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    parents = {relative(path): sha256(path) for path in protected_paths()}
    write_json(
        OUT / "protocol_frozen.json",
        {
            "protocol_version": "post_unblinding_mechanism_attribution_v1",
            "frozen_at_utc": stamp(),
            "analysis_status": "exploratory_post_unblinding",
            "questions": [
                "Which action and flow mechanisms explain the LLM pipeline time penalty?",
                "Can decision-time observations predict whether LLM is beneficial?",
            ],
            "replay_cells": 2_880,
            "new_ppo_training": 0,
            "new_llm_requests": 0,
            "same_models_scenarios_population_seeds": True,
            "outcome_replay_tolerance": 1e-12,
            "prediction_unit": "scenario x behavior model x noise level, averaged over five training seeds",
            "cross_validation": "eight folds grouped by scenario_id; every scenario appears in exactly one test fold",
            "feature_sets": {
                "public_observation": "initial 34-vector excluding five LLM advice fields, plus known behavior/noise indicators",
                "full_with_llm": "complete initial 34-vector plus known behavior/noise indicators",
            },
            "fixed_classifier": "L2 logistic regression, penalty=1.0, no tuning",
            "gate_success_rule": "for both LLM-vs-pure and LLM-vs-rule: AUC CI lower bound >0.60 and Brier improves at least 5% over fold prevalence baseline",
            "claim_boundary": "hypothesis generation only; final scenarios are already unblinded",
        },
    )
    write_json(
        OUT / "analysis_plan_frozen.json",
        {
            "primary_condition": {"behavior_model": PRIMARY_BEHAVIOR, "noise_level": PRIMARY_NOISE},
            "comparisons": [list(value) for value in COMPARISONS],
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "paired_uncertainty": "crossed scenario x training-seed bootstrap, 5000 draws",
            "benefit_label": "lower incapacitation; when tied, lower mean cumulative FED",
            "predictive_metrics": ["balanced_accuracy", "roc_auc", "brier", "ece_5bin"],
            "predictive_uncertainty": "scenario-cluster bootstrap, 5000 draws",
            "no_final_data_hyperparameter_search": True,
        },
    )
    write_csv(OUT / "replay_schedule_frozen.csv", build_schedule())
    write_json(OUT / "protected_parent_hashes.json", parents)
    frozen = [
        OUT / "protocol_frozen.json",
        OUT / "analysis_plan_frozen.json",
        OUT / "replay_schedule_frozen.csv",
        OUT / "protected_parent_hashes.json",
    ]
    write_json(OUT / "freeze_manifest.json", {relative(path): sha256(path) for path in frozen})
    write_json(
        OUT / "run_status.json",
        {"state": "frozen_ready", "replays_completed": 0, "new_ppo_training": 0, "new_llm_requests": 0},
    )
    print("Frozen 2,880-cell mechanism replay and grouped-CV plan.")


def verify_frozen() -> None:
    for name, digest in read_json(OUT / "freeze_manifest.json").items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"frozen file changed: {name}")
    for name, digest in read_json(OUT / "protected_parent_hashes.json").items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"protected input changed: {name}")


def load_models():
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    result = {}
    for seed in SEEDS:
        for method in METHODS:
            model = PPO.load(pipeline.model_path(seed, method), device="cpu")
            if model.action_space.n != 9 or model.observation_space.shape != (34,):
                raise RuntimeError(f"model interface mismatch: {seed}/{method}")
            model.policy.set_training_mode(False)
            result[(seed, method)] = model
    return result


def replay_episode(env, model, scenario, population_seed: int, llm_target: int) -> dict[str, object]:
    observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
    initial = np.asarray(observation, dtype=float).copy()
    if initial.shape != (34,):
        raise RuntimeError(initial.shape)
    terminated = truncated = False
    step = 0
    requested_nonwait = 0
    requested_wait = 0
    release_sum_nonwait = 0.0
    actual_release_sum = 0.0
    full_release = 0
    low_release = 0
    north = south = 0
    safe = source = target = 0
    first_people_release = None
    rooms_clear = None
    evacuated_50 = None
    evacuated_90 = None
    active_person_steps = 0.0
    room_person_steps = 0.0
    junction_person_steps = 0.0
    corridor_person_steps = 0.0
    total_released = 0
    max_room = max_junction = max_corridor = 0
    dt = float(env.unwrapped.domain_env.config.dt_seconds)
    population = int(env.unwrapped.domain_env.config.population)
    safe_branch = SOUTH if scenario.source_branch == NORTH else NORTH
    while not (terminated or truncated):
        action, _ = model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
        requested = int(np.asarray(action).item())
        branch, release = decode_absolute_action(requested)
        if branch is None:
            requested_wait += 1
        else:
            requested_nonwait += 1
            release_sum_nonwait += float(release)
            full_release += int(float(release) >= 1.0 - 1e-12)
            low_release += int(float(release) <= 0.5 + 1e-12)
            north += int(branch == NORTH)
            south += int(branch == SOUTH)
            safe += int(branch == safe_branch)
            source += int(branch == scenario.source_branch)
            target += int(branch == llm_target)
        observation, _, terminated, truncated, info = env.step(requested)
        step += 1
        actual_release_sum += float(info["actual_release_fraction"])
        released = int(info["room_release_count"])
        total_released += released
        if released > 0 and first_people_release is None:
            first_people_release = step
        counts = [int(value) for value in info["zone_counts"]]
        room_count = counts[0] + counts[1]
        junction_count = counts[2]
        corridor_count = counts[3] + counts[4]
        active = int(info["remaining"])
        active_person_steps += active
        room_person_steps += room_count
        junction_person_steps += junction_count
        corridor_person_steps += corridor_count
        max_room = max(max_room, room_count)
        max_junction = max(max_junction, junction_count)
        max_corridor = max(max_corridor, corridor_count)
        if room_count == 0 and rooms_clear is None:
            rooms_clear = step
        if int(info["evacuated"]) >= math.ceil(0.5 * population) and evacuated_50 is None:
            evacuated_50 = step
        if int(info["evacuated"]) >= math.ceil(0.9 * population) and evacuated_90 is None:
            evacuated_90 = step
    nonwait = max(1, requested_nonwait)
    result = {
        "steps": step,
        "evacuation_time_seconds": step * dt,
        "evacuation_rate": float(info["evacuation_rate"]),
        "incapacitation_rate": float(info["incapacitation_rate"]),
        "remaining_rate": float(info["remaining"]) / population,
        "mean_cumulative_fed": float(info["mean_cumulative_fed"]),
        "cumulative_congestion": float(info["cumulative_congestion"]),
        "sign_switches": int(info["sign_switches"]),
        "truncated": int(truncated),
        "wait_fraction_replay": requested_wait / step,
        "mean_requested_release_fraction_nonwait": release_sum_nonwait / nonwait,
        "mean_actual_release_fraction": actual_release_sum / step,
        "full_release_fraction_nonwait": full_release / nonwait,
        "low_release_fraction_nonwait": low_release / nonwait,
        "north_branch_fraction_nonwait": north / nonwait,
        "south_branch_fraction_nonwait": south / nonwait,
        "safe_branch_fraction_nonwait": safe / nonwait,
        "source_branch_fraction_nonwait": source / nonwait,
        "llm_target_fraction_nonwait": target / nonwait,
        "first_people_release_seconds": (first_people_release or step + 1) * dt,
        "rooms_clear_seconds": (rooms_clear or step + 1) * dt,
        "evacuated_50_seconds": (evacuated_50 or step + 1) * dt,
        "evacuated_90_seconds": (evacuated_90 or step + 1) * dt,
        "mean_active_people": active_person_steps / step,
        "room_person_seconds": room_person_steps * dt,
        "junction_person_seconds": junction_person_steps * dt,
        "corridor_person_seconds": corridor_person_steps * dt,
        "total_people_released": total_released,
        "peak_room_occupancy": max_room,
        "peak_junction_occupancy": max_junction,
        "peak_corridor_occupancy": max_corridor,
    }
    result.update({f"initial_{name}": float(value) for name, value in zip(OBSERVATION_NAMES, initial)})
    return result


def run_replays() -> pd.DataFrame:
    verify_frozen()
    schedule = pd.read_csv(OUT / "replay_schedule_frozen.csv")
    scenarios = pipeline.load_scenarios()
    advice = pipeline.load_advice()
    scenario_map = {item.scenario_id: item for item in scenarios}
    models = load_models()
    envs = {
        (behavior, noise, method): pipeline.make_env(scenarios, behavior, noise, method, advice)
        for behavior in BEHAVIORS
        for noise in NOISES
        for method in METHODS
    }
    path = OUT / "replay_diagnostics.csv"
    if path.exists():
        existing = pd.read_csv(path)
        rows = existing.to_dict("records")
        done = set(existing.run_id.astype(str))
    else:
        rows, done = [], set()
    try:
        for _, planned in schedule.sort_values("diagnostic_execution_order").iterrows():
            run_id = str(planned.run_id)
            if run_id in done:
                continue
            method = str(planned.method)
            scenario = scenario_map[str(planned.scenario_id)]
            result = replay_episode(
                envs[(str(planned.behavior_model), str(planned.noise_level), method)],
                models[(int(planned.training_seed), method)],
                scenario,
                int(planned.population_seed),
                advice[scenario.scenario_id].target_branch,
            )
            rows.append({**planned.to_dict(), "conflict_type": scenario.conflict_type, **result})
            done.add(run_id)
            if len(rows) % 48 == 0:
                write_csv(path, rows)
                write_json(
                    OUT / "run_status.json",
                    {
                        "state": "replaying",
                        "replays_completed": len(rows),
                        "new_ppo_training": 0,
                        "new_llm_requests": 0,
                    },
                )
                print(f"Mechanism replays {len(rows)}/2880", flush=True)
        write_csv(path, rows)
    finally:
        for env in envs.values():
            env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 2_880 or frame.run_id.nunique() != 2_880:
        raise RuntimeError("replay matrix incomplete")
    return frame


def validate_replay(frame: pd.DataFrame) -> dict[str, float]:
    original = pd.read_csv(PIPELINE_OUT / "combined_episode_results.csv")
    metrics = (
        "steps",
        "evacuation_time_seconds",
        "evacuation_rate",
        "incapacitation_rate",
        "remaining_rate",
        "mean_cumulative_fed",
        "cumulative_congestion",
        "sign_switches",
        "truncated",
    )
    merged = frame[["run_id", *metrics]].merge(
        original[["run_id", *metrics]], on="run_id", suffixes=("_replay", "_original"), validate="one_to_one"
    )
    errors = {}
    for metric in metrics:
        error = float(
            np.max(
                np.abs(
                    merged[f"{metric}_replay"].to_numpy(float)
                    - merged[f"{metric}_original"].to_numpy(float)
                )
            )
        )
        errors[metric] = error
        if error > 1e-12:
            raise RuntimeError(f"replay outcome mismatch: {metric}={error}")
    return errors


def paired_matrix(frame, behavior, noise, candidate, baseline, metric):
    subset = frame[(frame.behavior_model == behavior) & (frame.noise_level == noise)]
    pivot = subset.pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    difference = (pivot[candidate] - pivot[baseline]).unstack("training_seed")
    return difference.reindex(index=SCENARIOS, columns=SEEDS).to_numpy(float)


def crossed_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenarios = rng.integers(0, 48, 48)
        seeds = rng.integers(0, 5, 5)
        estimates[index] = matrix[np.ix_(scenarios, seeds)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def mechanism_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 0
    for behavior in BEHAVIORS:
        for noise in NOISES:
            for candidate, baseline in COMPARISONS:
                for metric in DIAGNOSTIC_METRICS:
                    values = paired_matrix(frame, behavior, noise, candidate, baseline, metric)
                    if values.shape != (48, 5) or np.isnan(values).any():
                        raise RuntimeError(f"invalid mechanism matrix: {behavior}/{noise}/{metric}")
                    scenario_values = values.mean(axis=1)
                    low, high = crossed_bootstrap(values, BOOTSTRAP_SEED + counter)
                    rows.append(
                        {
                            "behavior_model": behavior,
                            "noise_level": noise,
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
    return pd.DataFrame(rows)


def grouped_folds(groups: np.ndarray, folds: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    rng = np.random.default_rng(CV_SEED)
    unique = unique[rng.permutation(len(unique))]
    bins = np.array_split(unique, folds)
    result = []
    for test_groups in bins:
        test = np.isin(groups, test_groups)
        result.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    keep = scale > 1e-10
    transformed = (x[:, keep] - mean[keep]) / scale[keep]
    design = np.column_stack([np.ones(len(x)), transformed])

    def objective(beta):
        z = design @ beta
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * penalty * np.sum(beta[1:] ** 2) / len(y)
        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += penalty * beta[1:] / len(y)
        return float(loss), gradient

    fitted = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(design.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"logistic fit failed: {fitted.message}")
    return fitted.x, mean, np.where(keep, scale, np.nan)


def predict_logistic(x: np.ndarray, beta: np.ndarray, mean: np.ndarray, scale_marker: np.ndarray) -> np.ndarray:
    keep = np.isfinite(scale_marker)
    transformed = (x[:, keep] - mean[keep]) / scale_marker[keep]
    z = np.column_stack([np.ones(len(x)), transformed]) @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    positive = y == 1
    negative = y == 0
    sensitivity = float(prediction[positive].mean()) if positive.any() else math.nan
    specificity = float((~prediction[negative]).mean()) if negative.any() else math.nan
    balanced = (sensitivity + specificity) / 2
    ranks = rankdata(probability, method="average")
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    auc = float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    brier = float(np.mean((probability - y) ** 2))
    edges = np.linspace(0, 1, 6)
    ece = 0.0
    for index in range(5):
        in_bin = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 4 else probability < edges[index + 1]
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(probability[in_bin].mean() - y[in_bin].mean()))
    return {
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "brier": brier,
        "ece_5bin": ece,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def cluster_bootstrap_metrics(y, probability, groups, seed: int) -> dict[str, tuple[float, float]]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("balanced_accuracy", "roc_auc", "brier", "ece_5bin")}
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        current = binary_metrics(y[indices], probability[indices])
        for key in values:
            values[key].append(current[key])
    return {key: tuple(float(v) for v in np.quantile(item, (0.025, 0.975))) for key, item in values.items()}


def prediction_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observation_columns = [f"initial_{name}" for name in OBSERVATION_NAMES]
    llm = frame[frame.method == "trained_llm_advice_ppo"].copy()
    group_columns = ["scenario_id", "behavior_model", "noise_level"]
    initial = llm.groupby(group_columns, as_index=False)[observation_columns].mean()
    outcomes = (
        frame.groupby([*group_columns, "method"], as_index=False)[
            ["incapacitation_rate", "mean_cumulative_fed", "evacuation_time_seconds"]
        ]
        .mean()
        .pivot(index=group_columns, columns="method")
    )
    target_rows = []
    for index, values in outcomes.iterrows():
        record = dict(zip(group_columns, index))
        for baseline in PREDICTION_BASELINES:
            incap_difference = float(
                values[("incapacitation_rate", "trained_llm_advice_ppo")]
                - values[("incapacitation_rate", baseline)]
            )
            fed_difference = float(
                values[("mean_cumulative_fed", "trained_llm_advice_ppo")]
                - values[("mean_cumulative_fed", baseline)]
            )
            time_difference = float(
                values[("evacuation_time_seconds", "trained_llm_advice_ppo")]
                - values[("evacuation_time_seconds", baseline)]
            )
            beneficial = int(
                incap_difference < -1e-15
                or (abs(incap_difference) <= 1e-15 and fed_difference < 0)
            )
            record[f"beneficial_vs_{baseline}"] = beneficial
            record[f"incap_difference_vs_{baseline}"] = incap_difference
            record[f"fed_difference_vs_{baseline}"] = fed_difference
            record[f"time_difference_vs_{baseline}"] = time_difference
        target_rows.append(record)
    targets = pd.DataFrame(target_rows)
    dataset = initial.merge(targets, on=group_columns, validate="one_to_one")
    dataset["behavior_mixed_logit"] = (dataset.behavior_model == "mixed_logit").astype(float)
    dataset["noise_medium"] = (dataset.noise_level == "medium").astype(float)
    extra = ["behavior_mixed_logit", "noise_medium"]
    public = [column for index, column in enumerate(observation_columns) if index not in range(28, 33)] + extra
    full = observation_columns + extra
    feature_sets = {"public_observation": public, "full_with_llm": full}
    groups = dataset.scenario_id.to_numpy(str)
    folds = grouped_folds(groups)
    prediction_rows = []
    metric_rows = []
    counter = 0
    for baseline in PREDICTION_BASELINES:
        target_name = f"beneficial_vs_{baseline}"
        y = dataset[target_name].to_numpy(int)
        for feature_set, columns in feature_sets.items():
            x = dataset[columns].to_numpy(float)
            probability = np.empty(len(dataset))
            prevalence_probability = np.empty(len(dataset))
            for fold, (train, test) in enumerate(folds):
                beta, mean, scale = fit_logistic(x[train], y[train], penalty=1.0)
                probability[test] = predict_logistic(x[test], beta, mean, scale)
                prevalence_probability[test] = y[train].mean()
                for row_index in test:
                    prediction_rows.append(
                        {
                            "scenario_id": dataset.iloc[row_index].scenario_id,
                            "behavior_model": dataset.iloc[row_index].behavior_model,
                            "noise_level": dataset.iloc[row_index].noise_level,
                            "baseline": baseline,
                            "feature_set": feature_set,
                            "fold": fold,
                            "observed_beneficial": int(y[row_index]),
                            "predicted_probability": float(probability[row_index]),
                            "fold_prevalence_probability": float(prevalence_probability[row_index]),
                        }
                    )
            measured = binary_metrics(y, probability)
            prevalence = binary_metrics(y, prevalence_probability)
            intervals = cluster_bootstrap_metrics(y, probability, groups, CV_SEED + counter)
            metric_rows.append(
                {
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "rows": len(y),
                    "unique_scenarios": len(np.unique(groups)),
                    "beneficial_prevalence": float(y.mean()),
                    **measured,
                    **{f"{key}_ci95_low": value[0] for key, value in intervals.items()},
                    **{f"{key}_ci95_high": value[1] for key, value in intervals.items()},
                    "prevalence_baseline_brier": prevalence["brier"],
                    "brier_improvement_percent": 100 * (prevalence["brier"] - measured["brier"]) / prevalence["brier"],
                    "passes_frozen_gate": int(
                        intervals["roc_auc"][0] > 0.60
                        and 100 * (prevalence["brier"] - measured["brier"]) / prevalence["brier"] >= 5.0
                    ),
                }
            )
            counter += 1
    return dataset, pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def mechanism_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame[(frame.behavior_model == PRIMARY_BEHAVIOR) & (frame.noise_level == PRIMARY_NOISE)]
    metrics = (
        "mean_requested_release_fraction_nonwait",
        "mean_actual_release_fraction",
        "full_release_fraction_nonwait",
        "safe_branch_fraction_nonwait",
        "source_branch_fraction_nonwait",
        "rooms_clear_seconds",
        "evacuated_50_seconds",
        "evacuated_90_seconds",
        "mean_active_people",
        "room_person_seconds",
        "junction_person_seconds",
        "corridor_person_seconds",
        "sign_switches",
        "wait_fraction_replay",
    )
    index = ["scenario_id", "training_seed"]
    rows = []
    for baseline in PREDICTION_BASELINES:
        pivots = {
            metric: primary.pivot(index=index, columns="method", values=metric) for metric in (*metrics, "evacuation_time_seconds")
        }
        time_difference = (
            pivots["evacuation_time_seconds"]["trained_llm_advice_ppo"]
            - pivots["evacuation_time_seconds"][baseline]
        ).groupby("scenario_id").mean()
        for metric in metrics:
            difference = (
                pivots[metric]["trained_llm_advice_ppo"] - pivots[metric][baseline]
            ).groupby("scenario_id").mean()
            rho, p = spearmanr(difference, time_difference)
            rows.append(
                {
                    "baseline": baseline,
                    "mechanism_metric": metric,
                    "scenario_spearman_rho_with_time_difference": float(rho),
                    "two_sided_p_descriptive": float(p),
                    "scenarios": len(difference),
                }
            )
    return pd.DataFrame(rows)


def conflict_summary(frame: pd.DataFrame) -> pd.DataFrame:
    primary = frame[(frame.behavior_model == PRIMARY_BEHAVIOR) & (frame.noise_level == PRIMARY_NOISE)]
    rows = []
    for baseline in PREDICTION_BASELINES:
        for conflict, current in primary.groupby("conflict_type"):
            pivot = current.pivot(index=["scenario_id", "training_seed"], columns="method", values=["incapacitation_rate", "mean_cumulative_fed", "evacuation_time_seconds"])
            rows.append(
                {
                    "baseline": baseline,
                    "conflict_type": conflict,
                    "scenarios": current.scenario_id.nunique(),
                    "llm_minus_baseline_incapacitation_points": float(
                        100 * (pivot[("incapacitation_rate", "trained_llm_advice_ppo")] - pivot[("incapacitation_rate", baseline)]).mean()
                    ),
                    "llm_minus_baseline_fed": float(
                        (pivot[("mean_cumulative_fed", "trained_llm_advice_ppo")] - pivot[("mean_cumulative_fed", baseline)]).mean()
                    ),
                    "llm_minus_baseline_time_seconds": float(
                        (pivot[("evacuation_time_seconds", "trained_llm_advice_ppo")] - pivot[("evacuation_time_seconds", baseline)]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["baseline", "conflict_type"])


def make_figures(effects: pd.DataFrame, prediction: pd.DataFrame) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    selected = [
        ("mean_requested_release_fraction_nonwait", "平均请求放行档位"),
        ("full_release_fraction_nonwait", "满负荷动作比例"),
        ("safe_branch_fraction_nonwait", "安全方向动作比例"),
        ("rooms_clear_seconds", "房间清空时间（秒）"),
        ("evacuated_50_seconds", "50%撤离时间（秒）"),
        ("evacuated_90_seconds", "90%撤离时间（秒）"),
    ]
    current = effects[
        (effects.behavior_model == PRIMARY_BEHAVIOR)
        & (effects.noise_level == PRIMARY_NOISE)
        & (effects.candidate == "trained_llm_advice_ppo")
        & (effects.baseline == "trained_pure_ppo")
    ].set_index("metric")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (metric, label) in zip(axes.flat, selected):
        row = current.loc[metric]
        mean, low, high = row.mean_difference, row.ci95_low, row.ci95_high
        axis.errorbar(mean, 0, xerr=[[mean - low], [high - mean]], fmt="o", capsize=5, color="#0072B2")
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks([])
        axis.set_title(label, fontproperties=font)
        axis.set_xlabel("LLM管线−纯PPO管线", fontproperties=font)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("主要条件下的动作与进程机制差异", fontproperties=font)
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_effects_zh.png", dpi=220)
    plt.close(fig)

    labels = {
        "public_observation": "公开观测",
        "full_with_llm": "公开观测＋LLM字段",
    }
    baseline_labels = {
        "trained_pure_ppo": "相对纯PPO",
        "trained_rule_advice_ppo": "相对规则PPO",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for axis, baseline in zip(axes, PREDICTION_BASELINES):
        current = prediction[prediction.baseline == baseline]
        for y, feature_set in enumerate(("public_observation", "full_with_llm")):
            row = current[current.feature_set == feature_set].iloc[0]
            axis.errorbar(
                row.roc_auc,
                y,
                xerr=[[row.roc_auc - row.roc_auc_ci95_low], [row.roc_auc_ci95_high - row.roc_auc]],
                fmt="o",
                capsize=4,
            )
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.axvline(0.6, color="#D55E00", linestyle=":", linewidth=1)
        axis.set_yticks((0, 1))
        axis.set_yticklabels([labels[key] for key in ("public_observation", "full_with_llm")], fontproperties=font)
        axis.set_xlim(0.25, 0.8)
        axis.set_xlabel("场景分组交叉验证AUC", fontproperties=font)
        axis.set_title(baseline_labels[baseline], fontproperties=font)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("能否在决策时预测LLM是否有益", fontproperties=font)
    fig.tight_layout()
    fig.savefig(OUT / "predictability_auc_zh.png", dpi=220)
    plt.close(fig)


def write_report(effects, prediction, correlations, conflicts, replay_errors) -> None:
    primary = effects[
        (effects.behavior_model == PRIMARY_BEHAVIOR)
        & (effects.noise_level == PRIMARY_NOISE)
        & (effects.candidate == "trained_llm_advice_ppo")
        & (effects.baseline == "trained_pure_ppo")
    ].set_index("metric")
    release = primary.loc["mean_requested_release_fraction_nonwait"]
    full = primary.loc["full_release_fraction_nonwait"]
    safe = primary.loc["safe_branch_fraction_nonwait"]
    room = primary.loc["rooms_clear_seconds"]
    half = primary.loc["evacuated_50_seconds"]
    ninety = primary.loc["evacuated_90_seconds"]
    active = primary.loc["mean_active_people"]
    passes = int(prediction.passes_frozen_gate.sum())
    strongest = correlations[correlations.baseline == "trained_pure_ppo"].iloc[
        correlations[correlations.baseline == "trained_pure_ppo"]["scenario_spearman_rho_with_time_difference"].abs().argmax()
    ]
    lines = [
        "# LLM＋PPO机制归因与收益可预测性报告",
        "",
        "日期：2026-10-20。状态：冻结模型在原2,880个配对单元上完成动作级重放；没有重新训练PPO，没有新增LLM请求。重放结局逐指标复核通过。",
        "",
        "> **证据等级：解封后探索性。** 本分析用于解释已经观察到的结果和生成下一轮假设，不能作为新的确认性检验。预测器只接受决策时可见的初始观测，并按场景分组交叉验证。",
        "",
        "## 1. LLM为什么更慢",
        "",
        "主要条件下，LLM管线没有等待动作，但它的动作序列、放行档位和路线分配仍可能延长尾部疏散。相对独立训练纯PPO，主要机制差异如下：",
        "",
        "|机制指标|LLM−纯PPO|95% CI|",
        "|---|---:|---:|",
        f"|平均请求放行档位|{release.mean_difference:.4f}|[{release.ci95_low:.4f}, {release.ci95_high:.4f}]|",
        f"|满负荷动作比例|{full.mean_difference:.4f}|[{full.ci95_low:.4f}, {full.ci95_high:.4f}]|",
        f"|安全方向动作比例|{safe.mean_difference:.4f}|[{safe.ci95_low:.4f}, {safe.ci95_high:.4f}]|",
        f"|房间清空时间|{room.mean_difference:.2f}秒|[{room.ci95_low:.2f}, {room.ci95_high:.2f}]|",
        f"|50%撤离时间|{half.mean_difference:.2f}秒|[{half.ci95_low:.2f}, {half.ci95_high:.2f}]|",
        f"|90%撤离时间|{ninety.mean_difference:.2f}秒|[{ninety.ci95_low:.2f}, {ninety.ci95_high:.2f}]|",
        f"|平均在场人数|{active.mean_difference:.3f}|[{active.ci95_low:.3f}, {active.ci95_high:.3f}]|",
        "",
        f"与总疏散时间差关联最强的场景级机制变量是`{strongest.mechanism_metric}`，Spearman ρ={strongest.scenario_spearman_rho_with_time_difference:.3f}。该相关是解释性线索，不表示单一因果机制。",
        "",
        "![机制效应](../outputs/final_mechanism_attribution_20261020/mechanism_effects_zh.png)",
        "",
        "## 2. 能否提前识别LLM有益场景",
        "",
        "收益标签按安全优先定义：先比较五个训练种子平均失能率；失能率相同时再比较FED。使用固定L2逻辑回归，八折交叉验证严格按48个场景分组。",
        "",
        "|比较基线|特征|有益比例|平衡准确率|AUC及95% CI|Brier|相对频率基线改善|通过冻结门槛|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"public_observation": "公开观测", "full_with_llm": "公开观测＋LLM字段"}
    baselines = {"trained_pure_ppo": "纯PPO", "trained_rule_advice_ppo": "规则PPO"}
    for _, row in prediction.iterrows():
        lines.append(
            f"|{baselines[row.baseline]}|{labels[row.feature_set]}|{100 * row.beneficial_prevalence:.1f}%|{row.balanced_accuracy:.3f}|{row.roc_auc:.3f} [{row.roc_auc_ci95_low:.3f}, {row.roc_auc_ci95_high:.3f}]|{row.brier:.3f}|{row.brier_improvement_percent:.1f}%|{'是' if row.passes_frozen_gate else '否'}|"
        )
    lines.extend(
        [
            "",
            f"冻结判据要求两种比较都达到AUC区间下限>0.60且Brier较频率基线改善至少5%。四个模型中通过数为{passes}/4。若完整LLM字段仍不能通过，当前数据不足以支持可靠的实时收益门控。",
            "",
            "![收益可预测性](../outputs/final_mechanism_attribution_20261020/predictability_auc_zh.png)",
            "",
            "## 3. 按冲突类型的探索性结果",
            "",
            "|比较基线|冲突类型|场景数|失能率差（百分点）|FED差|时间差（秒）|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in conflicts.iterrows():
        lines.append(
            f"|{baselines[row.baseline]}|{row.conflict_type}|{int(row.scenarios)}|{row.llm_minus_baseline_incapacitation_points:.4f}|{row.llm_minus_baseline_fed:.4f}|{row.llm_minus_baseline_time_seconds:.2f}|"
        )
    lines.extend(
        [
            "",
            "冲突类型子组很小，只用于提出下一轮假设。不能从其中挑选最有利子组宣称LLM有效。",
            "",
            "## 4. 复核与下一步判据",
            "",
            f"重放与原结果的最大绝对误差为{max(replay_errors.values()):.3g}，无截断回合。完整诊断效应、场景级相关和逐折预测均保存为CSV。",
            "",
            "若门控未通过，应停止在这48个场景上反复设计新门控；将本轮最明确的机制假设带回历史开发集，冻结规则后再生成全新的场景银行验证。若机制差异本身也不稳定，则应结束LLM实时控制路线，把LLM限定为离线候选生成和结果解释。",
            "",
            "## 5. 可复核文件",
            "",
            "- 冻结协议：`outputs/final_mechanism_attribution_20261020/protocol_frozen.json`",
            "- 动作级重放：`outputs/final_mechanism_attribution_20261020/replay_diagnostics.csv`",
            "- 机制配对效应：`outputs/final_mechanism_attribution_20261020/mechanism_effects.csv`",
            "- 场景收益数据：`outputs/final_mechanism_attribution_20261020/prediction_dataset.csv`",
            "- 交叉验证预测：`outputs/final_mechanism_attribution_20261020/grouped_cv_predictions.csv`",
            "- 可预测性指标：`outputs/final_mechanism_attribution_20261020/predictability_metrics.csv`",
            "- 完整性核查：`outputs/final_mechanism_attribution_20261020/verification.json`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    frame = run_replays()
    replay_errors = validate_replay(frame)
    effects = mechanism_effects(frame)
    dataset, predictions, predictability = prediction_table(frame)
    correlations = mechanism_correlations(frame)
    conflicts = conflict_summary(frame)
    write_csv(OUT / "mechanism_effects.csv", effects.to_dict("records"))
    write_csv(OUT / "prediction_dataset.csv", dataset.to_dict("records"))
    write_csv(OUT / "grouped_cv_predictions.csv", predictions.to_dict("records"))
    write_csv(OUT / "predictability_metrics.csv", predictability.to_dict("records"))
    write_csv(OUT / "mechanism_time_correlations.csv", correlations.to_dict("records"))
    write_csv(OUT / "conflict_type_summary.csv", conflicts.to_dict("records"))
    make_figures(effects, predictability)
    write_report(effects, predictability, correlations, conflicts, replay_errors)
    verify_frozen()
    verification = {
        "status": "passed",
        "analysis_status": "exploratory_post_unblinding",
        "replay_rows": len(frame),
        "unique_run_ids": int(frame.run_id.nunique()),
        "scenarios": int(frame.scenario_id.nunique()),
        "training_seeds": int(frame.training_seed.nunique()),
        "truncated_episodes": int(frame.truncated.sum()),
        "replay_outcome_max_abs_errors": replay_errors,
        "prediction_rows": len(dataset),
        "prediction_unique_scenarios": int(dataset.scenario_id.nunique()),
        "group_leakage": False,
        "predictors_passing_frozen_gate": int(predictability.passes_frozen_gate.sum()),
        "new_ppo_training": 0,
        "new_llm_requests": 0,
        "verified_at_utc": stamp(),
    }
    write_json(OUT / "verification.json", verification)
    write_json(
        OUT / "run_status.json",
        {
            "state": "complete",
            "replays_completed": len(frame),
            "report": relative(REPORT),
            "new_ppo_training": 0,
            "new_llm_requests": 0,
        },
    )
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest.json"] + [REPORT]
    write_json(
        OUT / "completion_manifest.json",
        {"status": "complete", "files": {relative(path): sha256(path) for path in sorted(files)}},
    )
    print("Mechanism attribution and grouped predictability audit complete.")


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
