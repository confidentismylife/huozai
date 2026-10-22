from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_mechanism_attribution_20261020"
SOURCE = ROOT / "outputs" / "final_pipeline_context_20261020" / "combined_episode_results.csv"
REPORT = ROOT / "reports" / "LLM_PPO机制归因与可预测性报告_2026-10-20.md"
SCENARIOS = tuple(f"SCF{i:04d}" for i in range(1, 49))
SEEDS = tuple(range(601, 606))
BOOTSTRAP_SEED = 2026102022
CV_SEED = 2026102023
BOOTSTRAP_SAMPLES = 5_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def paired_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
    subset = frame[(frame.behavior_model == "calibrated_logit") & (frame.noise_level == "none")]
    pivot = subset.pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    difference = (
        pivot["trained_llm_advice_ppo"] - pivot["trained_pure_ppo"]
    ).unstack("training_seed")
    return difference.reindex(index=SCENARIOS, columns=SEEDS).to_numpy(float)


def crossed_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenarios = rng.integers(0, 48, 48)
        seeds = rng.integers(0, 5, 5)
        estimates[index] = matrix[np.ix_(scenarios, seeds)].mean()
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    positive, negative = y == 1, y == 0
    sensitivity = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    ranks = rankdata(probability, method="average")
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    auc = float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    brier = float(np.mean((probability - y) ** 2))
    edges = np.linspace(0, 1, 6)
    ece = 0.0
    for index in range(5):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 4 else probability < edges[index + 1]
        )
        if selected.any():
            ece += float(selected.mean()) * abs(float(probability[selected].mean() - y[selected].mean()))
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": auc,
        "brier": brier,
        "ece_5bin": ece,
    }


def cluster_bootstrap(y, probability, groups, seed: int):
    unique = np.unique(groups)
    index = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    stored = {key: [] for key in ("balanced_accuracy", "roc_auc", "brier", "ece_5bin")}
    for _ in range(BOOTSTRAP_SAMPLES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([index[group] for group in selected_groups])
        if len(np.unique(y[selected])) < 2:
            continue
        values = binary_metrics(y[selected], probability[selected])
        for key in stored:
            stored[key].append(values[key])
    return {key: tuple(float(value) for value in np.quantile(values, (0.025, 0.975))) for key, values in stored.items()}


def main() -> None:
    freeze_mismatches = [
        name
        for name, digest in read_json(OUT / "freeze_manifest.json").items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    parent_mismatches = [
        name
        for name, digest in read_json(OUT / "protected_parent_hashes.json").items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    if freeze_mismatches or parent_mismatches:
        raise RuntimeError({"freeze": freeze_mismatches, "parents": parent_mismatches})

    schedule = pd.read_csv(OUT / "replay_schedule_frozen.csv")
    replay = pd.read_csv(OUT / "replay_diagnostics.csv")
    source = pd.read_csv(SOURCE)
    if len(schedule) != 2_880 or len(replay) != 2_880:
        raise RuntimeError("replay count failed")
    if set(schedule.run_id) != set(replay.run_id) or replay.run_id.nunique() != 2_880:
        raise RuntimeError("replay identity failed")
    outcome_metrics = (
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
    merged = replay[["run_id", *outcome_metrics]].merge(
        source[["run_id", *outcome_metrics]], on="run_id", suffixes=("_replay", "_source"), validate="one_to_one"
    )
    replay_errors = {}
    for metric in outcome_metrics:
        error = float(
            np.max(
                np.abs(
                    merged[f"{metric}_replay"].to_numpy(float)
                    - merged[f"{metric}_source"].to_numpy(float)
                )
            )
        )
        replay_errors[metric] = error
        if error > 1e-12:
            raise RuntimeError(f"outcome mismatch: {metric}")

    stored_effects = pd.read_csv(OUT / "mechanism_effects.csv")
    diagnostic_order = read_json(OUT / "analysis_plan_frozen.json")["diagnostic_metrics"]
    selected_metrics = (
        "mean_requested_release_fraction_nonwait",
        "full_release_fraction_nonwait",
        "safe_branch_fraction_nonwait",
        "rooms_clear_seconds",
        "evacuated_50_seconds",
        "evacuated_90_seconds",
    )
    recomputed_effects = []
    for metric in selected_metrics:
        values = paired_matrix(replay, metric)
        counter = diagnostic_order.index(metric)
        low, high = crossed_bootstrap(values, BOOTSTRAP_SEED + counter)
        row = stored_effects[
            (stored_effects.behavior_model == "calibrated_logit")
            & (stored_effects.noise_level == "none")
            & (stored_effects.candidate == "trained_llm_advice_ppo")
            & (stored_effects.baseline == "trained_pure_ppo")
            & (stored_effects.metric == metric)
        ].iloc[0]
        current = [float(values.mean()), low, high]
        expected = [float(row.mean_difference), float(row.ci95_low), float(row.ci95_high)]
        if not np.allclose(current, expected, rtol=0, atol=1e-14):
            raise RuntimeError(f"mechanism effect mismatch: {metric}")
        recomputed_effects.append(
            {"metric": metric, "mean_difference": current[0], "ci95_low": low, "ci95_high": high}
        )

    predictions = pd.read_csv(OUT / "grouped_cv_predictions.csv")
    metrics = pd.read_csv(OUT / "predictability_metrics.csv")
    if len(predictions) != 768:
        raise RuntimeError("prediction row count failed")
    grouped = predictions.groupby(["baseline", "feature_set", "scenario_id"]).fold.nunique()
    if not (grouped == 1).all():
        raise RuntimeError("scenario crossed validation folds")
    recomputed_predictions = []
    order = [
        ("trained_pure_ppo", "public_observation"),
        ("trained_pure_ppo", "full_with_llm"),
        ("trained_rule_advice_ppo", "public_observation"),
        ("trained_rule_advice_ppo", "full_with_llm"),
    ]
    for counter, (baseline, feature_set) in enumerate(order):
        current = predictions[(predictions.baseline == baseline) & (predictions.feature_set == feature_set)]
        y = current.observed_beneficial.to_numpy(int)
        probability = current.predicted_probability.to_numpy(float)
        groups = current.scenario_id.to_numpy(str)
        values = binary_metrics(y, probability)
        intervals = cluster_bootstrap(y, probability, groups, CV_SEED + counter)
        row = metrics[(metrics.baseline == baseline) & (metrics.feature_set == feature_set)].iloc[0]
        for key in values:
            if abs(values[key] - float(row[key])) > 1e-12:
                raise RuntimeError(f"prediction metric mismatch: {baseline}/{feature_set}/{key}")
        for key, (low, high) in intervals.items():
            if abs(low - float(row[f"{key}_ci95_low"])) > 1e-12 or abs(high - float(row[f"{key}_ci95_high"])) > 1e-12:
                raise RuntimeError(f"prediction interval mismatch: {baseline}/{feature_set}/{key}")
        recomputed_predictions.append(
            {
                "baseline": baseline,
                "feature_set": feature_set,
                **values,
                "roc_auc_ci95_low": intervals["roc_auc"][0],
                "roc_auc_ci95_high": intervals["roc_auc"][1],
                "passes_frozen_gate": int(row.passes_frozen_gate),
            }
        )
    if int(metrics.passes_frozen_gate.sum()) != 0:
        raise RuntimeError("unexpected gate pass")

    audit = {
        "status": "passed",
        "analysis_status": "exploratory_post_unblinding",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "schedule_rows": len(schedule),
        "replay_rows": len(replay),
        "replay_outcome_max_abs_errors": replay_errors,
        "freeze_mismatches": freeze_mismatches,
        "protected_parent_mismatches": parent_mismatches,
        "selected_mechanism_effects_exactly_recomputed": recomputed_effects,
        "all_four_predictive_metric_sets_exactly_recomputed": recomputed_predictions,
        "scenario_group_leakage": False,
        "predictors_passing_frozen_gate": 0,
    }
    write_json(OUT / "independent_verification.json", audit)
    paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest_v2.json"]
    paths.append(REPORT)
    write_json(
        OUT / "completion_manifest_v2.json",
        {
            "status": "complete_and_independently_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {
                path.resolve().relative_to(ROOT.resolve()).as_posix(): sha256(path)
                for path in sorted(paths)
            },
        },
    )
    print("Independent mechanism and predictability verification passed.")


if __name__ == "__main__":
    main()
