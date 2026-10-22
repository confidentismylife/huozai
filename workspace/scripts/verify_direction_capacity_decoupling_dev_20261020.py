from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "direction_capacity_decoupling_dev_20261020_v2"
FAILED_V1 = ROOT / "outputs" / "direction_capacity_decoupling_dev_20261020"
SOURCE_RESULTS = ROOT / "outputs" / "guarded_confirmatory_v2" / "final_episode_results.csv"
SCENARIO_FILE = ROOT / "outputs" / "guarded_confirmatory_v2" / "final_scenarios.json"
REPORT = ROOT / "reports" / "方向容量解耦开发实验报告_2026-10-20.md"
SEEDS = tuple(range(601, 606))
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
)
BOOTSTRAP_SEED = 2026102032
BOOTSTRAP_SAMPLES = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def paired_matrix(frame, scenario_order, candidate, baseline, metric):
    averaged = (
        frame.groupby(["scenario_id", "training_seed", "method"], as_index=False)[metric]
        .mean()
        .pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    )
    result = (averaged[candidate] - averaged[baseline]).unstack("training_seed")
    return result.reindex(index=scenario_order, columns=SEEDS).to_numpy(float)


def crossed_bootstrap(matrix: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenarios = rng.integers(0, 60, 60)
        seeds = rng.integers(0, 5, 5)
        values[index] = matrix[np.ix_(scenarios, seeds)].mean()
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


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

    failed_status = read_json(FAILED_V1 / "run_status.json")
    failed_result_files = [path.name for path in FAILED_V1.glob("*episode_results.csv")]
    if failed_status.get("new_rollouts_completed") != 0 or failed_result_files:
        raise RuntimeError("v1 was not a pre-outcome failure")

    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    hybrids = pd.read_csv(OUT / "hybrid_episode_results.csv")
    combined = pd.read_csv(OUT / "combined_episode_results.csv")
    if len(schedule) != 4_500 or len(hybrids) != 4_500 or set(schedule.run_id) != set(hybrids.run_id):
        raise RuntimeError("new rollout plan/result mismatch")
    if len(combined) != 7_500 or combined.run_id.nunique() != 7_500:
        raise RuntimeError("combined matrix mismatch")
    keys = ["scenario_id", "population_seed", "training_seed"]
    cell_counts = combined.groupby(keys).method.nunique()
    if len(cell_counts) != 1_500 or not (cell_counts == 5).all():
        raise RuntimeError("five-method pairing mismatch")

    source = pd.read_csv(SOURCE_RESULTS)
    source = source[
        (source.test_behavior == "calibrated_logit")
        & (source.noise_level == "none")
        & source.arm.isin(("pure_ppo", "raw_llm_ppo"))
    ].copy()
    names = {"pure_ppo": "trained_pure_ppo", "raw_llm_ppo": "trained_llm_joint"}
    source["method"] = source.arm.map(names)
    reused = combined[combined.method.isin(names.values())]
    match = reused.merge(
        source,
        on=["scenario_id", "population_seed", "training_seed", "method"],
        suffixes=("_combined", "_source"),
        validate="one_to_one",
    )
    if len(match) != 3_000:
        raise RuntimeError("baseline reuse count failed")
    for metric in ("steps", "evacuation_rate", "incapacitation_rate", "mean_cumulative_fed", "cumulative_congestion"):
        if not np.allclose(
            match[f"{metric}_combined"].to_numpy(float),
            match[f"{metric}_source"].to_numpy(float),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"baseline reuse mismatch: {metric}")

    conservation = combined.evacuation_rate + combined.incapacitation_rate + combined.remaining_rate
    if not np.allclose(conservation, 1.0, rtol=0, atol=1e-12) or int(combined.truncated.sum()) != 0:
        raise RuntimeError("outcome integrity failed")
    scenarios = [record["scenario_id"] for record in read_json(SCENARIO_FILE)]
    stored = pd.read_csv(OUT / "paired_effects.csv")
    recomputed = []
    counter = 0
    for candidate, baseline in COMPARISONS:
        for metric in METRICS:
            matrix = paired_matrix(combined, scenarios, candidate, baseline, metric)
            low, high = crossed_bootstrap(matrix, BOOTSTRAP_SEED + counter)
            row = stored[
                (stored.candidate == candidate)
                & (stored.baseline == baseline)
                & (stored.metric == metric)
            ].iloc[0]
            values = [float(matrix.mean()), low, high]
            expected = [float(row.mean_difference), float(row.ci95_low), float(row.ci95_high)]
            if not np.allclose(values, expected, rtol=0, atol=1e-14):
                raise RuntimeError(f"effect mismatch: {candidate}/{baseline}/{metric}")
            if candidate == "llm_policy_direction_pure_capacity" and baseline == "trained_llm_joint":
                recomputed.append(
                    {
                        "metric": metric,
                        "mean_difference": values[0],
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
            counter += 1

    primary = {row["metric"]: row for row in recomputed}
    gate = {
        "time_ci_upper_below_zero": primary["evacuation_time_seconds"]["ci95_high"] < 0,
        "incapacitation_ci_upper_below_0_005": primary["incapacitation_rate"]["ci95_high"] < 0.005,
        "fed_ci_upper_below_0_01": primary["mean_cumulative_fed"]["ci95_high"] < 0.01,
    }
    gate["all_pass"] = all(gate.values())
    if gate != read_json(OUT / "development_gate.json") or gate["all_pass"]:
        raise RuntimeError("development gate mismatch")

    audit = {
        "status": "passed",
        "analysis_status": "development_hypothesis_test",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "failed_v1_rollouts": 0,
        "failed_v1_result_files": failed_result_files,
        "new_hybrid_rows": len(hybrids),
        "reused_baseline_rows_exactly_matched": len(match),
        "combined_rows": len(combined),
        "paired_cells": len(cell_counts),
        "freeze_mismatches": freeze_mismatches,
        "protected_parent_mismatches": parent_mismatches,
        "population_conservation_max_abs_error": float(np.abs(conservation - 1).max()),
        "truncated_episodes": int(combined.truncated.sum()),
        "all_25_effect_rows_exactly_recomputed": True,
        "primary_effects": recomputed,
        "development_gate": gate,
        "scf_final_scenarios_used": 0,
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
    print("Independent direction-capacity development verification passed.")


if __name__ == "__main__":
    main()
