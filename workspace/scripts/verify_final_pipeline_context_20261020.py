from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_pipeline_context_20261020"
BLIND = ROOT / "outputs" / "final_blind_evaluation_20261020"
REPORT = ROOT / "reports" / "LLM_PPO完整管线补充对照_2026-10-20.md"
SCENARIOS = tuple(f"SCF{i:04d}" for i in range(1, 49))
SEEDS = tuple(range(601, 606))
BEHAVIORS = ("calibrated_logit", "mixed_logit")
NOISES = ("none", "medium")
METHODS = ("trained_pure_ppo", "trained_rule_advice_ppo", "trained_llm_advice_ppo")
COMPARISONS = (
    ("trained_llm_advice_ppo", "trained_pure_ppo"),
    ("trained_llm_advice_ppo", "trained_rule_advice_ppo"),
    ("trained_rule_advice_ppo", "trained_pure_ppo"),
)
METRICS = (
    "incapacitation_rate",
    "mean_cumulative_fed",
    "evacuation_time_seconds",
    "max_congestion",
    "cumulative_congestion",
)
BOOTSTRAP_SEED = 2026102012
BOOTSTRAP_SAMPLES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def crossed_bootstrap(matrix: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        scenario_index = rng.integers(0, 48, 48)
        training_index = rng.integers(0, 5, 5)
        values[index] = matrix[np.ix_(scenario_index, training_index)].mean()
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def matrix(frame, behavior, noise, candidate, baseline, metric):
    subset = frame[(frame.behavior_model == behavior) & (frame.noise_level == noise)]
    pivot = subset.pivot(index=["scenario_id", "training_seed"], columns="method", values=metric)
    result = (pivot[candidate] - pivot[baseline]).unstack("training_seed")
    return result.reindex(index=SCENARIOS, columns=SEEDS).to_numpy(float)


def redraw(effects: pd.DataFrame) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    current = effects[
        (effects.behavior_model == "calibrated_logit")
        & (effects.noise_level == "none")
        & effects.metric.isin(("incapacitation_rate", "mean_cumulative_fed"))
    ]
    labels = {
        ("trained_llm_advice_ppo", "trained_pure_ppo"): "LLM管线−纯PPO管线",
        ("trained_llm_advice_ppo", "trained_rule_advice_ppo"): "LLM管线−规则管线",
        ("trained_rule_advice_ppo", "trained_pure_ppo"): "规则管线−纯PPO管线",
    }
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, metric, title in zip(
        axes,
        ("incapacitation_rate", "mean_cumulative_fed"),
        ("失能率差（百分点）", "人均累计FED差"),
    ):
        rows = current[current.metric == metric]
        for y, ((candidate, baseline), color) in enumerate(zip(COMPARISONS, colors)):
            row = rows[(rows.candidate == candidate) & (rows.baseline == baseline)].iloc[0]
            scale = 100 if metric == "incapacitation_rate" else 1
            mean, low, high = row.mean_difference * scale, row.ci95_low * scale, row.ci95_high * scale
            axis.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=4)
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(range(3))
        axis.set_yticklabels([labels[item] for item in COMPARISONS], fontproperties=font)
        axis.set_title(title, fontproperties=font)
        axis.set_xlabel("负值表示前一管线指标更低；95%交叉bootstrap区间", fontproperties=font)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("完整训练管线补充对照（解封后探索性）", fontproperties=font)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_primary_effects_zh.png", dpi=220)
    plt.close(fig)


def main() -> None:
    freeze_mismatches = []
    for name, expected in read(OUT / "freeze_manifest.json").items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            freeze_mismatches.append(name)
    parent_mismatches = []
    for name, expected in read(OUT / "protected_parent_hashes.json").items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            parent_mismatches.append(name)
    if freeze_mismatches or parent_mismatches:
        raise RuntimeError({"freeze_mismatches": freeze_mismatches, "parent_mismatches": parent_mismatches})

    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    new = pd.read_csv(OUT / "new_episode_results.csv")
    combined = pd.read_csv(OUT / "combined_episode_results.csv")
    original = pd.read_csv(BLIND / "episode_results.csv")
    stored = pd.read_csv(OUT / "paired_effects.csv")
    if len(schedule) != 1_920 or schedule.run_id.nunique() != 1_920:
        raise RuntimeError("schedule count/identity failed")
    if set(schedule.run_id) != set(new.run_id):
        raise RuntimeError("new result IDs differ from frozen schedule")
    joined = schedule.merge(
        new[["run_id", "scenario_id", "training_seed", "behavior_model", "noise_level", "method", "population_seed"]],
        on="run_id",
        suffixes=("_plan", "_result"),
        validate="one_to_one",
    )
    for column in ("scenario_id", "training_seed", "behavior_model", "noise_level", "method", "population_seed"):
        if not (joined[f"{column}_plan"].astype(str) == joined[f"{column}_result"].astype(str)).all():
            raise RuntimeError(f"schedule field differs: {column}")
    if len(combined) != 2_880 or combined.run_id.nunique() != 2_880:
        raise RuntimeError("combined result count/identity failed")
    cells = combined.groupby(["scenario_id", "training_seed", "behavior_model", "noise_level"])
    if cells.ngroups != 960 or not (cells.method.nunique() == 3).all():
        raise RuntimeError("combined pairing failed")

    original_llm = original[original.method == "llm_advice_ppo"].copy()
    combined_llm = combined[combined.method == "trained_llm_advice_ppo"].copy()
    key = ["scenario_id", "training_seed", "behavior_model", "noise_level"]
    reused = combined_llm.merge(original_llm, on=key, suffixes=("_combined", "_original"), validate="one_to_one")
    reuse_columns = [
        "incapacitation_rate",
        "mean_cumulative_fed",
        "evacuation_time_seconds",
        "max_congestion",
        "cumulative_congestion",
        "evacuation_rate",
        "remaining_rate",
    ]
    for column in reuse_columns:
        if not np.allclose(
            reused[f"{column}_combined"].to_numpy(float),
            reused[f"{column}_original"].to_numpy(float),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"reused LLM outcome differs: {column}")

    recomputed = []
    counter = 0
    for behavior in BEHAVIORS:
        for noise in NOISES:
            for candidate, baseline in COMPARISONS:
                for metric in METRICS:
                    values = matrix(combined, behavior, noise, candidate, baseline, metric)
                    if values.shape != (48, 5) or np.isnan(values).any():
                        raise RuntimeError("paired matrix invalid")
                    mean = float(values.mean())
                    low, high = crossed_bootstrap(values, BOOTSTRAP_SEED + counter)
                    row = stored[
                        (stored.behavior_model == behavior)
                        & (stored.noise_level == noise)
                        & (stored.candidate == candidate)
                        & (stored.baseline == baseline)
                        & (stored.metric == metric)
                    ]
                    if len(row) != 1:
                        raise RuntimeError("stored effect missing or duplicated")
                    row = row.iloc[0]
                    if not np.allclose(
                        [mean, low, high],
                        [row.mean_difference, row.ci95_low, row.ci95_high],
                        rtol=0,
                        atol=1e-14,
                    ):
                        raise RuntimeError("effect recomputation mismatch")
                    if behavior == "calibrated_logit" and noise == "none" and metric in METRICS[:2]:
                        recomputed.append(
                            {
                                "candidate": candidate,
                                "baseline": baseline,
                                "metric": metric,
                                "mean_difference": mean,
                                "ci95_low": low,
                                "ci95_high": high,
                            }
                        )
                    counter += 1

    conservation = combined.evacuation_rate + combined.incapacitation_rate + combined.remaining_rate
    if not np.allclose(conservation, 1, rtol=0, atol=1e-12) or int(combined.truncated.sum()) != 0:
        raise RuntimeError("outcome integrity failed")
    redraw(stored)
    audit = {
        "status": "passed",
        "analysis_status": "exploratory_post_unblinding",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_schedule_rows": len(schedule),
        "new_result_rows": len(new),
        "combined_rows": len(combined),
        "paired_blocks": cells.ngroups,
        "reused_llm_rows_exactly_matched": len(reused),
        "freeze_mismatches": freeze_mismatches,
        "protected_parent_mismatches": parent_mismatches,
        "truncated_episodes": int(combined.truncated.sum()),
        "population_conservation_max_abs_error": float(np.abs(conservation - 1).max()),
        "all_60_effect_rows_exactly_recomputed": True,
        "primary_effects": recomputed,
        "interpretation": "Whole-pipeline context only; protocol was frozen after the shared-controller results were visible.",
    }
    write(OUT / "independent_verification.json", audit)
    paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest_v2.json"]
    paths.append(REPORT)
    write(
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
    print("Independent pipeline verification passed; 60 effect rows reproduced exactly.")


if __name__ == "__main__":
    main()
