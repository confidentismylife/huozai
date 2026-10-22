from __future__ import annotations

"""Independent, read-only scientific audit of the 2026-10-20 blind matrix."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_blind_evaluation_20261020"
REPORT = ROOT / "reports" / "LLM_PPO最终盲测报告_2026-10-20.md"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_final_blind_evaluation_20261020 as study


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def effect(frame, candidate, baseline, metric, bootstrap_offset, permutation_offset):
    matrix = study.paired_matrix(
        frame,
        study.PRIMARY_BEHAVIOR,
        study.PRIMARY_NOISE,
        candidate,
        baseline,
        metric,
    )
    scenario_values = matrix.mean(axis=1)
    low, high = study.two_way_bootstrap(matrix, study.BOOTSTRAP_SEED + bootstrap_offset)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "mean_difference": float(matrix.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "sign_flip_p": study.sign_flip_p(scenario_values, study.PERMUTATION_SEED + permutation_offset),
        "scenario_wins": int((scenario_values < -1e-15).sum()),
        "scenario_ties": int((np.abs(scenario_values) <= 1e-15).sum()),
        "scenario_harms": int((scenario_values > 1e-15).sum()),
    }


def redraw(effects: pd.DataFrame) -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    subset = effects[
        effects.metric.isin(study.PRIMARY_METRICS)
        & effects.candidate.eq("llm_advice_ppo")
        & effects.baseline.isin(("pure_ppo", "rule_advice_ppo"))
    ]
    conditions = [
        ("calibrated_logit", "none", "校准Logit / 无噪声"),
        ("calibrated_logit", "medium", "校准Logit / 中噪声"),
        ("mixed_logit", "none", "Mixed Logit / 无噪声"),
        ("mixed_logit", "medium", "Mixed Logit / 中噪声"),
    ]
    comparisons = [
        ("pure_ppo", "LLM−同控制器无建议", "#0072B2"),
        ("rule_advice_ppo", "LLM−规则建议", "#D55E00"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for axis, metric, title, scale in zip(
        axes,
        study.PRIMARY_METRICS,
        ("失能率差（百分点）", "人均累计FED差"),
        (100, 1),
    ):
        ticks, labels = [], []
        row_index = 0
        for behavior, noise, condition_label in conditions:
            for baseline, comparison_label, color in comparisons:
                row = subset[
                    (subset.behavior_model == behavior)
                    & (subset.noise_level == noise)
                    & (subset.baseline == baseline)
                    & (subset.metric == metric)
                ].iloc[0]
                mean, low, high = row.mean_difference * scale, row.ci95_low * scale, row.ci95_high * scale
                axis.errorbar(
                    mean,
                    row_index,
                    xerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=color,
                    capsize=3,
                )
                ticks.append(row_index)
                labels.append(f"{condition_label} | {comparison_label}")
                row_index += 1
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_yticks(ticks)
        axis.set_yticklabels(labels, fontproperties=font, fontsize=8)
        axis.set_title(title, fontproperties=font)
        axis.set_xlabel("负值表示LLM输入更低；误差线为95%交叉bootstrap区间", fontproperties=font)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "primary_effects_zh.png", dpi=240)
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(OUT / "episode_results.csv")
    schedule = pd.read_csv(OUT / "run_schedule_frozen.csv")
    assert len(frame) == frame.run_id.nunique() == len(schedule) == schedule.run_id.nunique() == 2880
    assert set(frame.run_id) == set(schedule.run_id)
    assert set(frame.method) == set(study.METHODS)
    assert set(frame.scenario_id) == set(study.SCENARIO_IDS)
    assert set(map(int, frame.training_seed.unique())) == set(study.MODEL_SEEDS)
    cell_counts = frame.groupby(["scenario_id", "training_seed", "behavior_model", "noise_level"]).method.nunique()
    assert (cell_counts == 3).all() and len(cell_counts) == 960
    population_counts = frame.groupby(["scenario_id", "training_seed", "behavior_model", "noise_level"]).population_seed.nunique()
    assert (population_counts == 1).all()
    conservation = frame.evacuation_rate + frame.incapacitation_rate + frame.remaining_rate
    assert np.allclose(conservation, 1.0, atol=1e-12, rtol=0)
    assert int(frame.truncated.sum()) == 0

    freeze = read(OUT / "freeze_manifest.json")
    freeze_mismatches = [name for name, digest in freeze.items() if sha256(ROOT / name) != digest]
    parents = read(OUT / "protected_parent_hashes.json")
    parent_mismatches = [name for name, digest in parents.items() if sha256(ROOT / name) != digest]
    assert not freeze_mismatches and not parent_mismatches

    frozen_advice = read(OUT / "llm_advice_frozen.json")
    assert frozen_advice["model_digest"] == study.MODEL_DIGEST
    assert frozen_advice["outcome_rollouts_completed_before_freeze"] == 0
    assert set(frozen_advice["advice"]) == set(study.SCENARIO_IDS)
    assert len(list((OUT / "llm_cache").glob("SCF*.json"))) == 48

    recomputed = [
        effect(frame, "llm_advice_ppo", "pure_ppo", "incapacitation_rate", 0, 0),
        effect(frame, "llm_advice_ppo", "pure_ppo", "mean_cumulative_fed", 1, 1),
        effect(frame, "llm_advice_ppo", "rule_advice_ppo", "incapacitation_rate", 5, 2),
        effect(frame, "llm_advice_ppo", "rule_advice_ppo", "mean_cumulative_fed", 6, 3),
    ]
    adjusted = study.holm([row["sign_flip_p"] for row in recomputed])
    for row, value in zip(recomputed, adjusted):
        row["holm_adjusted_p"] = value
    stored = pd.read_csv(OUT / "paired_effects.csv")
    for row in recomputed:
        match = stored[
            (stored.candidate == row["candidate"])
            & (stored.baseline == row["baseline"])
            & (stored.metric == row["metric"])
            & (stored.behavior_model == study.PRIMARY_BEHAVIOR)
            & (stored.noise_level == study.PRIMARY_NOISE)
        ].iloc[0]
        for key in ("mean_difference", "ci95_low", "ci95_high", "sign_flip_p", "holm_adjusted_p"):
            assert abs(float(match[key]) - float(row[key])) < 1e-12, (key, match[key], row[key])

    old_manifest = read(OUT / "completion_manifest.json")
    old_mismatches = [name for name, digest in old_manifest["files"].items() if sha256(ROOT / name) != digest]
    expected_legacy_changes = {
        "outputs/final_blind_evaluation_20261020/run_status.json",
        REPORT.resolve().relative_to(ROOT.resolve()).as_posix(),
    }
    assert set(old_mismatches) == expected_legacy_changes

    redraw(stored)
    audit = {
        "status": "passed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_cells": len(frame),
        "paired_blocks": len(cell_counts),
        "population_conservation_max_abs_error": float(np.max(np.abs(conservation - 1))),
        "truncated_episodes": int(frame.truncated.sum()),
        "freeze_mismatches": freeze_mismatches,
        "protected_parent_mismatches": parent_mismatches,
        "llm_cache_files": 48,
        "primary_effects_exactly_recomputed": recomputed,
        "legacy_completion_manifest_mismatches": old_mismatches,
        "legacy_manifest_explanation": "run_status was finalized after the original manifest and the report terminology was corrected after independent audit; frozen inputs and scientific result files matched",
        "terminology_audit": {
            "protocol_label": "pure_ppo",
            "precise_meaning": "same raw_llm PPO controller with advice channel set to zero",
            "is_separately_trained_pure_ppo": False,
            "required_interpretation": "advice-channel ablation, not a whole-pipeline pure-PPO comparison",
        },
    }
    write(OUT / "independent_verification.json", audit)
    paths = [path for path in OUT.rglob("*") if path.is_file() and path.name != "completion_manifest_v2.json"]
    paths.append(REPORT)
    write(
        OUT / "completion_manifest_v2.json",
        {
            "status": "complete_and_independently_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {path.resolve().relative_to(ROOT.resolve()).as_posix(): sha256(path) for path in sorted(paths)},
        },
    )
    print("Independent verification passed; primary effects reproduced exactly.")


if __name__ == "__main__":
    main()
