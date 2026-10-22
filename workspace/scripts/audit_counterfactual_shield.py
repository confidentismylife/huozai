"""Audit a preserved development checkpoint without retraining or changing labels."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.counterfactual_shield import (
    ShieldPredictor, action_mask_metrics, effective_action_mask,
    evaluate_counterfactual_actions, material_safety_labels,
    mirror_action_values, mirror_shield_observation, reactive_followup_action,
)
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def meets(m):
    return (m["unsafe_action_recall"] is not None and m["unsafe_action_recall"] >= .95
            and m["safe_action_false_rejection_rate"] is not None and m["safe_action_false_rejection_rate"] <= .10
            and m["top_action_unsafe_rate"] <= .05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/conditional_shield_development")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/conditional_shield_audit_v1")
    parser.add_argument("--scenarios-per-conflict", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new audit output; existing results are preserved")
    args.output.mkdir(parents=True)
    checkpoint = args.source / "shield_predictor.pt"
    # Explicit migration of this project's own locally generated NumPy checkpoint.
    original = ShieldPredictor.load(checkpoint, trusted_legacy=True)
    original.save(args.output / "shield_predictor.pt")
    predictor = ShieldPredictor.load(args.output / "shield_predictor.pt")
    data = np.load(args.source / "counterfactual_shield_dataset.npz", allow_pickle=False)
    x = data["observations"]
    y = data["safe_masks"]
    validation = data["splits"] == "validation"
    with (args.source / "counterfactual_action_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != len(x) * 9:
        raise ValueError("CSV and NPZ state counts disagree")
    for offset in range(0, len(records), 9):
        group = records[offset:offset + 9]
        if [int(r["action"]) for r in group] != list(range(9)):
            raise ValueError("CSV state alignment requires ordered nine-action groups")
        for key in ("scenario_id", "population_seed", "step", "split"):
            if len({r[key] for r in group}) != 1:
                raise ValueError("CSV state group is inconsistent")
        if group[0]["split"] != str(data["splits"][offset // 9]):
            raise ValueError("CSV and NPZ splits disagree")
        np.testing.assert_array_equal([int(r["safety_label"]) for r in group], y[offset // 9])
    conflicts = np.asarray([r["conflict_type"] for r in records[::9]])[validation]
    vx = x[validation]
    vy = y[validation]
    scores = np.asarray([predictor.probabilities(v) for v in vx])
    np.savez_compressed(args.output / "validation_predictions.npz", observations=vx, labels=vy, scores=scores, conflicts=conflicts)
    overall = action_mask_metrics(vy, scores, predictor.threshold)
    by_conflict = {name: action_mask_metrics(vy[conflicts == name], scores[conflicts == name], predictor.threshold)
                   for name in sorted(set(conflicts))}
    threshold_rows = []
    for threshold in np.linspace(.05, .95, 181):
        m = action_mask_metrics(vy, scores, float(threshold))
        per_group = [action_mask_metrics(vy[conflicts == name], scores[conflicts == name], float(threshold))
                     for name in sorted(set(conflicts))]
        threshold_rows.append({"threshold": float(threshold), **m,
                               "aggregate_pass": meets(m), "all_conflicts_pass": all(meets(v) for v in per_group)})
    write_csv(threshold_rows, args.output / "threshold_audit.csv")
    mirror_scores = np.asarray([predictor.probabilities(mirror_shield_observation(v)) for v in vx])
    mirrored_scores = np.asarray([mirror_action_values(v) for v in scores])
    changed = vx.copy()
    changed[:, 28:33] = [1, 0, .99, 1, 1]
    advice_scores = np.asarray([predictor.probabilities(v) for v in changed])
    interface = {
        "checkpoint_roundtrip_max_error": float(max(np.max(np.abs(original.probabilities(v) - scores[i])) for i, v in enumerate(vx))),
        "advice_channel_max_error": float(np.max(np.abs(advice_scores - scores))),
        "mirror_score_max_error": float(np.max(np.abs(mirror_scores - mirrored_scores))),
        "mirror_effective_mask_consistency": float(np.mean(effective_action_mask(mirror_scores, predictor.threshold) == effective_action_mask(mirrored_scores, predictor.threshold))),
    }
    print(json.dumps({"runtime_metrics": overall, "by_conflict": by_conflict}, ensure_ascii=False), flush=True)

    # Development-only teacher audit: exact paired start state, fixed continuation,
    # short horizon vs terminal rollout, and two sampled future RNG streams.
    scenarios = tuple(DynamicConflictScenario(**v) for v in json.loads((ROOT / "outputs/confirmatory_protocol/development_scenarios.json").read_text(encoding="utf-8")))
    selected = []
    for conflict in sorted({s.conflict_type for s in scenarios}):
        selected.extend(sorted((s for s in scenarios if s.conflict_type == conflict), key=lambda s: s.scenario_id)[:args.scenarios_per_conflict])
    env = ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), scenarios), "none")
    teacher_rows = []
    sensitivity = []
    for index, scenario in enumerate(selected):
        observation, _ = env.reset(seed=1_300_000 + index, options={"scenario": scenario})
        step = 0
        done = False
        while not done and step <= 40:
            if step in (0, 10, 20, 40):
                short = evaluate_counterfactual_actions(env, horizon=20)
                full = evaluate_counterfactual_actions(env, horizon=env.unwrapped.domain_env.config.max_steps)
                alternate = copy.deepcopy(env)
                alternate.unwrapped.domain_env.rng = np.random.default_rng(1_400_000 + index * 100 + step)
                full_other = evaluate_counterfactual_actions(alternate, horizon=env.unwrapped.domain_env.config.max_steps)
                alternate.close()
                labels = [material_safety_labels(r, env.unwrapped.domain_env.config.population) for r in (short, full, full_other)]
                known_pair = (labels[0] >= 0) & (labels[1] >= 0)
                future_pair = (labels[1] >= 0) & (labels[2] >= 0)
                current_scores = predictor.probabilities(observation)
                runtime_mask = effective_action_mask(current_scores, predictor.threshold)
                sensitivity.append({
                    "scenario_id": scenario.scenario_id, "conflict_type": scenario.conflict_type, "step": step,
                    "horizon_labels_changed": int(np.sum(labels[0] != labels[1])),
                    "horizon_known_pairs": int(np.sum(known_pair)),
                    "horizon_known_flips": int(np.sum(known_pair & (labels[0] != labels[1]))),
                    "short_safe_full_unsafe": int(np.sum((labels[0] == 1) & (labels[1] == 0))),
                    "future_labels_changed": int(np.sum(labels[1] != labels[2])),
                    "future_known_pairs": int(np.sum(future_pair)),
                    "future_known_flips": int(np.sum(future_pair & (labels[1] != labels[2]))),
                    "first_hold_blocks": sum(r.first_action_blocked_by_hold for r in full),
                    "short_rollouts_nonterminal": sum(not r.terminal_reached for r in short),
                    "full_rollouts_nonterminal": sum(not r.terminal_reached for r in full),
                })
                for name, result_set, label_set in zip(("horizon20", "terminal", "terminal_other_rng"), (short, full, full_other), labels):
                    for result in result_set:
                        teacher_rows.append({"scenario_id": scenario.scenario_id, "conflict_type": scenario.conflict_type,
                                             "step": step, "teacher": name, "safety_label": int(label_set[result.action]),
                                             "allowed": bool(runtime_mask[result.action]), "score": float(current_scores[result.action]),
                                             **asdict(result)})
            action = reactive_followup_action(observation) if index % 2 == 0 else int(np.random.default_rng(1_500_000 + index * 100 + step).integers(9))
            observation, _, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1
        print(f"Teacher audit {index + 1}/{len(selected)}: {scenario.scenario_id}", flush=True)
    env.close()
    write_csv(teacher_rows, args.output / "teacher_action_results.csv")
    write_csv(sensitivity, args.output / "teacher_sensitivity_by_state.csv")
    totals = {k: sum(int(row[k]) for row in sensitivity) for k in list(sensitivity[0])[3:]}
    totals.update({"states": len(sensitivity), "actions_per_teacher": len(sensitivity) * 9,
                   "horizon_known_flip_rate": totals["horizon_known_flips"] / max(1, totals["horizon_known_pairs"]),
                   "future_known_flip_rate": totals["future_known_flips"] / max(1, totals["future_known_pairs"])})
    summary = {
        "evaluation_role": "development_audit_not_independent_qualification", "freeze_allowed": False,
        "checkpoint_source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_dataset_sha256": hashlib.sha256((args.source / "counterfactual_shield_dataset.npz").read_bytes()).hexdigest(),
        "threshold_unchanged": predictor.threshold,
        "runtime_metrics": overall, "by_conflict": by_conflict, "interface_checks": interface,
        "aggregate_thresholds_passing": sum(r["aggregate_pass"] for r in threshold_rows),
        "all_conflict_thresholds_passing": sum(r["aggregate_pass"] and r["all_conflicts_pass"] for r in threshold_rows),
        "teacher_sensitivity": totals,
        "limitations": ["threshold selected on reused development data", "two sampled futures are not risk bounds",
                        "fixed reactive continuation is not a global optimum", "classifier cannot identify indeterminate labels",
                        "hidden FED/persona/RNG/sign timer absent from predictor inputs", "no compound conflict in original development bank"],
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
