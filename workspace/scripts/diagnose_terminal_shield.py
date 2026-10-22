"""Posthoc development diagnostics; does not change frozen pilot results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv
from fire_evac.terminal_shield import ObservableHistoryWrapper, TernaryShieldPredictor, terminal_counterfactual_teachers, ternary_action_mask


def unguarded_unknown_mask(probabilities, threshold):
    risk = probabilities[..., 0]
    mask = risk < threshold
    empty = ~mask.any(axis=-1, keepdims=True)
    return mask | (empty & (risk == risk.min(axis=-1, keepdims=True))), empty.squeeze(-1)


def metrics(labels, probabilities, threshold, allow_unknown):
    if allow_unknown:
        mask, _, empty = ternary_action_mask(probabilities, threshold)
    else:
        mask, empty = unguarded_unknown_mask(probabilities, threshold)
    def rejected(selector):
        return float((~mask)[selector].mean()) if np.any(selector) else None
    chosen = np.argmin(np.where(mask, probabilities[..., 0], np.inf), axis=-1)
    top = labels[np.arange(len(labels)), chosen]
    return {"threshold": float(threshold), "unsafe_action_recall": rejected(labels == 0),
            "safe_action_false_rejection_rate": rejected(labels == 1),
            "top_action_unsafe_rate": float((top == 0).mean()),
            "indeterminate_action_rejection_rate": rejected(labels < 0),
            "empty_raw_mask_rate": float(empty.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/terminal_shield_development_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terminal_shield_diagnosis_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier posthoc diagnosis; choose another output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    data = np.load(args.source / "dataset.npz", allow_pickle=False)
    audit = data["splits"] == "audit"
    calibration = data["splits"] == "calibration"
    effects = {}
    for arm in ("no_history", "with_history"):
        predictor = TernaryShieldPredictor.load(args.source / arm / "shield_predictor.pt")
        probabilities = predictor.probabilities(data["observations"][audit])
        labels = data["labels"][audit]
        mask, abstain, empty = ternary_action_mask(probabilities, predictor.threshold)
        missed = (labels == 0) & mask
        cp = predictor.probabilities(data["observations"][calibration])
        cy = data["labels"][calibration]
        thresholds = [metrics(cy, cp, float(t), False) for t in np.linspace(.05, .95, 181)]
        def penalty(v):
            return max(0, .95 - (v["unsafe_action_recall"] or 0)) + max(0, (v["safe_action_false_rejection_rate"] or 0) - .10) + max(0, v["top_action_unsafe_rate"] - .05)
        chosen = min(thresholds, key=penalty)
        effects[arm] = {"audit_unsafe_actions": int((labels == 0).sum()), "missed_unsafe_actions": int(missed.sum()),
                        "missed_unsafe_predicted_unknown": int((missed & abstain).sum()),
                        "missed_unsafe_forced_fallback": int((missed & empty[:, None]).sum()),
                        "original_rule": metrics(labels, probabilities, predictor.threshold, True),
                        "remove_unknown_passthrough_same_threshold": metrics(labels, probabilities, predictor.threshold, False),
                        "remove_unknown_passthrough_calibration_selection": chosen,
                        "remove_unknown_passthrough_audit_posthoc": metrics(labels, probabilities, chosen["threshold"], False)}
    manifest = json.loads((args.source / "development_manifest.json").read_text(encoding="utf-8"))
    scenarios = tuple(DynamicConflictScenario(**s) for s in manifest["scenario_records"][:manifest["original_scenarios"]])
    env = ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), scenarios), "none"))
    rows = []
    for index, scenario in enumerate(scenarios):
        cached = np.load(args.source / "states" / f"{scenario.scenario_id}_00_000.npz", allow_pickle=False)
        # Same forecast, same capacities/blockage, same people and sensors at t=0;
        # flip only the hidden actual source. At t=0 smoke has not grown yet.
        twin = replace(scenario, source_branch=1 - scenario.source_branch)
        observation, _ = env.reset(seed=1_600_000 + index * manifest["repetitions"], options={"scenario": twin})
        np.testing.assert_array_equal(observation, cached["observation"])
        labels, _, _ = terminal_counterfactual_teachers(env, tuple(int(v) for v in cached["future_seeds"]))
        original = cached["labels"]
        known = (original >= 0) & (labels >= 0)
        for action in range(9):
            rows.append({"scenario_id": scenario.scenario_id, "action": action,
                         "original_label": int(original[action]), "hidden_source_flipped_label": int(labels[action]),
                         "same_observation": True, "known_label_conflict": bool(known[action] and original[action] != labels[action])})
        print(f"Hidden-source twin {index + 1}/{len(scenarios)}", flush=True)
    env.close()
    with (args.output / "identical_observation_twins.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"evaluation_role": "posthoc_development_diagnosis_not_new_validation", "freeze_allowed": False,
               "unknown_passthrough_effects": effects,
               "hidden_source_twins": {"states": len(scenarios), "action_pairs": len(rows),
                                       "known_label_conflicts": sum(r["known_label_conflict"] for r in rows),
                                       "any_label_changes": sum(r["original_label"] != r["hidden_source_flipped_label"] for r in rows),
                                       "states_with_known_conflicts": len({r["scenario_id"] for r in rows if r["known_label_conflict"]}),
                                       "scope": "constructed simulator twins, not observed population frequency; demonstrates label aliasing, not impossibility of robust control"}}
    (args.output / "diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
