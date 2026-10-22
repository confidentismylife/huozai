"""Predeclared development pilot: terminal consensus teacher and input ablation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.advice_protocol_v2 import mirror_scenario
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.counterfactual_shield import reactive_followup_action
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_domain_randomized_trust_bank
from fire_evac.terminal_shield import (
    ACTION_MIRROR, ObservableHistoryWrapper, TernaryShieldNetwork, TernaryShieldPredictor,
    mirror_history_observation, shield_input, terminal_counterfactual_teachers,
    ternary_action_mask, ternary_mask_metrics,
)


def write_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def meets(metrics):
    return (metrics["unsafe_action_recall"] is not None and metrics["unsafe_action_recall"] >= .95
            and metrics["safe_action_false_rejection_rate"] is not None and metrics["safe_action_false_rejection_rate"] <= .10
            and metrics["top_action_unsafe_rate"] <= .05)


def grouped_split(scenarios):
    assignment = {}
    for name in sorted({s.conflict_type for s in scenarios}):
        items = sorted((s for s in scenarios if s.conflict_type == name), key=lambda s: s.scenario_id)
        if len(items) < 6:
            raise ValueError("at least six original scenarios per conflict required")
        for index, s in enumerate(items):
            assignment[s.scenario_id] = "audit" if index == len(items) - 1 else "calibration" if index == len(items) - 2 else "train"
    return assignment


def fit_model(x, y, splits, seed, use_history, epochs, members, output):
    train = splits == "train"
    calibration = splits == "calibration"
    train_x = np.concatenate([x[train], np.asarray([mirror_history_observation(v) for v in x[train]])])
    train_y = np.concatenate([y[train], y[train][:, ACTION_MIRROR]])
    train_x = shield_input(train_x, use_history)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1
    classes = np.where(train_y < 0, 2, train_y).astype(np.int64)
    counts = np.bincount(classes.ravel(), minlength=3)
    weight = classes.size / (3 * np.maximum(1, counts))
    tx = torch.as_tensor((train_x - mean) / scale)
    ty = torch.as_tensor(classes)
    tw = torch.as_tensor(weight, dtype=torch.float32)
    models = []
    curve = []
    for member in range(members):
        torch.manual_seed(seed + member * 997)
        model = TernaryShieldNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(epochs):
            indices = torch.randperm(len(tx))
            total = 0.0
            for start in range(0, len(tx), 128):
                index = indices[start:start + 128]
                optimizer.zero_grad()
                logits = model(tx[index])
                loss = nn.functional.cross_entropy(logits.reshape(-1, 3), ty[index].reshape(-1), weight=tw)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(index)
            curve.append({"member": member, "epoch": epoch + 1, "loss": total / len(tx)})
        model.eval()
        models.append(model)
        print(f"Trained {'history' if use_history else 'no_history'} member {member + 1}/{members}", flush=True)
    predictor = TernaryShieldPredictor(tuple(models), mean, scale, .5, use_history)
    probabilities = predictor.probabilities(x[calibration])
    candidates = []
    for threshold in np.linspace(.05, .95, 181):
        metrics = ternary_mask_metrics(y[calibration], probabilities, float(threshold))
        candidates.append({"threshold": float(threshold), **metrics, "aggregate_pass": meets(metrics)})
    feasible = [v for v in candidates if v["aggregate_pass"]]
    if feasible:
        chosen = min(feasible, key=lambda v: (v["safe_action_false_rejection_rate"], -v["unsafe_action_recall"], v["threshold"]))
    else:
        chosen = min(candidates, key=lambda v: max(0, .95 - (v["unsafe_action_recall"] or 0)) + max(0, (v["safe_action_false_rejection_rate"] or 0) - .10) + max(0, v["top_action_unsafe_rate"] - .05))
    predictor.threshold = chosen["threshold"]
    predictor.save(output / "shield_predictor.pt")
    restored = TernaryShieldPredictor.load(output / "shield_predictor.pt")
    np.testing.assert_allclose(predictor.probabilities(x[:3]), restored.probabilities(x[:3]), atol=1e-7, rtol=0)
    write_csv(curve, output / "training_curve.csv")
    write_csv(candidates, output / "calibration_thresholds.csv")
    return restored, {"class_counts_train_after_mirror": counts.tolist(), "class_weights": weight.tolist(), "selected_calibration_metrics": chosen,
                      "parameter_count_per_member": sum(p.numel() for p in models[0].parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terminal_shield_development_v1")
    parser.add_argument("--scenarios", type=int, default=36)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--futures", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261301)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.scenarios < 36 or args.scenarios % 6 or min(args.repetitions, args.epochs, args.members) < 1 or args.futures < 2:
        raise ValueError("balanced 36+ original scenarios, positive budgets, and 2+ futures required")
    if args.output.exists() and not args.resume:
        raise FileExistsError("preserve prior results; choose a new output or use --resume with matching manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    original = make_domain_randomized_trust_bank(args.scenarios, args.seed, "TSDEV")
    scenarios = original + tuple(mirror_scenario(s) for s in original)
    assignment = grouped_split(original)
    collection_steps = (0, 8, 16, 24, 40, 64, 96)
    code_files = ("scripts/build_terminal_shield.py", "src/fire_evac/terminal_shield.py", "src/fire_evac/counterfactual_shield.py", "src/fire_evac/confirmatory.py", "src/fire_evac/dynamic_conflicts.py", "src/fire_evac/behavior.py", "src/fire_evac/env.py")
    config = {
        "evaluation_role": "development_pilot_not_final_or_independent_qualification", "freeze_allowed": False,
        "seed": args.seed, "original_scenarios": args.scenarios, "repetitions": args.repetitions,
        "futures": args.futures, "continuations": ["graded_reactive", "full_release_reactive"],
        "collection_steps": collection_steps, "members": args.members, "epochs": args.epochs,
        "scenario_records": [asdict(s) for s in scenarios], "original_scenario_split": assignment,
        "observation_contract": "49: original34 + clock + remaining_hold + actual_sign3 + observed_smoke_integrals5 + differences5",
        "labels": "unchanged material tolerances; unanimous safe/unsafe across all sampled futures/plans; disagreement unknown; no forced winner",
        "abstention_threshold": .5, "threshold_grid": [.05, .95, 181],
        "criteria": {"unsafe_recall_min": .95, "safe_false_rejection_max": .10, "top_unsafe_max": .05, "mirror_mask_consistency_min": .95},
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code_files},
        "behavior_artifact_sha256": hashlib.sha256((ROOT / "outputs/behavior_pipeline/behavior_model.json").read_bytes()).hexdigest(),
    }
    config = json.loads(json.dumps(config))
    manifest_path = args.output / "development_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume manifest differs; use a new output")
    else:
        manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    state_dir = args.output / "states"
    state_dir.mkdir(exist_ok=True)
    env = ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(CalibratedLogitBehavior(ROOT / "outputs/behavior_pipeline/behavior_model.json"), scenarios), "none"))
    metadata = []
    start = time.monotonic()
    for index, scenario in enumerate(scenarios):
        source_id = scenario.scenario_id.removesuffix("M")
        # Alternate trajectory policies by source family, preserving mode in mirrors.
        family = int(source_id.removeprefix("TSDEV"))
        for repetition in range(args.repetitions):
            population_seed = 1_600_000 + index * args.repetitions + repetition
            observation, _ = env.reset(seed=population_seed, options={"scenario": scenario})
            done = False
            step = 0
            while not done:
                if step in collection_steps:
                    state_id = f"{scenario.scenario_id}_{repetition:02d}_{step:03d}"
                    path = state_dir / f"{state_id}.npz"
                    if not path.exists():
                        future_seeds = tuple(2_000_000 + index * 10_000 + repetition * 1000 + step * 10 + f for f in range(args.futures))
                        labels, teachers, evaluations = terminal_counterfactual_teachers(env, future_seeds)
                        fields = ["incapacitation_delta", "fed_delta", "congestion_per_step", "evacuation_delta", "total_reward", "rollout_steps", "terminal_reached", "first_actual_joint_action", "first_action_blocked_by_hold"]
                        outcomes = np.asarray([[[getattr(r, k) for k in fields] for r in results] for _, _, results in evaluations], dtype=np.float64)
                        np.savez_compressed(path, observation=observation, labels=labels, teacher_labels=teachers, outcomes=outcomes, outcome_fields=np.asarray(fields), future_seeds=np.asarray(future_seeds))
                    else:
                        cached = np.load(path, allow_pickle=False)
                        np.testing.assert_array_equal(cached["observation"], observation)
                    metadata.append({"state_id": state_id, "scenario_id": scenario.scenario_id, "source_id": source_id,
                                     "conflict_type": scenario.conflict_type, "split": assignment[source_id],
                                     "population_seed": population_seed, "step": step,
                                     "trajectory_policy": "reactive" if ((family - 1) // 6 + repetition) % 2 == 0 else "random"})
                action = reactive_followup_action(observation) if ((family - 1) // 6 + repetition) % 2 == 0 else int(np.random.default_rng(3_000_000 + index * 10_000 + repetition * 1000 + step).integers(9))
                observation, _, term, trunc, _ = env.step(action)
                step += 1
                done = term or trunc
        print(f"Collected {index + 1}/{len(scenarios)} {scenario.scenario_id}: {len(metadata)} states, {time.monotonic() - start:.1f}s", flush=True)
        (args.output / "progress.json").write_text(json.dumps({"phase": "collection", "completed_scenarios": index + 1, "states": len(metadata), "elapsed_seconds": time.monotonic() - start}), encoding="utf-8")
    env.close()
    write_csv(metadata, args.output / "state_metadata.csv")
    datasets = [np.load(state_dir / f"{m['state_id']}.npz", allow_pickle=False) for m in metadata]
    x = np.asarray([d["observation"] for d in datasets])
    y = np.asarray([d["labels"] for d in datasets])
    teachers = np.asarray([d["teacher_labels"] for d in datasets])
    outcomes = np.asarray([d["outcomes"] for d in datasets])
    splits = np.asarray([m["split"] for m in metadata])
    conflicts = np.asarray([m["conflict_type"] for m in metadata])
    np.savez_compressed(args.output / "dataset.npz", observations=x, labels=y, teacher_labels=teachers,
                        outcomes=outcomes, outcome_fields=datasets[0]["outcome_fields"], splits=splits, conflicts=conflicts,
                        source_ids=np.asarray([m["source_id"] for m in metadata]), steps=np.asarray([m["step"] for m in metadata]))
    summaries = {}
    for arm, use_history in (("no_history", False), ("with_history", True)):
        arm_dir = args.output / arm
        arm_dir.mkdir(exist_ok=True)
        predictor, fit_info = fit_model(x, y, splits, args.seed, use_history, args.epochs, args.members, arm_dir)
        audit = splits == "audit"
        probabilities = predictor.probabilities(x[audit])
        reflected = predictor.probabilities(np.asarray([mirror_history_observation(v) for v in x[audit]]))
        direct_mask = ternary_action_mask(probabilities, predictor.threshold)[0]
        mirror_mask = ternary_action_mask(reflected, predictor.threshold)[0]
        mirror_rate = float(np.mean(mirror_mask == direct_mask[:, ACTION_MIRROR]))
        changed = x[audit].copy()
        changed[:, 28:33] = [1, 0, .99, 1, 1]
        advice_error = float(np.max(np.abs(predictor.probabilities(changed) - probabilities)))
        aggregate = ternary_mask_metrics(y[audit], probabilities, predictor.threshold)
        audit_conflicts = conflicts[audit]
        per_conflict = {name: ternary_mask_metrics(y[audit][audit_conflicts == name], probabilities[audit_conflicts == name], predictor.threshold)
                        for name in sorted(set(audit_conflicts))}
        counts = np.zeros((3, 3), dtype=int)
        truth_class = np.where(y[audit] < 0, 2, y[audit])
        predicted_class = probabilities.argmax(axis=-1)
        np.add.at(counts, (truth_class.ravel(), predicted_class.ravel()), 1)
        passed = meets(aggregate) and all(meets(v) for v in per_conflict.values()) and mirror_rate >= .95
        summaries[arm] = {"status": "development_pass" if passed else "development_failed", "freeze_allowed": False,
                          "threshold": predictor.threshold, "audit_metrics": aggregate, "by_conflict": per_conflict,
                          "mirror_effective_mask_consistency": mirror_rate, "advice_channel_max_error": advice_error,
                          "confusion_matrix_order": ["unsafe", "safe", "unknown"], "confusion_matrix": counts.tolist(), **fit_info}
        np.savez_compressed(arm_dir / "audit_predictions.npz", probabilities=probabilities, labels=y[audit], observations=x[audit], conflicts=audit_conflicts)
        print(json.dumps({"arm": arm, **summaries[arm]}, ensure_ascii=False), flush=True)
    disagreement = np.any(teachers != teachers[:, :1, :], axis=1)
    summary = {"evaluation_role": config["evaluation_role"], "freeze_allowed": False, "states": len(x),
               "states_by_split": {s: int((splits == s).sum()) for s in ("train", "calibration", "audit")},
               "counterfactual_rollouts": len(x) * args.futures * 2 * 9,
               "mean_safe_actions": float((y == 1).sum(axis=1).mean()), "mean_unsafe_actions": float((y == 0).sum(axis=1).mean()),
               "mean_unknown_actions": float((y < 0).sum(axis=1).mean()),
               "teacher_disagreement_action_rate": float(disagreement.mean()),
               "terminal_rollout_fraction": float(outcomes[..., 6].mean()),
               "first_hold_block_rate": float(outcomes[..., 8].mean()),
               "states_by_step": {str(s): int(sum(m["step"] == s for m in metadata)) for s in collection_steps},
               "arms": summaries, "elapsed_seconds": time.monotonic() - start,
               "limitations": ["finite consensus not a risk bound", "same latent population across sampled futures", "only two fixed continuation plans",
                               "calibrated behavior and noiseless sensors only", "six audit scenario families; correlated actions/states/mirrors",
                               "privileged teacher sees hidden future and individual FED", "no independent qualification or formal final bank"]}
    (args.output / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "progress.json").write_text(json.dumps({"phase": "complete", "states": len(x), "freeze_allowed": False}), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
