from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.advice_protocol_v2 import mirror_scenario
from fire_evac.behavior import CalibratedLogitBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.counterfactual_shield import (
    ShieldNetwork,
    ShieldPredictor,
    admissible_action_mask,
    evaluate_counterfactual_actions,
    reactive_followup_action,
    mirror_action_values,
    mirror_shield_observation,
    material_safety_labels,
    action_mask_metrics,
    effective_action_mask,
)
from fire_evac.dynamic_conflicts import DynamicConflictScenario, GymDynamicConflictEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a counterfactual action-set shield.")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--state-stride", type=int, default=2)
    parser.add_argument("--maximum-collected-step", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261221)
    parser.add_argument(
        "--development-scenarios", type=Path,
        default=ROOT / "outputs" / "confirmatory_protocol" / "development_scenarios.json",
    )
    parser.add_argument(
        "--calibration-artifact", type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "conditional_shield_development"
    )
    return parser.parse_args()


def split_ids(originals: tuple[DynamicConflictScenario, ...]) -> dict[str, str]:
    by_conflict: dict[str, list[str]] = {}
    for scenario in originals:
        by_conflict.setdefault(scenario.conflict_type, []).append(scenario.scenario_id)
    result = {}
    for conflict, ids in by_conflict.items():
        ordered = sorted(ids)
        validation_id = ordered[-1]
        for scenario_id in ordered:
            result[scenario_id] = "validation" if scenario_id == validation_id else "train"
    return result


def main() -> None:
    args = parse_args()
    if (args.output / "validation_summary.json").exists():
        raise FileExistsError("preserve prior development results: choose a new --output directory")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.development_scenarios.read_text(encoding="utf-8"))
    originals = tuple(DynamicConflictScenario(**item) for item in payload)
    scenarios = originals + tuple(mirror_scenario(item) for item in originals)
    split = split_ids(originals)
    env = ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(CalibratedLogitBehavior(args.calibration_artifact), scenarios), "none"
    )
    observations = []
    masks = []
    ranks = []
    splits = []
    records = []
    for scenario_index, scenario in enumerate(scenarios):
        source_id = scenario.scenario_id.removesuffix("M")
        for repetition in range(args.repetitions):
            population_seed = 1_100_000 + scenario_index * args.repetitions + repetition
            observation, info = env.reset(seed=population_seed, options={"scenario": scenario})
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                if step <= args.maximum_collected_step and step % args.state_stride == 0:
                    results = evaluate_counterfactual_actions(env, horizon=args.horizon)
                    _, rank = admissible_action_mask(results)
                    labels = material_safety_labels(results, env.unwrapped.domain_env.config.population)
                    observations.append(observation.copy()); masks.append(labels); ranks.append(rank)
                    splits.append(split[source_id])
                    for result in results:
                        records.append(
                            {
                                "scenario_id": scenario.scenario_id,
                                "source_id": source_id,
                                "split": split[source_id],
                                "conflict_type": scenario.conflict_type,
                                "population_seed": population_seed,
                                "step": step,
                                "action": result.action,
                                "safety_label": int(labels[result.action]),
                                "risk_rank": float(rank[result.action]),
                                **asdict(result),
                            }
                        )
                if repetition % 2 == 0:
                    action = reactive_followup_action(observation)
                else:
                    action = int(rng.integers(0, 9))
                observation, _, terminated, truncated, info = env.step(action)
                step += 1
    env.close()
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)
    rank_values = np.asarray(ranks, dtype=np.float32)
    split_array = np.asarray(splits)
    np.savez_compressed(
        args.output / "counterfactual_shield_dataset.npz",
        observations=x, safe_masks=y, risk_ranks=rank_values, splits=split_array,
        scenario_ids=np.asarray([r["scenario_id"] for r in records[::9]]),
        conflict_types=np.asarray([r["conflict_type"] for r in records[::9]]),
        population_seeds=np.asarray([r["population_seed"] for r in records[::9]]),
        steps=np.asarray([r["step"] for r in records[::9]]),
    )
    with (args.output / "counterfactual_action_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)

    train_mask = split_array == "train"
    validation_mask = split_array == "validation"
    # Exact reflection augmentation makes direction symmetry an explicit training invariant.
    train_x = np.concatenate([x[train_mask], np.asarray([mirror_shield_observation(v) for v in x[train_mask]])])
    train_y = np.concatenate([y[train_mask], np.asarray([mirror_action_values(v) for v in y[train_mask]])])
    train_rank = np.concatenate(
        [rank_values[train_mask], np.asarray([mirror_action_values(v) for v in rank_values[train_mask]])]
    )
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (train_x - mean) / scale
    known_train = train_y >= 0
    positive_rate = np.clip(
        np.sum((train_y == 1) & known_train, axis=0) / np.maximum(1, known_train.sum(axis=0)),
        1e-3,
        1 - 1e-3,
    )
    positive_weight = torch.as_tensor((1 - positive_rate) / positive_rate, dtype=torch.float32)
    tx = torch.as_tensor(normalized, dtype=torch.float32)
    ty = torch.as_tensor(np.clip(train_y, 0, 1), dtype=torch.float32)
    tk = torch.as_tensor(known_train, dtype=torch.float32)
    tr = torch.as_tensor(1.0 - train_rank, dtype=torch.float32)
    curve = []
    models = []
    for member in range(5):
        torch.manual_seed(args.seed + member * 997)
        model = ShieldNetwork(x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(args.epochs):
            permutation = torch.randperm(len(tx))
            epoch_loss = 0.0
            for start in range(0, len(tx), 128):
                index = permutation[start:start + 128]
                optimizer.zero_grad()
                logits = model(tx[index])
                element_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, ty[index], reduction="none", pos_weight=positive_weight
                )
                supervised_loss = (element_loss * tk[index]).sum() / tk[index].sum().clamp_min(1.0)
                loss = supervised_loss + 0.1 * nn.functional.mse_loss(torch.sigmoid(logits), tr[index])
                loss.backward(); optimizer.step()
                epoch_loss += float(loss.item()) * len(index)
            curve.append({"member": member, "epoch": epoch + 1, "loss": epoch_loss / len(tx)})
        model.eval(); models.append(model)
    predictor_for_validation = ShieldPredictor(
        tuple(models), mean.astype(np.float32), scale.astype(np.float32), 0.5
    )
    probabilities = np.asarray([predictor_for_validation.probabilities(v) for v in x[validation_mask]])
    validation_labels = y[validation_mask]
    truth = validation_labels == 1
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = action_mask_metrics(validation_labels, probabilities, float(threshold))
        unsafe_recall = metrics["unsafe_action_recall"]
        safe_false_rejection = metrics["safe_action_false_rejection_rate"]
        candidates.append((threshold, unsafe_recall, safe_false_rejection))
    feasible = [item for item in candidates if item[1] >= 0.95 and item[2] <= 0.10]
    if feasible:
        threshold, unsafe_recall, safe_false_rejection = min(feasible, key=lambda z: z[2])
    else:
        threshold, unsafe_recall, safe_false_rejection = min(
            candidates, key=lambda z: max(0.0, 0.95 - z[1]) + max(0.0, z[2] - 0.10)
        )
    top_labels = validation_labels[np.arange(len(truth)), np.argmax(probabilities, axis=1)]
    top_action_unsafe = float(np.mean(top_labels == 0))
    mirror_x = np.asarray([mirror_shield_observation(v) for v in x[validation_mask]])
    mirror_probabilities = np.asarray([predictor_for_validation.probabilities(v) for v in mirror_x])
    mirrored_original = np.asarray([mirror_action_values(v) for v in probabilities])
    symmetry_mae = float(np.mean(np.abs(mirror_probabilities - mirrored_original)))
    symmetry_rate = float(np.mean(effective_action_mask(mirror_probabilities, threshold) == effective_action_mask(mirrored_original, threshold)))
    predictor = ShieldPredictor(tuple(models), mean.astype(np.float32), scale.astype(np.float32), float(threshold))
    predictor.save(args.output / "shield_predictor.pt")
    loaded = ShieldPredictor.load(args.output / "shield_predictor.pt")
    np.testing.assert_allclose(loaded.probabilities(x[validation_mask][0]), probabilities[0], atol=1e-7)
    with (args.output / "training_curve.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys()); writer.writeheader(); writer.writerows(curve)
    conflicts = np.asarray([r["conflict_type"] for r in records[::9]])[validation_mask]
    by_conflict = {
        name: action_mask_metrics(validation_labels[conflicts == name], probabilities[conflicts == name], float(threshold))
        for name in sorted(set(conflicts))
    }
    def meets_criteria(m):
        return (m["unsafe_action_recall"] is not None and m["unsafe_action_recall"] >= 0.95
                and m["safe_action_false_rejection_rate"] is not None and m["safe_action_false_rejection_rate"] <= 0.10
                and m["top_action_unsafe_rate"] <= 0.05)
    aggregate = action_mask_metrics(validation_labels, probabilities, float(threshold))
    passed = meets_criteria(aggregate) and all(meets_criteria(m) for m in by_conflict.values()) and symmetry_rate >= 0.95
    summary = {
        "status": "passed" if passed else "failed",
        "freeze_allowed": False,
        "evaluation_role": "development_threshold_selection_not_independent_qualification",
        "requires_independent_qualification": True,
        "runtime_metrics": aggregate,
        "by_conflict": by_conflict,
        "states": len(x),
        "train_states_before_mirror_augmentation": int(train_mask.sum()),
        "validation_states": int(validation_mask.sum()),
        "counterfactual_rollouts": len(x) * 9,
        "horizon": args.horizon,
        "threshold": float(threshold),
        "unsafe_action_recall": unsafe_recall,
        "safe_action_false_rejection_rate": safe_false_rejection,
        "top_action_unsafe_rate": top_action_unsafe,
        "mirror_probability_mae": symmetry_mae,
        "mirror_mask_consistency": symmetry_rate,
        "mean_clearly_safe_actions": float(np.mean(np.sum(y == 1, axis=1))),
        "mean_clearly_unsafe_actions": float(np.mean(np.sum(y == 0, axis=1))),
        "mean_indeterminate_actions": float(np.mean(np.sum(y < 0, axis=1))),
        "criteria": {
            "unsafe_action_recall_min": 0.95,
            "safe_action_false_rejection_rate_max": 0.10,
            "top_action_unsafe_rate_max": 0.05,
            "mirror_mask_consistency_min": 0.95,
        },
    }
    (args.output / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
