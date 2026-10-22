"""Observable plan ranking against a fixed PPO baseline, for development only.

Scores are signed predicted differences in six teacher priorities. They are
neither calibrated probabilities nor action masks. Fitting accepts arrays from
training and calibration families; it does not accept an environment or actor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .plan_distillation import (
    PRIORITY_FIELDS, STUDENT_SEEDS, PlanScoreNetwork, choose_priority,
    family_weights, observable_features,
)


BASELINE_PLAN_INDEX = 12  # All twelve manual plans precede frozen pure PPO 601.
PRIMARY_MSE_WEIGHT = 1.0
PAIRWISE_WEIGHT = 0.5
AUXILIARY_MSE_WEIGHT = 0.1
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
EVALUATION_INTERVAL = 10
REGRET_TOLERANCE = 1e-12
OBJECTIVE_NAME = "baseline_relative_primary_mse_gap_weighted_ranking_v1"
MSE_OBJECTIVE_NAME = "min_relative_six_priority_mse_v1"


def baseline_relative_targets(targets, baseline_plan_index=BASELINE_PLAN_INDEX):
    """Subtract one fixed plan in every field, preserving lexicographic order."""
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim not in (2, 3) or values.shape[-2:] != (17, 6) or not np.isfinite(values).all():
        raise ValueError("finite seventeen-plan six-field targets required")
    if baseline_plan_index != BASELINE_PLAN_INDEX:
        raise ValueError("the predeclared baseline must be plan 12, pure PPO 601")
    return values - values[..., baseline_plan_index:baseline_plan_index + 1, :]


def _relative_predictions(predictions):
    return predictions - predictions[:, BASELINE_PLAN_INDEX:BASELINE_PLAN_INDEX + 1, :]


def ranking_loss(predictions, targets, weights):
    """Family-weighted loss on standardized, baseline-relative priorities.

For row r, let p and y be standardized predictions and targets. The loss is
mean_i (p[i,0]-y[i,0])**2 + 0.5 * R + 0.1 * mean_{i,k>0}
(p[i,k]-y[i,k])**2. R is the weighted mean, over unordered plan pairs, of
softplus(-sign(y[i,0]-y[j,0]) * (p[i,0]-p[j,0])), with weight
abs(y[i,0]-y[j,0]). Rows with no primary differences contribute zero R.
The outer weighted sum gives each scene family equal total weight.
"""
    if predictions.shape != targets.shape or predictions.ndim != 3 or predictions.shape[1:] != (17, 6):
        raise ValueError("aligned [N,17,6] standardized score tensors required")
    if weights.shape != (len(predictions),):
        raise ValueError("one family weight per row required")
    primary = (predictions[..., 0] - targets[..., 0]).square().mean(dim=1)
    auxiliary = (predictions[..., 1:] - targets[..., 1:]).square().mean(dim=(1, 2))
    first, second = torch.triu_indices(17, 17, offset=1, device=predictions.device)
    true_gaps = targets[:, first, 0] - targets[:, second, 0]
    predicted_gaps = predictions[:, first, 0] - predictions[:, second, 0]
    gaps = true_gaps.abs()
    numerator = (gaps * F.softplus(-true_gaps.sign() * predicted_gaps)).sum(dim=1)
    pairwise = numerator / gaps.sum(dim=1).clamp_min(1e-12)
    terms = {"primary_mse": (primary * weights).sum(),
             "pairwise_ranking": (pairwise * weights).sum(),
             "auxiliary_mse": (auxiliary * weights).sum()}
    total = (PRIMARY_MSE_WEIGHT * terms["primary_mse"] +
             PAIRWISE_WEIGHT * terms["pairwise_ranking"] +
             AUXILIARY_MSE_WEIGHT * terms["auxiliary_mse"])
    return total, terms


def teacher_primary_regret(predictions, signed_targets, weights):
    """Family-equal teacher regret of lexicographically selected plans.

Teacher priorities are from planning futures; this metric makes no claim
about independent simulator outcomes or true risk probabilities.
"""
    scores, targets = np.asarray(predictions), np.asarray(signed_targets)
    row_weights = np.asarray(weights, dtype=np.float64)
    if scores.shape != targets.shape or scores.ndim != 3 or scores.shape[1:] != (17, 6):
        raise ValueError("aligned [N,17,6] priorities required")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("finite priorities required")
    if row_weights.shape != (len(scores),):
        raise ValueError("one family weight per row required")
    chosen = np.asarray([choose_priority(row) for row in scores])
    regret = targets[np.arange(len(scores)), chosen, 0] - targets[:, :, 0].min(axis=1)
    return float(np.sum(row_weights * regret))


@dataclass
class ObservableScoreStudent:
    network: PlanScoreNetwork
    input_mean: np.ndarray
    input_std: np.ndarray
    target_scale: np.ndarray
    forecast_only: bool
    training_seed: int
    selected_epoch: int
    objective: str = "relative_rank"

    def priorities(self, observation):
        values = observable_features(observation, self.forecast_only)
        if values.shape != (49,):
            raise ValueError("one observation required for selecting a plan")
        self.network.eval()
        with torch.inference_mode():
            predictions = self.network(torch.as_tensor(
                (values - self.input_mean) / self.input_std).unsqueeze(0))
            result = (_relative_predictions(predictions) if self.objective == "relative_rank"
                      else predictions)[0].cpu().numpy()
        return result * self.target_scale

    def choose(self, observation):
        return choose_priority(self.priorities(observation))

    def save(self, path):
        torch.save({"state_dict": self.network.state_dict(),
                    "input_mean": torch.as_tensor(self.input_mean),
                    "input_std": torch.as_tensor(self.input_std),
                    "target_scale": torch.as_tensor(self.target_scale),
                    "forecast_only": self.forecast_only,
                    "training_seed": self.training_seed,
                    "selected_epoch": self.selected_epoch,
                    "plan_count": 17, "priority_fields": PRIORITY_FIELDS,
                    "baseline_plan_index": BASELINE_PLAN_INDEX if self.objective == "relative_rank" else None,
                    "objective": self.objective,
                    "objective_name": OBJECTIVE_NAME if self.objective == "relative_rank" else MSE_OBJECTIVE_NAME}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        objective = saved["objective"]
        expected_name = OBJECTIVE_NAME if objective == "relative_rank" else MSE_OBJECTIVE_NAME
        expected_baseline = BASELINE_PLAN_INDEX if objective == "relative_rank" else None
        if (objective not in ("mse", "relative_rank") or saved["plan_count"] != 17 or
                tuple(saved["priority_fields"]) != PRIORITY_FIELDS or
                saved["baseline_plan_index"] != expected_baseline or saved["objective_name"] != expected_name):
            raise ValueError("relative student plan or objective contract differs")
        network = PlanScoreNetwork()
        network.load_state_dict(saved["state_dict"])
        network.eval()
        return cls(network, saved["input_mean"].numpy(), saved["input_std"].numpy(),
                   saved["target_scale"].numpy(), saved["forecast_only"],
                   saved["training_seed"], saved["selected_epoch"], objective)


RelativePlanRankingStudent = ObservableScoreStudent


def fit_score_student(observations, targets, families, train_mask, calibration_mask,
                      training_seed, forecast_only=False, epochs=600, patience=100,
                      objective="relative_rank", selection_mask=None):
    """Train from observable arrays; select only by calibration teacher regret.

The defaults are predeclared: CPU/full-batch AdamW, seeds 801--805, at most
600 epochs, evaluation each ten epochs, patience 100. The epoch/patience
arguments allow smaller bounded unit tests, not automatic hyperparameter search.
"""
    if not 1 <= int(epochs) <= 600 or epochs != int(epochs) or int(patience) < 1:
        raise ValueError("epochs must be 1..600 and patience positive")
    if training_seed not in STUDENT_SEEDS:
        raise ValueError("predeclared student seeds are 801..805")
    x = observable_features(observations, forecast_only)
    if x.ndim != 2:
        raise ValueError("training observations must be [N,49]")
    count = len(x)
    original_y = np.asarray(targets, dtype=np.float32)
    if original_y.shape != (count, 17, 6) or not np.isfinite(original_y).all():
        raise ValueError("aligned finite relative plan targets required")
    if objective not in ("mse", "relative_rank"):
        raise ValueError("objective must be mse or relative_rank")
    y = baseline_relative_targets(original_y) if objective == "relative_rank" else original_y.copy()
    families = np.asarray(families)
    train_mask, calibration_mask = np.asarray(train_mask, bool), np.asarray(calibration_mask, bool)
    if families.shape != (count,) or train_mask.shape != (count,) or calibration_mask.shape != (count,):
        raise ValueError("one family and split membership per observation required")
    if np.any(train_mask & calibration_mask) or not train_mask.any() or not calibration_mask.any():
        raise ValueError("nonempty disjoint training and calibration required")
    if not np.all(train_mask | calibration_mask):
        raise ValueError("fit input must contain only training and calibration rows")
    train_families, cal_families = set(families[train_mask]), set(families[calibration_mask])
    if train_families & cal_families:
        raise ValueError("a scenario family cannot span training and calibration")
    selection_mask = (calibration_mask.copy() if selection_mask is None
                      else np.asarray(selection_mask, bool))
    if (selection_mask.shape != (count,) or not selection_mask.any() or
            np.any(selection_mask & ~calibration_mask)):
        raise ValueError("nonempty checkpoint selection must be a subset of calibration")
    weights, cal_weights = family_weights(families[train_mask]), family_weights(families[calibration_mask])
    selection_weights = family_weights(families[selection_mask])
    selection_in_calibration = selection_mask[calibration_mask]
    mean = np.sum(x[train_mask] * weights[:, None], axis=0)
    std = np.sqrt(np.sum((x[train_mask] - mean) ** 2 * weights[:, None], axis=0))
    std = np.where(std < 1e-3, 1, std).astype(np.float32)
    target_mean = np.sum(y[train_mask].mean(axis=1) * weights[:, None], axis=0)
    variance = np.sum(((y[train_mask] - target_mean) ** 2).mean(axis=1) * weights[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), 1e-3).astype(np.float32)
    train_x, cal_x = torch.as_tensor((x[train_mask] - mean) / std), torch.as_tensor((x[calibration_mask] - mean) / std)
    train_y, cal_y = torch.as_tensor(y[train_mask] / scale), torch.as_tensor(y[calibration_mask] / scale)
    train_weights, calibration_weights = torch.as_tensor(weights), torch.as_tensor(cal_weights)
    torch.manual_seed(training_seed)
    network = PlanScoreNetwork().cpu()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    def objective_loss(predictions, standardized_targets, row_weights):
        if objective == "relative_rank":
            return ranking_loss(predictions, standardized_targets, row_weights)
        terms = {"six_priority_mse": ((predictions - standardized_targets).square().mean(dim=(1, 2)) * row_weights).sum()}
        return terms["six_priority_mse"], terms

    def scores(raw_predictions):
        return _relative_predictions(raw_predictions) if objective == "relative_rank" else raw_predictions

    best_regret, best_loss, best_epoch, best_state, curve = np.inf, np.inf, 0, None, []
    for epoch in range(1, int(epochs) + 1):
        network.train()
        optimizer.zero_grad()
        train_loss, train_terms = objective_loss(scores(network(train_x)), train_y, train_weights)
        train_loss.backward()
        optimizer.step()
        record = {"epoch": epoch, "training_loss": float(train_loss.detach()),
                  **{f"training_{name}": float(value.detach()) for name, value in train_terms.items()},
                  "calibration_loss": None, "calibration_primary_regret": None,
                  "calibration_all_state_primary_regret": None,
                  "calibration_primary_mse": None, "calibration_pairwise_ranking": None,
                  "calibration_auxiliary_mse": None, "calibration_six_priority_mse": None,
                  "selected_checkpoint": False}
        if epoch % EVALUATION_INTERVAL == 0 or epoch == epochs:
            network.eval()
            with torch.inference_mode():
                predictions = scores(network(cal_x))
                cal_loss, cal_terms = objective_loss(predictions, cal_y, calibration_weights)
            calibration_loss = float(cal_loss)
            recovered_predictions = predictions.numpy() * scale
            regret = teacher_primary_regret(recovered_predictions[selection_in_calibration],
                                            original_y[selection_mask], selection_weights)
            all_state_regret = teacher_primary_regret(recovered_predictions, original_y[calibration_mask], cal_weights)
            record.update({"calibration_loss": calibration_loss, "calibration_primary_regret": regret,
                           "calibration_all_state_primary_regret": all_state_regret,
                           **{f"calibration_{name}": float(value) for name, value in cal_terms.items()}})
            improves = regret < best_regret - REGRET_TOLERANCE
            if improves:
                best_regret, best_loss, best_epoch = regret, calibration_loss, epoch
                best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
                record["selected_checkpoint"] = True
        curve.append(record)
        if best_state is not None and epoch - best_epoch >= patience:
            break
    network.load_state_dict(best_state)
    network.eval()
    student = ObservableScoreStudent(network, mean.astype(np.float32), std, scale,
                                     forecast_only, int(training_seed), best_epoch, objective)
    metadata = {"objective": objective,
                "objective_name": OBJECTIVE_NAME if objective == "relative_rank" else MSE_OBJECTIVE_NAME,
                "baseline_plan_index": BASELINE_PLAN_INDEX if objective == "relative_rank" else None,
                "baseline_name": "pure_ppo_601" if objective == "relative_rank" else None,
                "best_calibration_primary_regret": best_regret,
                "best_calibration_loss": best_loss, "selected_epoch": best_epoch,
                "actual_epochs": epoch, "maximum_epochs": int(epochs), "patience": int(patience),
                "evaluation_interval": EVALUATION_INTERVAL,
                "checkpoint_selection": "selection_subset_family_equal_teacher_primary_regret_only_ties_keep_earliest",
                "regret_tolerance": REGRET_TOLERANCE,
                "training_seed": int(training_seed), "forecast_only": bool(forecast_only),
                "training_families": len(train_families), "calibration_families": len(cal_families),
                "training_family_ids": sorted(str(value) for value in train_families),
                "calibration_family_ids": sorted(str(value) for value in cal_families),
                "training_rows": int(train_mask.sum()), "calibration_rows": int(calibration_mask.sum()),
                "selection_rows": int(selection_mask.sum()),
                "selection_families": len(set(families[selection_mask])),
                "selection_family_ids": sorted(str(value) for value in set(families[selection_mask])),
                "normalization": "train_family_equal_input_mean_std_and_target_std_across_plans",
                "loss_weights": ({"primary_mse": PRIMARY_MSE_WEIGHT,
                                 "gap_weighted_pairwise_logistic": PAIRWISE_WEIGHT,
                                 "auxiliary_mse": AUXILIARY_MSE_WEIGHT} if objective == "relative_rank"
                                 else {"six_priority_mse": 1.0}),
                "pairwise_weighting": ("abs_standardized_true_primary_gap_normalized_per_row_ties_zero"
                                       if objective == "relative_rank" else None),
                "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "batch": "full", "device": "cpu", "plan_count": 17,
                "priority_fields": list(PRIORITY_FIELDS), "architecture": [49, 64, 64, 102],
                "output_contract": ("six_signed_priority_differences_baseline_exact_zero_no_probability_or_mask"
                                    if objective == "relative_rank"
                                    else "six_min_relative_priorities_no_probability_or_mask")}
    return student, curve, metadata


def fit_relative_student(observations, targets, families, train_mask, calibration_mask,
                         training_seed, forecast_only=False, epochs=600, patience=100,
                         selection_mask=None):
    """Compatibility entry point for the fixed baseline-relative objective."""
    return fit_score_student(observations, targets, families, train_mask, calibration_mask,
                             training_seed, forecast_only, epochs, patience,
                             objective="relative_rank", selection_mask=selection_mask)
