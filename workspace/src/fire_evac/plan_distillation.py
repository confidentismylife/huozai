"""Observable finite-plan score distillation; no calibrated safety probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


PRIORITY_FIELDS = ("worst_incapacitation", "mean_incapacitation", "worst_fed", "mean_fed",
                   "worst_remaining", "worst_congestion")
STUDENT_SEEDS = tuple(range(801, 806))


def observable_features(observations, forecast_only=False):
    result = np.asarray(observations, dtype=np.float32).copy()
    if result.shape[-1] != 49 or not np.isfinite(result).all():
        raise ValueError("finite 49-input observations required")
    result[..., 28:33] = 0
    if forecast_only:
        forecasts = result[..., 19:28].copy()
        result[...] = 0
        result[..., 19:28] = forecasts
    return result


def relative_priority_targets(risks, supported):
    risks = np.asarray(risks, dtype=np.float64)
    supported = np.asarray(supported, dtype=bool)
    if risks.ndim != 4 or risks.shape[0] != 2 or risks.shape[1] != 17 or supported.shape != (2,) or not supported.any():
        raise ValueError("two-world seventeen-plan risks and nonempty support required")
    means = risks[supported].mean(axis=2)
    if not np.isfinite(means).all():
        raise ValueError("supported planning scores must be finite")
    average, worst = means.mean(axis=0), means.max(axis=0)
    scores = np.stack([worst[:, 0], average[:, 0], worst[:, 1], average[:, 1],
                       worst[:, 3], worst[:, 4]], axis=1)
    return scores - scores.min(axis=0, keepdims=True)


def choose_priority(scores):
    scores = np.asarray(scores)
    if scores.shape != (17, 6) or not np.isfinite(scores).all():
        raise ValueError("seventeen finite six-field plan priorities required")
    return min(range(17), key=lambda p: tuple(scores[p]) + (p,))


def family_weights(families):
    families = np.asarray(families)
    unique, counts = np.unique(families, return_counts=True)
    lookup = dict(zip(unique, counts))
    return np.asarray([1 / (len(unique) * lookup[f]) for f in families], dtype=np.float32)


def stratified_family_split(scenario_records, seed=20261600):
    rng = np.random.default_rng(seed)
    split = {}
    for kind in sorted({s["conflict_type"] for s in scenario_records}):
        ids = sorted(s["scenario_id"] for s in scenario_records if s["conflict_type"] == kind)
        if len(ids) != 6:
            raise ValueError("six old families per conflict required for fixed 4/2 split")
        ids = np.asarray(ids)[rng.permutation(6)]
        split.update({str(s): "train" if i < 4 else "calibration" for i, s in enumerate(ids)})
    return split


class PlanScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(49, 64), nn.ReLU(), nn.Linear(64, 64),
                                    nn.ReLU(), nn.Linear(64, 17 * 6))

    def forward(self, values):
        return self.layers(values).reshape(-1, 17, 6)


@dataclass
class ObservablePlanStudent:
    network: PlanScoreNetwork
    input_mean: np.ndarray
    input_std: np.ndarray
    target_scale: np.ndarray
    forecast_only: bool
    training_seed: int
    selected_epoch: int

    def priorities(self, observation):
        x = observable_features(observation, self.forecast_only)
        if x.shape != (49,):
            raise ValueError("one observation required for selecting a plan")
        self.network.eval()
        with torch.inference_mode():
            result = self.network(torch.as_tensor((x - self.input_mean) / self.input_std).unsqueeze(0))[0]
        return result.numpy() * self.target_scale

    def choose(self, observation):
        return choose_priority(self.priorities(observation))

    def save(self, path):
        torch.save({"state_dict": self.network.state_dict(), "input_mean": torch.as_tensor(self.input_mean),
                    "input_std": torch.as_tensor(self.input_std), "target_scale": torch.as_tensor(self.target_scale),
                    "forecast_only": self.forecast_only, "training_seed": self.training_seed,
                    "selected_epoch": self.selected_epoch, "plan_count": 17, "priority_fields": PRIORITY_FIELDS}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["plan_count"] != 17 or tuple(saved["priority_fields"]) != PRIORITY_FIELDS:
            raise ValueError("student plan contract differs")
        network = PlanScoreNetwork()
        network.load_state_dict(saved["state_dict"])
        network.eval()
        return cls(network, saved["input_mean"].numpy(), saved["input_std"].numpy(), saved["target_scale"].numpy(),
                   saved["forecast_only"], saved["training_seed"], saved["selected_epoch"])


def fit_student(observations, targets, families, train_mask, calibration_mask, training_seed,
                forecast_only=False, epochs=600, patience=100):
    """Only old train/calibration arrays enter; no audit input or simulator API."""
    train_mask, calibration_mask = np.asarray(train_mask, bool), np.asarray(calibration_mask, bool)
    families = np.asarray(families)
    if np.any(train_mask & calibration_mask) or not train_mask.any() or not calibration_mask.any():
        raise ValueError("nonempty disjoint training and calibration required")
    if set(families[train_mask]) & set(families[calibration_mask]):
        raise ValueError("a scenario family cannot span training and calibration")
    x = observable_features(observations, forecast_only)
    y = np.asarray(targets, dtype=np.float32)
    if y.shape != (len(x), 17, 6) or not np.isfinite(y).all():
        raise ValueError("aligned finite relative plan targets required")
    weights = family_weights(families[train_mask])
    cal_weights = family_weights(families[calibration_mask])
    mean = np.sum(x[train_mask] * weights[:, None], axis=0)
    std = np.sqrt(np.sum((x[train_mask] - mean) ** 2 * weights[:, None], axis=0))
    std = np.where(std < 1e-3, 1, std).astype(np.float32)
    target_mean = np.sum(y[train_mask].mean(axis=1) * weights[:, None], axis=0)
    target_variance = np.sum(((y[train_mask] - target_mean) ** 2).mean(axis=1) * weights[:, None], axis=0)
    target_scale = np.maximum(np.sqrt(target_variance), 1e-3).astype(np.float32)
    train_x = torch.as_tensor((x[train_mask] - mean) / std)
    cal_x = torch.as_tensor((x[calibration_mask] - mean) / std)
    train_y, cal_y = torch.as_tensor(y[train_mask] / target_scale), torch.as_tensor(y[calibration_mask] / target_scale)
    train_weights, calibration_weights = torch.as_tensor(weights), torch.as_tensor(cal_weights)
    torch.manual_seed(training_seed)
    network = PlanScoreNetwork()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=.001)
    best_loss, best_epoch, best_state, curve = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        network.train()
        optimizer.zero_grad()
        train_loss = (((network(train_x) - train_y) ** 2).mean(dim=(1, 2)) * train_weights).sum()
        train_loss.backward()
        optimizer.step()
        if epoch % 10 != 0 and epoch != epochs:
            continue
        network.eval()
        with torch.inference_mode():
            calibration_loss = float(((((network(cal_x) - cal_y) ** 2).mean(dim=(1, 2))) * calibration_weights).sum())
        curve.append({"epoch": epoch, "training_loss": float(train_loss.detach()), "calibration_loss": calibration_loss})
        if calibration_loss < best_loss - 1e-8:
            best_loss, best_epoch = calibration_loss, epoch
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    network.load_state_dict(best_state)
    network.eval()
    student = ObservablePlanStudent(network, mean.astype(np.float32), std, target_scale, forecast_only, training_seed, best_epoch)
    return student, curve, {"best_calibration_loss": best_loss, "selected_epoch": best_epoch,
                            "actual_epochs": epoch, "training_families": len(set(families[train_mask])),
                            "calibration_families": len(set(families[calibration_mask]))}
