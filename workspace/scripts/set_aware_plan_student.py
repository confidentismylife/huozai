"""Development plan-set supervision and empirical ensemble override margin.

This module fits observable arrays only, without an environment, audit outcomes,
or an LLM. Advice softmax scores and margin are not safety probabilities/bounds.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.plan_distillation import observable_features, family_weights, choose_priority

BASE = 12
SET_TOLERANCE = .001
SCALE_FLOOR = .001
MIN_GAIN = .001
SEEDS = tuple(range(801, 806))
OBJECTIVES = ("single", "set")


class SetGainNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(49, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 119))

    def forward(self, x):
        raw = self.layers(x).reshape(-1, 17, 7)
        gains = raw[..., :6] - raw[:, BASE:BASE + 1, :6]
        return gains, raw[..., 6]


def training_targets(priorities, objective):
    y = np.asarray(priorities, np.float32)
    if y.ndim != 3 or y.shape[1:] != (17, 6) or not np.isfinite(y).all() or objective not in OBJECTIVES:
        raise ValueError("finite [N,17,6] priorities and single/set objective required")
    gains = y[:, BASE:BASE + 1] - y
    if objective == "set":
        acceptable = y[..., 0] - y[..., 0].min(axis=1, keepdims=True) <= SET_TOLERANCE + 1e-12
    else:
        acceptable = np.zeros(y.shape[:2], bool)
        acceptable[np.arange(len(y)), [choose_priority(row) for row in y]] = True
    return gains, acceptable


def set_mass_loss(logits, acceptable):
    if logits.shape != acceptable.shape or logits.ndim != 2 or not torch.all(acceptable.any(dim=1)):
        raise ValueError("aligned logits and nonempty acceptable plan sets required")
    log_probs = F.log_softmax(logits, dim=1)
    return -torch.logsumexp(log_probs.masked_fill(~acceptable, -torch.inf), dim=1)


def nominate(mean_gains, mean_probabilities):
    gains, probabilities = np.asarray(mean_gains), np.asarray(mean_probabilities)
    if gains.shape != (17, 6) or probabilities.shape != (17,) or not np.isfinite(gains).all() or not np.isfinite(probabilities).all():
        raise ValueError("finite 17-plan advice required")
    return min(range(17), key=lambda p: (-probabilities[p],) + tuple(-gains[p]) + (p,))


def ensemble_decision(gains, probabilities, k=1.):
    gains, probabilities = np.asarray(gains, float), np.asarray(probabilities, float)
    if gains.ndim != 3 or gains.shape[1:] != (17, 6) or len(gains) < 2 or probabilities.shape != gains.shape[:2]:
        raise ValueError("at least two aligned model predictions required")
    if not np.isfinite(gains).all() or not np.isfinite(probabilities).all() or not np.isfinite(k) or k < 1:
        raise ValueError("finite predictions and k>=1 required")
    mean = gains.mean(axis=0)
    std = gains[..., 0].std(axis=0, ddof=1)
    nominee = nominate(mean, probabilities.mean(axis=0))
    lower = float(mean[nominee, 0] - k * max(float(std[nominee]), SCALE_FLOOR))
    guarded = nominee if nominee != BASE and lower > MIN_GAIN else BASE
    return nominee, guarded, {"nominee_mean_gain_pp": float(mean[nominee, 0] * 100),
        "nominee_model_std_pp": float(std[nominee] * 100), "nominee_lower_score_pp": lower * 100,
        "k": float(k), "allowed_override": guarded != BASE, "blocked_nomination": nominee != BASE and guarded == BASE}


def calibrate_margin(gains, actual_gains, families):
    # Each family's worst optimistic residual across states AND all candidates.
    gains, actual_gains, families = np.asarray(gains, float), np.asarray(actual_gains, float), np.asarray(families)
    if gains.ndim != 4 or gains.shape[0] < 2 or gains.shape[2:] != (17, 6) or actual_gains.shape != gains.shape[1:]:
        raise ValueError("model x states x plans x six gains and actual gains required")
    if len(families) != len(actual_gains) or not np.isfinite(gains).all() or not np.isfinite(actual_gains).all():
        raise ValueError("aligned finite calibration values required")
    mean, std = gains[..., 0].mean(axis=0), gains[..., 0].std(axis=0, ddof=1)
    residual = (mean - actual_gains[..., 0]) / np.maximum(std, SCALE_FLOOR)
    scores = {str(f): float(residual[families == f].max()) for f in np.unique(families)}
    q = float(np.quantile(list(scores.values()), .80, method="higher"))
    return max(1., q), {"family_max_optimistic_residuals": scores, "quantile": .80, "method": "higher",
                        "empirical_quantile": q, "k": max(1., q), "no_safety_coverage_guarantee": True}


@dataclass
class SetGainStudent:
    network: SetGainNetwork
    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    objective: str
    seed: int
    selected_epoch: int

    def predictions(self, observation):
        x = observable_features(observation)
        if x.shape != (49,):
            raise ValueError("one observable state required")
        self.network.eval()
        with torch.inference_mode():
            gains, logits = self.network(torch.as_tensor((x - self.mean) / self.std).unsqueeze(0))
            return gains[0].numpy() * self.scale, torch.softmax(logits, dim=1)[0].numpy()

    def choose(self, observation):
        return nominate(*self.predictions(observation))

    def save(self, path):
        torch.save({"state_dict": self.network.state_dict(), "mean": torch.as_tensor(self.mean), "std": torch.as_tensor(self.std),
                    "scale": torch.as_tensor(self.scale), "objective": self.objective, "seed": self.seed,
                    "selected_epoch": self.selected_epoch, "contract": "observable_set_gain_17x7_v1"}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["contract"] == "observable_set_gain_17x7_v1"
        assert saved["objective"] in OBJECTIVES and saved["seed"] in SEEDS
        network = SetGainNetwork()
        network.load_state_dict(saved["state_dict"])
        network.eval()
        return cls(network, saved["mean"].numpy(), saved["std"].numpy(), saved["scale"].numpy(), saved["objective"], saved["seed"], saved["selected_epoch"])


def fit(observations, priorities, families, steps, train_mask, checkpoint_mask, seed, objective):
    x = observable_features(observations)
    families, steps = np.asarray(families), np.asarray(steps)
    train_mask, checkpoint_mask = np.asarray(train_mask, bool), np.asarray(checkpoint_mask, bool)
    if not np.all(train_mask | checkpoint_mask) or np.any(train_mask & checkpoint_mask):
        raise ValueError("fit accepts training and checkpoint families only")
    if set(families[train_mask]) & set(families[checkpoint_mask]) or not train_mask.any() or not checkpoint_mask.any():
        raise ValueError("nonempty disjoint families required")
    if seed not in SEEDS or objective not in OBJECTIVES:
        raise ValueError("frozen seed/objective required")
    gains, acceptable = training_targets(priorities, objective)
    w = family_weights(families[train_mask])
    mean = (x[train_mask] * w[:, None]).sum(axis=0)
    std = np.sqrt(((x[train_mask] - mean) ** 2 * w[:, None]).sum(axis=0))
    std = np.where(std < 1e-3, 1, std).astype(np.float32)
    y_mean = (gains[train_mask].mean(axis=1) * w[:, None]).sum(axis=0)
    scale = np.maximum(np.sqrt((((gains[train_mask] - y_mean) ** 2).mean(axis=1) * w[:, None]).sum(axis=0)), 1e-3).astype(np.float32)
    tx = torch.as_tensor((x[train_mask] - mean) / std)
    ty = torch.as_tensor(gains[train_mask] / scale)
    ta = torch.as_tensor(acceptable[train_mask])
    tw = torch.as_tensor(w)
    selection = checkpoint_mask & (steps == 0)
    cx = torch.as_tensor((x[selection] - mean) / std)
    cw = family_weights(families[selection])
    assert selection.any()
    torch.manual_seed(seed)
    network = SetGainNetwork()
    initial_hash = hashlib.sha256(b"".join(name.encode() + value.detach().numpy().tobytes()
        for name, value in sorted(network.state_dict().items()))).hexdigest()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=.001)
    best_regret, best_epoch, best_state, curve = np.inf, 0, None, []
    for epoch in range(1, 301):
        network.train()
        optimizer.zero_grad()
        prediction, logits = network(tx)
        primary = (prediction[..., 0] - ty[..., 0]).square().mean(dim=1)
        auxiliary = (prediction[..., 1:] - ty[..., 1:]).square().mean(dim=(1, 2))
        classification = set_mass_loss(logits, ta)
        loss = ((primary + .1 * auxiliary + classification) * tw).sum()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            network.eval()
            with torch.inference_mode():
                pred, logit = network(cx)
                pred = pred.numpy() * scale
                probability = torch.softmax(logit, dim=1).numpy()
            chosen = np.asarray([nominate(g, p) for g, p in zip(pred, probability)])
            primary_cost = np.asarray(priorities)[selection, :, 0]
            regret = float(((primary_cost[np.arange(len(chosen)), chosen] - primary_cost.min(axis=1)) * cw).sum())
            selected = regret < best_regret - 1e-12
            if selected:
                best_regret, best_epoch = regret, epoch
                best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
            curve.append({"epoch": epoch, "train_loss": float(loss.detach()), "train_primary_mse": float((primary * tw).sum().detach()),
                          "train_auxiliary_mse": float((auxiliary * tw).sum().detach()),
                          "train_classification_loss": float((classification * tw).sum().detach()),
                          "checkpoint_primary_regret_pp": regret * 100, "selected_checkpoint": selected})
        if best_state is not None and epoch - best_epoch >= 80:
            break
    network.load_state_dict(best_state)
    network.eval()
    return SetGainStudent(network, mean, std, scale, objective, seed, best_epoch), curve, {
        "selected_epoch": best_epoch, "actual_epochs": epoch, "best_checkpoint_primary_regret_pp": best_regret * 100,
        "training_families": len(set(families[train_mask])), "checkpoint_families": len(set(families[checkpoint_mask])),
        "fit_families": sorted(set(families.tolist())), "train_states": int(train_mask.sum()),
        "checkpoint_states": int(checkpoint_mask.sum()), "checkpoint_selection_states": int(selection.sum()),
        "initial_network_sha256": initial_hash}
