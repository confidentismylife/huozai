"""Lightweight three-class worth-of-proposal model; no safety probabilities."""
from dataclasses import dataclass
import numpy as np
import torch

BASE = 12
MIN_GAIN = .001
REGULARIZATION = .01
WORTH_THRESHOLD = .80
HARM_THRESHOLD = .10


def proposal(gain):
    gain = np.asarray(gain, float)
    if gain.shape != (17,) or not np.isfinite(gain).all() or gain[BASE] != 0:
        raise ValueError("finite relative17 gain with zero baseline required")
    best = min(range(17), key=lambda p: (-gain[p], p != BASE, p))
    return best if gain[best] > MIN_GAIN else BASE


def meta_features(observation, gain, plan):
    x = np.asarray(observation, float).copy()
    gain = np.asarray(gain, float)
    if x.shape != (49,) or gain.shape != (17,) or plan != proposal(gain) or plan == BASE:
        raise ValueError("one aligned eligible observable proposal required")
    if not np.isfinite(x).all():
        raise ValueError("finite observation required")
    x[28:33] = 0
    onehot = np.eye(17)[plan]
    gap = gain[plan] - np.max(np.delete(gain, plan))
    return np.concatenate([x, onehot, [gain[plan], gap]])


def labels(gain):
    gain = np.asarray(gain, float)
    if not np.isfinite(gain).all():
        raise ValueError("finite empirical gains required")
    return np.where(gain < -1e-12, 0, np.where(gain > MIN_GAIN, 2, 1)).astype(int)


def filtered_choice(existing_choice, probabilities):
    p = np.asarray(probabilities, float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("three advice class scores required")
    return int(existing_choice) if p[2] >= WORTH_THRESHOLD and p[0] <= HARM_THRESHOLD else BASE


@dataclass
class WorthModel:
    mean: np.ndarray
    std: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray

    def probabilities(self, features):
        scores = ((np.asarray(features, float) - self.mean)/self.std) @ self.coefficient + self.intercept
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=-1, keepdims=True)

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, std=self.std, coefficient=self.coefficient, intercept=self.intercept)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k].copy() for k in ("mean", "std", "coefficient", "intercept")))


def fit_worth(features, target, families):
    x, y, families = np.asarray(features, float), np.asarray(target, int), np.asarray(families)
    if x.shape != (len(y), 68) or len(families) != len(y) or not np.isfinite(x).all() or set(y) != {0, 1, 2}:
        raise ValueError("aligned finite68 features and all three classes required")
    unique, counts = np.unique(families, return_counts=True)
    lookup = dict(zip(unique, counts))
    w = np.asarray([1/(len(unique)*lookup[f]) for f in families])
    mean = (x*w[:, None]).sum(axis=0)
    std = np.sqrt(((x-mean)**2*w[:, None]).sum(axis=0)); std = np.where(std < .001, 1., std)
    tx, ty, tw = torch.as_tensor((x-mean)/std), torch.as_tensor(y), torch.as_tensor(w)
    coefficient = torch.zeros((68, 3), dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficient, intercept], lr=1., max_iter=300,
        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.cross_entropy(tx@coefficient+intercept, ty, reduction="none")*tw).sum()
        loss = loss + .5*REGULARIZATION*coefficient.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    loss = float(closure().detach())
    gradient = max(float(coefficient.grad.abs().max()), float(intercept.grad.abs().max()))
    state = optimizer.state[coefficient]
    return WorthModel(mean, std, coefficient.detach().numpy(), intercept.detach().numpy()), {
        "states": len(y), "families": len(unique), "class_counts": {str(c): int((y == c).sum()) for c in range(3)},
        "iterations": int(state["n_iter"]), "loss": loss, "max_abs_gradient": gradient,
        "training_family_ids": sorted(set(families.tolist())), "regularization": REGULARIZATION}
