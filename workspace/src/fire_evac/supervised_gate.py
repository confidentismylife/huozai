from __future__ import annotations

from collections.abc import Mapping

import numpy as np


PLANS = ("trust", "override", "wait_then_trust")


def plan_action(plan: str, blockage_steps_remaining: int) -> int:
    if plan == "trust":
        return 0
    if plan == "override":
        return 1
    if plan == "wait_then_trust":
        return 2 if blockage_steps_remaining > 0 else 0
    raise ValueError(f"unknown counterfactual plan: {plan}")


def select_safest_plan(results: Mapping[str, Mapping[str, float]]) -> str:
    """Lexicographic safety objective: incapacity, FED, then reward."""
    if set(results) != set(PLANS):
        raise ValueError(f"results must contain exactly {PLANS}")
    for plan in PLANS:
        required = {"incapacitation_rate", "mean_cumulative_fed", "total_reward"}
        if not required.issubset(results[plan]):
            raise ValueError(f"missing counterfactual metrics for {plan}")
    return min(
        PLANS,
        key=lambda plan: (
            float(results[plan]["incapacitation_rate"]),
            float(results[plan]["mean_cumulative_fed"]),
            -float(results[plan]["total_reward"]),
            PLANS.index(plan),
        ),
    )


def balanced_class_weights(actions: np.ndarray, classes: int | None = None) -> np.ndarray:
    labels = np.asarray(actions, dtype=int)
    if classes is None:
        classes = int(labels.max()) + 1 if labels.size else 0
    if labels.ndim != 1 or labels.size == 0 or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("actions must be a non-empty vector of valid class indices")
    counts = np.bincount(labels, minlength=classes).astype(float)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = labels.size / (classes * counts[present])
    return weights


def policy_accuracy(model, observations: np.ndarray, actions: np.ndarray, batch_size: int = 1024) -> float:
    import torch

    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)
    predictions: list[np.ndarray] = []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], device=model.device)
            distribution = model.policy.get_distribution(batch)
            predictions.append(distribution.distribution.logits.argmax(dim=1).cpu().numpy())
    return float(np.mean(np.concatenate(predictions) == y))


def pretrain_ppo_gate(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    import torch
    import torch.nn.functional as functional

    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("observations/actions have incompatible shapes")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    classes = int(getattr(model.action_space, "n", int(y.max()) + 1))
    weights = torch.as_tensor(balanced_class_weights(y, classes=classes), device=model.device)
    rng = np.random.default_rng(seed)
    logs: list[dict[str, float | int]] = []
    model.policy.set_training_mode(True)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        losses: list[float] = []
        correct = 0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = torch.as_tensor(x[indices], device=model.device)
            batch_y = torch.as_tensor(y[indices], device=model.device)
            distribution = model.policy.get_distribution(batch_x)
            logits = distribution.distribution.logits
            loss = functional.cross_entropy(logits, batch_y, weight=weights)
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            seen += len(indices)
        logs.append(
            {
                "epoch": epoch,
                "cross_entropy": float(np.mean(losses)),
                "training_accuracy": correct / seen,
            }
        )
    model.policy.set_training_mode(False)
    return logs
