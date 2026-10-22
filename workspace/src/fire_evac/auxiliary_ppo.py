from __future__ import annotations

import numpy as np

from .supervised_gate import balanced_class_weights


def _require_sb3():
    try:
        import torch
        import torch.nn.functional as functional
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("AuxiliarySupervisedPPO requires the project's RL dependencies") from exc
    return torch, functional, PPO


_, _, _PPO = _require_sb3()


class AuxiliarySupervisedPPO(_PPO):
    """PPO with balanced behavior-cloning refreshes after every PPO update."""

    def configure_auxiliary_supervision(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        coefficient: float = 0.3,
        batches_per_update: int = 8,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        x = np.asarray(observations, dtype=np.float32)
        y = np.asarray(actions, dtype=np.int64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
            raise ValueError("observations/actions have incompatible shapes")
        if coefficient <= 0 or batches_per_update < 1 or batch_size < 1:
            raise ValueError("auxiliary hyperparameters must be positive")
        self.auxiliary_observations = x
        self.auxiliary_actions = y
        self.auxiliary_coefficient = float(coefficient)
        self.auxiliary_batches_per_update = int(batches_per_update)
        self.auxiliary_batch_size = int(batch_size)
        self.auxiliary_rng = np.random.default_rng(seed)
        classes = int(getattr(self.action_space, "n", int(y.max()) + 1))
        self.auxiliary_class_weights = balanced_class_weights(y, classes=classes)
        self.auxiliary_update_count = 0

    def _auxiliary_train(self) -> tuple[float, float]:
        torch, functional, _ = _require_sb3()
        required = (
            "auxiliary_observations",
            "auxiliary_actions",
            "auxiliary_class_weights",
            "auxiliary_rng",
        )
        if not all(hasattr(self, name) for name in required):
            return float("nan"), float("nan")
        self.policy.set_training_mode(True)
        weights = torch.as_tensor(self.auxiliary_class_weights, device=self.device)
        losses: list[float] = []
        correct = seen = 0
        for _ in range(self.auxiliary_batches_per_update):
            replace = len(self.auxiliary_actions) < self.auxiliary_batch_size
            indices = self.auxiliary_rng.choice(
                len(self.auxiliary_actions), size=self.auxiliary_batch_size, replace=replace
            )
            batch_x = torch.as_tensor(self.auxiliary_observations[indices], device=self.device)
            batch_y = torch.as_tensor(self.auxiliary_actions[indices], device=self.device)
            logits = self.policy.get_distribution(batch_x).distribution.logits
            cross_entropy = functional.cross_entropy(logits, batch_y, weight=weights)
            loss = self.auxiliary_coefficient * cross_entropy
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(float(cross_entropy.detach().cpu()))
            correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            seen += len(indices)
            self.auxiliary_update_count += 1
        return float(np.mean(losses)), correct / seen

    def train(self) -> None:
        super().train()
        loss, accuracy = self._auxiliary_train()
        if np.isfinite(loss):
            self.logger.record("train/auxiliary_cross_entropy", loss)
            self.logger.record("train/auxiliary_batch_accuracy", accuracy)
            self.logger.record("train/auxiliary_updates", self.auxiliary_update_count)
