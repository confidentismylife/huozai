"""Development teacher and ternary predictor. No certified safety guarantees."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .counterfactual_shield import (
    ShieldNetwork, evaluate_counterfactual_actions, material_safety_labels,
    mirror_shield_observation, reactive_followup_action,
)

OBSERVATION_DIM = 49
UNSAFE, SAFE, UNKNOWN = 0, 1, 2
ACTION_MIRROR = np.asarray([4, 5, 6, 7, 0, 1, 2, 3, 8])


class ObservableHistoryWrapper(gym.Wrapper):
    """49 inputs from sensors, elapsed time, and known actuator feedback.

    Columns 34:36: clock, remaining sign hold; 36:39: actual sign feedback;
    39:44: accumulated observed zone smoke / maximum episode steps;
    44:49: observed smoke differences. No individual FED or latent attributes.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.observation_space.shape != (34,):
            raise ValueError("history wrapper requires the fair 34-input environment")
        config = self.unwrapped.domain_env.config
        self.max_steps = config.max_steps
        self.min_hold = config.min_sign_hold_steps
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, np.zeros(10), np.full(5, -np.inf)]).astype(np.float32),
            high=np.concatenate([env.observation_space.high, np.ones(5), np.full(10, np.inf)]).astype(np.float32),
            dtype=np.float32,
        )
        self.elapsed = 0
        self.sign = 2
        self.since_switch = self.min_hold
        self.smoke_sum = np.zeros(5, dtype=np.float32)
        self.previous_smoke = np.zeros(5, dtype=np.float32)

    def _augment(self, observation, difference):
        hold = max(0, self.min_hold - self.since_switch) / max(1, self.min_hold)
        sign = np.eye(3, dtype=np.float32)[self.sign]
        history = np.concatenate([[self.elapsed / self.max_steps, hold], sign,
                                  self.smoke_sum / self.max_steps, difference])
        return np.concatenate([observation, history]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.elapsed = 0
        self.sign = int(info["last_action"])
        self.since_switch = self.min_hold
        self.smoke_sum = np.zeros(5, dtype=np.float32)
        self.previous_smoke = observation[5:10].copy()
        return self._augment(observation, np.zeros(5)), info

    def step(self, action):
        observation, reward, term, trunc, info = self.env.step(action)
        actual = int(info["actual_sign_action"])
        self.since_switch = 0 if actual != self.sign else self.since_switch + 1
        self.sign = actual
        self.elapsed += 1
        # Integrate only the last sensor reading, observed before this transition.
        self.smoke_sum += self.previous_smoke
        difference = observation[5:10] - self.previous_smoke
        self.previous_smoke = observation[5:10].copy()
        return self._augment(observation, difference), reward, term, trunc, info


def mirror_history_observation(observation):
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (OBSERVATION_DIM,):
        raise ValueError("49 inputs required")
    result = mirror_shield_observation(raw)
    for left, right in ((36, 37), (42, 43), (47, 48)):
        result[left], result[right] = raw[right], raw[left]
    return result


def shield_input(observation, use_history=True):
    result = np.asarray(observation, dtype=np.float32).copy()
    if result.shape[-1] != OBSERVATION_DIM:
        raise ValueError("49 inputs required")
    result[..., 28:33] = 0
    if not use_history:
        result[..., 34:] = 0
    return result


def full_release_followup(observation):
    action = reactive_followup_action(observation)
    return action if action == 8 else (action // 4) * 4 + 3


def consensus_labels(teacher_labels):
    """Unanimous relative safe/unsafe classifications; everything else unknown.

    Does not force a safe winner when sampled teachers disagree. A consensus
    across a finite set of sampled continuations is not a risk bound.
    """
    labels = np.asarray(teacher_labels)
    if labels.ndim != 2 or labels.shape[1] != 9 or len(labels) < 2:
        raise ValueError("at least two nine-action teachers required")
    consensus = np.full(9, -1, dtype=np.int8)
    consensus[np.all(labels == SAFE, axis=0)] = SAFE
    consensus[np.all(labels == UNSAFE, axis=0)] = UNSAFE
    return consensus


def terminal_counterfactual_teachers(env, future_seeds):
    """Candidate actions share each sampled future's starting RNG stream.

    Re-seeds only future simulator decisions and sensor noise; already sampled
    individual attributes remain fixed. Different actions may consume draws
    differently. Both continuation plans run through termination/truncation.
    """
    if len(future_seeds) < 2 or len(set(future_seeds)) != len(future_seeds):
        raise ValueError("at least two distinct future seeds required")
    population = env.unwrapped.domain_env.config.population
    horizon = env.unwrapped.domain_env.config.max_steps
    evaluations = []
    labels = []
    for seed in future_seeds:
        for name, continuation in (("graded_reactive", reactive_followup_action), ("full_release_reactive", full_release_followup)):
            clone = copy.deepcopy(env)
            clone.unwrapped.domain_env.rng = np.random.default_rng(seed)
            # Confirmatory sensor RNG is a separate stream.
            base = clone
            while isinstance(base, gym.Wrapper):
                if "rng" in vars(base):
                    base.rng = np.random.default_rng(seed + 9_000_001)
                base = base.env
            results = evaluate_counterfactual_actions(clone, horizon, continuation)
            clone.close()
            if not all(r.terminal_reached for r in results):
                raise RuntimeError("terminal teacher failed to reach the episode boundary")
            labels.append(material_safety_labels(results, population))
            evaluations.append((int(seed), name, results))
    labels = np.asarray(labels, dtype=np.int8)
    return consensus_labels(labels), labels, tuple(evaluations)


class TernaryShieldNetwork(ShieldNetwork):
    def __init__(self):
        super().__init__(OBSERVATION_DIM)
        self.action_head[-1] = nn.Linear(64, 3)


def ternary_action_mask(probabilities, unsafe_threshold, abstention_threshold=.5):
    """Explicit predicted-unknown abstention and empty-set fallback.

    Abstention is based on the *predicted* unknown class, not on oracle labels.
    If all actions are excluded, retain those with minimum unsafe score and log
    that forced fallback; the chosen action has no safety guarantee.
    """
    probabilities = np.asarray(probabilities)
    if probabilities.shape[-2:] != (9, 3) or not np.all(np.isfinite(probabilities)):
        raise ValueError("finite nine-action, three-class predictions required")
    abstain = probabilities[..., UNKNOWN] >= abstention_threshold
    rejected = (probabilities[..., UNSAFE] >= unsafe_threshold) & ~abstain
    mask = ~rejected
    empty = ~mask.any(axis=-1, keepdims=True)
    unsafe = probabilities[..., UNSAFE]
    mask |= empty & (unsafe == unsafe.min(axis=-1, keepdims=True))
    return mask, abstain, empty.squeeze(-1)


def ternary_mask_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels)
    mask, abstain, empty = ternary_action_mask(probabilities, threshold)
    def rejected(selector):
        return float((~mask)[selector].mean()) if np.any(selector) else None
    chosen = np.argmin(np.where(mask, probabilities[..., UNSAFE], np.inf), axis=-1)
    top = labels[np.arange(len(labels)), chosen]
    return {
        "states": len(labels), "unsafe_actions": int((labels == 0).sum()),
        "safe_actions": int((labels == 1).sum()), "indeterminate_actions": int((labels < 0).sum()),
        "unsafe_action_recall": rejected(labels == 0),
        "safe_action_false_rejection_rate": rejected(labels == 1),
        "top_action_unsafe_rate": float((top == 0).mean()),
        "top_action_indeterminate_rate": float((top < 0).mean()),
        "indeterminate_action_rejection_rate": rejected(labels < 0),
        "predicted_abstention_rate": float(abstain.mean()),
        "empty_raw_mask_rate": float(empty.mean()), "mean_allowed_actions": float(mask.sum(axis=-1).mean()),
        "known_label_coverage": float((labels >= 0).mean()),
    }


@dataclass
class TernaryShieldPredictor:
    models: tuple[TernaryShieldNetwork, ...]
    mean: np.ndarray
    scale: np.ndarray
    threshold: float
    use_history: bool = True

    def probabilities(self, observations):
        raw = np.asarray(observations, dtype=np.float32)
        single = raw.ndim == 1
        if single:
            raw = raw[None, :]
        direct = shield_input(raw, self.use_history)
        reflected = np.asarray([mirror_history_observation(x) for x in raw])
        reflected = shield_input(reflected, self.use_history)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                forward = torch.softmax(model(torch.as_tensor((direct - self.mean) / self.scale)), dim=-1).cpu().numpy()
                backward = torch.softmax(model(torch.as_tensor((reflected - self.mean) / self.scale)), dim=-1).cpu().numpy()
                predictions.append(.5 * (forward + backward[:, ACTION_MIRROR, :]))
        result = np.mean(predictions, axis=0)
        return result[0] if single else result

    def mask(self, observation):
        return ternary_action_mask(self.probabilities(observation), self.threshold)[0]

    def save(self, path):
        torch.save({"format_version": 1, "input_contract": "observable_history_49",
                    "state_dicts": [m.state_dict() for m in self.models],
                    "mean": torch.as_tensor(self.mean), "scale": torch.as_tensor(self.scale),
                    "threshold": float(self.threshold), "use_history": self.use_history}, Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["input_contract"] != "observable_history_49":
            raise ValueError("incompatible ternary shield format")
        models = []
        for state in payload["state_dicts"]:
            model = TernaryShieldNetwork()
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        return cls(tuple(models), payload["mean"].numpy(), payload["scale"].numpy(), float(payload["threshold"]), bool(payload["use_history"]))
