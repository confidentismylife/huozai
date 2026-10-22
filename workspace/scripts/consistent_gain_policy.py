"""One score for plan nomination and override; cached development only."""
from dataclasses import dataclass
import numpy as np

BASE = 12
FLOOR = .001
MIN_GAIN = .001
ALPHA = .01


def validate_gain(gain):
    gain = np.asarray(gain, float)
    if gain.shape != (17,) or not np.isfinite(gain).all() or gain[BASE] != 0:
        raise ValueError("17 finite primary gains with exactly zero baseline required")
    return gain


def winner(scores):
    scores = validate_gain(scores)
    return min(range(17), key=lambda p: (-scores[p], p != BASE, p))


def decide(gains, penalties):
    gains = validate_gain(gains)
    penalty = np.asarray(penalties, float)
    if penalty.ndim == 0:
        penalty = np.full(17, float(penalty))
    if penalty.shape != (17,) or not np.isfinite(penalty).all() or np.any(penalty < 0):
        raise ValueError("finite nonnegative penalty required")
    scores = gains - penalty
    scores[BASE] = 0.
    nominee = winner(scores)
    execution = nominee if scores[nominee] > MIN_GAIN else BASE
    return {"raw_gain_choice": winner(gains), "consistent_nominee": nominee, "execution": execution,
            "scores": scores, "selected_score_pp": float(scores[execution] * 100),
            "nominee_score_pp": float(scores[nominee] * 100)}


def ensemble_scores(gains, k):
    gains = np.asarray(gains, float)
    if gains.ndim != 3 or gains.shape[1:] != (17, 6) or len(gains) < 2 or not np.isfinite(gains).all() or k < 1 or not np.isfinite(k):
        raise ValueError("model x17x6 finite gains and k>=1 required")
    mean = gains[..., 0].mean(axis=0)
    sd = gains[..., 0].std(axis=0, ddof=1)
    return decide(mean, k * np.maximum(sd, FLOOR))


def family_margin(prediction, actual, families):
    prediction, actual, families = np.asarray(prediction, float), np.asarray(actual, float), np.asarray(families)
    if prediction.shape != actual.shape or prediction.ndim != 2 or prediction.shape[1] != 17 or len(families) != len(prediction):
        raise ValueError("aligned state x17 gains and families required")
    if not np.isfinite(prediction).all() or not np.isfinite(actual).all():
        raise ValueError("finite residual inputs required")
    chosen = np.asarray([winner(g) for g in prediction])
    residual = prediction[np.arange(len(chosen)), chosen] - actual[np.arange(len(chosen)), chosen]
    scores = {str(f): float(residual[families == f].max()) for f in sorted(set(families))}
    quantile = float(np.quantile(list(scores.values()), .8, method="higher"))
    return max(FLOOR, quantile), {"family_max_nominee_optimism": scores, "empirical_quantile": quantile,
        "nominees": chosen, "quantile": .8, "method": "higher", "no_safety_coverage_guarantee": True}


def features(x):
    x = np.asarray(x, float).copy()
    if x.shape[-1] != 49 or not np.isfinite(x).all():
        raise ValueError("finite observable49 inputs required")
    x[..., 28:33] = 0
    return x


def weights(families):
    names, counts = np.unique(families, return_counts=True)
    count_map = dict(zip(names, counts))
    return np.asarray([1. / (len(names) * count_map[f]) for f in families])


@dataclass
class GainRidge:
    mean: np.ndarray
    std: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, observation):
        prediction = ((features(observation) - self.mean) / self.std) @ self.coefficients + self.intercept
        prediction[..., BASE] = 0
        return prediction

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, std=self.std, intercept=self.intercept, coefficients=self.coefficients)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k].copy() for k in ("mean", "std", "intercept", "coefficients")))


def fit_gain(observations, targets, families, forbidden_families=()):
    x, y, families = features(observations), np.asarray(targets, float), np.asarray(families)
    if y.shape != (len(x), 17) or not np.isfinite(y).all() or np.any(y[:, BASE] != 0):
        raise ValueError("finite primary gains with zero baseline required")
    if len(families) != len(x) or not len(x) or set(families).intersection(forbidden_families):
        raise ValueError("fit rows must exclude all held-out families")
    w = weights(families)
    mean = (x * w[:, None]).sum(axis=0)
    std = np.sqrt(((x - mean)**2 * w[:, None]).sum(axis=0))
    std = np.where(std < 1e-3, 1., std)
    standardized = (x - mean) / std
    intercept = (y * w[:, None]).sum(axis=0)
    coefficients = np.linalg.solve(standardized.T @ (w[:, None] * standardized) + ALPHA * np.eye(49),
                                   standardized.T @ (w[:, None] * (y - intercept)))
    coefficients[:, BASE] = 0.; intercept[BASE] = 0.
    return GainRidge(mean, std, intercept, coefficients)
