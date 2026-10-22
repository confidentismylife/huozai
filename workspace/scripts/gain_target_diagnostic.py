"""Arithmetic diagnosis only; this module is never a control policy."""
import numpy as np


def decompose(predicted, risks, actual, supported, true_source):
    risks, actual = np.asarray(risks, float), np.asarray(actual, float)
    support = np.asarray(supported)
    if (risks.ndim != 4 or risks.shape[:2] != (2, 2) or risks.shape[2] < 1
            or risks.shape[3] < 1 or actual.ndim != 3 or actual.shape[0] != 2
            or actual.shape[1] < 1 or actual.shape[2] != risks.shape[3]
            or support.shape != (2,) or support.dtype != bool or not support.any()
            or true_source not in (0, 1) or not support[true_source]
            or not np.isfinite(predicted) or not np.isfinite(risks).all()
            or not np.isfinite(actual).all()):
        raise ValueError("finite two-world/two-policy risks and supported diagnostic true source required")
    means = risks.mean(axis=2)[:, :, 0]
    world_gains = means[:, 0] - means[:, 1]
    worst = means[support].max(axis=0)
    robust = float(worst[0] - worst[1])
    truth = float(world_gains[true_source])
    a = actual.mean(axis=1)[:, 0]
    observed = float(a[0] - a[1])
    sources = np.flatnonzero(support)
    worst_sources = sources[np.argmax(means[support], axis=0)]
    return {"predicted_gain_pp": float(predicted * 100), "robust_gain_pp": robust * 100,
        "true_planning_gain_pp": truth * 100, "actual_gain_pp": observed * 100,
        "prediction_to_robust_pp": float((predicted - robust) * 100),
        "robust_to_true_pp": (robust - truth) * 100,
        "true_to_actual_pp": (truth - observed) * 100,
        "total_error_pp": float((predicted - observed) * 100),
        "minimum_world_gain_pp": float(world_gains[support].min() * 100),
        "world0_gain_pp": float(world_gains[0] * 100), "world1_gain_pp": float(world_gains[1] * 100),
        "baseline_worst_source": int(worst_sources[0]), "proposal_worst_source": int(worst_sources[1]),
        "different_worst_sources": bool(worst_sources[0] != worst_sources[1])}


def gain_class(gain_pp):
    if not np.isfinite(gain_pp):
        raise ValueError("finite gain required")
    return "harmful" if gain_pp < -1e-10 else "beneficial" if gain_pp > .1 else "neutral_or_small"


def paired_future_interval(gains_pp, seed, draws=2000):
    gains = np.asarray(gains_pp, float)
    if gains.ndim != 2 or min(gains.shape) < 1 or not np.isfinite(gains).all():
        raise ValueError("population x future paired gains required")
    future_means = gains.mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = future_means[rng.integers(len(future_means), size=(draws, len(future_means)))].mean(axis=1)
    return np.quantile(samples, [.025, .975])
