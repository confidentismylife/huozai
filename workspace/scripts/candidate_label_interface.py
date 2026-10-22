"""All-five candidate labels from planning data, never nominated-only data."""
import numpy as np

PPO_NAMES = tuple(f"pure_ppo_{seed}" for seed in range(601, 606))
DIAGNOSTIC_ROLE = "viewed_development_diagnostic_not_training"


def candidate_indices(names):
    names = tuple(names)
    if len(names) != 17 or len(set(names)) != 17 or not all(name in names for name in PPO_NAMES):
        raise ValueError("unique complete17 registry with all five frozen PPO names required")
    return np.array([names.index(name) for name in PPO_NAMES], dtype=int)


def observable_features(observation):
    x = np.asarray(observation, float).copy()
    if x.shape != (49,) or not np.isfinite(x).all():
        raise ValueError("finite observable49 required")
    x[28:33] = 0
    return x


def planning_labels(planning, names):
    """Read only observation, supported, planning_risks; emit every PPO candidate."""
    indices = candidate_indices(names)
    x = observable_features(planning["observation"])
    risks = np.asarray(planning["planning_risks"], float)
    support = np.asarray(planning["supported"])
    if (risks.ndim != 4 or risks.shape[:2] != (2, 17) or risks.shape[2] < 1 or risks.shape[3] < 1 or
            not np.isfinite(risks).all() or support.shape != (2,) or support.dtype != bool or not support.any()):
        raise ValueError("finite two-world17-plan planning data and nonempty boolean support required")
    losses = risks[:, :, :, 0][:, indices, :]
    if np.any(losses < 0) or np.any(losses > 1):
        raise ValueError("incapacitation losses must be fractions between zero and one")
    means = losses[support].mean(axis=2)
    worst = means.max(axis=0)
    R = worst[0] - worst
    M = (means[:, 0, None] - means).min(axis=0)
    R[0] = M[0] = 0
    return {"features": np.c_[np.tile(x, (5, 1)), np.eye(5)], "target_R": R, "target_M": M,
            "paired_future_gains": losses[:, 0, None, :] - losses,
            "world_mean_losses": losses.mean(axis=2), "supported": support.copy(), "plan_indices": indices}


def choose_candidate(gains, margin=0.):
    g = np.asarray(gains, float)
    if g.shape != (5,) or not np.isfinite(g).all() or g[0] != 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("five relative scores with zero601 and nonnegative margin required")
    scores = g - margin
    scores[0] = 0
    best = float(scores.max())
    tied = np.flatnonzero(best - scores <= 1e-12)
    selected = int(tied[0])
    return selected if scores[selected] > .001 else 0


def family_weights(families):
    f = np.asarray(families)
    if f.ndim != 1 or not len(f):
        raise ValueError("nonempty one-dimensional family IDs required")
    names, counts = np.unique(f, return_counts=True)
    sizes = dict(zip(names, counts))
    return np.array([1 / (len(names) * sizes[name]) for name in f])


def require_training_partition(role, families, permitted_training_families, heldout_families):
    """Fail closed for this viewed audit; grouping checks also apply to future data."""
    observed, permitted, heldout = set(families), set(permitted_training_families), set(heldout_families)
    if role != "fresh_family_disjoint_training":
        raise ValueError("viewed diagnostic labels cannot become fresh training data")
    if not observed or not permitted or observed - permitted or permitted & heldout or observed & heldout:
        raise ValueError("entire families must remain in one independently registered partition")
    return True
