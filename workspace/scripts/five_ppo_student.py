"""Observable multi-head relative gains with exact baseline and pooled labels."""
from dataclasses import dataclass
import numpy as np
from candidate_label_interface import observable_features, choose_candidate, require_training_partition

ALPHA = .01


def pooled_labels(risks, support):
    r = np.asarray(risks, float); s = np.asarray(support)
    if (r.ndim != 5 or r.shape[:3] != (2, 2, 5) or min(r.shape[3:]) < 1 or not np.isfinite(r).all()
            or s.shape != (2,) or s.dtype != bool or not s.any()):
        raise ValueError("two populations x two worlds x five plans x futures x outcomes required")
    loss = r[..., 0]
    if np.any(loss < 0) or np.any(loss > 1): raise ValueError("fraction loss required")
    means = loss.mean(axis=(0, 3))
    active = means[s]; worst = active.max(axis=0)
    R = worst[0] - worst; M = (active[:, 0, None] - active).min(axis=0)
    R[0] = M[0] = 0
    return np.stack((R, M)), means


def kernel(x, z, gamma):
    d = np.maximum(0, (x*x).sum(axis=1)[:, None] + (z*z).sum(axis=1)[None, :] - 2*x@z.T)
    return np.exp(-gamma*d)


@dataclass
class MultiGain:
    kind: str
    mean: np.ndarray
    std: np.ndarray
    intercept: np.ndarray
    coefficient: np.ndarray
    train: np.ndarray
    weights: np.ndarray
    kernel_row_mean: np.ndarray
    kernel_grand_mean: np.ndarray
    gamma: np.ndarray

    def predict(self, observation):
        raw = np.asarray(observation, float)
        single = raw.ndim == 1
        rows = np.atleast_2d(raw)
        if rows.ndim != 2 or rows.shape[1] != 49: raise ValueError("observable49 required")
        x = np.array([observable_features(row) for row in rows]); x = (x-self.mean)/self.std
        if self.kind == "linear": y = x@self.coefficient + self.intercept
        elif self.kind == "kernel":
            k = kernel(x, self.train, float(self.gamma))
            centered = k - (k@self.weights)[:, None] - self.kernel_row_mean[None, :] + float(self.kernel_grand_mean)
            y = centered@self.coefficient + self.intercept
        else: raise ValueError("unknown model")
        result = np.c_[np.zeros(len(x)), y]
        return result[0] if single else result

    def save(self, path): np.savez_compressed(path, **{k: np.asarray(v) for k, v in vars(self).items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z: values = {k: z[k].copy() for k in z.files}
        values["kind"] = str(values["kind"]); return cls(**values)


def fit(observations, targets, families, kind, permitted_families, heldout_families):
    require_training_partition("fresh_family_disjoint_training", families, permitted_families, heldout_families)
    x = np.array([observable_features(row) for row in observations]); y = np.asarray(targets, float)
    if x.shape != (len(families), 49) or y.shape != (len(x), 5) or len(set(families)) != len(x) or not np.isfinite(y).all() or y[:, 0].any():
        raise ValueError("one pooled row per training family with exact zero601 required")
    w = np.full(len(x), 1/len(x)); mean = w@x; std = np.sqrt(w@((x-mean)**2)); std = np.where(std < .001, 1., std)
    tx = (x-mean)/std; b = w@y[:, 1:]
    if kind == "linear":
        matrix = tx.T@(w[:, None]*tx) + ALPHA*np.eye(49)
        rhs = tx.T@(w[:, None]*(y[:, 1:]-b)); coefficient = np.linalg.solve(matrix, rhs)
        residual = matrix@coefficient-rhs
        model = MultiGain(kind, mean, std, b, coefficient, np.empty((0,49)), w, np.empty(0), np.array(0.), np.array(0.))
    elif kind == "kernel":
        active = max(1, int((np.var(tx, axis=0)>1e-12).sum())); gamma = np.array(1/(2*active))
        k = kernel(tx, tx, float(gamma)); row = k@w; grand = np.array(w@row)
        centered = k-row[:,None]-row[None,:]+grand; root = np.sqrt(w)
        matrix = root[:,None]*centered*root[None,:]+ALPHA*np.eye(len(x)); rhs = root[:,None]*(y[:,1:]-b)
        dual = np.linalg.solve(matrix, rhs); residual = matrix@dual-rhs
        model = MultiGain(kind, mean, std, b, root[:,None]*dual, tx, w, row, grand, gamma)
    else: raise ValueError("linear or kernel required")
    return model, {"families": len(x), "alpha": ALPHA, "kind": kind, "gamma": float(model.gamma),
        "training_family_ids": list(families), "normal_equation_max_residual": float(np.abs(residual).max()),
        "training_MAE_pp": float(np.abs(model.predict(x)-y).mean()*100)}


def calibrate(predictions, empirical, families, prior):
    p, e = np.asarray(predictions, float), np.asarray(empirical, float)
    if p.shape != e.shape or p.shape != (len(families), 5) or len(set(families)) != len(families) or not np.isfinite(p).all() or not np.isfinite(e).all() or not np.isfinite(prior) or prior < 0:
        raise ValueError("finite one-row-per-calibration-family candidate gains required")
    residuals = (p[:, 1:]-e[:, 1:]).max(axis=1)
    q = float(np.quantile(residuals, .95, method="higher"))
    return max(prior, .001, q), {str(f): float(v) for f, v in zip(families, residuals)}, q


def choices(gains, margin):
    return {"point": choose_candidate(gains), "margin": choose_candidate(gains, margin)}
