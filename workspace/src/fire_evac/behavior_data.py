from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REQUIRED_COLUMNS = (
    "participant_id",
    "trial_id",
    "hazard_north",
    "hazard_south",
    "congestion_north",
    "congestion_south",
    "sign_action",
    "familiarity_branch",
    "previous_flow_north",
    "previous_flow_south",
    "choice",
    "data_source",
)

FEATURE_NAMES = (
    "intercept",
    "relative_safety_north",
    "relative_space_north",
    "sign_points_north",
    "familiarity_north",
    "crowd_moves_north",
)


@dataclass(frozen=True)
class BehaviorObservation:
    participant_id: str
    trial_id: str
    hazard_north: float
    hazard_south: float
    congestion_north: float
    congestion_south: float
    sign_action: str
    familiarity_branch: str
    previous_flow_north: float
    previous_flow_south: float
    choice: str
    data_source: str


@dataclass(frozen=True)
class ParticipantSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def assert_disjoint(self) -> None:
        train, validation, test = set(self.train), set(self.validation), set(self.test)
        if train & validation or train & test or validation & test:
            raise ValueError("participant leakage detected across data splits")


@dataclass
class HierarchicalLogitModel:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    participant_intercepts: dict[str, float]
    l2: float

    def predict_probability(self, rows: Sequence[BehaviorObservation], use_random_effects: bool = False) -> np.ndarray:
        x, _ = design_matrix(rows, self.feature_mean, self.feature_scale)
        logits = x @ self.coefficients
        if use_random_effects:
            logits += np.array([self.participant_intercepts.get(row.participant_id, 0.0) for row in rows])
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "hierarchical_binary_logit",
            "target": "choice_north",
            "feature_names": list(self.feature_names),
            "coefficients": self.coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "participant_intercepts": self.participant_intercepts,
            "l2": self.l2,
            "warning": "Participant effects are estimated only for training participants; use zero for unseen participants.",
        }


def _bounded_float(value: str, name: str, low: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < low:
        raise ValueError(f"{name} must be finite and >= {low}")
    return parsed


def parse_observation(row: dict[str, str]) -> BehaviorObservation:
    missing = [name for name in REQUIRED_COLUMNS if name not in row or row[name] == ""]
    if missing:
        raise ValueError(f"missing required values: {', '.join(missing)}")
    sign = row["sign_action"].strip().lower()
    familiarity = row["familiarity_branch"].strip().lower()
    choice = row["choice"].strip().lower()
    if sign not in {"north", "south", "neutral"}:
        raise ValueError("sign_action must be north, south, or neutral")
    if familiarity not in {"north", "south"}:
        raise ValueError("familiarity_branch must be north or south")
    if choice not in {"north", "south"}:
        raise ValueError("choice must be north or south")
    return BehaviorObservation(
        participant_id=row["participant_id"].strip(),
        trial_id=row["trial_id"].strip(),
        hazard_north=_bounded_float(row["hazard_north"], "hazard_north"),
        hazard_south=_bounded_float(row["hazard_south"], "hazard_south"),
        congestion_north=_bounded_float(row["congestion_north"], "congestion_north"),
        congestion_south=_bounded_float(row["congestion_south"], "congestion_south"),
        sign_action=sign,
        familiarity_branch=familiarity,
        previous_flow_north=_bounded_float(row["previous_flow_north"], "previous_flow_north"),
        previous_flow_south=_bounded_float(row["previous_flow_south"], "previous_flow_south"),
        choice=choice,
        data_source=row["data_source"].strip(),
    )


def read_behavior_csv(path: str | Path) -> list[BehaviorObservation]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("behavior CSV has no header")
        absent = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if absent:
            raise ValueError(f"behavior CSV missing columns: {', '.join(absent)}")
        rows = [parse_observation(row) for row in reader]
    if not rows:
        raise ValueError("behavior CSV contains no observations")
    pairs = [(row.participant_id, row.trial_id) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate participant_id/trial_id pairs")
    return rows


def write_behavior_csv(rows: Iterable[BehaviorObservation], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in REQUIRED_COLUMNS})


def split_participants(
    rows: Sequence[BehaviorObservation], seed: int = 20260928, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ParticipantSplit:
    participants = sorted({row.participant_id for row in rows})
    if len(participants) < 5:
        raise ValueError("at least five participants are required for participant-level splitting")
    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0):
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than one")
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    n = len(participants)
    train_end = max(1, int(round(n * train_fraction)))
    validation_end = min(n - 1, train_end + max(1, int(round(n * validation_fraction))))
    split = ParticipantSplit(
        train=tuple(sorted(participants[:train_end])),
        validation=tuple(sorted(participants[train_end:validation_end])),
        test=tuple(sorted(participants[validation_end:])),
    )
    if not split.test:
        raise ValueError("test split is empty")
    split.assert_disjoint()
    return split


def rows_for_participants(rows: Sequence[BehaviorObservation], participant_ids: Sequence[str]) -> list[BehaviorObservation]:
    allowed = set(participant_ids)
    return [row for row in rows if row.participant_id in allowed]


def raw_features(rows: Sequence[BehaviorObservation]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    x[:, 0] = 1.0
    for i, row in enumerate(rows):
        x[i, 1] = row.hazard_south - row.hazard_north
        x[i, 2] = row.congestion_south - row.congestion_north
        x[i, 3] = 1.0 if row.sign_action == "north" else -1.0 if row.sign_action == "south" else 0.0
        x[i, 4] = 1.0 if row.familiarity_branch == "north" else -1.0
        total_flow = row.previous_flow_north + row.previous_flow_south
        x[i, 5] = (row.previous_flow_north - row.previous_flow_south) / max(1.0, total_flow)
        y[i] = 1.0 if row.choice == "north" else 0.0
    return x, y


def design_matrix(
    rows: Sequence[BehaviorObservation], mean: np.ndarray | None = None, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    x, y = raw_features(rows)
    if mean is None:
        mean = x.mean(axis=0)
        mean[0] = 0.0
    if scale is None:
        scale = x.std(axis=0)
        scale[0] = 1.0
        scale[scale < 1e-9] = 1.0
    x[:, 1:] = (x[:, 1:] - mean[1:]) / scale[1:]
    return x, y


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_hierarchical_logit(
    rows: Sequence[BehaviorObservation], l2: float = 1.0, random_effect_l2: float = 4.0, max_iter: int = 100
) -> HierarchicalLogitModel:
    if len(rows) < len(FEATURE_NAMES) * 2:
        raise ValueError("insufficient training rows for behavior model")
    x_raw, _ = raw_features(rows)
    mean = x_raw.mean(axis=0)
    mean[0] = 0.0
    scale = x_raw.std(axis=0)
    scale[0] = 1.0
    scale[scale < 1e-9] = 1.0
    x, y = design_matrix(rows, mean, scale)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(x @ beta)
        weights = np.maximum(p * (1.0 - p), 1e-6)
        gradient = x.T @ (y - p) - penalty @ beta
        hessian = x.T @ (weights[:, None] * x) + penalty
        step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-9, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break

    participant_intercepts: dict[str, float] = {}
    base_logits = x @ beta
    for participant_id in sorted({row.participant_id for row in rows}):
        indices = np.array([i for i, row in enumerate(rows) if row.participant_id == participant_id])
        delta = 0.0
        for _ in range(30):
            p = _sigmoid(base_logits[indices] + delta)
            gradient = float(np.sum(y[indices] - p) - random_effect_l2 * delta)
            curvature = float(np.sum(p * (1.0 - p)) + random_effect_l2)
            step = gradient / max(curvature, 1e-9)
            delta += step
            if abs(step) < 1e-8:
                break
        participant_intercepts[participant_id] = delta
    return HierarchicalLogitModel(FEATURE_NAMES, beta, mean, scale, participant_intercepts, l2)


def classification_metrics(rows: Sequence[BehaviorObservation], probabilities: np.ndarray) -> dict[str, float]:
    _, y = raw_features(rows)
    p = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return {
        "n": float(len(rows)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "north_rate": float(np.mean(y)),
        "predicted_north_rate": float(np.mean(p)),
    }


def save_model(model: HierarchicalLogitModel, path: str | Path, metadata: dict[str, object] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    payload["metadata"] = metadata or {}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_synthetic_behavior_data(
    participants: int = 60, trials_per_participant: int = 20, seed: int = 20260928
) -> list[BehaviorObservation]:
    """Create an explicitly labelled demo dataset for pipeline testing only."""
    rng = np.random.default_rng(seed)
    rows: list[BehaviorObservation] = []
    true_beta = np.array([-0.05, 1.7, 0.9, 0.8, 0.55, 0.45])
    for participant_index in range(participants):
        participant_id = f"SYN-{participant_index + 1:03d}"
        random_intercept = float(rng.normal(0.0, 0.55))
        familiarity = "north" if rng.random() < 0.5 else "south"
        for trial_index in range(trials_per_participant):
            hazard_north, hazard_south = rng.uniform(0.0, 0.25, size=2)
            congestion_north, congestion_south = rng.uniform(0.0, 1.8, size=2)
            sign = str(rng.choice(["north", "south", "neutral"], p=[0.4, 0.4, 0.2]))
            flow_north, flow_south = rng.integers(0, 16, size=2)
            provisional = BehaviorObservation(
                participant_id=participant_id,
                trial_id=f"T{trial_index + 1:03d}",
                hazard_north=float(hazard_north),
                hazard_south=float(hazard_south),
                congestion_north=float(congestion_north),
                congestion_south=float(congestion_south),
                sign_action=sign,
                familiarity_branch=familiarity,
                previous_flow_north=float(flow_north),
                previous_flow_south=float(flow_south),
                choice="north",
                data_source="synthetic_demo_not_human_evidence",
            )
            x, _ = raw_features([provisional])
            probability = float(_sigmoid(x @ true_beta + random_intercept)[0])
            choice = "north" if rng.random() < probability else "south"
            rows.append(BehaviorObservation(**{**provisional.__dict__, "choice": choice}))
    return rows
