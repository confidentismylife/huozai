from __future__ import annotations

from typing import Sequence

import numpy as np

from .dynamic_conflicts import JOINT_RELEASE_LEVELS


def copy_previous_joint_action(observations: np.ndarray) -> np.ndarray:
    """Decode the previous executed route/capacity from joint-gate observations.

    This is a diagnostic baseline, not a controller. It exposes how much of a
    step-level supervised score can be obtained from temporal persistence.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or x.shape[1] < 29:
        raise ValueError("joint observations must be a two-dimensional array with at least 29 columns")
    previous_sign = np.argmax(x[:, 10:13], axis=1)
    advice_target = np.argmax(x[:, 19:21], axis=1)
    release = x[:, 28]
    levels = np.asarray(JOINT_RELEASE_LEVELS, dtype=float)
    level_index = np.argmin(np.abs(release[:, None] - levels[None, :]), axis=1)
    copied = np.where(previous_sign == advice_target, level_index, 4 + level_index)
    copied = np.where((previous_sign == 2) | (release <= 1e-8), 8, copied)
    return copied.astype(np.int64)


def copy_baseline_diagnostics(
    observations: np.ndarray,
    labels: np.ndarray,
    conflict_types: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    x = np.asarray(observations, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("observations and labels must have equal length")
    copied = copy_previous_joint_action(x)
    previous_neutral = np.argmax(x[:, 10:13], axis=1) == 2
    initial_occupancy = (x[:, 0] + x[:, 1] > 0.99) & (x[:, 2:5].sum(axis=1) < 1e-8)
    first_decision = previous_neutral & initial_occupancy
    contradiction = x[:, 25] > 0.5
    label_switch = np.zeros(len(y), dtype=bool)
    first_contradiction = np.zeros(len(y), dtype=bool)

    # Reopening is observable when the normalized remaining-blockage feature
    # crosses from positive to zero. Initial observations delimit episodes.
    reopening = np.zeros(len(y), dtype=bool)
    for index in range(len(y)):
        if not first_decision[index] and index > 0:
            label_switch[index] = y[index] != y[index - 1]
            first_contradiction[index] = contradiction[index] and not contradiction[index - 1]
        elif contradiction[index]:
            first_contradiction[index] = True
        if index > 0 and not first_decision[index] and x[index - 1, 18] > 0 and x[index, 18] <= 0:
            reopening[index] = True

    masks = {
        "all_steps": np.ones(len(y), dtype=bool),
        "label_switch_steps": label_switch,
        "first_decision": first_decision,
        "first_smoke_contradiction": first_contradiction,
        "exit_reopening": reopening,
    }
    rows: list[dict[str, object]] = []
    for stratum, mask in masks.items():
        count = int(mask.sum())
        rows.append(
            {
                "stratum": stratum,
                "conflict_type": "all",
                "n": count,
                "accuracy": float(np.mean(copied[mask] == y[mask])) if count else float("nan"),
            }
        )
    if conflict_types is not None:
        conflicts = np.asarray(conflict_types).astype(str)
        if len(conflicts) != len(y):
            raise ValueError("conflict_types and labels must have equal length")
        for conflict in sorted(set(conflicts.tolist())):
            mask = conflicts == conflict
            rows.append(
                {
                    "stratum": "all_steps",
                    "conflict_type": conflict,
                    "n": int(mask.sum()),
                    "accuracy": float(np.mean(copied[mask] == y[mask])),
                }
            )
    return rows
