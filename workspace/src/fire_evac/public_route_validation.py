"""Low-cost, participant-held-out validation on the Snopkova route-choice data.

The source study used a visual, time-limited virtual-building task. This module
reconstructs only the documented corridor attributes; text prompts are therefore
a modality-shift pilot, not a validation of behavior in real building fires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import re
from zipfile import ZipFile

import numpy as np


TASK_PATTERN = re.compile(r"^\d+_L-([WNE])([SML])([01])_R-([WNE])([SML])([01])$")
WIDTH_METERS = {"W": 4, "N": 2, "E": 2}
LENGTH_METERS = {"S": 10, "M": 15, "L": 20}
VALID_CHOICES = {"corridor_left": "left", "corridor_right": "right"}


@dataclass(frozen=True)
class RouteTrial:
    participant_id: str
    task_id: str
    choice: str
    left_width_m: int
    left_length_m: int
    left_stairs: bool
    right_width_m: int
    right_length_m: int
    right_stairs: bool

    def public_features(self) -> dict[str, int | bool]:
        return {key: value for key, value in asdict(self).items() if key.startswith(("left_", "right_"))}


def parse_trial(row: dict[str, str]) -> RouteTrial | None:
    task_id = row["taskId"]
    if task_id.endswith("_tr") or row["AOI"] not in VALID_CHOICES:
        return None
    match = TASK_PATTERN.fullmatch(task_id)
    if match is None:
        return None
    lw, ll, ls, rw, rl, rs = match.groups()
    return RouteTrial(
        participant_id=row["userId"],
        task_id=task_id,
        choice=VALID_CHOICES[row["AOI"]],
        left_width_m=WIDTH_METERS[lw],
        left_length_m=LENGTH_METERS[ll],
        left_stairs=ls == "1",
        right_width_m=WIDTH_METERS[rw],
        right_length_m=LENGTH_METERS[rl],
        right_stairs=rs == "1",
    )


def read_public_trials(zip_path: str) -> tuple[list[RouteTrial], dict[str, int]]:
    with ZipFile(zip_path) as archive:
        with archive.open("SciData_export/responses.csv") as binary:
            rows = list(csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig")))
    trials = [trial for row in rows if (trial := parse_trial(row)) is not None]
    keys = [(trial.participant_id, trial.task_id) for trial in trials]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate participant/task observations")
    counts = {
        "source_rows": len(rows),
        "training_or_no_choice_rows": sum(
            row["taskId"].endswith("_tr") or row["AOI"] not in VALID_CHOICES for row in rows
        ),
        "undecodable_main_rows": sum(
            not row["taskId"].endswith("_tr")
            and row["AOI"] in VALID_CHOICES
            and TASK_PATTERN.fullmatch(row["taskId"]) is None
            for row in rows
        ),
        "usable_rows": len(trials),
        "participants": len({trial.participant_id for trial in trials}),
        "tasks": len({trial.task_id for trial in trials}),
    }
    return trials, counts


def participant_splits(trials: list[RouteTrial], seed: int = 20260928) -> dict[str, str]:
    participants = np.array(sorted({trial.participant_id for trial in trials}))
    np.random.default_rng(seed).shuffle(participants)
    n = len(participants)
    n_train, n_validation = int(round(0.6 * n)), int(round(0.2 * n))
    split = {}
    for name, ids in (
        ("train", participants[:n_train]),
        ("validation", participants[n_train : n_train + n_validation]),
        ("test", participants[n_train + n_validation :]),
    ):
        split.update({str(participant_id): name for participant_id in ids})
    return split


def route_prompt(trial: RouteTrial) -> str:
    return (
        "一项虚拟建筑疏散任务中，一名成年人走到T形路口，需要在左、右走廊中选择一条。"
        "只根据以下可见的走廊属性，预测类似参与者选择左走廊的概率。"
        "没有给出烟雾位置、人群队列、动态指示牌或其他信息，不要自行补充。\n"
        f"左走廊：宽{trial.left_width_m}米，长{trial.left_length_m}米，"
        f"{'尽头有' if trial.left_stairs else '尽头没有'}楼梯。\n"
        f"右走廊：宽{trial.right_width_m}米，长{trial.right_length_m}米，"
        f"{'尽头有' if trial.right_stairs else '尽头没有'}楼梯。\n"
        "只返回JSON对象，其中p_left是0到1之间的数字。"
    )


def baseline_probabilities(train: list[RouteTrial], trial: RouteTrial) -> dict[str, float]:
    left_count = sum(item.choice == "left" for item in train)
    overall = (left_count + 1) / (len(train) + 2)
    task_rows = [item for item in train if item.task_id == trial.task_id]
    task_left = sum(item.choice == "left" for item in task_rows)
    task_rate = (task_left + 1) / (len(task_rows) + 2) if task_rows else overall
    left_score = (-trial.left_length_m, -int(trial.left_stairs), trial.left_width_m)
    right_score = (-trial.right_length_m, -int(trial.right_stairs), trial.right_width_m)
    heuristic = 0.75 if left_score > right_score else 0.25 if right_score > left_score else 0.5
    return {"global_majority": overall, "geometry_rule": heuristic, "train_task_rate": task_rate}


def score_predictions(
    trials: list[RouteTrial], split: dict[str, str], llm_by_task: dict[str, float],
    *, evaluation_split: str = "test",
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float | int]]]:
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    train = [trial for trial in trials if split[trial.participant_id] == "train"]
    test = [trial for trial in trials if split[trial.participant_id] == evaluation_split]
    rows: list[dict[str, str | float]] = []
    for trial in test:
        if trial.task_id not in llm_by_task:
            continue
        target = float(trial.choice == "left")
        probabilities = {"local_llm": llm_by_task[trial.task_id], **baseline_probabilities(train, trial)}
        for method, probability in probabilities.items():
            probability = float(probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid probability for {method}: {probability}")
            rows.append(
                {
                    "participant_id": trial.participant_id,
                    "task_id": trial.task_id,
                    "actual_choice": trial.choice,
                    "method": method,
                    "p_left": probability,
                    "correct": float((probability >= 0.5) == bool(target)),
                    "brier": (probability - target) ** 2,
                }
            )
    metrics: list[dict[str, str | float | int]] = []
    for method in ("local_llm", "global_majority", "geometry_rule", "train_task_rate"):
        items = [row for row in rows if row["method"] == method]
        if not items:
            continue
        metrics.append(
            {
                "method": method,
                f"{evaluation_split}_participants": len({row["participant_id"] for row in items}),
                f"{evaluation_split}_decisions": len(items),
                "accuracy": float(np.mean([row["correct"] for row in items])),
                "brier": float(np.mean([row["brier"] for row in items])),
            }
        )
    return rows, metrics


def participant_bootstrap_differences(
    prediction_rows: list[dict[str, str | float]],
    *,
    samples: int = 5000,
    seed: int = 20260928,
) -> list[dict[str, str | float | int]]:
    """Paired LLM-minus-baseline intervals resampling participants as units."""
    if samples < 1:
        raise ValueError("samples must be positive")
    lookup = {
        (str(row["participant_id"]), str(row["task_id"]), str(row["method"])): row
        for row in prediction_rows
    }
    participants = sorted({str(row["participant_id"]) for row in prediction_rows})
    rng = np.random.default_rng(seed)
    output: list[dict[str, str | float | int]] = []
    for baseline in ("global_majority", "geometry_rule", "train_task_rate"):
        for metric in ("accuracy", "brier"):
            column = "correct" if metric == "accuracy" else "brier"
            sums = np.zeros(len(participants))
            counts = np.zeros(len(participants))
            for i, person in enumerate(participants):
                tasks = sorted(
                    {task for p, task, method in lookup if p == person and method == "local_llm"}
                )
                for task in tasks:
                    candidate = lookup.get((person, task, "local_llm"))
                    reference = lookup.get((person, task, baseline))
                    if candidate is None or reference is None:
                        continue
                    sums[i] += float(candidate[column]) - float(reference[column])
                    counts[i] += 1
            if not counts.any():
                continue
            draws = rng.integers(0, len(participants), size=(samples, len(participants)))
            replicate_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            output.append(
                {
                    "candidate": "local_llm",
                    "baseline": baseline,
                    "metric": metric,
                    "participants": int((counts > 0).sum()),
                    "paired_decisions": int(counts.sum()),
                    "mean_difference": float(sums.sum() / counts.sum()),
                    "ci_2_5": float(np.quantile(replicate_means, 0.025)),
                    "ci_97_5": float(np.quantile(replicate_means, 0.975)),
                }
            )
    return output
