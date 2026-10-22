from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


DecisionMode = Literal[
    "first_individual",
    "first_subgroup",
    "later_individual",
    "later_subgroup",
]


@dataclass(frozen=True)
class ExitAttributes:
    name: str
    npc: int
    distance_m: float
    smoke: int
    familiar: int


@dataclass(frozen=True)
class LiExitScenario:
    scenario_id: int
    exits: tuple[ExitAttributes, ExitAttributes, ExitAttributes]


@dataclass(frozen=True)
class UtilityCoefficients:
    npc: float
    distance: float
    smoke: float
    familiar: float
    groupmate: float


@dataclass(frozen=True)
class SplitPattern:
    group_size: int
    pattern: str
    count: int
    expected_trials: int

    @property
    def reported_trials(self) -> int:
        return sum(p.count for p in PUBLISHED_SPLIT_PATTERNS if p.group_size == self.group_size)

    @property
    def proportion(self) -> float:
        return self.count / self.reported_trials


PUBLISHED_SCENARIOS: tuple[LiExitScenario, ...] = (
    LiExitScenario(1, (ExitAttributes("A", 0, 6.0, 0, 1), ExitAttributes("B", 10, 3.6, 1, 0), ExitAttributes("C", 5, 4.6, 1, 0))),
    LiExitScenario(2, (ExitAttributes("A", 5, 6.0, 1, 1), ExitAttributes("B", 0, 3.6, 1, 0), ExitAttributes("C", 10, 3.0, 0, 0))),
    LiExitScenario(3, (ExitAttributes("A", 1, 6.0, 1, 1), ExitAttributes("B", 1, 3.6, 0, 0), ExitAttributes("C", 10, 3.0, 0, 0))),
    LiExitScenario(4, (ExitAttributes("A", 10, 6.0, 0, 1), ExitAttributes("B", 0, 3.6, 0, 0), ExitAttributes("C", 1, 4.6, 1, 0))),
    LiExitScenario(5, (ExitAttributes("A", 10, 6.0, 0, 1), ExitAttributes("B", 1, 3.6, 1, 0), ExitAttributes("C", 0, 4.6, 0, 0))),
    LiExitScenario(6, (ExitAttributes("A", 5, 6.0, 1, 1), ExitAttributes("B", 10, 3.6, 0, 0), ExitAttributes("C", 0, 3.0, 1, 0))),
    LiExitScenario(7, (ExitAttributes("A", 1, 6.0, 1, 1), ExitAttributes("B", 5, 3.6, 0, 0), ExitAttributes("C", 1, 3.0, 0, 0))),
    LiExitScenario(8, (ExitAttributes("A", 0, 6.0, 0, 1), ExitAttributes("B", 5, 3.6, 1, 0), ExitAttributes("C", 5, 4.6, 1, 0))),
)


PUBLISHED_COEFFICIENTS: dict[DecisionMode, UtilityCoefficients] = {
    "first_individual": UtilityCoefficients(0.064, 0.178, -0.494, 0.790, 0.000),
    "first_subgroup": UtilityCoefficients(0.064, 0.178, -1.820, 0.790, 0.805),
    "later_individual": UtilityCoefficients(0.000, -0.341, -1.549, 0.790, 0.000),
    "later_subgroup": UtilityCoefficients(0.000, -0.341, -2.875, 0.790, 0.805),
}


PUBLISHED_SPLIT_PATTERNS: tuple[SplitPattern, ...] = (
    SplitPattern(2, "1+1", 57, 80),
    SplitPattern(2, "2_intact", 23, 80),
    SplitPattern(3, "1+1+1", 35, 96),
    SplitPattern(3, "2+1", 38, 96),
    SplitPattern(3, "3_intact", 23, 96),
    SplitPattern(4, "1+1+1+1", 27, 112),
    SplitPattern(4, "2+1+1", 56, 112),
    SplitPattern(4, "3+1", 20, 112),
    SplitPattern(4, "2+2", 6, 112),
    SplitPattern(4, "4_intact", 2, 112),
)


def exit_utilities(
    scenario: LiExitScenario,
    mode: DecisionMode,
    groupmates: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    if mode not in PUBLISHED_COEFFICIENTS:
        raise ValueError(f"unknown decision mode: {mode}")
    if len(groupmates) != 3 or any(value < 0 for value in groupmates):
        raise ValueError("groupmates must contain three non-negative values")
    coefficients = PUBLISHED_COEFFICIENTS[mode]
    utilities = np.zeros(3, dtype=float)
    for index, exit_attributes in enumerate(scenario.exits):
        utilities[index] = (
            coefficients.npc * exit_attributes.npc
            + coefficients.distance * exit_attributes.distance_m
            + coefficients.smoke * exit_attributes.smoke
            + coefficients.familiar * exit_attributes.familiar
            + coefficients.groupmate * groupmates[index]
        )
    return utilities


def choice_probabilities(
    scenario: LiExitScenario,
    mode: DecisionMode,
    groupmates: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    utilities = exit_utilities(scenario, mode, groupmates)
    shifted = utilities - np.max(utilities)
    weights = np.exp(shifted)
    return weights / weights.sum()


def reproduce_exit_choice_model(
    draws_per_condition: int = 10_000,
    seed: int = 20260929,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if draws_per_condition <= 0:
        raise ValueError("draws_per_condition must be positive")
    rng = np.random.default_rng(seed)
    scenario_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    simulation_errors: list[float] = []

    for scenario in PUBLISHED_SCENARIOS:
        for mode in PUBLISHED_COEFFICIENTS:
            utilities = exit_utilities(scenario, mode)
            probabilities = choice_probabilities(scenario, mode)
            simulated_counts = rng.multinomial(draws_per_condition, probabilities)
            simulated_rates = simulated_counts / draws_per_condition
            simulation_errors.extend((simulated_rates - probabilities).tolist())
            scenario_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "mode": mode,
                    "groupmate_assumption": "0,0,0",
                    "utility_A": float(utilities[0]),
                    "utility_B": float(utilities[1]),
                    "utility_C": float(utilities[2]),
                    "probability_A": float(probabilities[0]),
                    "probability_B": float(probabilities[1]),
                    "probability_C": float(probabilities[2]),
                    "simulated_rate_A": float(simulated_rates[0]),
                    "simulated_rate_B": float(simulated_rates[1]),
                    "simulated_rate_C": float(simulated_rates[2]),
                    "simulated_count_A": int(simulated_counts[0]),
                    "simulated_count_B": int(simulated_counts[1]),
                    "simulated_count_C": int(simulated_counts[2]),
                }
            )

        for mode in ("first_subgroup", "later_subgroup"):
            baseline = choice_probabilities(scenario, mode)
            for exit_index, exit_name in enumerate(("A", "B", "C")):
                groupmates = [0, 0, 0]
                groupmates[exit_index] = 1
                influenced = choice_probabilities(scenario, mode, groupmates)
                sensitivity_rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "mode": mode,
                        "target_exit": exit_name,
                        "baseline_probability": float(baseline[exit_index]),
                        "probability_with_one_groupmate": float(influenced[exit_index]),
                        "probability_change": float(influenced[exit_index] - baseline[exit_index]),
                    }
                )

    error_array = np.asarray(simulation_errors)
    metrics: dict[str, object] = {
        "paper": "Lu et al. (2026), Safety Science, DOI 10.1016/j.ssci.2026.107111",
        "model": "published_four_multinomial_logit_utility_equations",
        "seed": seed,
        "draws_per_condition": draws_per_condition,
        "scenarios": len(PUBLISHED_SCENARIOS),
        "decision_modes": len(PUBLISHED_COEFFICIENTS),
        "scenario_mode_conditions": len(scenario_rows),
        "monte_carlo_probability_mae": float(np.mean(np.abs(error_array))),
        "monte_carlo_probability_max_error": float(np.max(np.abs(error_array))),
        "probability_sum_max_error": float(
            max(abs(sum(float(row[f"probability_{name}"]) for name in ("A", "B", "C")) - 1.0) for row in scenario_rows)
        ),
        "minimum_one_groupmate_probability_increase": float(
            min(float(row["probability_change"]) for row in sensitivity_rows)
        ),
        "coefficients": {name: asdict(value) for name, value in PUBLISHED_COEFFICIENTS.items()},
        "interpretation": (
            "The paper does not publish observed exit shares for all eight scenarios. "
            "These outputs reproduce the published equations and verify their numerical behavior; "
            "they are not an independent accuracy estimate."
        ),
    }
    return scenario_rows, sensitivity_rows, metrics


def reproduce_group_splitting(
    draws_per_group_size: int = 10_000,
    seed: int = 20260930,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if draws_per_group_size <= 0:
        raise ValueError("draws_per_group_size must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    errors: list[float] = []
    discrepancies: dict[str, int] = {}

    for group_size in (2, 3, 4):
        patterns = [p for p in PUBLISHED_SPLIT_PATTERNS if p.group_size == group_size]
        reported = sum(p.count for p in patterns)
        expected = patterns[0].expected_trials
        discrepancies[str(group_size)] = reported - expected
        probabilities = np.asarray([p.count / reported for p in patterns], dtype=float)
        simulated_counts = rng.multinomial(draws_per_group_size, probabilities)
        simulated_rates = simulated_counts / draws_per_group_size
        for pattern, probability, count, rate in zip(patterns, probabilities, simulated_counts, simulated_rates):
            error = float(rate - probability)
            errors.append(error)
            rows.append(
                {
                    "group_size": group_size,
                    "pattern": pattern.pattern,
                    "reported_count": pattern.count,
                    "reported_trials": reported,
                    "expected_trials_from_design": expected,
                    "count_discrepancy": reported - expected,
                    "reported_proportion": float(probability),
                    "simulated_draws": draws_per_group_size,
                    "simulated_count": int(count),
                    "simulated_proportion": float(rate),
                    "simulation_error": error,
                    "absolute_simulation_error": abs(error),
                }
            )

    error_array = np.asarray(errors, dtype=float)
    metrics: dict[str, object] = {
        "draws_per_group_size": draws_per_group_size,
        "seed": seed,
        "simulation_mae": float(np.mean(np.abs(error_array))),
        "simulation_max_absolute_error": float(np.max(np.abs(error_array))),
        "reported_minus_expected_trials": discrepancies,
        "paper_text_correction": (
            "The three-person intact count is 23/96 = 23.96% (24.0% in Figure 5), "
            "not 28.75% as stated in the paragraph."
        ),
    }
    return rows, metrics


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_validation_outputs(
    scenario_rows: Sequence[dict[str, object]],
    sensitivity_rows: Sequence[dict[str, object]],
    choice_metrics: dict[str, object],
    split_rows: Sequence[dict[str, object]],
    split_metrics: dict[str, object],
    output_directory: str | Path,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(scenario_rows, output / "scenario_probabilities.csv")
    _write_csv(sensitivity_rows, output / "groupmate_sensitivity.csv")
    _write_csv(split_rows, output / "group_splitting_validation.csv")
    (output / "metrics.json").write_text(
        json.dumps({"exit_choice": choice_metrics, "group_splitting": split_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Lu et al. (2026) 三出口与群体拆分复现结果",
        "",
        "## 三出口模型",
        "",
        "下表给出8个论文场景在4种决策模式下的理论选择概率。小组成员计数先固定为 GM=(0,0,0)，其影响另见 `groupmate_sensitivity.csv`。",
        "",
        "| 场景 | 决策模式 | P(A) | P(B) | P(C) | 最高概率出口 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in scenario_rows:
        probabilities = [float(row[f"probability_{name}"]) for name in ("A", "B", "C")]
        report.append(
            f"| {row['scenario_id']} | {row['mode']} | {probabilities[0]:.2%} | {probabilities[1]:.2%} | "
            f"{probabilities[2]:.2%} | {('A', 'B', 'C')[int(np.argmax(probabilities))]} |"
        )
    report.extend(
        [
            "",
            "说明：论文没有公布8个场景逐项的真实出口比例，因此这里验证的是公式实现、Softmax概率与蒙特卡洛一致性，不能称为独立准确率验证。",
            "",
            "## 群体拆分",
            "",
            "| 组规模 | 拆分结构 | 论文计数 | 论文比例 | 模拟比例 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in split_rows:
        report.append(
            f"| {row['group_size']} | {row['pattern']} | {row['reported_count']} | "
            f"{float(row['reported_proportion']):.2%} | {float(row['simulated_proportion']):.2%} |"
        )
    report.extend(
        [
            "",
            "## 核对发现",
            "",
            "- 三人组保持完整的计数为23/96，即23.96%，与图5的24.0%一致；正文中的28.75%是算术/排版错误。",
            "- 四人组图示计数合计111次，实验设计应有14组×8=112次，存在1次未解释的缺口。",
            f"- 三出口蒙特卡洛概率 MAE：{float(choice_metrics['monte_carlo_probability_mae']):.4f}。",
            f"- 群体拆分蒙特卡洛比例 MAE：{float(split_metrics['simulation_mae']):.4f}。",
            f"- 加入1名同伴后，目标出口概率的最小增幅：{float(choice_metrics['minimum_one_groupmate_probability_increase']):.4f}。",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

