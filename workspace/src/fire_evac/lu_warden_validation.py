from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


WardenType = Literal["none", "metro_staff", "firefighter"]


@dataclass(frozen=True)
class LuWardenCoefficients:
    """Published mean coefficients from Lu et al. (2025), Table 3."""

    intercept_right: float = 0.263
    beta_npc: float = 0.035
    beta_warden: float = 2.739
    beta_firefighter: float = 0.559


@dataclass(frozen=True)
class LuWardenScenario:
    scenario_id: int
    npc_right: int
    warden: WardenType
    observed_left_count: int
    observed_right_count: int

    @property
    def observations(self) -> int:
        return self.observed_left_count + self.observed_right_count

    @property
    def observed_right_rate(self) -> float:
        return self.observed_right_count / self.observations

    @property
    def warden_present(self) -> int:
        return int(self.warden != "none")

    @property
    def firefighter_present(self) -> int:
        return int(self.warden == "firefighter")


PUBLISHED_SCENARIOS: tuple[LuWardenScenario, ...] = (
    LuWardenScenario(1, 0, "none", 64, 67),
    LuWardenScenario(2, 1, "none", 46, 84),
    LuWardenScenario(3, 10, "none", 47, 84),
    LuWardenScenario(4, 20, "none", 37, 93),
    LuWardenScenario(5, 0, "metro_staff", 62, 4),
    LuWardenScenario(6, 1, "metro_staff", 61, 4),
    LuWardenScenario(7, 10, "metro_staff", 57, 9),
    LuWardenScenario(8, 20, "metro_staff", 56, 10),
    LuWardenScenario(9, 0, "firefighter", 61, 4),
    LuWardenScenario(10, 1, "firefighter", 64, 1),
    LuWardenScenario(11, 10, "firefighter", 59, 6),
    LuWardenScenario(12, 20, "firefighter", 60, 5),
)


def exit_utilities(
    scenario: LuWardenScenario,
    coefficients: LuWardenCoefficients = LuWardenCoefficients(),
) -> tuple[float, float]:
    """Return (left, right) utility using the published experimental coding.

    NPCs always move right. The warden, when present, always points left.
    The alternative-specific intercept is assigned only to the right exit.
    """

    left = (
        coefficients.beta_warden * scenario.warden_present
        + coefficients.beta_firefighter * scenario.firefighter_present
    )
    right = coefficients.intercept_right + coefficients.beta_npc * scenario.npc_right
    return left, right


def right_exit_probability(
    scenario: LuWardenScenario,
    coefficients: LuWardenCoefficients = LuWardenCoefficients(),
) -> float:
    left, right = exit_utilities(scenario, coefficients)
    difference = float(np.clip(right - left, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-difference))


def validate_published_scenarios(
    draws_per_scenario: int = 10_000,
    seed: int = 20260929,
    scenarios: Sequence[LuWardenScenario] = PUBLISHED_SCENARIOS,
    coefficients: LuWardenCoefficients = LuWardenCoefficients(),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if draws_per_scenario <= 0:
        raise ValueError("draws_per_scenario must be positive")
    if not scenarios:
        raise ValueError("at least one scenario is required")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_observations = 0
    weighted_brier = 0.0
    weighted_log_loss = 0.0
    analytic_errors: list[float] = []
    simulation_errors: list[float] = []
    majority_matches = 0

    for scenario in scenarios:
        probability = right_exit_probability(scenario, coefficients)
        simulated_right_count = int(rng.binomial(draws_per_scenario, probability))
        simulated_rate = simulated_right_count / draws_per_scenario
        observed_rate = scenario.observed_right_rate
        analytic_error = probability - observed_rate
        simulation_error = simulated_rate - observed_rate
        analytic_errors.append(analytic_error)
        simulation_errors.append(simulation_error)
        majority_matches += int((probability >= 0.5) == (observed_rate >= 0.5))

        n = scenario.observations
        p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
        weighted_brier += n * (observed_rate * (1.0 - p) ** 2 + (1.0 - observed_rate) * p**2)
        weighted_log_loss += -(
            scenario.observed_right_count * math.log(p)
            + scenario.observed_left_count * math.log(1.0 - p)
        )
        total_observations += n

        left_utility, right_utility = exit_utilities(scenario, coefficients)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "npc_right": scenario.npc_right,
                "warden": scenario.warden,
                "observed_n": n,
                "observed_left_count": scenario.observed_left_count,
                "observed_right_count": scenario.observed_right_count,
                "observed_right_rate": observed_rate,
                "left_utility": left_utility,
                "right_utility": right_utility,
                "analytic_right_probability": probability,
                "simulated_draws": draws_per_scenario,
                "simulated_right_count": simulated_right_count,
                "simulated_right_rate": simulated_rate,
                "analytic_error": analytic_error,
                "absolute_analytic_error": abs(analytic_error),
                "simulation_error": simulation_error,
                "absolute_simulation_error": abs(simulation_error),
            }
        )

    analytic_array = np.asarray(analytic_errors, dtype=float)
    simulation_array = np.asarray(simulation_errors, dtype=float)
    worst_index = int(np.argmax(np.abs(analytic_array)))
    metrics: dict[str, object] = {
        "paper": "Lu et al. (2025), Fire Technology, DOI 10.1007/s10694-025-01746-6",
        "model": "published_multinomial_logit_mean_coefficients",
        "seed": seed,
        "draws_per_scenario": draws_per_scenario,
        "scenarios": len(scenarios),
        "published_observations": total_observations,
        "paper_claimed_observations": 1048,
        "table_count_discrepancy": total_observations - 1048,
        "analytic_mae": float(np.mean(np.abs(analytic_array))),
        "analytic_rmse": float(np.sqrt(np.mean(analytic_array**2))),
        "analytic_max_absolute_error": float(np.max(np.abs(analytic_array))),
        "analytic_max_error_scenario": int(scenarios[worst_index].scenario_id),
        "simulated_mae": float(np.mean(np.abs(simulation_array))),
        "simulated_rmse": float(np.sqrt(np.mean(simulation_array**2))),
        "weighted_brier_score": weighted_brier / total_observations,
        "weighted_log_loss": weighted_log_loss / total_observations,
        "majority_direction_accuracy": majority_matches / len(scenarios),
        "coefficients": asdict(coefficients),
        "interpretation": (
            "This validates the published mean-choice equation against aggregated scenario rates. "
            "It is not an independent participant-level replication and does not include random-coefficient heterogeneity."
        ),
    }
    return rows, metrics


def write_validation_outputs(
    rows: Sequence[dict[str, object]],
    metrics: dict[str, object],
    output_directory: str | Path,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty validation result")

    with (output / "scenario_validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Lu et al. (2025) 双出口场景复现结果",
        "",
        "模型使用论文表 3 的多项 Logit 均值系数。NPC 始终向右出口移动；引导员始终指向左出口。",
        "",
        "| 场景 | NPC→右 | 引导员 | 论文右出口率 | 理论概率 | 模拟比例 | 绝对误差 |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {scenario_id} | {npc_right} | {warden} | {observed_right_rate:.2%} | "
            "{analytic_right_probability:.2%} | {simulated_right_rate:.2%} | "
            "{absolute_analytic_error:.2%} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "## 汇总指标",
            "",
            f"- 理论概率 MAE：{float(metrics['analytic_mae']):.4f}",
            f"- 理论概率 RMSE：{float(metrics['analytic_rmse']):.4f}",
            f"- 最大绝对误差：{float(metrics['analytic_max_absolute_error']):.4f}（场景 {metrics['analytic_max_error_scenario']}）",
            f"- 蒙特卡洛 MAE：{float(metrics['simulated_mae']):.4f}",
            f"- 加权 Brier 分数：{float(metrics['weighted_brier_score']):.4f}",
            f"- 加权 Log Loss：{float(metrics['weighted_log_loss']):.4f}",
            f"- 多数选择方向准确率：{float(metrics['majority_direction_accuracy']):.2%}",
            f"- 表 5 计数合计：{metrics['published_observations']}；正文声称：{metrics['paper_claimed_observations']}；差异：{metrics['table_count_discrepancy']}",
            "",
            "注意：论文正文报告 1048 次选择，但表 5 的逐场景人数合计为 1045。这里按表 5 原始计数计算，未补造缺失记录。",
            "",
            "这些结果是对论文已发表均值系数的场景级复现，不是使用原始参与者数据重新估计模型。",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

