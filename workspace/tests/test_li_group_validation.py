from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.li_group_validation import (
    PUBLISHED_SCENARIOS,
    PUBLISHED_SPLIT_PATTERNS,
    choice_probabilities,
    reproduce_exit_choice_model,
)


class LiGroupValidationTests(unittest.TestCase):
    def test_published_scenarios_have_expected_attributes(self) -> None:
        self.assertEqual(len(PUBLISHED_SCENARIOS), 8)
        first = PUBLISHED_SCENARIOS[0]
        self.assertEqual([(e.npc, e.distance_m, e.smoke, e.familiar) for e in first.exits], [
            (0, 6.0, 0, 1),
            (10, 3.6, 1, 0),
            (5, 4.6, 1, 0),
        ])

    def test_probabilities_sum_to_one_and_groupmate_increases_target(self) -> None:
        for scenario in PUBLISHED_SCENARIOS:
            baseline = choice_probabilities(scenario, "later_subgroup")
            self.assertAlmostEqual(float(baseline.sum()), 1.0)
            for index in range(3):
                groupmates = [0, 0, 0]
                groupmates[index] = 1
                influenced = choice_probabilities(scenario, "later_subgroup", groupmates)
                self.assertGreater(float(influenced[index]), float(baseline[index]))

    def test_split_counts_preserve_paper_discrepancies(self) -> None:
        by_size = {
            size: sum(pattern.count for pattern in PUBLISHED_SPLIT_PATTERNS if pattern.group_size == size)
            for size in (2, 3, 4)
        }
        self.assertEqual(by_size, {2: 80, 3: 96, 4: 111})
        intact_three = next(p for p in PUBLISHED_SPLIT_PATTERNS if p.pattern == "3_intact")
        self.assertAlmostEqual(intact_three.proportion, 23 / 96)

    def test_monte_carlo_reproduction_is_deterministic(self) -> None:
        rows_a, sensitivity_a, metrics_a = reproduce_exit_choice_model(2000, 31)
        rows_b, sensitivity_b, metrics_b = reproduce_exit_choice_model(2000, 31)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(sensitivity_a, sensitivity_b)
        self.assertEqual(metrics_a, metrics_b)
        self.assertLess(float(metrics_a["monte_carlo_probability_mae"]), 0.02)
        self.assertTrue(np.isclose(float(metrics_a["probability_sum_max_error"]), 0.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()

