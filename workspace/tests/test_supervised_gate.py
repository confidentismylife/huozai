from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.supervised_gate import balanced_class_weights, plan_action, select_safest_plan


class SupervisedGateTests(unittest.TestCase):
    def test_wait_plan_switches_after_reopening(self) -> None:
        self.assertEqual(plan_action("wait_then_trust", 3), 2)
        self.assertEqual(plan_action("wait_then_trust", 0), 0)
        self.assertEqual(plan_action("override", 8), 1)

    def test_plan_selection_prioritizes_safety(self) -> None:
        results = {
            "trust": {"incapacitation_rate": 0.1, "mean_cumulative_fed": 0.2, "total_reward": 2.0},
            "override": {"incapacitation_rate": 0.0, "mean_cumulative_fed": 0.4, "total_reward": -2.0},
            "wait_then_trust": {"incapacitation_rate": 0.0, "mean_cumulative_fed": 0.3, "total_reward": -3.0},
        }
        self.assertEqual(select_safest_plan(results), "wait_then_trust")

    def test_balanced_weights_upweight_rare_actions(self) -> None:
        weights = balanced_class_weights(np.asarray([0, 0, 0, 0, 1, 2]))
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(float(weights[1]), float(weights[2]))


if __name__ == "__main__":
    unittest.main()
