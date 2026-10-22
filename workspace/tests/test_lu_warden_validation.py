from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.lu_warden_validation import (
    PUBLISHED_SCENARIOS,
    exit_utilities,
    right_exit_probability,
    validate_published_scenarios,
)


class LuWardenValidationTests(unittest.TestCase):
    def test_table_counts_preserve_published_discrepancy(self) -> None:
        # The paper states 1048 choices (131 x 8), but Table 5 sums to 1045.
        # Preserve the reported table rather than inventing three observations.
        self.assertEqual(sum(s.observations for s in PUBLISHED_SCENARIOS), 1045)

    def test_experimental_coding_matches_paper(self) -> None:
        no_warden = PUBLISHED_SCENARIOS[3]
        firefighter = PUBLISHED_SCENARIOS[11]
        left_none, right_none = exit_utilities(no_warden)
        left_firefighter, right_firefighter = exit_utilities(firefighter)
        self.assertAlmostEqual(left_none, 0.0)
        self.assertAlmostEqual(right_none, 0.263 + 0.035 * 20)
        self.assertAlmostEqual(left_firefighter, 2.739 + 0.559)
        self.assertAlmostEqual(right_firefighter, right_none)
        self.assertGreater(right_exit_probability(no_warden), right_exit_probability(firefighter))

    def test_validation_is_reproducible_and_tracks_majority_choice(self) -> None:
        rows_a, metrics_a = validate_published_scenarios(draws_per_scenario=2000, seed=17)
        rows_b, metrics_b = validate_published_scenarios(draws_per_scenario=2000, seed=17)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(metrics_a, metrics_b)
        self.assertGreaterEqual(float(metrics_a["majority_direction_accuracy"]), 0.90)
        self.assertLess(float(metrics_a["analytic_mae"]), 0.05)


if __name__ == "__main__":
    unittest.main()

