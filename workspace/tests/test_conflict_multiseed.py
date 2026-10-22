from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.conflict_multiseed import (
    aggregate_paired_differences,
    aggregate_seed_summaries,
    bootstrap_mean_interval,
    paired_seed_differences,
)


class ConflictMultiseedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = []
        for seed, ppo_reward, guided_reward in ((101, 0.4, 0.6), (102, 0.5, 0.7)):
            for policy, reward, optimal in (
                ("ppo", ppo_reward, 0.75),
                ("llm_guided_ppo", guided_reward, 1.0),
            ):
                self.rows.append(
                    {
                        "training_seed": seed,
                        "policy": policy,
                        "mean_reward": reward,
                        "optimal_choice_rate": optimal,
                        "smoke_choice_rate": 0.0,
                        "mean_distance": 4.0,
                    }
                )

    def test_bootstrap_is_reproducible_and_bounded(self) -> None:
        first = bootstrap_mean_interval([1.0, 2.0, 3.0], samples=500, seed=9)
        second = bootstrap_mean_interval([1.0, 2.0, 3.0], samples=500, seed=9)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 1.0)
        self.assertLessEqual(first[1], 3.0)

    def test_aggregate_and_paired_differences(self) -> None:
        aggregate = aggregate_seed_summaries(self.rows, bootstrap_samples=500)
        reward = next(row for row in aggregate if row["policy"] == "ppo" and row["metric"] == "mean_reward")
        self.assertAlmostEqual(float(reward["mean"]), 0.45)
        self.assertEqual(reward["training_seeds"], 2)

        paired = paired_seed_differences(self.rows)
        reward_differences = [row["difference"] for row in paired if row["metric"] == "mean_reward"]
        self.assertEqual(len(reward_differences), 2)
        self.assertTrue(all(abs(float(value) - 0.2) < 1e-12 for value in reward_differences))
        paired_aggregate = aggregate_paired_differences(paired, bootstrap_samples=500)
        paired_reward = next(row for row in paired_aggregate if row["metric"] == "mean_reward")
        self.assertAlmostEqual(float(paired_reward["mean"]), 0.2)


if __name__ == "__main__":
    unittest.main()
