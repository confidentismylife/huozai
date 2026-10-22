from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.dynamic_multiseed import DYNAMIC_METRICS, cross_seed_statistics, summarize_cross_episodes


class DynamicMultiseedTests(unittest.TestCase):
    def test_cross_summary_keeps_behavior_and_seed_pairing(self) -> None:
        rows = []
        for behavior in ("rule", "logit"):
            for seed in (1, 2):
                for policy, offset in (("ppo", 0.0), ("llm_guided_ppo", -0.1)):
                    for episode in range(3):
                        row = {
                            "test_behavior": behavior,
                            "training_seed": seed,
                            "policy": policy,
                        }
                        for metric in DYNAMIC_METRICS:
                            row[metric] = float(seed + episode + offset)
                        rows.append(row)
        per_seed = summarize_cross_episodes(rows)
        self.assertEqual(len(per_seed), 8)
        aggregate, paired, paired_aggregate = cross_seed_statistics(per_seed, bootstrap_samples=500)
        self.assertEqual(len(aggregate), 2 * 2 * len(DYNAMIC_METRICS))
        self.assertEqual(len(paired), 2 * 2 * len(DYNAMIC_METRICS))
        self.assertEqual(len(paired_aggregate), 2 * len(DYNAMIC_METRICS))
        differences = [float(row["mean"]) for row in paired_aggregate]
        self.assertTrue(all(abs(value + 0.1) < 1e-12 for value in differences))


if __name__ == "__main__":
    unittest.main()
