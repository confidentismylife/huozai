from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fire_evac.public_route_validation import parse_trial, participant_splits, route_prompt, score_predictions
from run_mirror_validation import swapped


class PublicRouteValidationTests(unittest.TestCase):
    def test_prompt_cannot_reveal_observed_choice(self) -> None:
        left = parse_trial(
            {"taskId": "01_L-EL0_R-EM0", "AOI": "corridor_left", "userId": "person-1"}
        )
        self.assertIsNotNone(left)
        right = replace(left, choice="right")
        self.assertEqual(route_prompt(left), route_prompt(right))
        self.assertIn("左走廊：宽2米，长20米", route_prompt(left))

    def test_training_tasks_and_undocumented_codes_are_excluded(self) -> None:
        self.assertIsNone(
            parse_trial({"taskId": "01_L-EL0_R-EM0_tr", "AOI": "corridor_left", "userId": "p"})
        )
        self.assertIsNone(
            parse_trial({"taskId": "39_L-EE0_R-EE1", "AOI": "corridor_left", "userId": "p"})
        )

    def test_participant_split_keeps_repeated_trials_together(self) -> None:
        trials = []
        for index in range(20):
            first = parse_trial(
                {"taskId": "01_L-EL0_R-EM0", "AOI": "corridor_left", "userId": str(index)}
            )
            trials.extend([first, replace(first, task_id="02_L-EL0_R-ES0")])
        split = participant_splits(trials, seed=99)
        self.assertEqual(len(split), 20)
        self.assertEqual({split[trial.participant_id] for trial in trials if trial.participant_id == "5"}, {split["5"]})
        self.assertEqual({name: list(split.values()).count(name) for name in ("train", "validation", "test")},
                         {"train": 12, "validation": 4, "test": 4})

    def test_mirror_is_reversible_and_does_not_use_choice(self) -> None:
        left = parse_trial({"taskId": "03_L-EL0_R-ES1", "AOI": "corridor_left", "userId": "p"})
        self.assertEqual(swapped(swapped(left)), left)
        self.assertEqual(route_prompt(swapped(left)), route_prompt(swapped(replace(left, choice="right"))))
        self.assertIn("左走廊：宽2米，长10米，尽头有楼梯", route_prompt(swapped(left)))

    def test_validation_scoring_excludes_test_participant(self) -> None:
        train = parse_trial({"taskId": "01_L-EL0_R-EM0", "AOI": "corridor_left", "userId": "train"})
        validation = replace(train, participant_id="validation", choice="right")
        test = replace(train, participant_id="test")
        rows, metrics = score_predictions(
            [train, validation, test],
            {"train": "train", "validation": "validation", "test": "test"},
            {train.task_id: 0.3}, evaluation_split="validation",
        )
        self.assertEqual({r["participant_id"] for r in rows}, {"validation"})
        self.assertEqual(metrics[0]["validation_decisions"], 1)


if __name__ == "__main__":
    unittest.main()
