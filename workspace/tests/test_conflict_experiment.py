from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.conflict_experiment import (
    LEFT,
    RIGHT,
    ConflictChoiceEnv,
    action_reward,
    conflict_prompt,
    generate_conflict_scenarios,
    optimal_action,
    parse_llm_advice,
)


class ConflictExperimentTests(unittest.TestCase):
    def test_scenarios_are_reproducible_and_unique(self) -> None:
        first = generate_conflict_scenarios(24, 7)
        second = generate_conflict_scenarios(24, 7)
        self.assertEqual(first, second)
        self.assertEqual(len({scenario.to_dict().__repr__() for scenario in first}), 24)

    def test_reward_prioritizes_clear_exit(self) -> None:
        scenario = next(s for s in generate_conflict_scenarios(12, 9) if s.smoke_left != s.smoke_right)
        clear_action = LEFT if scenario.smoke_left < scenario.smoke_right else RIGHT
        smoky_action = RIGHT if clear_action == LEFT else LEFT
        self.assertGreater(action_reward(scenario, clear_action), action_reward(scenario, smoky_action))
        self.assertEqual(optimal_action(scenario), clear_action)

    def test_llm_schema_and_environment(self) -> None:
        advice = parse_llm_advice(
            {"recommended_exit": "left", "confidence": 0.9, "reason_codes": ["avoid_smoke", "resolve_conflict"]}
        )
        scenarios = generate_conflict_scenarios(8, 12)
        env = ConflictChoiceEnv(scenarios, {s.scenario_id: advice for s in scenarios})
        observation, info = env.reset(seed=2)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn("scenario_id", info)
        _, reward, terminated, truncated, step_info = env.step(LEFT)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["chosen_exit"], "left")
        self.assertIn("Prioritize avoiding visible smoke", conflict_prompt(scenarios[0]))

        guided = ConflictChoiceEnv(scenarios, {s.scenario_id: advice for s in scenarios}, advice_alignment_bonus=0.25)
        guided.reset(seed=2, options={"scenario_id": scenarios[0].scenario_id})
        _, _, _, _, guided_info = guided.step(LEFT)
        self.assertAlmostEqual(float(guided_info["advice_alignment_bonus"]), 0.225)


if __name__ == "__main__":
    unittest.main()

