from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.dynamic_llm_rl import (
    LLMGuidedObservationWrapper,
    ScenarioBankHazardProvider,
    dynamic_advice_prompt,
    make_scenario_bank,
    parse_dynamic_advice,
    safe_forecast_target,
)
from fire_evac.gym_env import GymFireEvacuationEnv


class DynamicLLMRLTests(unittest.TestCase):
    def test_scenario_bank_is_reproducible_and_balanced(self) -> None:
        first = make_scenario_bank(12, 8, "T")
        second = make_scenario_bank(12, 8, "T")
        self.assertEqual(first, second)
        self.assertEqual(sum(s.source_branch == 0 for s in first), 6)
        self.assertEqual(len({s.scenario_id for s in first}), 12)

    def test_parser_and_prompt(self) -> None:
        scenario = make_scenario_bank(2, 4, "P")[0]
        advice = parse_dynamic_advice(
            {
                "target_exit": "south",
                "confidence": 0.9,
                "reason_codes": ["avoid_predicted_smoke", "early_guidance"],
            }
        )
        self.assertEqual(advice.target_branch, 1)
        self.assertIn("lower-level reinforcement-learning controller", dynamic_advice_prompt(scenario))
        self.assertIn("not an evacuation origin", dynamic_advice_prompt(scenario))
        self.assertNotEqual(safe_forecast_target(scenario), scenario.source_branch)

    def test_guided_wrapper_augments_without_changing_reward(self) -> None:
        scenarios = make_scenario_bank(2, 2, "W")
        advice = {
            s.scenario_id: parse_dynamic_advice(
                {
                    "target_exit": "south" if s.source_branch == 0 else "north",
                    "confidence": 0.8,
                    "reason_codes": ["avoid_predicted_smoke"],
                }
            )
            for s in scenarios
        }
        provider = ScenarioBankHazardProvider(scenarios)
        base = GymFireEvacuationEnv(MixedLogitBehavior(), hazard_provider=provider)
        guided = LLMGuidedObservationWrapper(
            GymFireEvacuationEnv(MixedLogitBehavior(), hazard_provider=provider), advice
        )
        base_obs, _ = base.reset(seed=7, options={"scenario": scenarios[0]})
        guided_obs, info = guided.reset(seed=7, options={"scenario": scenarios[0]})
        self.assertEqual(base_obs.shape, (14,))
        self.assertEqual(guided_obs.shape, (17,))
        self.assertTrue(guided.observation_space.contains(guided_obs))
        self.assertIn("llm_target_exit", info)
        _, base_reward, *_ = base.step(2)
        _, guided_reward, _, _, step_info = guided.step(2)
        self.assertAlmostEqual(base_reward, guided_reward)
        self.assertTrue(np.isfinite(guided_reward))
        self.assertIn("action_matches_llm_target", step_info)


if __name__ == "__main__":
    unittest.main()
