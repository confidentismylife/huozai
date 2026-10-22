from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import MixedLogitBehavior
from fire_evac.dynamic_conflicts import (
    CONFLICT_TYPES,
    DOMAIN_RANDOMIZED_TYPES,
    ContradictionAwareLLMTrustGateWrapper,
    GymDynamicConflictEnv,
    HoldEnabledActionWrapper,
    LLMTrustGateWrapper,
    JointRouteReleaseGateWrapper,
    ReopenForecastObservationWrapper,
    TRUST_GATE_TYPES,
    conflict_llm_advice,
    make_dynamic_conflict_bank,
    make_domain_randomized_trust_bank,
    make_trust_gate_bank,
)
from fire_evac.dynamic_llm_rl import LLMGuidedObservationWrapper, safe_forecast_target


class DynamicConflictTests(unittest.TestCase):
    def test_bank_is_reproducible_and_covers_conflicts(self) -> None:
        first = make_dynamic_conflict_bank(12, 9, "C")
        second = make_dynamic_conflict_bank(12, 9, "C")
        self.assertEqual(first, second)
        self.assertEqual({scenario.conflict_type for scenario in first}, set(CONFLICT_TYPES))

    def test_forecast_error_points_to_actual_fire_source(self) -> None:
        scenario = next(s for s in make_dynamic_conflict_bank(12, 5, "E") if s.conflict_type == "forecast_error")
        advice = conflict_llm_advice(scenario)
        self.assertEqual(advice.target_branch, scenario.source_branch)
        self.assertNotEqual(advice.target_branch, safe_forecast_target(scenario))

    def test_blockage_is_observed_and_recovers(self) -> None:
        scenario = next(
            s for s in make_dynamic_conflict_bank(12, 7, "B") if s.conflict_type == "safe_exit_temporarily_blocked"
        )
        env = GymDynamicConflictEnv(MixedLogitBehavior(), [scenario])
        observation, info = env.reset(seed=1, options={"scenario": scenario})
        self.assertEqual(observation.shape, (18,))
        self.assertTrue(info["blockage_active"])
        blocked_feature = 16 if scenario.blocked_branch == 0 else 17
        self.assertEqual(float(observation[blocked_feature]), 1.0)
        for _ in range(scenario.blockage_end_step):
            observation, _, terminated, truncated, info = env.step(2)
            self.assertFalse(terminated or truncated)
        self.assertFalse(info["blockage_active"])
        self.assertEqual(float(observation[blocked_feature]), 0.0)

    def test_guided_conflict_observation_has_21_features(self) -> None:
        scenarios = make_dynamic_conflict_bank(12, 11, "G")
        advice = {scenario.scenario_id: conflict_llm_advice(scenario) for scenario in scenarios}
        env = LLMGuidedObservationWrapper(
            GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice
        )
        observation, _ = env.reset(seed=3, options={"scenario": scenarios[0]})
        self.assertEqual(observation.shape, (21,))
        self.assertTrue(env.observation_space.contains(observation))

    def test_trust_bank_and_reopen_forecast(self) -> None:
        scenarios = make_trust_gate_bank(10, 13, "T")
        self.assertEqual({scenario.conflict_type for scenario in scenarios}, set(TRUST_GATE_TYPES))
        scenario = next(s for s in scenarios if s.conflict_type == "safe_exit_short_block")
        env = ReopenForecastObservationWrapper(GymDynamicConflictEnv(MixedLogitBehavior(), scenarios))
        observation, info = env.reset(seed=2, options={"scenario": scenario})
        self.assertEqual(observation.shape, (19,))
        self.assertAlmostEqual(float(observation[-1]), info["blockage_steps_remaining"] / 12.0)
        next_observation, *_ = env.step(2)
        self.assertLess(float(next_observation[-1]), float(observation[-1]))

    def test_hold_and_trust_gate_actions(self) -> None:
        scenarios = make_trust_gate_bank(10, 15, "H")
        scenario = next(s for s in scenarios if s.conflict_type == "safe_exit_short_block")
        hold_env = HoldEnabledActionWrapper(GymDynamicConflictEnv(MixedLogitBehavior(), scenarios))
        hold_env.reset(seed=4, options={"scenario": scenario})
        _, _, _, _, hold_info = hold_env.step(3)
        self.assertEqual(hold_info["zone_counts"][2], 0)
        self.assertTrue(hold_info["hold_action_requested"])

        advice = {item.scenario_id: conflict_llm_advice(item) for item in scenarios}
        gate = LLMTrustGateWrapper(GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice)
        observation, _ = gate.reset(seed=4, options={"scenario": scenario})
        self.assertEqual(observation.shape, (22,))
        _, _, _, _, wait_info = gate.step(2)
        self.assertEqual(wait_info["trust_gate_action"], "wait")
        self.assertEqual(wait_info["zone_counts"][2], 0)

    def test_domain_randomized_bank_is_reproducible_and_broad(self) -> None:
        first = make_domain_randomized_trust_bank(60, 101, "R")
        second = make_domain_randomized_trust_bank(60, 101, "R")
        self.assertEqual(first, second)
        self.assertEqual({scenario.conflict_type for scenario in first}, set(DOMAIN_RANDOMIZED_TYPES))
        self.assertGreater(max(scenario.growth_rate for scenario in first), 0.03)
        self.assertGreater(max(scenario.onset_step for scenario in first), 7)

    def test_contradiction_wrapper_exposes_wrong_smoke_forecast(self) -> None:
        scenarios = make_trust_gate_bank(10, 31, "X")
        scenario = next(item for item in scenarios if item.conflict_type == "forecast_error")
        advice = {item.scenario_id: conflict_llm_advice(item) for item in scenarios}
        env = ContradictionAwareLLMTrustGateWrapper(
            GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice
        )
        observation, _ = env.reset(seed=2, options={"scenario": scenario})
        self.assertEqual(observation.shape, (28,))
        self.assertTrue(env.observation_space.contains(observation))
        for _ in range(scenario.onset_step + 2):
            observation, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated or truncated)
        self.assertGreater(float(observation[-4]), 0.0)
        self.assertEqual(float(observation[-3]), 1.0)

    def test_joint_gate_controls_route_and_release_capacity(self) -> None:
        scenarios = make_trust_gate_bank(10, 41, "J")
        scenario = next(item for item in scenarios if item.conflict_type == "safe_exit_congested")
        advice = {item.scenario_id: conflict_llm_advice(item) for item in scenarios}
        env = JointRouteReleaseGateWrapper(GymDynamicConflictEnv(MixedLogitBehavior(), scenarios), advice)
        observation, _ = env.reset(seed=3, options={"scenario": scenario})
        self.assertEqual(observation.shape, (29,))
        self.assertEqual(env.action_space.n, 9)
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["trust_gate_action"], "trust")
        self.assertEqual(info["release_fraction"], 0.25)
        self.assertEqual(info["room_release_capacity"], 5)


if __name__ == "__main__":
    unittest.main()
