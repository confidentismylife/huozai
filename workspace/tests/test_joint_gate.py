from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.joint_gate import (
    WAIT_ACTION,
    JointGatePlan,
    encode_joint_action,
    joint_plan_action,
)
from fire_evac.action_diagnostics import copy_previous_joint_action
from fire_evac.behavior import RuleBehavior
from fire_evac.dynamic_conflicts import (
    GymDynamicConflictEnv,
    JointRouteReleaseGateWrapper,
    conflict_llm_advice,
    make_trust_gate_bank,
)
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.advice_protocol_v2 import (
    advice_prompt_v2,
    expected_semantic_target,
    mirror_scenario,
    safety_validate_advice,
)
from fire_evac.dynamic_llm_rl import DynamicLLMAdvice
from fire_evac.counterfactual_shield import CounterfactualActionResult, admissible_action_mask


class JointGateTests(unittest.TestCase):
    def test_action_encoding(self) -> None:
        self.assertEqual(encode_joint_action("trust", 0.25), 0)
        self.assertEqual(encode_joint_action("trust", 1.0), 3)
        self.assertEqual(encode_joint_action("override", 0.25), 4)
        self.assertEqual(encode_joint_action("override", 1.0), 7)

    def test_reactive_plan_uses_smoke_before_capacity(self) -> None:
        plan = JointGatePlan(
            "reactive", "reactive", release_fraction=0.5, reactive_capacity=True
        )
        self.assertEqual(
            joint_plan_action(
                plan,
                step=5,
                blockage_steps_remaining=0,
                smoke_contradiction=True,
                target_capacity_difference=-0.5,
            ),
            7,
        )

    def test_reactive_plan_throttles_safe_congested_route(self) -> None:
        plan = JointGatePlan(
            "reactive", "reactive", release_fraction=0.5, reactive_capacity=True
        )
        self.assertEqual(
            joint_plan_action(
                plan,
                step=5,
                blockage_steps_remaining=0,
                smoke_contradiction=False,
                target_capacity_difference=-0.5,
            ),
            1,
        )

    def test_reactive_plan_waits_for_blockage(self) -> None:
        plan = JointGatePlan(
            "reactive", "reactive", release_fraction=0.5, reactive_capacity=True
        )
        self.assertEqual(
            joint_plan_action(
                plan,
                step=1,
                blockage_steps_remaining=3,
                smoke_contradiction=False,
                target_capacity_difference=0.0,
            ),
            WAIT_ACTION,
        )

    def test_copy_previous_action_decoder(self) -> None:
        observations = np.zeros((3, 29), dtype=np.float32)
        observations[:, 19] = 1.0  # advice points north
        observations[0, 10] = 1.0
        observations[0, 28] = 0.5
        observations[1, 11] = 1.0
        observations[1, 28] = 1.0
        observations[2, 12] = 1.0
        observations[2, 28] = 1.0
        np.testing.assert_array_equal(copy_previous_joint_action(observations), [1, 7, 8])

    def test_joint_wrapper_records_requested_and_executed_actions(self) -> None:
        scenario = make_trust_gate_bank(5, 17, "T")[0]
        advice = {scenario.scenario_id: conflict_llm_advice(scenario)}
        env = JointRouteReleaseGateWrapper(GymDynamicConflictEnv(RuleBehavior(), (scenario,)), advice)
        env.reset(seed=3, options={"scenario": scenario})
        _, _, _, _, first = env.step(3)  # trust, full release; first switch is allowed
        _, _, _, _, second = env.step(7)  # immediate override request is held
        self.assertEqual(first["requested_gate_action"], "trust")
        self.assertEqual(first["actual_gate_action"], "trust")
        self.assertEqual(second["requested_gate_action"], "override")
        self.assertEqual(second["actual_gate_action"], "trust")
        self.assertTrue(second["action_blocked_by_sign_hold"])
        self.assertEqual(second["requested_joint_action"], 7)
        self.assertEqual(second["actual_joint_action"], 3)
        env.close()

    def test_confirmatory_arms_have_identical_spaces_and_absolute_actions(self) -> None:
        scenario = make_trust_gate_bank(5, 21, "C")[0]
        advice = {scenario.scenario_id: conflict_llm_advice(scenario)}
        envs = [
            ConfirmatoryJointControlWrapper(
                GymDynamicConflictEnv(RuleBehavior(), (scenario,)), mode, advice
            )
            for mode in ("none", "rule", "llm")
        ]
        observations = [env.reset(seed=5, options={"scenario": scenario})[0] for env in envs]
        self.assertTrue(all(env.action_space.n == 9 for env in envs))
        self.assertTrue(all(obs.shape == (34,) for obs in observations))
        # Every arm exposes identical physical/raw-forecast fields.
        np.testing.assert_allclose(observations[0][:28], observations[1][:28])
        np.testing.assert_allclose(observations[0][:28], observations[2][:28])
        _, _, _, _, info = envs[0].step(7)  # south at full release
        self.assertEqual(info["requested_sign_action"], 1)
        self.assertEqual(info["requested_release_fraction"], 1.0)
        for env in envs:
            env.close()

    def test_v2_prompt_and_safety_guard_preserve_fire_source_semantics(self) -> None:
        scenario = make_trust_gate_bank(5, 31, "V")[0]
        prompt = advice_prompt_v2(scenario)
        self.assertIn("danger location", prompt)
        unsafe = DynamicLLMAdvice(
            scenario.forecast_source_branch, 0.95, ("avoid_predicted_smoke",)
        )
        corrected, changed = safety_validate_advice(scenario, unsafe)
        self.assertTrue(changed)
        self.assertEqual(corrected.target_branch, expected_semantic_target(scenario))
        self.assertLessEqual(corrected.confidence, 0.60)

    def test_mirror_scenario_swaps_directional_fields(self) -> None:
        scenario = make_trust_gate_bank(5, 32, "V")[0]
        mirrored = mirror_scenario(scenario)
        self.assertNotEqual(mirrored.source_branch, scenario.source_branch)
        self.assertNotEqual(mirrored.forecast_source_branch, scenario.forecast_source_branch)
        self.assertEqual(mirrored.north_exit_capacity, scenario.south_exit_capacity)
        self.assertEqual(mirrored.south_exit_capacity, scenario.north_exit_capacity)

    def test_counterfactual_mask_keeps_three_lexicographically_safest_actions(self) -> None:
        results = tuple(
            CounterfactualActionResult(
                action=index,
                incapacitation_delta=0.0 if index < 4 else 0.1,
                fed_delta=float(index) / 100.0,
                congestion_per_step=0.0,
                evacuation_delta=0.0,
                total_reward=0.0,
            )
            for index in range(9)
        )
        mask, ranks = admissible_action_mask(results)
        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1, 2])
        self.assertEqual(int(np.argmin(ranks)), 0)


if __name__ == "__main__":
    unittest.main()
