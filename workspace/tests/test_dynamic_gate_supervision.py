from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.dynamic_gate_supervision import (
    DYNAMIC_GATE_PLANS,
    DynamicGatePlan,
    dynamic_plan_action,
    select_safest_dynamic_plan,
)


class DynamicGateSupervisionTests(unittest.TestCase):
    def test_wait_duration_then_override(self) -> None:
        plan = DynamicGatePlan("wait4_override", "test", wait_steps=4, trust_fraction=0.0)
        self.assertEqual(dynamic_plan_action(plan, step=0, blockage_steps_remaining=8), 2)
        self.assertEqual(dynamic_plan_action(plan, step=3, blockage_steps_remaining=5), 2)
        self.assertEqual(dynamic_plan_action(plan, step=4, blockage_steps_remaining=4), 1)

    def test_wait_is_skipped_without_an_active_blockage(self) -> None:
        plan = DynamicGatePlan("wait4_trust", "test", wait_steps=4)
        self.assertEqual(dynamic_plan_action(plan, step=0, blockage_steps_remaining=0), 0)

    def test_half_split_uses_two_step_sign_slots(self) -> None:
        plan = DynamicGatePlan("half", "test", trust_fraction=0.5)
        actions = [dynamic_plan_action(plan, step=step, blockage_steps_remaining=0) for step in range(8)]
        self.assertEqual(actions, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_dynamic_bank_contains_all_requested_dimensions(self) -> None:
        families = {plan.family for plan in DYNAMIC_GATE_PLANS}
        self.assertTrue(
            {"timed_override", "wait_then_override", "split", "wait_then_split", "throttle"} <= families
        )

    def test_throttle_holds_without_rerouting(self) -> None:
        plan = DynamicGatePlan("throttle_half", "test", release_fraction=0.5)
        actions = [dynamic_plan_action(plan, step=step, blockage_steps_remaining=0) for step in range(8)]
        self.assertEqual(actions, [0, 0, 0, 0, 2, 2, 2, 2])

    def test_safety_selection_accepts_dynamic_plan_bank(self) -> None:
        plans = DYNAMIC_GATE_PLANS[:2]
        results = {
            plans[0].name: {"incapacitation_rate": 0.1, "mean_cumulative_fed": 0.1, "total_reward": 2.0},
            plans[1].name: {"incapacitation_rate": 0.0, "mean_cumulative_fed": 0.3, "total_reward": -2.0},
        }
        self.assertEqual(select_safest_dynamic_plan(results, plans).name, plans[1].name)


if __name__ == "__main__":
    unittest.main()
