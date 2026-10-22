from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_evac.behavior import RuleBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank
from fire_evac.terminal_shield import ObservableHistoryWrapper, ternary_action_mask
from fire_evac.terminal_shield_runtime import TerminalShieldWrapper


class FixedPredictor:
    threshold = .4

    def probabilities(self, observation):
        scores = np.tile([.9, .05, .05], (9, 1)).astype(np.float32)
        scores[0] = [.1, .7, .2]
        scores[8] = [.2, .1, .7]
        return scores


def test_runtime_matches_validated_mask_and_keeps_requested_applied_actual_separate():
    base = ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(RuleBehavior(), make_trust_gate_bank(5, 13, "T")), "none"))
    p = FixedPredictor()
    env = TerminalShieldWrapper(base, p, enforce=True)
    observation, _ = env.reset(seed=33)
    assert observation.shape == (94,)
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(env.current_mask, ternary_action_mask(p.probabilities(observation[:49]), p.threshold)[0])
    _, _, _, _, info = env.step(7)
    assert info["shield_requested_action"] == 7
    assert info["shield_applied_action"] == 0
    assert info["shield_intervened"]
    assert info["shield_actual_joint_action"] == info["actual_joint_action"]
    # Waiting is a predicted abstention and passes without substitution.
    _, _, _, _, info = env.step(8)
    assert info["shield_applied_action"] == 8
    assert not info["shield_intervened"]
    assert info["shield_predicted_abstention_count"] == 1
    with pytest.raises(ValueError):
        env.step(9)
    env.expose = False
    observation, _ = env.reset(seed=33)
    np.testing.assert_array_equal(observation[49:], np.zeros(45))
    env.close()
