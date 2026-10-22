from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fire_evac.behavior import RuleBehavior
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper
from fire_evac.counterfactual_shield import (
    ConditionalSafetyShieldWrapper, ShieldNetwork, ShieldPredictor,
    action_mask_metrics, effective_action_mask, evaluate_counterfactual_actions,
    mirror_action_values, mirror_shield_observation,
)
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank


def predictor():
    torch.manual_seed(17)
    model = ShieldNetwork()
    model.eval()
    return ShieldPredictor(model, np.zeros(34, np.float32), np.ones(34, np.float32), 0.7)


def environment():
    return ConfirmatoryJointControlWrapper(GymDynamicConflictEnv(RuleBehavior(), make_trust_gate_bank(5, 12, "T")), "none")


def test_checkpoint_roundtrip_and_advice_invariance(tmp_path):
    original = predictor()
    observation = np.linspace(0, 1, 34, dtype=np.float32)
    path = tmp_path / "shield.pt"
    original.save(path)
    loaded = ShieldPredictor.load(path)
    np.testing.assert_array_equal(original.probabilities(observation), loaded.probabilities(observation))
    changed = observation.copy()
    changed[28:33] = [1, 0, 0.99, 1, 1]
    np.testing.assert_array_equal(loaded.probabilities(observation), loaded.probabilities(changed))
    with pytest.raises(ValueError, match="34-dimensional"):
        loaded.probabilities(np.zeros(52))


def test_mirror_mask_including_empty_ties():
    scores = np.asarray([0.4, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1])
    mask = effective_action_mask(scores, 0.8)
    assert set(np.flatnonzero(mask)) == {0, 4}
    np.testing.assert_array_equal(effective_action_mask(mirror_action_values(scores), 0.8), mirror_action_values(mask))
    p = predictor()
    observation = np.linspace(0, 1, 34, dtype=np.float32)
    np.testing.assert_array_equal(p.mask(mirror_shield_observation(observation)), mirror_action_values(p.mask(observation)))


def test_metrics_count_forced_unsafe_fallback_and_unknown_rejections():
    labels = np.asarray([[0, 1, -1, 1, 1, 1, 1, 1, 1]])
    scores = np.asarray([[0.4, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
    m = action_mask_metrics(labels, scores, 0.8)
    assert m["unsafe_action_recall"] == 0.0
    assert m["empty_raw_mask_rate"] == 1.0
    assert m["indeterminate_action_rejection_rate"] == 1.0


def test_counterfactual_clone_does_not_change_original_state_or_rng():
    import copy
    env = environment()
    env.reset(seed=72)
    env.step(3)
    reference = copy.deepcopy(env)
    evaluate_counterfactual_actions(env, horizon=5)
    for action in (7, 8, 0, 6):
        actual = env.step(action)
        expected = reference.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    env.close()
    reference.close()


def test_wrapper_matches_central_mask_and_distinguishes_executed_action():
    base = environment()
    p = predictor()
    wrapper = ConditionalSafetyShieldWrapper(base, p, enforce=True)
    observation, _ = wrapper.reset(seed=81)
    assert observation.shape == (52,)
    np.testing.assert_array_equal(wrapper.current_mask, p.mask(observation[:34]))
    blocked = np.flatnonzero(~wrapper.current_mask)
    assert len(blocked)
    allowed = wrapper.current_mask.copy()
    _, _, _, _, info = wrapper.step(int(blocked[0]))
    assert info["shield_intervened"]
    assert allowed[info["shield_applied_action"]]
    assert info["shield_actual_joint_action"] == info["actual_joint_action"]
    assert info["shield_applied_actual_match"] == (info["shield_applied_action"] == info["actual_joint_action"])
    wrapper.close()
