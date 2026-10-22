from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_evac.behavior import RuleBehavior
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper, NOISE_LEVELS
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank
from fire_evac.terminal_shield import (
    ACTION_MIRROR, ObservableHistoryWrapper, TernaryShieldNetwork, TernaryShieldPredictor,
    consensus_labels, mirror_history_observation, terminal_counterfactual_teachers,
    ternary_action_mask, ternary_mask_metrics,
)


def environment(noise="none"):
    return ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(RuleBehavior(), make_trust_gate_bank(5, 12, "T"), EnvConfig(max_steps=10)),
        "none", noise=NOISE_LEVELS[noise]))


def test_observable_memory_tracks_actual_actuator_and_resets():
    env = environment("medium")
    observation, _ = env.reset(seed=72)
    assert observation.shape == (49,)
    assert env.observation_space.contains(observation)
    assert observation[35] == 0
    observed_smoke = observation[5:10].copy()
    for action in (3, 7, 7, 7, 8, 0, 4):
        observation, _, _, _, info = env.step(action)
        domain = env.unwrapped.domain_env
        expected = max(0, domain.config.min_sign_hold_steps - domain.steps_since_switch) / domain.config.min_sign_hold_steps
        assert observation[35] == expected
        assert np.argmax(observation[36:39]) == info["actual_sign_action"]
        assert observation[34] == pytest.approx(info["step"] / 10)
    reset, _ = env.reset(seed=72)
    np.testing.assert_array_equal(reset[39:49], np.zeros(10))
    # No future/person-level exposure enters the history at reset.
    np.testing.assert_array_equal(reset[5:10], observed_smoke)
    env.close()


def test_sensor_integral_is_causal_even_with_delay():
    env = environment("medium")
    observation, _ = env.reset(seed=92)
    previous = observation.copy()
    accumulated = np.zeros(5, dtype=np.float32)
    for action in (3, 3, 7):
        observation, *_ = env.step(action)
        accumulated += previous[5:10]
        np.testing.assert_allclose(observation[39:44], accumulated / 10, atol=1e-7)
        np.testing.assert_allclose(observation[44:49], observation[5:10] - previous[5:10], atol=1e-7)
        previous = observation
    env.close()


def test_consensus_keeps_disagreement_unknown_without_forcing_winner():
    labels = np.asarray([[1, 0, -1, 1, 0, -1, 1, 0, -1], [1, 0, -1, 0, 1, 1, -1, -1, 0]])
    np.testing.assert_array_equal(consensus_labels(labels), [1, 0, -1, -1, -1, -1, -1, -1, -1])
    assert not np.any(consensus_labels(np.stack([np.eye(9)[0], np.eye(9)[4]])) == 1)


def test_terminal_teachers_reproducible_and_do_not_mutate_environment():
    import copy
    env = environment()
    env.reset(seed=81)
    env.step(3)
    reference = copy.deepcopy(env)
    labels, teachers, evaluations = terminal_counterfactual_teachers(env, (12, 29))
    labels2, teachers2, evaluations2 = terminal_counterfactual_teachers(env, (12, 29))
    np.testing.assert_array_equal(labels, labels2)
    np.testing.assert_array_equal(teachers, teachers2)
    assert evaluations == evaluations2
    assert teachers.shape == (4, 9)
    assert all(r.terminal_reached for _, _, results in evaluations for r in results)
    for action in (7, 8):
        actual = env.step(action)
        expected = reference.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    with pytest.raises(ValueError):
        terminal_counterfactual_teachers(env, (1, 1))
    env.close()
    reference.close()


def test_ternary_roundtrip_symmetry_advice_and_input_ablation(tmp_path):
    torch.manual_seed(4)
    model = TernaryShieldNetwork().eval()
    observation = np.linspace(0, 1, 49, dtype=np.float32)
    predictor = TernaryShieldPredictor((model,), np.zeros(49, np.float32), np.ones(49, np.float32), .4)
    path = tmp_path / "ternary.pt"
    predictor.save(path)
    restored = TernaryShieldPredictor.load(path)
    p = restored.probabilities(observation)
    np.testing.assert_array_equal(predictor.probabilities(observation), p)
    mirrored = restored.probabilities(mirror_history_observation(observation))
    np.testing.assert_allclose(mirrored, p[ACTION_MIRROR], atol=1e-7)
    changed = observation.copy()
    changed[28:33] = [1, 0, .99, 1, 1]
    np.testing.assert_array_equal(restored.probabilities(changed), p)
    np.testing.assert_array_equal(mirror_history_observation(mirror_history_observation(observation)), observation)
    restored.use_history = False
    ablated = restored.probabilities(changed)
    changed[34:] += 10
    np.testing.assert_array_equal(restored.probabilities(changed), ablated)


def test_predicted_unknown_abstention_and_forced_fallback_metrics():
    probabilities = np.tile([.8, .1, .1], (1, 9, 1))
    labels = np.zeros((1, 9), dtype=int)
    metrics = ternary_mask_metrics(labels, probabilities, .5)
    assert metrics["empty_raw_mask_rate"] == 1
    assert metrics["unsafe_action_recall"] == 0
    # All tied fallback candidates remain, so the mask has no directional bias.
    assert ternary_action_mask(probabilities, .5)[0].all()
    probabilities[0, 4] = [.3, .1, .6]
    mask, abstain, empty = ternary_action_mask(probabilities, .2)
    assert mask[0, 4] and abstain[0, 4]
    assert not empty[0]
    assert mask.sum() == 1
