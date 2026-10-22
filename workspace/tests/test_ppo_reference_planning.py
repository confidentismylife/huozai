import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.behavior import RuleBehavior
from fire_evac.config import EnvConfig
from fire_evac.confirmatory import ConfirmatoryJointControlWrapper, NOISE_LEVELS
from fire_evac.dynamic_conflicts import GymDynamicConflictEnv, make_trust_gate_bank
from fire_evac.terminal_shield import ObservableHistoryWrapper
from fire_evac.source_world_planning import PLANS, evaluate_plan, select_reference_plan
from fire_evac.ppo_reference_planning import (
    FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, pure_actor_observation,
)


def environment(noise="none"):
    return ObservableHistoryWrapper(ConfirmatoryJointControlWrapper(
        GymDynamicConflictEnv(RuleBehavior(), make_trust_gate_bank(5, 12, "T"),
                              EnvConfig(max_steps=10)), "none", noise=NOISE_LEVELS[noise]))


def actor(env):
    return PPO("MlpPolicy", env.env, seed=76, n_steps=8, batch_size=4,
               policy_kwargs={"net_arch": [8]}, device="cpu")


def test_adapter_preserves_physical_forecast_and_release_memory_only():
    original = np.arange(49, dtype=np.float32)
    before = original.copy()
    adapted = pure_actor_observation(original)
    np.testing.assert_array_equal(adapted[:28], original[:28])
    np.testing.assert_array_equal(adapted[28:33], np.zeros(5))
    assert adapted[33] == original[33]
    changed = original.copy()
    changed[28:33] = -100
    changed[34:] = 300
    np.testing.assert_array_equal(pure_actor_observation(changed), adapted)
    np.testing.assert_array_equal(original, before)
    with pytest.raises(ValueError, match="49-dimensional"):
        pure_actor_observation(np.zeros(34))
    with pytest.raises(ValueError, match="finite"):
        pure_actor_observation(np.full(49, np.nan))


def test_real_sb3_continuation_is_reproducible_does_not_copy_or_update_actor():
    torch.set_num_threads(1)
    env = environment("medium")
    observation, _ = env.reset(seed=77)
    model = actor(env)
    continuation = FrozenPPOContinuation(601, model)
    initial_hash = policy_parameter_sha256(model)
    initial_steps = model.num_timesteps
    model.__deepcopy__ = lambda memo: pytest.fail("actor must not be copied")
    untouched = copy.deepcopy(env)
    before_observation = observation.copy()
    trace = []
    outcome = evaluate_continuation(env, observation, continuation, 151, trace)
    assert evaluate_continuation(env, observation, continuation, 151) == outcome
    assert len(trace) == outcome.steps
    assert trace[0][1] == outcome.first_requested_action
    assert trace[0][2] == outcome.first_actual_joint_action
    assert np.mean([r[4] for r in trace]) == pytest.approx(outcome.mean_release)
    assert np.mean([r[5] for r in trace]) == pytest.approx(outcome.hold_block_rate)
    assert policy_parameter_sha256(model) == initial_hash
    assert model.num_timesteps == initial_steps
    np.testing.assert_array_equal(observation, before_observation)
    for action in (7, 3, 8):
        actual = env.step(action)
        expected = untouched.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    env.close()
    untouched.close()


def test_new_rollout_matches_frozen_manual_evaluator_and_records_neutral_release():
    env = environment()
    observation, _ = env.reset(seed=78)
    for plan in (PLANS[0], PLANS[3], PLANS[9]):
        assert evaluate_continuation(env, observation, plan, 152) == evaluate_plan(env, observation, plan, 152)
    # Enter a reachable neutral sign state, then request release within its hold.
    for action in (3, 3, 3, 8):
        observation, *_ = env.step(action)
    trace = []
    outcome = evaluate_continuation(env, observation, PLANS[0], 152, trace)
    assert trace[0][1] in range(8)
    assert trace[0][2] == outcome.first_actual_joint_action == -1
    assert trace[0][3] == 2 and trace[0][4] > 0 and trace[0][5] == 1
    env.close()


def test_frozen_actor_rejects_wrong_observation_schema():
    env = environment()
    wrong_model = PPO("MlpPolicy", env, n_steps=8, batch_size=4,
                      policy_kwargs={"net_arch": [8]}, device="cpu")
    with pytest.raises(ValueError, match="34 inputs"):
        FrozenPPOContinuation(601, wrong_model)
    env.close()


def test_seventeen_candidate_selection_uses_whole_library_without_source_leak():
    risks = np.zeros((2, 17, 3, 11))
    risks[..., 0] = .2
    risks[:, 12, :, 0] = .08
    risks[0, 16, :, 0] = .02
    risks[1, 16, :, 0] = .13
    assert select_reference_plan(risks, [True, True], 0, "point") == 16
    assert select_reference_plan(risks, [True, True], 0, "robust") == 12
    assert select_reference_plan(risks, [True, True], 1, "robust") == 12
    assert select_reference_plan(risks[:, :12], [True, True], 0, "robust") == 0


def test_runner_namespaces_and_selection_keep_all_five_fixed_baselines():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_ppo_extended_world_planning import seed_schedule, select_indices
    schedule = [seed_schedule(i) for i in range(249)]
    planning = {seed for pair in schedule for seed in pair[0]}
    evaluation = {seed for pair in schedule for seed in pair[1]}
    assert len(planning) == len(evaluation) == 249 * 3
    streams = (planning, evaluation, {s + 9_000_001 for s in planning}, {s + 9_000_001 for s in evaluation})
    for i, seeds in enumerate(streams):
        assert all(not seeds.intersection(other) for other in streams[i + 1:])
    risks = np.zeros((2, 17, 3, 11))
    risks[..., 0] = .1
    risks[0, 12, :, 0] = .01
    risks[1, 16, :, 0] = .01
    north = select_indices(risks, [True, True], 0, 0, 7)
    south = select_indices(risks, [True, True], 0, 1, 7)
    assert len(north) == 15
    assert north[8:13] == south[8:13] == list(range(12, 17))
    for i in range(15):
        if i not in (3, 7):
            assert north[i] == south[i]  # Only oracle may depend on the label.
    assert north[3] == 12 and south[3] == 16


def test_virtual_fixed_ppo_expectation_is_outcome_average_not_action_vote():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_ppo_extended_world_planning import append_virtual_mean, FIXED_ARMS, METRICS, VIRTUAL_MEAN
    rows = [{"state_id": "T0", "future_seed": 1, "arm": arm, "selected_plan": arm,
             **{metric: i / 10 for metric in METRICS}, "first_requested_action": i}
            for i, arm in enumerate(FIXED_ARMS)]
    combined = append_virtual_mean(rows)
    assert len(combined) == 6 and len(rows) == 5
    assert combined[-1]["arm"] == VIRTUAL_MEAN
    assert combined[-1]["final_incapacitation"] == pytest.approx(.2)
    assert "first_requested_action" not in combined[-1]
    with pytest.raises(ValueError, match="five unique"):
        append_virtual_mean(rows[:-1])
