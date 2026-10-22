import numpy as np
import pytest
import torch
from fire_evac.safestep_shared_observation import (
    encode_observation, effective_observation, masked_td_targets, training_reward)


def test_blinding_changes_only_current_FER_and_preserves_own_dose_clock():
    common = encode_observation(13, np.arange(24)*.0001, 9, .03)
    blind = effective_observation(common, 'hazard_blind')
    assert np.array_equal(blind[:24], common[:24])
    assert np.array_equal(blind[48:], common[48:])
    assert np.all(blind[24:48] == 0) and np.any(common[24:48] != 0)
    assert np.array_equal(effective_observation(common, 'hazard_full'), common)
    assert common[48] == pytest.approx(9/128)


def test_identical_current_observations_cannot_encode_different_future_onsets():
    now = np.full(24, .00005)
    assert np.array_equal(encode_observation(23, now, 0, 0), encode_observation(23, now.copy(), 0, 0))


def test_TD_target_cannot_use_illegal_high_Q_and_terminal_does_not_bootstrap():
    rewards = torch.tensor([-1., -2.])
    terminated = torch.tensor([False, True])
    q = torch.tensor([[100., 2., 3., 200.], [5., 6., 7., 8.]])
    legal = torch.tensor([[False, True, True, False], [False, False, False, False]])
    result = masked_td_targets(rewards, terminated, q, legal)
    assert result.tolist() == [2., -2.]
    with pytest.raises(ValueError):
        masked_td_targets(rewards, torch.tensor([False, False]), q, legal)


def test_successful_return_equals_negative_scaled_J_and_timeout_has_explicit_cost():
    cfg = {'time_cost_in_FED_per_step': .00125,
           'training': {'reward_scale': 20, 'timeout_penalty_in_FED_units': 2}}
    doses = [.001, .002, .003]
    rewards = [training_reward(d, k == 2, k == 2, cfg) for k, d in enumerate(doses)]
    assert sum(rewards) == pytest.approx(-20*(sum(doses)+3*.00125))
    assert training_reward(.003, False, True, cfg) == pytest.approx(rewards[-1]-40)
    assert training_reward(.003, False, False, cfg) == pytest.approx(rewards[-1])
