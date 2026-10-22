"""Shared current-observation contract and masked finite-horizon DQN helpers."""
import numpy as np
import torch
from torch import nn


def encode_observation(state, current_rates, steps, personal_fed, cap=128, divisor=.012):
    rates = np.asarray(current_rates, dtype=np.float32)
    if rates.shape != (24,) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Expected 24 nonnegative current FER values')
    if not 0 <= state < 24 or not 0 <= steps <= cap or personal_fed < 0:
        raise ValueError('Invalid position, clock or personal dose')
    result = np.zeros(50, dtype=np.float32)
    result[state] = 1
    result[24:48] = rates / divisor
    result[48], result[49] = steps/cap, personal_fed
    return result


def effective_observation(common, arm):
    result = np.array(common, dtype=np.float32, copy=True)
    if result.shape != (50,):
        raise ValueError('Expected shared 50-feature observation')
    if arm == 'hazard_blind':
        result[24:48] = 0
    elif arm != 'hazard_full':
        raise ValueError('Unknown observation arm')
    return result


def build_network(hidden=(64, 64)):
    layers, width = [], 50
    for size in hidden:
        layers.extend([nn.Linear(width, size), nn.ReLU()])
        width = size
    return nn.Sequential(*layers, nn.Linear(width, 4))


def masked_td_targets(rewards, terminated, next_q, next_legal, discount=1.):
    if torch.any(~next_legal.any(dim=1) & ~terminated):
        raise ValueError('Nonterminal transition without legal actions')
    future = next_q.masked_fill(~next_legal, -torch.inf).max(dim=1).values
    return rewards + discount*torch.where(terminated, torch.zeros_like(future), future)


def rate_table(grid, scenario, cfg):
    x = np.array([point[0] for point in grid.coordinates], dtype=float)
    profile = scenario['peak'] * np.exp(-.5*((x-scenario['center'])/cfg['spread_cells'])**2)
    rates = np.full((cfg['max_steps']+1, 24), cfg['background_rate_per_second'])
    rates[scenario['onset']:] += profile
    rates[:, list(grid.exits)] = 0
    return rates


def legal_matrix(grid):
    matrix = np.zeros((24, 4), dtype=bool)
    for state in range(24):
        matrix[state, [a for a, _ in grid.valid_neighbors(state)]] = True
    return matrix


def training_reward(dose_increment, exit_reached, task_done, cfg):
    cost = dose_increment + cfg['time_cost_in_FED_per_step']
    if task_done and not exit_reached:
        cost += cfg['training']['timeout_penalty_in_FED_units']
    return -cfg['training']['reward_scale'] * cost
