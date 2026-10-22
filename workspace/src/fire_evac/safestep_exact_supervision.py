"""Exact targets depend only on the current shared observation in a static task."""
import numpy as np
import torch
from fire_evac.safestep_q_cache_audit import exact_action_values


def snapshot_teacher(grid, normalized_current_rates, source_cfg, dt):
    current = np.asarray(normalized_current_rates, dtype=np.float32)
    if current.shape != (24,) or not np.isfinite(current).all() or (current < 0).any():
        raise ValueError('Invalid current normalized FER snapshot')
    decoded = current.astype(float)*source_cfg['observation']['FER_divisor']
    rates = np.repeat(decoded[None, :], source_cfg['max_steps']+1, axis=0)
    tr = source_cfg['training']
    return exact_action_values(grid, rates, dt, source_cfg['time_cost_in_FED_per_step'],
                               tr['reward_scale'], tr['timeout_penalty_in_FED_units'])


def legal_mse(predictions, labels, legal):
    if predictions.shape != labels.shape or predictions.shape != legal.shape or not legal.any():
        raise ValueError('Predictions, labels and nonempty legal mask must agree')
    return ((predictions-labels).square()*legal).sum()/legal.sum()


def quality(predictions, labels, legal, tolerance):
    masked_pred = np.where(legal, predictions, -np.inf)
    actions = masked_pred.argmax(axis=1)
    best = np.where(legal, labels, -np.inf).max(axis=1)
    chosen_truth = labels[np.arange(len(actions)), actions]
    errors = predictions[legal]-labels[legal]
    return {'states': int(len(actions)), 'legal_action_values': int(legal.sum()),
        'legal_Q_MAE': float(np.abs(errors).mean()), 'legal_Q_RMSE': float(np.sqrt(np.mean(errors**2))),
        'optimal_action_accuracy': float(np.mean(best-chosen_truth <= tolerance)),
        'mean_action_regret': float(np.mean(best-chosen_truth)),
        'positive_selected_Q_fraction': float(np.mean(masked_pred.max(axis=1) > tolerance))}
