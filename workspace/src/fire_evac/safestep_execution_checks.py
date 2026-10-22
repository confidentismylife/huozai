"""A geometry-only execution check and deterministic cycle witnesses."""
from __future__ import annotations
import numpy as np


def select_action(q_values, legal_actions=None):
    scores = np.asarray(q_values, dtype=float)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError('Expected four finite Q scores')
    if legal_actions is None:
        return int(scores.argmax())
    legal = sorted(set(legal_actions))
    if not legal or any(a not in [0, 1, 2, 3] for a in legal):
        raise ValueError('No legal moves or invalid legal-action indices')
    masked = np.full(4, -np.inf)
    masked[legal] = scores[legal]
    return int(masked.argmax())


def policy_graph_witness(grid, action_map, start):
    """Prove exit or repetition within a finite position-only functional graph."""
    states, seen = [], {}
    state = start
    while state not in grid.exits:
        if state in seen:
            cycle = states[seen[state]:]
            return {'graph_reaches_exit': False, 'first_repeat_at_transition': len(states),
                'cycle_states': cycle, 'cycle_labels': [grid.labels[s] for s in cycle],
                'cycle_length': len(cycle),
                'cycle_type': 'wall_self_loop' if len(cycle) == 1 else 'legal_movement_cycle',
                'precycle_states': states[:seen[state]]}
        seen[state] = len(states)
        states.append(state)
        state = grid.transition(state, action_map[state])
    return {'graph_reaches_exit': True, 'first_repeat_at_transition': None,
            'cycle_states': [], 'cycle_labels': [], 'cycle_length': 0, 'cycle_type': 'none',
            'precycle_states': states, 'exit_label': grid.labels[state]}
