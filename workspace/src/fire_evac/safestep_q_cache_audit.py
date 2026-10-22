"""Finite-horizon diagnostic reference and positional closed-walk witnesses."""
import numpy as np


def exact_action_values(grid, rates, dt, step_cost, reward_scale, timeout_cost):
    cap, n = len(rates)-1, len(grid.labels)
    values = np.full((cap+1, n), -reward_scale*timeout_cost, dtype=float)
    values[:, list(grid.exits)] = 0
    q = np.full((cap, n, 4), np.nan)
    for k in range(cap-1, -1, -1):
        for state in range(n):
            if state in grid.exits:
                continue
            immediate = -reward_scale*(dt*rates[k, state]+step_cost)
            for action, new in grid.valid_neighbors(state):
                q[k, state, action] = immediate+values[k+1, new]
            values[k, state] = np.nanmax(q[k, state])
    return q


def closed_walk_members(trace, onset):
    members = np.zeros(len(trace), dtype=bool)
    last, witnesses = {}, []
    for k, step in enumerate(trace):
        if k < onset:
            continue
        old, new = step['from_state'], step['to_state']
        last[old] = k
        if new in last:
            begin = last[new]
            members[begin:k+1] = True
            witnesses.append({'begin_k': begin, 'end_k_exclusive': k+1,
                'length': k+1-begin,
                'cells': [trace[begin]['from_cell']]+[row['to_cell'] for row in trace[begin:k+1]]})
        last[new] = k+1
    return members, witnesses


def returns_to_go(rewards):
    return np.cumsum(np.asarray(rewards, dtype=float)[::-1])[::-1]
