"""Bounded synthetic observation probe; policies receive explicit observations."""
import heapq
import numpy as np


def scenario_rates(grid, scenario, cfg):
    """Piecewise constant FER per control interval, in FED/second."""
    values = np.full((cfg['max_steps'] + 1, len(grid.labels)), cfg['background_rate_per_second'])
    x = np.array([point[0] for point in grid.coordinates], dtype=float)
    peak = cfg['peak_rates_per_second'][scenario['severity']]
    center = cfg['danger_centers'][scenario['danger_side']]
    profile = peak * np.exp(-.5 * ((x - center) / cfg['spread_cells']) ** 2)
    values[cfg['reveal_steps'][scenario['reveal']]:] += profile
    values[:, list(grid.exits)] = 0
    return values


class ControlTickField:
    def __init__(self, rates, tau, dt):
        self.rates, self.tau, self.dt = rates, tau, dt

    def rate(self, state, time_s):
        raw = (time_s - self.tau) / self.dt
        k = int(np.floor(raw + 1e-9))
        if k < 0 or k >= len(self.rates):
            raise ValueError('Query outside frozen control-tick field')
        return float(self.rates[k, state])


def snapshot_path(grid, state, rates, dt, time_cost, exit_target=None):
    """Dijkstra with nonnegative departure-node exposure and per-move cost."""
    if not np.isfinite(rates).all() or (np.asarray(rates) < 0).any() or time_cost <= 0:
        raise ValueError('Finite nonnegative FER and positive time cost required')
    queue, best = [(0., state, [])], {state: 0.}
    while queue:
        cost, old, path = heapq.heappop(queue)
        if cost != best[old]:
            continue
        if old in grid.exits:
            if exit_target is None or old == exit_target:
                return path
            continue  # Another exit is terminal, never transit through it.
        for action, new in grid.valid_neighbors(old):
            proposed = cost + dt * float(rates[old]) + time_cost
            if proposed < best.get(new, float('inf')):
                best[new] = proposed
                heapq.heappush(queue, (proposed, new, path + [action]))
    raise ValueError('No reachable target exit')


def future_j_plan(grid, rates, dt, time_cost, start):
    """Privileged finite-horizon reference, with no waiting/illegal movement."""
    cap, n = len(rates) - 1, len(grid.labels)
    costs = np.full((cap + 1, n), np.inf)
    actions = np.full((cap, n), -1, dtype=int)
    costs[:, list(grid.exits)] = 0
    for k in range(cap - 1, -1, -1):
        for state in range(n):
            if state in grid.exits:
                continue
            best, action = min((dt*rates[k, state] + time_cost + costs[k+1, new], action)
                for action, new in grid.valid_neighbors(state))
            costs[k, state], actions[k, state] = best, action
    path, state = [], start
    for k in range(cap):
        if state in grid.exits:
            return path, float(costs[0, start])
        action = int(actions[k, state])
        if action < 0 or not np.isfinite(costs[k, state]):
            raise ValueError('No finite-horizon exit route')
        path.append(action)
        state = grid.transition(state, action)
    if state not in grid.exits:
        raise ValueError('Oracle did not reach an exit')
    return path, float(costs[0, start])
