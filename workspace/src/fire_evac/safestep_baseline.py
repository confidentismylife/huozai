"""SafeStep equations and a 24-cell source-grounded map; synthetic FER is explicit.

This module does not reuse the project's crowd simulator. Rates supplied through
HazardField are FED per second, preventing implicit concentration/time units.
"""
from __future__ import annotations
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


@dataclass(frozen=True)
class SmallMap:
    labels: tuple[str, ...]
    coordinates: tuple[tuple[int, int], ...]
    start: int = 23
    exits: tuple[int, ...] = (20, 21)

    @classmethod
    def paper_case_map(cls):
        return cls(tuple([f'R{i+1}' for i in range(20)] + ['Exit_A', 'Exit_B', 'Room_A', 'Room_B']),
            tuple([(i, 0) for i in range(20)] + [(0, 1), (19, 1), (17, 1), (13, -1)]))

    def transition(self, state, action):
        dx, dy = ACTION_DELTAS[action]
        x, y = self.coordinates[state]
        destination = (x + dx, y + dy)
        return self.coordinates.index(destination) if destination in self.coordinates else state

    def valid_neighbors(self, state):
        return [(action, self.transition(state, action)) for action in range(4)
                if self.transition(state, action) != state]


@dataclass
class HazardField:
    times_s: np.ndarray
    fer_per_s: np.ndarray
    labels: tuple[str, ...]
    provenance: str

    def __post_init__(self):
        if self.fer_per_s.shape != (len(self.times_s), len(self.labels)):
            raise ValueError('FER shape mismatch')
        if len(self.times_s) < 2 or not np.all(np.diff(self.times_s) > 0):
            raise ValueError('Time series must be strictly increasing')
        if not np.isfinite(self.fer_per_s).all() or (self.fer_per_s < 0).any():
            raise ValueError('FER must be finite and nonnegative')

    def rate(self, state, time_s):
        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            raise ValueError('Hazard query beyond the supplied time domain')
        return float(np.interp(time_s, self.times_s, self.fer_per_s[:, state]))

    def stationary_fed(self, time_s):
        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            raise ValueError('Stationary FED requested outside profile')
        # Include the exact endpoint, avoiding a silently shortened integral.
        times = self.times_s[self.times_s < time_s]
        times = np.r_[times, time_s]
        values = np.column_stack([np.interp(times, self.times_s, self.fer_per_s[:, i])
                                  for i in range(len(self.labels))])
        return np.trapezoid(values, times, axis=0)

    @classmethod
    def from_csv(cls, path, grid):
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        times = np.array([float(r['time_s']) for r in rows])
        values = np.array([[float(r[label]) for label in grid.labels] for r in rows])
        return cls(times, values, grid.labels, f'Externally supplied FER-per-second CSV: {Path(path).name}')

    def to_csv(self, path):
        with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time_s', *self.labels])
            writer.writerows([float(t), *[float(v) for v in rates]] for t, rates in zip(self.times_s, self.fer_per_s))


def controlled_hazard(grid, cfg):
    times = np.arange(0, cfg['profile_end_seconds'] + 1, cfg['profile_sampling_seconds'], dtype=float)
    growth = np.minimum((times / cfg['growth_scale_seconds']) ** 2, 1)
    sigma = cfg['initial_spread_cells'] + cfg['spread_per_second'] * np.minimum(times, cfg['spread_saturation_seconds'])
    values = np.zeros((len(times), len(grid.labels)))
    for state, (x, _) in enumerate(grid.coordinates):
        if state in grid.exits: continue
        smoke = np.exp(-.5 * ((x - 17) / sigma) ** 2)
        values[:, state] = growth * (cfg['background_rate_per_s'] + cfg['source_rate_per_s'] * smoke)
        if grid.labels[state] == 'Room_A': values[:, state] *= cfg['fire_room_rate_multiplier']
    return HazardField(times, values, grid.labels, cfg['status'])


def toxic_utility(fed, settings):
    utilities = settings['tox_utilities']
    lo, hi = settings['FED_thresholds']
    return utilities['safe'] if fed == 0 else utilities['alert'] if fed < lo else utilities['critical'] if fed < hi else utilities['severe']


class SafeStepEpisode:
    def __init__(self, grid, hazard, settings, start_time_s, max_steps, start=None):
        self.grid, self.hazard, self.settings = grid, hazard, settings
        self.start_time_s, self.max_steps = start_time_s, max_steps
        self.state = grid.start if start is None else start
        self.steps, self.fed, self.visited = 0, 0.0, {self.state}
        self.done = self.state in grid.exits
        self.trace = []
        self.dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']

    def step(self, action):
        if self.done: raise ValueError('Stepping a terminated episode')
        old = self.state
        absolute_time = self.start_time_s + self.steps * self.dt
        dose = self.hazard.rate(old, absolute_time) * self.dt
        self.fed += dose
        new = self.grid.transition(old, action)
        invalid, revisit = new == old, new != old and new in self.visited
        end = new in self.grid.exits
        kind = 'impossible' if invalid else 'exit' if end else 'visited' if revisit else 'regular'
        path_component = self.settings['path_rewards'][kind]
        utility = toxic_utility(self.fed, self.settings)
        tox_component = utility * dose
        reward = self.settings['sigma_path'] * path_component + self.settings['sigma_tox'] * tox_component
        self.steps += 1
        self.state = new
        self.visited.add(new)
        self.done = end or self.steps >= self.max_steps
        info = {'step': self.steps, 'from_cell': self.grid.labels[old], 'to_cell': self.grid.labels[new],
            'from_state': old, 'to_state': new, 'action': int(action), 'absolute_time_s': absolute_time,
            'dose_increment': dose, 'cumulative_personal_FED': self.fed, 'path_reward_kind': kind,
            'path_reward': path_component, 'tox_utility': utility, 'tox_reward': tox_component,
            'reward': reward, 'invalid': bool(invalid), 'revisit': bool(revisit), 'exit_reached': bool(end)}
        self.trace.append(info)
        return new, float(reward), self.done, info


def shortest_path(grid, start, hazard=None, start_time_s=None, threshold=.3):
    blocked = set()
    if hazard is not None:
        stationary = hazard.stationary_fed(start_time_s)
        blocked = set(np.flatnonzero(stationary >= threshold).tolist())
    if start in blocked: return None, sorted(blocked)
    queue = [(0, start, [])]
    distances = {start: 0}
    while queue:
        distance, state, path = heapq.heappop(queue)
        if distance != distances[state]: continue
        if state in grid.exits: return path, sorted(blocked)
        for action, new in grid.valid_neighbors(state):
            if new in blocked: continue
            new_distance = distance + 1
            if new_distance < distances.get(new, float('inf')):
                distances[new] = new_distance
                heapq.heappush(queue, (new_distance, new, path + [action]))
    return None, sorted(blocked)


def future_fed_oracle(grid, hazard, settings, start_time_s, max_steps, start=None):
    start = grid.start if start is None else start
    dt = settings['cell_size_m'] / settings['walking_speed_m_per_s']
    n = len(grid.labels)
    cost = np.full((max_steps + 1, n), np.inf)
    choices = np.full((max_steps, n), -1, dtype=int)
    cost[:, list(grid.exits)] = 0
    for k in range(max_steps - 1, -1, -1):
        t = start_time_s + k * dt
        for state in range(n):
            if state in grid.exits: continue
            dose = hazard.rate(state, t) * dt
            candidates = [(dose + cost[k+1, new], action) for action, new in grid.valid_neighbors(state)]
            value, action = min(candidates)
            cost[k, state], choices[k, state] = value, action
    path, state = [], start
    for k in range(max_steps):
        if state in grid.exits: break
        action = int(choices[k, state])
        if action < 0 or not np.isfinite(cost[k, state]): return None, float('inf')
        path.append(action)
        state = grid.transition(state, action)
    return path, float(cost[0, start])


def evaluate_actions(grid, hazard, settings, tau, max_steps, actions, start=None):
    if actions is None:
        return {'exit_reached': False, 'no_path': True, 'steps': 0, 'time_seconds': None,
                'personal_FED': None, 'undiscounted_reward': None, 'discounted_reward': None,
                'invalid_actions': 0, 'revisits': 0, 'exit': None}, []
    env = SafeStepEpisode(grid, hazard, settings, tau, max_steps, start)
    for action in actions:
        if env.done: break
        env.step(action)
    trace = env.trace
    return {'exit_reached': env.state in grid.exits, 'no_path': False, 'steps': env.steps,
        'time_seconds': env.steps * env.dt, 'personal_FED': env.fed,
        'undiscounted_reward': sum(row['reward'] for row in trace),
        'discounted_reward': sum(settings['discount'] ** i * row['reward'] for i, row in enumerate(trace)),
        'invalid_actions': sum(row['invalid'] for row in trace), 'revisits': sum(row['revisit'] for row in trace),
        'exit': grid.labels[env.state] if env.state in grid.exits else None}, trace
