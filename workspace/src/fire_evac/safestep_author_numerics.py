"""Read-only adapter for the author SafeStep Beta data and stored DQN.

No published endpoint is used to construct hazards, routes, or model predictions.
Import the reviewed source text directly, avoiding downloaded __pycache__ files.
"""
from __future__ import annotations

import ast
import hashlib
import types
from collections import deque
from pathlib import Path

import numpy as np
import openpyxl
import torch
from torch import nn

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_reviewed_source(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(Path(path).resolve())
    exec(compile(Path(path).read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    return module


def literal_assign(path, name):
    """Extract only a literal assignment; never run main.py's demonstration."""
    root = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in root.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "array":
                value = value.args[0]
            return ast.literal_eval(value)
    raise ValueError(f"Assignment {name} absent")


def load_sheet(path, sheet):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook[sheet].values)
    workbook.close()
    header = rows[0]
    # The author rate worksheet has a completely blank final row.
    rows = [row for row in rows[1:] if row[0] is not None]
    data = np.asarray(rows, dtype=float)
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Nonfinite worksheet values or non-increasing times")
    return {str(key): data[:, i] for i, key in enumerate(header)}


def load_author_inputs(directory, start_time_s=300):
    directory = Path(directory)
    grid = np.asarray(literal_assign(directory / "main.py", "maze"), dtype=int)
    mapping = literal_assign(directory / "coordinates2areanumber.py", "mapping")
    rates = load_sheet(directory / "FED rate - room24.xlsx", "Sheet1")
    pattern = load_sheet(directory / "fire hazard zones FED - room24.xlsx", "Sheet1")
    matches = np.flatnonzero(rates["Time"] == start_time_s)
    if len(matches) != 1:
        raise ValueError("Evacuation start must be an exact unique rate timestamp")
    index = int(matches[0])
    # Match main.py exactly, including reuse of the rate index for pattern rows.
    streams = {xy: rates[label][index:].tolist() for xy, label in mapping.items() if label in rates}
    patterns = {xy: pattern[label][index:].tolist() for xy, label in mapping.items() if label in pattern}
    nrows, ncols = grid.shape
    exits = ((6, 4), (14, 2))
    empty = [(col, row) for col in range(ncols) for row in range(nrows)
             if grid[row, col] == 0 and (col, row) not in exits]
    assert set(empty) == set(streams)
    return dict(grid=grid, mapping=mapping, rates=rates, pattern=pattern,
                streams=streams, patterns=patterns, empty=empty, exits=exits,
                start_index=index)


class AuthorDQN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc1 = nn.Linear(n, n)
        self.out = nn.Linear(n, 4)

    def forward(self, x):
        return self.out(torch.relu(self.fc1(x)))


class StoredAuthorPolicy:
    def __init__(self, directory, empty, action_class):
        self.empty = list(empty)
        self.action_class = action_class
        # Explicit weights-only loading of the publicly downloaded state dict.
        state = torch.load(Path(directory) / "maze_dql.pth", map_location="cpu", weights_only=True)
        self.network = AuthorDQN(len(empty))
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def q(self, state):
        x = torch.zeros(len(self.empty))
        x[self.empty.index(tuple(int(v) for v in state))] = 1
        with torch.no_grad():
            return self.network(x).numpy()

    def predict(self, state):
        return self.action_class(int(np.argmax(self.q(state))))


class FixedActions:
    def __init__(self, actions, action_class):
        self.actions = list(actions)
        self.index = 0
        self.action_class = action_class

    def predict(self, state):
        action = self.actions[self.index]
        self.index += 1
        return self.action_class(action)


def shortest_to_exit(grid, start, goal):
    queue = deque([[tuple(start)]])
    visited = {tuple(start)}
    while queue:
        path = queue.popleft()
        if path[-1] == tuple(goal):
            return path
        for dx, dy in DELTAS:
            x, y = path[-1][0] + dx, path[-1][1] + dy
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]) and not grid[y, x] and (x, y) not in visited:
                visited.add((x, y))
                queue.append(path + [(x, y)])
    return None


def path_from_actions(grid, start, actions):
    path = [tuple(start)]
    for action in actions:
        dx, dy = DELTAS[int(action)]
        x, y = path[-1][0] + dx, path[-1][1] + dy
        if not (0 <= y < len(grid) and 0 <= x < len(grid[0])) or grid[y, x]:
            x, y = path[-1]
        path.append((x, y))
    return path


def independent_author_integral(path, streams, dt=2/.3):
    """Mirror author arrival-cell/floor sampling, with auditable per-step rows."""
    path = [tuple(p) for p in path]
    initial = streams[path[0]][0] * dt
    cumulative = initial
    trace = [{"step": 0, "cell": path[0], "relative_sample_s": 0,
              "rate": streams[path[0]][0], "increment": initial, "FED": cumulative}]
    for k, cell in enumerate(path[1:], 1):
        index = int(dt*k/5)
        rate = streams[cell][min(index, len(streams[cell])-1)] if cell in streams else 0.0
        increment = rate * dt
        cumulative += increment
        trace.append(dict(step=k, cell=cell, relative_sample_s=5*index,
                          rate=rate, increment=increment, FED=cumulative))
    return float(cumulative), trace


def author_rollout(module, inputs, policy, start):
    environment = module.Maze(inputs["grid"].copy(), inputs["streams"], inputs["patterns"])
    status, path, fed = environment.play(policy, tuple(start))
    path = [tuple(int(v) for v in state) for state in path]
    independent_fed, trace = independent_author_integral(path, inputs["streams"], module.Maze.time_duration_passing)
    assert np.isclose(fed, independent_fed, atol=1e-13, rtol=0)
    return dict(status=status.name, path=path, steps=len(path)-1,
                time_s=(len(path)-1)*module.Maze.time_duration_passing,
                FED=float(fed), independent_FED=independent_fed,
                revisits=len(path)-len(set(path)),
                invalid_steps=sum(a == b for a, b in zip(path, path[1:])),
                integration_trace=trace)
