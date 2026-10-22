"""Independent scalar inference, grid walk and dose audit of cached checkpoints.

Does not call the evaluator, author Maze, author_rollout, cycle helper, or
training/optimizer functions. No additional training is performed.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from fire_evac.safestep_author_numerics import load_author_inputs


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    cfg = read(ROOT / "schemas/safestep_checkpoint_diagnosis_20261016.json")
    out = ROOT / cfg["output_directory"]
    frozen = read(out / "frozen_manifest.json")
    for name, expected in frozen["input_sha256"].items():
        assert digest(ROOT / name) == expected, name
    data = load_author_inputs(ROOT / cfg["source_directory"])
    grid = data["grid"]
    exits = set(data["exits"])
    coords = [(x, y) for x in range(grid.shape[1]) for y in range(grid.shape[0])
              if grid[y, x] == 0 and (x, y) not in exits]
    assert len(coords) == 61 and coords == data["empty"]
    index = {xy: i for i, xy in enumerate(coords)}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    identity = torch.eye(61)
    details, all_rollouts = [], 0
    numeric = read((ROOT / cfg["source_directory"]).parent / "numeric_run/results.json")
    reference = numeric["stored_weight_runs"]
    for arm in cfg["arms"]:
        base = cfg["baseline_directory"] if arm == "full_anchor" else cfg["loss_directory"]
        source_arm = "every_step" if arm == "full_anchor" else arm
        for seed in cfg["seeds"]:
            directory = ROOT / base / "runs" / f"{source_arm}_seed{seed}"
            episodes = rows(directory / "episodes.csv")
            for step in cfg["checkpoints_at_steps"]:
                assert time.monotonic() - started < cfg["hard_timeout_seconds"]
                weight_path = directory / f"checkpoint_step{step:06d}.pth"
                if not weight_path.exists():
                    continue
                saved = read(out / "checkpoints" / f"{arm}_seed{seed}_step{step:06d}.json")
                assert digest(weight_path) == saved["checkpoint_sha256"]
                w = torch.load(weight_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    q = torch.stack([F.linear(torch.relu(F.linear(identity[i], w["fc1.weight"], w["fc1.bias"])),
                         w["out.weight"], w["out.bias"]) for i in range(61)]).numpy()
                assert np.isfinite(q).all()
                assert np.array_equal(q, np.asarray(saved["scalar_float32_Q"], dtype=np.float32))
                actions = q.argmax(1).tolist()
                assert actions == saved["greedy_actions"]
                categories = {"exit_arrival": 0, "blocked_self_loop": 0, "multi_cell_cycle": 0}
                invalid = revisits = 0
                assert len(saved["traces"]) == 61
                for start, trace in zip(coords, saved["traces"]):
                    assert trace["start"] == list(start)
                    xy, path = start, [start]
                    dt = 2 / .3
                    fed = data["streams"][start][0] * dt
                    for k in range(1, 302):
                        dx, dy = moves[actions[index[xy]]]
                        candidate = (xy[0] + dx, xy[1] + dy)
                        if (0 <= candidate[0] < grid.shape[1] and 0 <= candidate[1] < grid.shape[0]
                                and grid[candidate[1], candidate[0]] == 0):
                            xy = candidate
                        path.append(xy)
                        if xy not in exits:
                            stream = data["streams"][xy]
                            fed += stream[min(int(dt * k / 5), len(stream)-1)] * dt
                        if xy in exits:
                            break
                    status = "WIN" if xy in exits else "LOSE"
                    assert trace["path"] == list(map(list, path))
                    assert trace["status"] == status and trace["steps"] == len(path)-1
                    assert abs(trace["FED"] - fed) < 1e-13
                    assert abs(trace["independent_FED"] - fed) < 1e-13
                    assert trace["time_s"] == (len(path)-1)*dt
                    bad = sum(p == n for p, n in zip(path, path[1:]))
                    visited = len(path)-len(set(path))
                    assert bad == trace["invalid_steps"] and visited == trace["revisits"]
                    invalid += bad
                    revisits += visited
                    entry, period = None, 0
                    if status == "WIN":
                        category = "exit_arrival"
                        assert visited == bad == 0
                    else:
                        for j in range(1, len(path)):
                            if path[j] in path[:j]:
                                entry = path[:j].index(path[j])
                                period = j-entry
                                break
                        assert entry is not None and period > 0
                        segment = path[entry:entry+period]
                        assert all(p == segment[j % period] for j, p in enumerate(path[entry:]))
                        category = "blocked_self_loop" if period == 1 else "multi_cell_cycle"
                    assert (trace["category"], trace["cycle_entry_step"], trace["cycle_period"]) == (category, entry, period)
                    categories[category] += 1
                    all_rollouts += 1
                summary = saved["summary"]
                assert summary["arrivals"] == categories["exit_arrival"]
                assert summary["blocked_starts"] == categories["blocked_self_loop"]
                assert summary["cycle_starts"] == categories["multi_cell_cycle"]
                assert sum(categories.values()) == 61
                assert (summary["invalid_steps"], summary["revisits"]) == (invalid, revisits)
                completed = [r for r in episodes if int(r["global_steps"]) <= step and r["episode_status"] in ("WIN", "LOSE")]
                epsilon = 1.0
                for r in completed:
                    if int(r["global_steps"]) > 64:
                        epsilon *= .9995
                assert summary["completed_episodes"] == len(completed)
                assert abs(summary["epsilon_at_checkpoint"]-epsilon) < 1e-14
                assert summary["last_completed_episode_global_step"] == (int(completed[-1]["global_steps"]) if completed else 0)
                for i, expected in enumerate(reference, 1):
                    t = saved["traces"][coords.index(tuple(expected["path"][0]))]
                    match = t["path"] == expected["path"]
                    assert summary[f"start{i}_paper_path_match"] == match
                    assert summary[f"start{i}_status"] == t["status"]
                    assert summary[f"start{i}_FED"] == t["FED"]
                    assert summary[f"start{i}_steps"] == t["steps"]
                    if match:
                        assert round(t["FED"], 4) == expected["paper_FED"]
                if step == 200000:
                    final_w = torch.load(directory / "budget_final_model.pth", map_location="cpu", weights_only=True)
                    assert all(torch.equal(w[k], final_w[k]) for k in w)
                    old = read(directory / "evaluation_traces.json")
                    assert len(old) == len(saved["traces"]) == 61
                    for old_trace, new_trace in zip(old, saved["traces"]):
                        for key in ("path", "start", "status", "steps", "FED", "invalid_steps", "revisits", "time_s"):
                            assert old_trace[key] == new_trace[key], key
                    final_train = read(directory / "budget_training_complete.json")
                    assert final_train["completed_episodes"] == summary["completed_episodes"]
                    assert final_train["epsilon"] == summary["epsilon_at_checkpoint"]
                details.append(dict(arm=arm, seed=seed, checkpoint_step=step, verified_starts=61,
                                    categories=categories, endpoint_matches_previous=step == 200000))
                print(json.dumps(details[-1]), flush=True)
    assert len(details) == cfg["planned_checkpoints"] and all_rollouts == cfg["planned_rollouts"]
    csv_metrics = rows(out / "checkpoint_metrics.csv")
    csv_starts = rows(out / "start_outcomes.csv")
    assert len(csv_metrics) == 27 and len(csv_starts) == all_rollouts
    for r in csv_metrics:
        s = read(out / "checkpoints" / f"{r['arm']}_seed{r['seed']}_step{int(r['checkpoint_step']):06d}.json")["summary"]
        assert all(str(s[k]) == v for k, v in r.items())
    for name, expected in frozen["input_sha256"].items():
        assert digest(ROOT / name) == expected, name
    result = dict(status="PASS", created_at=datetime.now(timezone.utc).isoformat(),
        verified_checkpoints=len(details), verified_rollouts=all_rollouts, exact_final_endpoint_matches=9,
        unchanged_frozen_inputs=len(frozen["input_sha256"]), checks=details,
        verifier_sha256=digest(Path(__file__)), training_updates=0,
        old_final48_crowd_scenarios_accessed=False, wall_seconds=time.monotonic()-started)
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
