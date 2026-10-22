"""Post-hoc read-only diagnosis of already selected interrupted checkpoints.

No training, action repair, checkpoint search, or final evaluation is performed.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.safestep_author_numerics import AuthorDQN, DELTAS, load_author_inputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_stats(rows):
    wins = [r for r in rows if r["episode_status"] == "WIN"]
    failed = [r for r in rows if r["episode_status"] == "LOSE"]
    steps = sum(int(r["steps"]) for r in rows)
    fail_steps = sum(int(r["steps"]) for r in failed)
    return dict(episodes=len(rows), wins=len(wins), timeouts=len(failed),
                timeout_episode_share=len(failed) / len(rows), control_steps=steps,
                timeout_control_steps=fail_steps, timeout_control_step_share=fail_steps / steps)


def classify_trace(trace):
    path = list(map(tuple, trace["path"]))
    seen = {}
    first_repeat = None
    for k, cell in enumerate(path):
        if cell in seen:
            first_repeat = (seen[cell], k)
            break
        seen[cell] = k
    if trace["status"] == "WIN":
        assert first_repeat is None
        return dict(start=trace["start"], category="exit_arrival", steps=trace["steps"])
    assert trace["status"] == "LOSE" and trace["steps"] == 301 and first_repeat is not None
    a, b = first_repeat
    cycle = path[a:b]
    assert all(path[k] == cycle[(k - a) % len(cycle)] for k in range(a, len(path)))
    return dict(start=trace["start"], category="blocked_self_loop" if len(cycle) == 1 else "multi_cell_cycle",
                first_repeated_cell_step=b, cycle_entry_step=a, cycle_length=len(cycle), cycle=cycle,
                steps=trace["steps"], FED=trace["FED"])


def hidden_context(records, i, coordinates, inputs):
    r = records[i]
    ep = int(r["episode"])
    start = i - int(r["step"]) + 1
    assert start >= 0 and int(records[start]["episode"]) == ep
    prior = records[start:i]
    visited = set(prior["next_state"][prior["state"] != prior["next_state"]].tolist())
    s, a, nxt = int(r["state"]), int(r["action"]), int(r["next_state"])
    before_fed = float(records[i - 1]["FED"]) if len(prior) else inputs["streams"][coordinates[s]][0] * (2 / .3)
    return dict(transition_ID=i, episode=ep, step=int(r["step"]),
                relative_elapsed_s=int(r["step"]) * (2 / .3), cell=coordinates[s], action=a,
                next_cell=coordinates[nxt], reward=float(r["reward"]),
                FED_before_action=before_fed, FED_after_action=float(r["FED"]),
                destination_previously_visited=nxt in visited,
                invalid_action=s == nxt, terminal=int(r["status"]) != 2)


def main():
    cfg_path = ROOT / "schemas/safestep_stepwise_cached_diagnosis_20261015.json"
    cfg = read(cfg_path)
    audit_path = ROOT / cfg["input_audit"]
    audit = read(audit_path)
    assert audit["status"] == "INTERRUPTED_TRAINING_PREFIX_AUDIT_PASS"
    out = ROOT / cfg["output_directory"]
    out.mkdir(exist_ok=False)
    inputs = load_author_inputs(ROOT / "outputs/safestep_paper_numerics_20261015/author_repository")
    coordinates = inputs["empty"] + list(inputs["exits"])
    index = {xy: i for i, xy in enumerate(coordinates)}
    torch.set_num_threads(1)
    reports = []
    source_hashes = {str(audit_path.relative_to(ROOT)): digest(audit_path),
                     str(cfg_path.relative_to(ROOT)): digest(cfg_path)}
    for prior in audit["reports"]:
        seed, selection = prior["seed"], prior["selection"]
        checkpoint = ROOT / selection["checkpoint"]
        assert digest(checkpoint) == selection["checkpoint_sha256"]
        source_hashes[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
        trace_path = audit_path.parent / f"seed{seed}" / "checkpoint_rollout_traces.json"
        source_hashes[str(trace_path.relative_to(ROOT))] = digest(trace_path)
        traces = read(trace_path)
        network = AuthorDQN(61)
        network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        network.eval()
        with torch.no_grad():
            q = network(torch.eye(61)).numpy()
        actions = q.argmax(axis=1)
        neighbors = []
        for s, (x, y) in enumerate(inputs["empty"]):
            row = []
            for dx, dy in DELTAS:
                row.append(index.get((x + dx, y + dy), s))
            neighbors.append(row)
        invalid_cells = [coordinates[s] for s, a in enumerate(actions) if neighbors[s][a] == s]
        for t in traces:
            assert len(t["path"]) == t["steps"] + 1
            for a, b in zip(t["path"], t["path"][1:]):
                state = index[tuple(a)]
                assert coordinates[neighbors[state][actions[state]]] == tuple(b)
        classifications = [classify_trace(t) for t in traces]
        assert Counter(r["category"] for r in classifications)["exit_arrival"] == prior["checkpoint_success"]
        directory = ROOT / selection["source_directory"]
        arrays, n = [], selection["prefix_transitions"]
        for path in sorted((directory / "transitions").glob("chunk_*.npz")):
            with np.load(path) as data:
                arrays.append(data["records"])
            if sum(len(a) for a in arrays) >= n:
                break
        records = np.concatenate(arrays)[:n]
        assert len(records) == n and int(records[-1]["episode"]) == selection["checkpoint_episode"]
        groups = records["state"].astype(int) * 4 + records["action"]
        reward_groups = []
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            lo, hi = int(ids[np.argmin(records["reward"][ids])]), int(ids[np.argmax(records["reward"][ids])])
            spread = float(records["reward"][hi] - records["reward"][lo])
            assert np.all(records["next_state"][ids] == records["next_state"][ids[0]])
            reward_groups.append(dict(state=int(group // 4), action=int(group % 4), observations=len(ids),
                                      reward_range=spread, min_transition=lo, max_transition=hi))
        reward_groups.sort(key=lambda r: (-r["reward_range"], r["state"], r["action"]))
        examples = []
        for group in reward_groups[:5]:
            examples.append(group | dict(min_context=hidden_context(records, group["min_transition"], coordinates, inputs),
                                         max_context=hidden_context(records, group["max_transition"], coordinates, inputs)))
        with (directory / "episodes.csv").open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("global_steps") and r.get("episode_status")]
        prefix_rows = rows[:selection["checkpoint_episode"]]
        prefix_stats = episode_stats(prefix_rows)
        assert prefix_stats["control_steps"] == n
        report = dict(seed=seed, intermediate_checkpoint_episode=selection["checkpoint_episode"],
                      training_complete=False, diagnostic_only=True,
                      trajectory_categories=dict(Counter(r["category"] for r in classifications)),
                      failed_cycle_lengths=dict(Counter(r["cycle_length"] for r in classifications if "cycle_length" in r)),
                      invalid_greedy_action_cells=len(invalid_cells), invalid_cells=invalid_cells,
                      greedy_action_counts={str(a): int(np.sum(actions == a)) for a in range(4)},
                      q_min=float(q.min()), q_max=float(q.max()),
                      observed_state_action_groups=len(reward_groups),
                      groups_with_reward_range_over_1e_minus8=sum(g["reward_range"] > 1e-8 for g in reward_groups),
                      reward_variation_examples=examples,
                      verified_prefix_training=prefix_stats,
                      last1000_recorded_training=episode_stats(rows[-1000:]),
                      trace_classifications=classifications)
        (out / f"seed{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("trace_classifications", "reward_variation_examples", "invalid_cells")}), flush=True)
    result = dict(status="READ_ONLY_CACHED_DIAGNOSIS_COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
                  exploratory_after_checkpoint_scores=True, reports=reports,
                  training_updates=0, LLM_requests=0, old_final48_crowd_scenarios_accessed=False,
                  source_sha256=source_hashes, script_sha256=digest(__file__))
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for path, expected in source_hashes.items():
        assert digest(ROOT / path) == expected


if __name__ == "__main__":
    main()
