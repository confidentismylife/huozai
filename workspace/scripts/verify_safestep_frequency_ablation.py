"""Independent geometry, dose, schedules, replay and Adam audits for both arms."""
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.safestep_author_numerics import AuthorDQN, load_author_inputs
from fire_evac.safestep_author_training import tensor_fingerprint


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(directory, cfg, inputs):
    train = read(directory / "budget_training_complete.json")
    coordinates = list(map(tuple, train["coordinate_order"]))
    assert coordinates == inputs["empty"] + list(inputs["exits"])
    index = {xy: i for i, xy in enumerate(coordinates)}
    neighbors = np.zeros((63, 4), dtype=int)
    for i, (x, y) in enumerate(coordinates):
        for a, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            neighbors[i, a] = index.get((x + dx, y + dy), i)
    nt = len(next(iter(inputs["streams"].values())))
    fer = np.column_stack([inputs["streams"].get(xy, [0.] * nt) for xy in coordinates])
    arrays = []
    for path in sorted((directory / "transitions").glob("*.npz")):
        with np.load(path) as z:
            arrays.append(z["records"])
    records = np.concatenate(arrays)
    n = len(records)
    assert n == train["control_steps"]
    assert np.array_equal(neighbors[records["state"], records["action"]], records["next_state"])
    dt = 2 / .3
    doses = fer[np.minimum((dt * records["step"] / 5).astype(int), nt - 1), records["next_state"]] * dt
    boundaries = np.r_[0, np.flatnonzero(np.diff(records["episode"])) + 1, n]
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == len(boundaries) - 1 == train["episodes_started"]
    all_steps = np.arange(1, n + 1)
    completed_mask = records["status"] != 2
    update_steps = all_steps[(all_steps > 64) & ((train["arm"] == "every_step") | completed_mask)]
    syncs = [0]
    decay_count = 0
    reset_rng = np.random.default_rng(100000 + train["seed"])
    for ep, (a, b), row in zip(range(1, len(rows) + 1), zip(boundaries[:-1], boundaries[1:]), rows):
        block = records[a:b]
        assert np.all(block["episode"] == ep)
        assert np.array_equal(block["step"], np.arange(1, len(block) + 1))
        assert np.array_equal(block["state"][1:], block["next_state"][:-1])
        reset_seed = int(reset_rng.integers(0, 2**32, dtype=np.uint64))
        assert int(row["reset_seed"]) == reset_seed
        start = inputs["empty"][random.Random(reset_seed).randint(0, 60)]
        assert coordinates[block["state"][0]] == start == (int(row["start_x"]), int(row["start_y"]))
        initial = fer[0, block["state"][0]] * dt
        cumulative = np.cumsum(np.r_[initial, doses[a:b]])[1:]
        assert np.allclose(cumulative, block["FED"], rtol=1e-12, atol=5e-11)
        previous = np.r_[initial, block["FED"][:-1]]
        hazard = np.select([previous == 0, (previous > 0) & (previous < .3),
                            (previous > .3) & (previous < 1), previous > 1], [0., -100., -300., -800.], default=np.nan)
        seen, walking = set(), []
        for old, new in zip(block["state"], block["next_state"]):
            if old == new:
                walking.append(-1.)
            else:
                walking.append(cfg["reward_exit"] if new in (61, 62) else -.5 if new in seen else -.25)
                seen.add(int(new))
        expected = .1 * np.asarray(walking) + .2 * hazard * doses[a:b]
        assert np.allclose(expected, block["reward"], rtol=1e-12, atol=1e-10)
        status = np.where(np.isin(block["next_state"], (61, 62)), 0, np.where(block["step"] > 300, 1, 2))
        assert np.array_equal(status, block["status"])
        assert np.all(block["status"][:-1] == 2)
        ended = int(block["status"][-1]) != 2
        assert ended or (ep == len(rows) and row["episode_status"] == "BUDGET_CUT")
        assert row["episode_status"] == ({0: "WIN", 1: "LOSE", 2: "BUDGET_CUT"}[int(block["status"][-1])])
        assert row["environment_status"] == {0: "WIN", 1: "LOSE", 2: "PLAYING"}[int(block["status"][-1])]
        assert int(row["steps"]) == len(block) and int(row["global_steps"]) == b
        assert abs(float(row["FED"]) - block["FED"][-1]) < 1e-13
        assert abs(float(row["reward"]) - sum(block["reward"])) < 1e-8
        assert int(row["updates"]) == int(np.searchsorted(update_steps, b, side="right"))
        assert np.isclose(float(row["epsilon_before"]), cfg["epsilon_decay_per_completed_episode"] ** decay_count,
                          rtol=1e-12, atol=1e-15)
        if ended and b > 64:
            decay_count += 1
            if b - syncs[-1] > 50:
                syncs.append(int(b))
        assert np.isclose(float(row["epsilon_after"]), cfg["epsilon_decay_per_completed_episode"] ** decay_count,
                          rtol=1e-12, atol=1e-15)
    assert read(directory / "target_syncs.json") == syncs
    assert sum(r["episode_status"] != "BUDGET_CUT" for r in rows) == train["completed_episodes"]
    assert train["last_episode_budget_cut"] == (int(records[-1]["status"]) == 2)
    updates = wins_checked = timeout_labels = timeout_nonzero_bootstrap = 0
    replay_rng = random.Random(300000 + train["seed"])
    for path in sorted((directory / "optimizer").glob("*.npz")):
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        ids = data["ids"]
        count = len(ids)
        steps = update_steps[updates:updates + count]
        assert np.array_equal(data["global_step"], steps)
        assert np.array_equal(data["episode"], records[steps - 1]["episode"])
        assert np.all(ids < steps[:, None]) and np.all(ids >= np.maximum(0, steps - 1000)[:, None])
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        expected_ids = np.asarray([replay_rng.sample(range(max(0, int(s) - 1000), int(s)), 64) for s in steps], dtype=np.uint32)
        assert np.array_equal(ids, expected_ids)
        sync_indices = np.searchsorted(syncs, steps, side="left") - 1
        assert np.array_equal(data["target_last_sync_step"], np.asarray(syncs)[sync_indices])
        assert all(np.isfinite(data[key]).all() for key in ("q", "targets", "loss", "anchor_sse"))
        sampled = records[ids]
        rewards = sampled["reward"].astype(np.float32)
        wins = sampled["status"] == 0
        timeout = sampled["status"] == 1
        assert np.array_equal(data["targets"][wins], rewards[wins])
        wins_checked += int(wins.sum())
        timeout_labels += int(timeout.sum())
        timeout_nonzero_bootstrap += int(np.count_nonzero(data["targets"][timeout] != rewards[timeout]))
        derived_loss = (np.sum((data["q"] - data["targets"]) ** 2, axis=1, dtype=np.float32) + data["anchor_sse"]) / 256
        assert np.allclose(derived_loss, data["loss"], rtol=5e-5, atol=1e-6)
        updates += count
    assert updates == len(update_steps) == train["optimizer_updates"]
    contexts = []
    identity = torch.eye(61)
    for path in sorted((directory / "update_contexts").glob("*.pt")):
        c = torch.load(path, weights_only=True, map_location="cpu")
        batch = records[np.asarray(c["ids"], dtype=int)]
        policy, target = AuthorDQN(61), AuthorDQN(61)
        policy.load_state_dict(c["policy_before"])
        target.load_state_dict(c["target"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
        optimizer.load_state_dict(c["optimizer_before"])
        states, actions = torch.tensor(batch["state"].astype(int)), torch.tensor(batch["action"].astype(int))
        q_full = policy(identity[states])
        with torch.no_grad():
            labels = torch.from_numpy(batch["reward"].astype(np.float32))
            active = torch.tensor(batch["status"] != 0)
            ns = torch.tensor(batch["next_state"].astype(int))
            if active.any():
                labels[active] += cfg["gamma"] * target(identity[ns[active]]).max(1).values
            anchored = target(identity[states]).clone()
            anchored[torch.arange(64), actions] = labels
        q = q_full[torch.arange(64), actions]
        assert torch.equal(q.detach(), c["q"]) and torch.equal(labels, c["labels"])
        loss = ((q_full - anchored) ** 2).mean()
        assert loss.item() == c["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert tensor_fingerprint(policy.state_dict()) == tensor_fingerprint(c["policy_after"])
        contexts.append(c["update"])
    assert contexts == [k for k in cfg["audit_update_indices"] if k <= updates] == train["audited_update_indices"]
    model = AuthorDQN(61)
    model.load_state_dict(torch.load(directory / "budget_final_model.pth", weights_only=True, map_location="cpu"))
    assert tensor_fingerprint(model.state_dict()) == train["final_policy_fingerprint"]
    with torch.no_grad():
        actions = [int(model(identity[s]).argmax()) for s in range(61)]
    traces = read(directory / "evaluation_traces.json")
    summary = read(directory / "summary.json")
    categories = dict(exit_arrival=0, blocked_self_loop=0, multi_cell_cycle=0)
    invalid = revisits = 0
    for start, trace in enumerate(traces):
        s, path, fed = start, [start], fer[0, start] * dt
        for k in range(1, 302):
            s = int(neighbors[s, actions[s]])
            path.append(s)
            fed += fer[min(int(k * dt / 5), nt - 1), s] * dt
            if s in (61, 62):
                break
        assert trace["start"] == list(coordinates[start])
        assert trace["path"] == [list(coordinates[s]) for s in path]
        assert abs(trace["FED"] - fed) < 1e-13
        assert trace["status"] == ("WIN" if s in (61, 62) else "LOSE")
        invalid += sum(a == b for a, b in zip(path, path[1:]))
        revisits += len(path) - len(set(path))
        if s in (61, 62):
            categories["exit_arrival"] += 1
        else:
            seen = {}
            for i, s in enumerate(path):
                if s in seen:
                    first, period = seen[s], i - seen[s]
                    assert all(path[j] == path[first + (j - first) % period] for j in range(first, len(path)))
                    categories["blocked_self_loop" if period == 1 else "multi_cell_cycle"] += 1
                    break
                seen[s] = i
    assert len(traces) == 61 and summary["success"] == categories["exit_arrival"]
    assert summary["invalid_steps"] == invalid and summary["revisits"] == revisits
    reference = read(ROOT / "outputs/safestep_paper_numerics_20261015/numeric_run/results.json")["stored_weight_runs"]
    matches = []
    for expected, actual in zip(reference, summary["paper_starts"]):
        t = next(t for t in traces if t["start"] == expected["path"][0])
        match = t["path"] == expected["path"]
        assert actual["path_match"] == match and actual["steps"] == t["steps"] and actual["FED"] == t["FED"]
        if match:
            assert round(t["FED"], 4) == expected["paper_FED"]
        matches.append(match)
    assert summary["both_paper_routes_match"] == all(matches)
    assert (directory / "budget_final_model.pth").stat().st_mtime_ns <= (directory / "budget_training_complete.json").stat().st_mtime_ns
    assert (directory / "budget_training_complete.json").stat().st_mtime_ns <= (directory / "evaluation_traces.json").stat().st_mtime_ns
    return dict(run=directory.name, control_steps=n, completed_episodes=train["completed_episodes"],
        optimizer_updates=updates, replay_IDs_checked=updates * 64, exact_replay_RNG_verified=True,
        WIN_labels_checked=wins_checked, LOSE_labels_checked=timeout_labels,
        LOSE_labels_with_nonzero_bootstrap=timeout_nonzero_bootstrap, exactly_replayed_Adam_contexts=len(contexts),
        trajectory_categories=categories, final_rollouts=61, both_paper_routes_match=all(matches))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = read(ROOT / "schemas/safestep_frequency_ablation_20261015.json")
    out = ROOT / cfg["output_directory"]
    inputs = load_author_inputs(ROOT / cfg["source_directory"])
    if args.preflight:
        rows = [audit_run(out / f"preflight_{arm}_0", cfg, inputs) for arm in cfg["arms"]]
        result = dict(status="PASS", runs=rows, verifier_sha256=digest(Path(__file__)))
        (out / "preflight_independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    frozen = read(out / "frozen_manifest.json")
    for path, expected in frozen["files"].items():
        assert digest(ROOT / path) == expected, path
    status = read(out / "campaign_status.json")
    assert len(status) == 6 and all(v["status"] == "complete" for v in status.values())
    rows = []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            directory = out / "runs" / f"{arm}_seed{seed}"
            result = audit_run(directory, cfg, inputs)
            assert result["control_steps"] == cfg["control_step_budget"]
            rows.append(result)
            print(json.dumps(result), flush=True)
    for seed in cfg["seeds"]:
        pair = [read(out / "runs" / f"{arm}_seed{seed}" / "budget_training_complete.json") for arm in cfg["arms"]]
        assert pair[0]["initial_policy_fingerprint"] == pair[1]["initial_policy_fingerprint"]
        arrays = []
        for arm in cfg["arms"]:
            with np.load(out / "runs" / f"{arm}_seed{seed}/transitions/chunk_00000.npz") as z:
                arrays.append(z["records"][:65])
        assert np.array_equal(*arrays)
    old = read(ROOT / "outputs/safestep_paper_stepwise_timeout_audit_20261015/completion_manifest.json")
    for path, expected in old["artifact_sha256"].items():
        assert digest(ROOT / path) == expected, path
    result = dict(status="PASS", created_at=datetime.now(timezone.utc).isoformat(), runs=rows,
        total_control_steps=sum(r["control_steps"] for r in rows), total_updates=sum(r["optimizer_updates"] for r in rows),
        total_replay_IDs_checked=sum(r["replay_IDs_checked"] for r in rows),
        exactly_replayed_Adam_contexts=sum(r["exactly_replayed_Adam_contexts"] for r in rows),
        independent_final_rollouts=366, paired_initial_weights_and_first65_transitions=True,
        earlier_artifact_hashes_unchanged=len(old["artifact_sha256"]),
        old_final48_crowd_scenarios_accessed=False, verifier_sha256=digest(Path(__file__)))
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
