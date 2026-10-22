"""Stream independent replay of completed worlds while the frozen batch runs.

No future truth is provided to online policies. This process only validates
cached outcomes and gate decisions, writing a separate audit directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from fire_evac.safestep_public_benchmark import Graph, forward_min_fed_reference

RUN = ROOT / "outputs/safestep_robust_gate_20261018"
OUT = ROOT / "outputs/safestep_robust_gate_audit_20261018"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    assert not p.exists(), p
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def independent_costs(times, region_rates, cell_to_region, departure, dt, horizon):
    """Native-knot trapezoids, independent of RateField prefix integration."""
    answer = np.zeros((horizon, len(cell_to_region)))
    valid = cell_to_region >= 0
    for k in range(horizon):
        a, b = departure + k * dt, departure + (k + 1) * dt
        knots = np.concatenate(([a], times[(times > a) & (times < b)], [b]))
        index = np.minimum(np.searchsorted(times, knots, side="right") - 1, len(times) - 2)
        weight = (knots - times[index]) / (times[index + 1] - times[index])
        y = region_rates[index] * (1 - weight[:, None]) + region_rates[index + 1] * weight[:, None]
        area = np.sum((y[:-1] + y[1:]) * np.diff(knots)[:, None] / 2, axis=0)
        answer[k, valid] = area[cell_to_region[valid]]
    return answer


def snapshot_continuations(graph, actions, costs, destinations):
    values = np.full((len(costs) + 1, len(graph.cells)), np.inf)
    values[:, list(graph.exit_ids)] = 0
    for k in range(len(costs) - 1, -1, -1):
        for i in range(len(graph.cells)):
            if i in graph.exit_ids or actions[k, i] < 0:
                continue
            values[k, i] = costs[k, i] + values[k + 1, destinations[i, actions[k, i]]]
    return values


def audit_world(directory, spec, ordinal, graph, cfg):
    manifest = json.loads((directory / "completion_manifest.json").read_text("utf-8"))
    for rel, digest in manifest["artifact_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    with np.load(directory / "native_inputs.npz") as native:
        times = native["acquisition_times"]
        rates = native["truth_region_FER"]
        cell_to_region = native["cell_to_region"]
    assert np.isfinite(rates).all() and np.all(rates >= 0)
    destinations = np.full((len(graph.cells), 4), -1, dtype=int)
    for i, edges in enumerate(graph.edges):
        for a, dest in edges:
            destinations[i, a] = dest
    cache, real, continuation = {}, {}, {}
    reference_checks = 0
    for departure in cfg["departure_fire_times_s"]:
        with np.load(directory / f"tables_t{departure}.npz") as stored:
            cache[departure] = {key: stored[key] for key in stored.files}
        real[departure] = independent_costs(times, rates, cell_to_region, departure,
                                            cfg["dt_s"], cfg["maximum_actions"])
        assert np.max(np.abs(real[departure] - cache[departure]["true_costs"])) < 1e-10
        continuation[departure] = snapshot_continuations(graph,
            cache[departure]["first_snapshot_hold"], real[departure], destinations)
        timestamps = json.loads((directory / f"timestamps_t{departure}.json").read_text("utf-8"))
        latency = spec["sensor"]["latency_s"]
        for row in timestamps:
            assert row["latest_acquisition_s"] + latency <= row["decision_time"] + 1e-9
            assert row["latest_delivery_s"] == row["latest_acquisition_s"] + latency
        if ordinal % 10 == 0 or ordinal == cfg["execution"]["expected_worlds"] - 1:
            for cell in ((5, 1), (10, 9)):
                state = graph.index[cell]
                value, _ = forward_min_fed_reference(graph, real[departure], state)
                assert abs(value - cache[departure]["oracle_values"][0, state]) < 1e-10
                reference_checks += 1
    episodes, moves, gate_checks, max_error = 0, 0, 0, 0.
    classifications = {f"{decision}_{outcome}": 0 for decision in ("accepted", "rejected")
                       for outcome in ("helpful", "harmful", "neutral", "failure")}
    with gzip.open(directory / "episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            departure, method = record["departure_fire_s"], record["method"]
            table, costs = cache[departure], real[departure]
            path = np.asarray(record["path_ids"], dtype=int)
            actions = np.asarray(record["actions"], dtype=int)
            increments = np.asarray(record["FED_increments"], dtype=float)
            steps = record["steps"]
            assert len(path) == len(actions) + 1 == steps + 1
            assert len(increments) == steps and steps <= cfg["maximum_actions"]
            assert path[0] == record["start_id"]
            assert all(int(i) not in graph.exit_ids for i in path[:-1])
            assert (int(path[-1]) in graph.exit_ids) == record["reached"]
            assert np.array_equal(destinations[path[:-1], actions], path[1:])
            independent = costs[np.arange(steps), path[:-1]]
            assert np.max(np.abs(independent - increments), initial=0) < 1e-10
            error = abs(float(independent.sum()) - record["FED"])
            assert error < 1e-10
            max_error = max(max_error, error)
            assert record["immediate_reversals"] == int(np.sum(path[2:] == path[:-2]))
            assert record["FED"] >= table["oracle_values"][0, path[0]] - 1e-10
            if method in ("snapshot_hold", "trend30", "half30", "ensemble_mean"):
                expected = table[f"first_{method}"][np.arange(steps), path[:-1]]
                assert np.array_equal(actions, expected)
            elif method == "full_future_reference":
                assert np.array_equal(actions, table["oracle_actions"][np.arange(steps), path[:-1]])
            else:
                log = {r["step"]: r for r in record["gate_opportunities"]}
                previous_exit, switches, expected_opportunities = -1, 0, 0
                for k, state in enumerate(path[:-1]):
                    baseline = int(table["first_snapshot_hold"][k, state])
                    nominee = int(table["first_ensemble_mean"][k, state])
                    threshold = .05
                    if method == "switch_margin_gate" and previous_exit >= 0:
                        threshold = .01 if previous_exit == table["nominee_exit"][k, state] else .10
                    if method == "consensus_gate_low":
                        threshold = .02
                    if method == "consensus_gate_high":
                        threshold = .10
                    opportunity = baseline != nominee and baseline >= 0 and nominee >= 0
                    accepted = bool(opportunity and table["min_gain"][k, state] >= threshold)
                    selected_exit = int(table["nominee_exit" if accepted else "base_exit"][k, state])
                    assert actions[k] == (nominee if accepted else baseline)
                    assert record["selected_exit_ids"][k] == selected_exit
                    if selected_exit >= 0 and previous_exit >= 0:
                        switches += selected_exit != previous_exit
                    if selected_exit >= 0:
                        previous_exit = selected_exit
                    if opportunity:
                        expected_opportunities += 1
                        gate = log[k]
                        assert gate["accepted"] == accepted and gate["threshold"] == threshold
                        assert abs(gate["minimum_predicted_relative_gain"] - table["model_gains"][k, :, state].min()) < 1e-10
                        dest = destinations[state, nominee]
                        difference = float(costs[k, state] + continuation[departure][k + 1, dest]
                                           - continuation[departure][k, state])
                        assert abs(difference - gate["offline_one_action_then_snapshot_difference"]) < 1e-10
                        outcome = "helpful" if difference < -1e-10 else "harmful" if difference > 1e-10 else "neutral"
                        assert outcome == gate["offline_classification"]
                        classifications[("accepted" if accepted else "rejected") + "_" + outcome] += 1
                        gate_checks += 1
                assert len(log) == expected_opportunities
                assert switches == record["selected_plan_exit_changes"]
            episodes += 1
            moves += steps
    assert episodes == manifest["episodes"] and moves == manifest["moves"]
    return dict(world_id=spec["id"], episodes=episodes, moves=moves, gate_decisions_checked=gate_checks,
        maximum_FED_replay_error=max_error, independently_recomputed_reference_checks=reference_checks,
        gate_classifications=classifications, original_world_manifest_sha256=sha(directory / "completion_manifest.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    cfg = frozen["configuration"]
    if not OUT.exists():
        OUT.mkdir()
        save(OUT / "audit_plan.json", dict(frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
            status="INDEPENDENT_REPLAY_PROTOCOL_FROZEN_DURING_ONGOING_BATCH",
            first_inspection="Progress counts and original clean context summaries seen; no synthetic threshold changes",
            run_plan_sha256=sha(RUN / "frozen_plan.json"), script_sha256=sha(Path(__file__)),
            checks="Every cached trajectory and gated action; independent knot-split integration; every tenth scheduled world plus last gets forward-reference checks",
            new_policy_episodes=0, sealed_final48_access=False))
    graph_data = json.loads((RUN / "graph.json").read_text("utf-8"))
    graph = Graph(graph_data["cells"], graph_data["exits"])
    results = []
    start = time.perf_counter()
    for ordinal, spec in enumerate(frozen["world_schedule"]):
        audit_path = OUT / f"{spec['id']}.json"
        if audit_path.exists():
            result = json.loads(audit_path.read_text("utf-8"))
        else:
            directory = RUN / "worlds" / spec["id"]
            while not (directory / "completion_manifest.json").exists():
                # Wait only for actual unfinished computation, not for padding a task.
                if not args.watch:
                    print(json.dumps(dict(status="WAITING_FOR_COMPUTATION", next_world=spec["id"])), flush=True)
                    return
                if list(RUN.glob("failure_*.txt")) or time.perf_counter() - start > 7200:
                    raise RuntimeError("Main batch failed or exceeded watch budget; completed audits retained")
                time.sleep(10)
            result = audit_world(directory, spec, ordinal, graph, cfg)
            save(audit_path, result)
        results.append(result)
        if (ordinal + 1) % 20 == 0 or ordinal == 0:
            print(json.dumps(dict(audited_worlds=len(results), total_worlds=len(frozen["world_schedule"]),
                audited_episodes=sum(r["episodes"] for r in results))), flush=True)
    assert sum(r["episodes"] for r in results) == cfg["execution"]["expected_episodes"]
    save(OUT / "audit_results.json", dict(status="PASS_ALL_CACHED_TRAJECTORIES_AND_GATE_DECISIONS",
        worlds=len(results), episodes=sum(r["episodes"] for r in results),
        moves=sum(r["moves"] for r in results),
        gate_decisions_checked=sum(r["gate_decisions_checked"] for r in results),
        maximum_FED_replay_error=max(r["maximum_FED_replay_error"] for r in results),
        independent_forward_reference_checks=sum(r["independently_recomputed_reference_checks"] for r in results),
        elapsed_watch_and_audit_wall_s=time.perf_counter() - start,
        new_policy_episodes=0, physical_fire_count=1, sealed_final48_accessed=False))
    files = [p for p in OUT.iterdir() if p.is_file()]
    save(OUT / "completion_manifest.json", dict(status="COMPLETE_INDEPENDENT_REPLAY",
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", worlds=len(results))), flush=True)


if __name__ == "__main__":
    main()
