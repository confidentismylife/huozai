"""Execution-only two-process continuation of the frozen world matrix.

Imports unchanged per-world code. No method, seed, generator or threshold change.
Completed world artifacts are reused; interrupted partial attempts are preserved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import run_safestep_robust_gate as original

RUN = ROOT / "outputs/safestep_robust_gate_20261018"
WORKER = None


def initialize_worker(cfg):
    global WORKER
    data = original.load_author_inputs(ROOT / cfg["source_directory"])
    starts = [tuple(c) for c in data["empty"]]
    graph = original.Graph(starts + list(data["exits"]), data["exits"])
    by_cell = [data["mapping"][c] for c in graph.cells]
    regions = sorted({r for c, r in zip(graph.cells, by_cell) if c not in graph.exits}, key=lambda r: int(r[1:]))
    source = np.column_stack([data["rates"][r] for r in regions])
    distances = original.region_distances(graph, by_cell, regions)
    mapping = np.array([regions.index(r) if r in regions else -1 for r in by_cell])
    WORKER = cfg, graph, data["rates"]["Time"], source, distances, regions, mapping


def compute_world(spec):
    cfg, graph, times, source, distances, regions, mapping = WORKER
    directory = RUN / "worlds" / spec["id"]
    rows, n, moves, reached = original.run_world(spec, directory, graph, times,
        source, distances, regions, mapping, cfg)
    return spec["id"], rows, n, moves, reached


def prepare():
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    original.check(frozen["input_sha256"])
    original.save(RUN / "execution_amendment.json", dict(
        date_UTC=datetime.now(timezone.utc).isoformat(),
        reason="Serial runtime projects close to budget; use two conservative CPU process workers",
        prior_inspection="Progress counts and original clean context only; confirmation outcome tables not inspected",
        change="Independent worlds may execute concurrently in two processes; per-world frozen code is unchanged",
        frozen_execution_clause_amended="Process and write one world at a time",
        unchanged=["all forecast and gate equations", "all thresholds", "world and sensor generators",
                   "development and fresh confirmation seeds", "scoring", "scenario matrix", "shared native information"],
        worker_count=2, source_script_sha256=original.sha(Path(__file__)),
        frozen_method_input_sha256=frozen["input_sha256"],
        numerical_reproducibility="Each world RNG is seeded locally. Audit all actions and cached trajectories; original clean baseline agrees with earlier run.",
        timing_scope="Concurrent bundle timings are resource-contended; not an equal-compute-speed comparison across methods",
        total_original_compute_wall_budget_s=frozen["configuration"]["execution"]["max_compute_wall_s"]))
    print(json.dumps(dict(status="EXECUTION_AMENDMENT_FROZEN", workers=2)), flush=True)


def run():
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    amendment = json.loads((RUN / "execution_amendment.json").read_text("utf-8"))
    assert original.sha(Path(__file__)) == amendment["source_script_sha256"]
    original.check(frozen["input_sha256"])
    original.check(frozen["protected_previous_sha256"])
    cfg = frozen["configuration"]
    started = json.loads((RUN / "started.json").read_text("utf-8"))["started_UTC"]
    original_start = datetime.fromisoformat(started).timestamp()
    deadline = original_start + cfg["execution"]["max_compute_wall_s"]
    world_root = (RUN / "worlds").resolve()
    results, pending, preserved = {}, [], []
    for spec in frozen["world_schedule"]:
        directory = world_root / spec["id"]
        if (directory / "completion_manifest.json").exists():
            results[spec["id"]] = original.load_completed(directory)
        else:
            if directory.exists():
                preserved_directory = world_root / f"{spec['id']}_interrupted_{time.time_ns()}"
                # Verify both absolute move targets stay inside the intended world directory.
                assert directory.resolve().is_relative_to(world_root)
                assert preserved_directory.resolve().is_relative_to(world_root)
                assert directory != world_root and not preserved_directory.exists()
                directory.rename(preserved_directory)
                preserved.append(preserved_directory.relative_to(ROOT).as_posix())
            pending.append(spec)
    original.save(RUN / "parallel_started.json", dict(UTC=datetime.now(timezone.utc).isoformat(),
        resumed_completed_worlds=len(results), pending_worlds=len(pending),
        preserved_partial_attempts=preserved, workers=2,
        original_absolute_compute_deadline_UTC=datetime.fromtimestamp(deadline, timezone.utc).isoformat()))
    print(json.dumps(dict(status="TWO_WORKER_CONTINUATION", reused_worlds=len(results),
        remaining_worlds=len(pending))), flush=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(cfg,)) as executor:
        futures = {executor.submit(compute_world, spec): spec for spec in pending}
        for future in as_completed(futures, timeout=max(1., deadline - time.time() - 90)):
            world_id, rows, n, moves, reached = future.result()
            results[world_id] = rows, n, moves, reached
            print(json.dumps(dict(completed_worlds=len(results), total_worlds=len(frozen["world_schedule"]),
                latest_world=world_id, total_episodes=sum(x[1] for x in results.values()),
                total_moves=sum(x[2] for x in results.values()),
                elapsed_since_original_start_s=time.time() - original_start)), flush=True)
    assert len(results) == cfg["execution"]["expected_worlds"]
    rows, episodes, moves, reached = [], 0, 0, 0
    for spec in frozen["world_schedule"]:
        local, n, m, r = results[spec["id"]]
        rows.extend(local)
        episodes += n
        moves += m
        reached += r
    assert episodes == cfg["execution"]["expected_episodes"]
    original.csv_save(RUN / "all_world_departure_summary.csv", rows)
    original.check(frozen["input_sha256"])
    original.check(frozen["protected_previous_sha256"])
    original.save(RUN / "run_results.json", dict(status="COMPLETE_PLANNED_SYNTHETIC_MATRIX",
        worlds=len(results), episodes=episodes, executed_moves=moves, reached=reached,
        elapsed_wall_s=time.time() - original_start,
        execution_amendment="two CPU workers; identical frozen per-world method",
        original_physical_fires=1, independent_physical_fire_claim=False,
        LLM_requests=0, training_updates=0, FDS_runs=0, sealed_final48_accessed=False, unit_tests=19))
    files = [p for p in RUN.iterdir() if p.is_file()]
    files += [world_root / spec["id"] / "completion_manifest.json" for spec in frozen["world_schedule"]]
    original.save(RUN / "completion_manifest.json", dict(status="COMPLETE_COMPUTATION_PENDING_INDEPENDENT_REVIEW",
        completed_UTC=datetime.now(timezone.utc).isoformat(),
        protected_previous_files_verified=len(frozen["protected_previous_sha256"]),
        artifact_sha256={p.relative_to(ROOT).as_posix(): original.sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", worlds=len(results), episodes=episodes)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        try:
            run()
        except Exception:
            (RUN / f"failure_{time.time_ns()}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
    else:
        parser.error("Choose --prepare or --run")


if __name__ == "__main__":
    main()
