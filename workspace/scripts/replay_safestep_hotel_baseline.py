"""Replay the archived author hotel baseline into a new directory.

This is inference and dose verification only. An explicit, previously absent
output directory is required; historical outputs and author files are read-only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from fire_evac.safestep_author_numerics import (
    FixedActions, StoredAuthorPolicy, author_rollout, import_reviewed_source, load_author_inputs,
)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New output directory, relative to the project or absolute")
    args = parser.parse_args()
    cfg_path = ROOT / "schemas/safestep_author_numerics_20261015.json"
    cfg = json.loads(cfg_path.read_text("utf-8"))
    source = ROOT / cfg["source_directory"]
    previous = ROOT / cfg["output_directory"] / "results.json"
    recorded = json.loads(previous.read_text("utf-8"))
    out = Path(args.output)
    out = out if out.is_absolute() else ROOT / out
    out = out.resolve()
    if out.exists():
        parser.error("Output directory already exists; choose a new directory. Nothing was overwritten.")
    frozen_files = [cfg_path, previous, Path(__file__), ROOT / "src/fire_evac/safestep_author_numerics.py"]
    frozen_files += [source / n for n in ("main.py", "maze.py", "coordinates2areanumber.py", "maze_dql.pth",
                   "FED rate - room24.xlsx", "fire hazard zones FED - room24.xlsx")]
    historical = json.loads((ROOT / cfg["output_directory"] / "frozen_manifest.json").read_text("utf-8"))["files"]
    expected = {Path(k).as_posix(): v for k, v in historical.items()}
    for p in frozen_files:
        key = p.relative_to(ROOT).as_posix()
        if key in expected:
            assert digest(p) == expected[key], f"Archived input mismatch: {key}"
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in frozen_files}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "input_manifest.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
        author_commit=cfg["commit"], expected_scores_already_known=True, input_sha256=before))
    torch.set_num_threads(1)
    data = load_author_inputs(source, cfg["start_time_s"])
    maze = import_reviewed_source(source / "maze.py", "replay_archived_author_hotel")
    model = StoredAuthorPolicy(source, data["empty"], maze.Action)
    route_rows, fixed_traces = [], []
    for target in cfg["paper_targets"]:
        trace = author_rollout(maze, data, FixedActions(target["actions"], maze.Action), target["start"])
        old = next(t for t in recorded["fixed_routes"] if t["id"] == target["id"])
        assert trace["path"] == old["path"] or list(map(list, trace["path"])) == old["path"]
        assert trace["FED"] == old["FED"] and trace["steps"] == old["steps"]
        assert trace["status"] == "WIN"
        assert round(trace["FED"], 4) == target["FED"]
        assert trace["steps"] == target["steps"]
        assert abs(trace["time_s"] - target["time_s"]) <= .051
        route_rows.append(dict(id=target["id"], steps=trace["steps"], time_s=trace["time_s"],
            paper_FED=target["FED"], reproduced_FED=trace["FED"],
            absolute_difference_from_rounded_paper=abs(trace["FED"]-target["FED"]),
            paper_four_decimal_match=True, archived_numeric_result_exact_match=True))
        fixed_traces.append(dict(id=target["id"], **trace))
    starts = []
    for xy in data["empty"]:
        trace = author_rollout(maze, data, model, xy)
        trace.pop("integration_trace")
        trace["start"] = xy
        old = next(t for t in recorded["all_start_diagnostic"] if t["start"] == list(xy))
        for key in ("status", "steps", "time_s", "FED", "invalid_steps", "revisits"):
            assert old[key] == trace[key], (xy, key)
        assert old["path"] == list(map(list, trace["path"]))
        starts.append(trace)
    matches = []
    for expected_route in recorded["stored_weight_runs"]:
        trace = next(t for t in starts if list(t["start"]) == expected_route["path"][0])
        matches.append(list(map(list, trace["path"])) == expected_route["path"])
    for key, expected_hash in before.items():
        assert digest(ROOT / key) == expected_hash, key
    assert len(starts) == 61 and all(t["status"] == "WIN" for t in starts)
    assert all(matches)
    invalid = sum(t["invalid_steps"] for t in starts)
    revisits = sum(t["revisits"] for t in starts)
    assert invalid == revisits == 0
    write(out / "fixed_route_traces.json", fixed_traces)
    write(out / "all_start_traces.json", starts)
    with (out / "hotel_key_numbers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(route_rows[0]))
        writer.writeheader()
        writer.writerows(route_rows)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={p: importlib.metadata.version(p) for p in ("numpy", "torch", "openpyxl", "matplotlib")})
    write(out / "environment.json", environment)
    summary = dict(status="PASS", author_commit=cfg["commit"], fixed_route_FED_matches=4,
        stored_weight_paper_path_matches=sum(matches), all_start_arrivals=61, starts=61,
        invalid_steps=invalid, revisits=revisits, archived_start_trajectories_exactly_matched=61,
        unchanged_input_files=len(before), new_training_updates=0, LLM_requests=0,
        old_final48_crowd_scenarios_accessed=False,
        scope="Known hotel case replay only; not independent external validation, new training, FDS, or small-case62% result")
    write(out / "verification.json", summary)
    write(out / "completion_manifest.json", dict(status="HOTEL_BASELINE_REPLAY_COMPLETE",
        artifact_sha256={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
