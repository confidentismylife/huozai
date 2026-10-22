"""Rebuild every reported per-world metric independently from raw episodes."""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/safestep_robust_gate_20261018"
OUT = ROOT / "outputs/safestep_robust_summary_check_20261018"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    assert not p.exists(), p
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_world(spec):
    directory = RUN / "worlds" / spec["id"]
    by = defaultdict(list)
    with gzip.open(directory / "episodes.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            by[record["departure_fire_s"], record["method"]].append(record)
    with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == len(by)
    maximum_error, fields = 0., 0
    for row in summary:
        departure, method = int(row["departure_fire_s"]), row["method"]
        records = by[departure, method]
        assert len(records) == 61 and len({r["start_id"] for r in records}) == 61
        successful = [r for r in records if r["reached"]]
        rebuilt = dict(n=len(records), reached=len(successful),
            mean_FED=statistics.fmean(r["FED"] for r in successful),
            maximum_FED=max(r["FED"] for r in successful),
            mean_travel_s=statistics.fmean(r["travel_s"] for r in successful),
            immediate_reversals=sum(r["immediate_reversals"] for r in records),
            selected_plan_exit_changes=(sum(r["selected_plan_exit_changes"] for r in records)
                if records[0]["selected_plan_exit_changes"] is not None else None))
        for label, baseline in (("snapshot", "snapshot_hold"), ("ensemble", "ensemble_mean"), ("half", "half30")):
            comparator = {r["start_id"]: r for r in by[departure, baseline]}
            both = [(r, comparator[r["start_id"]]) for r in records if r["reached"] and comparator[r["start_id"]]["reached"]]
            differences = [r["FED"] - b["FED"] for r, b in both]
            rebuilt[f"paired_{label}"] = len(both)
            rebuilt[f"lower_vs_{label}"] = sum(x < -1e-10 for x in differences)
            rebuilt[f"equal_vs_{label}"] = sum(abs(x) <= 1e-10 for x in differences)
            rebuilt[f"higher_vs_{label}"] = sum(x > 1e-10 for x in differences)
            rebuilt[f"mean_difference_vs_{label}"] = statistics.fmean(differences)
            rebuilt[f"sum_comparator_FED_{label}"] = sum(b["FED"] for r, b in both)
            rebuilt[f"sum_paired_difference_{label}"] = sum(differences)
        events = [event for r in records for event in r["gate_opportunities"]]
        rebuilt["opportunities"] = len(events)
        rebuilt["accepted"] = sum(event["accepted"] for event in events)
        for decision in ("accepted", "rejected"):
            for outcome in ("helpful", "harmful", "neutral", "failure"):
                rebuilt[f"{decision}_{outcome}"] = sum(event["accepted"] == (decision == "accepted")
                    and event["offline_classification"] == outcome for event in events)
        for key, value in rebuilt.items():
            if value is None:
                assert row[key] == "", (spec["id"], method, key)
            else:
                error = abs(float(row[key]) - value)
                assert error < 1e-9, (spec["id"], method, key, row[key], value)
                maximum_error = max(maximum_error, error)
            fields += 1
    return dict(world_id=spec["id"], summary_rows_checked=len(summary), numeric_fields_checked=fields,
        maximum_numeric_difference=maximum_error, raw_episodes=sum(len(rows) for rows in by.values()))


def main():
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    if not OUT.exists():
        OUT.mkdir()
        save(OUT / "plan.json", dict(scope="Every per-world summary rebuilt from raw trajectories",
            method_changes=False, new_policy_runs=0,
            expected_worlds=len(frozen["world_schedule"]), script_sha256=sha(Path(__file__))))
    results = []
    began = time.perf_counter()
    for ordinal, spec in enumerate(frozen["world_schedule"]):
        path = OUT / f"{spec['id']}.json"
        if path.exists():
            result = json.loads(path.read_text("utf-8"))
        else:
            while not (RUN / "worlds" / spec["id"] / "completion_manifest.json").exists():
                if list(RUN.glob("failure_*.txt")) or time.perf_counter() - began > 7200:
                    raise RuntimeError("Batch stopped; prior checks retained")
                time.sleep(10)
            result = check_world(spec)
            save(path, result)
        results.append(result)
        if (ordinal + 1) % 100 == 0:
            print(json.dumps(dict(checked_worlds=len(results), expected=len(frozen["world_schedule"]))), flush=True)
    save(OUT / "results.json", dict(status="PASS_ALL_REPORTED_WORLD_METRICS_REBUILT_FROM_RAW",
        worlds=len(results), raw_episodes=sum(r["raw_episodes"] for r in results),
        summary_rows_checked=sum(r["summary_rows_checked"] for r in results),
        numeric_fields_checked=sum(r["numeric_fields_checked"] for r in results),
        maximum_numeric_difference=max(r["maximum_numeric_difference"] for r in results)))
    files = [p for p in OUT.iterdir() if p.is_file()]
    save(OUT / "completion_manifest.json", dict(status="COMPLETE",
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print(json.dumps(dict(status="COMPLETE", worlds=len(results))), flush=True)


if __name__ == "__main__":
    main()
