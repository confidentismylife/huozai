"""Monitored normal-end hotel run with verified passive-checkpoint capture."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

from prepare_safestep_clean_hotel120 import ROOT, OUT, PREFIX, sha, save
from analyze_safestep_restart_hrr_audit import checkpoint_metadata

PLAN = json.loads((OUT / "frozen_plan.json").read_text("utf-8"))
EXE = PLAN["FDS_executable"]
MESHES = [(220,12,12), (7,67,12), (152,9,12), (17,40,12), (9,122,12), (53,21,17)]


def signature(paths):
    return [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def framed_file_complete(path):
    """Check every unformatted record; do not assume stable size means complete."""
    size = path.stat().st_size; count = 0
    with path.open("rb") as stream:
        while header := stream.read(4):
            if len(header) != 4:
                return False
            length = struct.unpack("<i", header)[0]
            if length < 0 or stream.tell()+length+4 > size:
                return False
            stream.seek(length, 1)
            trailer = stream.read(4)
            if len(trailer) != 4 or struct.unpack("<i", trailer)[0] != length:
                return False
            count += 1
        return stream.tell() == size and count > 100


def native_csv_time(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size-32768))
        rows = stream.read().decode("ascii", errors="replace").splitlines()
    for line in reversed(rows):
        try:
            return float(line.split(",", 1)[0])
        except ValueError:
            continue
    return None


def capture_if_ready(case, paths, previous_signature, stable_since, attempts):
    if (OUT / "captured_checkpoint.json").exists() or not all(p.exists() for p in paths):
        return previous_signature, stable_since, attempts
    current = signature(paths)
    if current != previous_signature:
        return current, time.monotonic(), attempts
    if stable_since is None or time.monotonic()-stable_since < 1.0:
        return current, stable_since, attempts
    try:
        meta = [checkpoint_metadata(p, ijk, .5) for p, ijk in zip(paths, MESHES)]
    except (OSError, AssertionError, struct.error):
        return current, time.monotonic(), attempts
    times = [m["fire_time_s"] for m in meta]
    if len(set(times)) != 1 or not 100.0 <= times[0] < 105.0:
        return current, stable_since, attempts
    attempts += 1
    destination = OUT / "checkpoint_capture_attempts" / f"attempt_{attempts:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    before = signature(paths)
    source_hashes = {p.name: sha(p) for p in paths}
    for p in paths:
        shutil.copyfile(p, destination / p.name)
    copied = [destination / p.name for p in paths]
    after = signature(paths)
    copied_hashes = {p.name: sha(p) for p in copied}
    complete = all(framed_file_complete(p) for p in copied)
    copied_meta = [checkpoint_metadata(p, ijk, .5) for p, ijk in zip(copied, MESHES)] if complete else []
    passed = before == after and source_hashes == copied_hashes and complete and len({m["fire_time_s"] for m in copied_meta}) == 1
    record = {"captured_at": datetime.now(timezone.utc).isoformat(), "path": destination.relative_to(ROOT).as_posix(),
        "fire_time_s": times[0], "source_size_mtime_stable_during_copy": before == after,
        "source_sha256": source_hashes, "copied_sha256": copied_hashes,
        "every_Fortran_record_complete": complete, "six_mesh_times_identical": len({m["fire_time_s"] for m in copied_meta}) == 1,
        "snapshot_metadata": copied_meta, "no_stop_file": not (case / f"{PREFIX}.stop").exists(),
        "ROOM24_reaction_heat_field_nonzero": bool(copied_meta and copied_meta[-1]["snapshot_fields"]["Q"]["nonzero_count"] > 0),
        "passed": passed}
    save(destination / "capture_result.json", record)
    assert passed and record["no_stop_file"] and record["ROOM24_reaction_heat_field_nonzero"], "Inspect checkpoint capture; never substitute old stop checkpoints"
    save(OUT / "captured_checkpoint.json", record)
    print(json.dumps({"event": "passive_checkpoint_captured", "fire_time_s": times[0], "complete_files": 6}), flush=True)
    return current, stable_since, attempts


def run(case_name, hard_limit, capture=False):
    case = OUT / "runs" / case_name; inp = case / f"{case_name}.fds"
    assert inp.exists() and not (case / "process_start.json").exists()
    assert not (case / f"{PREFIX}.stop").exists()
    env = os.environ.copy(); bindir = str(Path(EXE).parent)
    env["PATH"] = os.pathsep.join([bindir + "/mpi", bindir, env["PATH"]])
    env["I_MPI_ROOT"] = bindir + "/mpi"; env["OMP_NUM_THREADS"] = PLAN["OMP_NUM_THREADS"]
    began = time.monotonic(); records = []; timeout = False; attempts = 0; prior_signature = None; stable_since = None
    paths = [case / f"{PREFIX}_{m}.restart" for m in range(1, 7)]
    next_record = began+20
    with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen([EXE, inp.name], cwd=case, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        start = {"command": [EXE, inp.name], "input_sha256": sha(inp), "pid": proc.pid,
            "started_at": datetime.now(timezone.utc).isoformat(), "hard_timeout_s": hard_limit,
            "OMP_NUM_THREADS": env["OMP_NUM_THREADS"], "stop_file_creation": False}
        save(case / "process_start.json", start)
        print(json.dumps({"event": "started", "case": case_name, "pid": proc.pid}), flush=True)
        while proc.poll() is None:
            elapsed = time.monotonic()-began
            if elapsed > hard_limit:
                print(json.dumps({"event": "hard_timeout_terminating", "case": case_name, "elapsed_s": elapsed}), flush=True)
                proc.kill(); timeout = True; break
            if capture and not (OUT / "captured_checkpoint.json").exists():
                prior_signature, stable_since, attempts = capture_if_ready(case, paths, prior_signature, stable_since, attempts)
            if time.monotonic() >= next_record:
                log = (case / "stderr.log").read_text("utf-8", errors="replace")
                times = re.findall(r"Simulation Time:\s*([\d.Ee+\-]+)\s*s", log)
                record = {"elapsed_s": round(elapsed, 2), "process_alive": True,
                    "native_device_time_s": native_csv_time(case / f"{PREFIX}_devc.csv"),
                    "last_logged_time_s": float(times[-1]) if times else None,
                    "existing_restart_files": sum(p.exists() for p in paths),
                    "captured_checkpoint": (OUT / "captured_checkpoint.json").exists()}
                records.append(record)
                save(case / "latest_progress.json", record)
                print(json.dumps({"case": case_name, **record}), flush=True)
                next_record += 20
            time.sleep(.25)
        proc.wait(timeout=15)
    log = (case / "stderr.log").read_text("utf-8", errors="replace")
    errors = [line.strip() for line in log.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
    success = "STOP: FDS completed successfully" in log
    final_time = native_csv_time(case / f"{PREFIX}_devc.csv")
    passed = proc.returncode == 0 and success and not timeout and not errors and not (case/f"{PREFIX}.stop").exists()
    result = {**start, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic()-began,
        "returncode": proc.returncode, "timed_out": timeout, "passed": passed,
        "FDS_success_marker": success, "FDS_errors": errors, "final_native_time_s": final_time,
        "progress_records": records, "stop_file_created": False,
        "warnings": [line.strip() for line in log.splitlines() if line.strip().startswith("WARNING:")]}
    save(case / "run_result.json", result)
    print(json.dumps({"event": "finished", "case": case_name, "passed": passed, "elapsed_s": result["elapsed_s"], "native_time_s": final_time}), flush=True)
    assert passed, "Run did not finish normally; preserve artifacts and inspect, no automatic in-place retry"
    if capture:
        assert (OUT / "captured_checkpoint.json").exists(), "No complete 100s periodic checkpoint captured"


def prepare_resume():
    parent = OUT / "runs" / PREFIX
    snapshot = json.loads((OUT / "captured_checkpoint.json").read_text("utf-8"))
    assert snapshot["passed"]
    case = OUT / "runs/hotel_clean_checkpoint_resume110"
    case.mkdir(exist_ok=False)
    copied = {}
    for p in parent.iterdir():
        if p.is_file() and p.name.startswith(PREFIX) and p.suffix not in (".stop", ".restart", ".core"):
            shutil.copyfile(p, case / p.name); copied[p.name] = sha(p)
    snapshot_dir = ROOT / snapshot["path"]
    for name, expected in snapshot["copied_sha256"].items():
        assert sha(snapshot_dir / name) == expected
        shutil.copyfile(snapshot_dir / name, case / name)
        assert sha(case / name) == expected
    original = (parent / f"{PREFIX}.fds").read_text("utf-8")
    resumed = original.replace("&MISC TMPA=20.0", "&MISC RESTART=.TRUE., TMPA=20.0")
    resumed = resumed.replace("&TIME T_END=120.0 /", "&TIME T_END=110.0 /")
    assert resumed != original and "RESTART=.TRUE." in resumed
    (case / f"{case.name}.fds").write_text(resumed, "utf-8")
    save(case / "derivation.json", {"parent": parent.relative_to(ROOT).as_posix(), "same_CHID": PREFIX,
        "input_changes": {"RESTART": True, "T_END_s": 110.0}, "physical_input_changes": False,
        "snapshot": snapshot["path"], "checkpoint_fire_time_s": snapshot["fire_time_s"],
        "checkpoint_sha256": snapshot["copied_sha256"], "copied_completed_parent_prefix_files": copied,
        "checkpoint_bytes_changed": False, "parent_outputs_modified": False})
    return case.name


if __name__ == "__main__":
    run(PREFIX, PLAN["hard_wall_limit_continuous_s"], capture=True)
    resumed_case = prepare_resume()
    run(resumed_case, PLAN["paired_restart"]["hard_wall_limit_s"])
    save(OUT / "execution_complete.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
        "normal_end120s": True, "passive_checkpoint_resume110s": True,
        "analysis_pending": True, "no_stop_files_created": True})
