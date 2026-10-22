"""Bounded same-checkpoint controls; only output cadence and stopping time differ."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from prepare_safestep_restart_hrr_audit import ROOT, OUT, OLD, sha, save

PREFIX = "hotel_literature_pilot120_v2"
PARENT = OLD / "runs" / PREFIX
CASES = {"hotel_repeat_hrr050": 0.5, "hotel_dense_hrr0025": 0.025}
EXE = "C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe"


def prepare():
    plan = {"frozen_at": datetime.now(timezone.utc).isoformat(), "checkpoint_s": 115.49118593879896,
            "end_s": 117.6, "cases": CASES, "wall_limit_per_case_s": 240,
            "physical_input_changes": False, "checkpoint_bytes_changed": False,
            "contrasts": ["Repeat prior 0.5s continuation", "Same physical trajectory with near-step HRR output"],
            "interpretation": "Output-only cause requires invariant physical observations plus compatible native energy; dense HRR alone cannot establish restart accuracy",
            "no_training_no_LLM_no_final_scenarios": True}
    save(OUT / "frozen_control_plan.json", plan)
    for name, cadence in CASES.items():
        dest = OUT / "runs" / name
        dest.mkdir(parents=True, exist_ok=False)
        copied = {}
        for src in PARENT.iterdir():
            if src.is_file() and src.name.startswith(PREFIX) and not src.name.endswith(".stop"):
                shutil.copyfile(src, dest / src.name)
                copied[src.name] = sha(src)
        original = (PARENT / f"{PREFIX}.fds").read_text("utf-8")
        new = original.replace("&MISC TMPA=20.0", "&MISC RESTART=.TRUE., TMPA=20.0")
        new = new.replace("&TIME T_END=120.0 /", "&TIME T_END=117.6 /")
        new = new.replace("DT_HRR=0.5", f"DT_HRR={cadence}")
        assert new != original and "RESTART=.TRUE." in new
        (dest / f"{name}.fds").write_text(new, "utf-8")
        save(dest / "derivation.json", {"parent": PARENT.relative_to(ROOT).as_posix(),
             "same_CHID": PREFIX, "changes": {"RESTART": True, "T_END": 117.6, "DT_HRR": cadence},
             "copied_parent_files_sha256": copied, "checkpoint_bytes_unchanged": True})
    print(json.dumps({"prepared_cases": list(CASES)}), flush=True)


def run_case(name):
    case = OUT / "runs" / name
    assert not (case / "process_start.json").exists(), "Each prepared case can run only once"
    env = os.environ.copy(); bindir = str(Path(EXE).parent)
    env["PATH"] = os.pathsep.join([bindir + "/mpi", bindir, env["PATH"]])
    env["I_MPI_ROOT"] = bindir + "/mpi"; env["OMP_NUM_THREADS"] = "2"
    command = [EXE, name + ".fds"]
    began = time.monotonic(); monitor = []; timeout = False
    with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=case, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                                stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        start = {"started_at": datetime.now(timezone.utc).isoformat(), "command": command,
                 "pid": proc.pid, "OMP_NUM_THREADS": 2, "hard_timeout_s": 240}
        save(case / "process_start.json", start)
        print(json.dumps({"event": "started", "case": name, "pid": proc.pid}), flush=True)
        next_monitor = began + 10
        while proc.poll() is None:
            elapsed = time.monotonic() - began
            if elapsed > 240:
                print(json.dumps({"event": "hard_timeout_terminating", "case": name}), flush=True)
                proc.kill(); timeout = True; break
            if time.monotonic() >= next_monitor:
                log = (case / "stderr.log").read_text("utf-8", errors="replace")
                times = re.findall(r"Simulation Time:\s*([\d.Ee+\-]+)\s*s", log)
                record = {"elapsed_s": round(elapsed, 2), "process_alive": True,
                          "last_logged_time_s": float(times[-1]) if times else None}
                monitor.append(record); print(json.dumps({"case": name, **record}), flush=True)
                next_monitor += 10
            time.sleep(0.2)
        proc.wait(timeout=15)
    log = (case / "stderr.log").read_text("utf-8", errors="replace")
    errors = [line for line in log.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
    success = "STOP: FDS completed successfully" in log
    result = {**start, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic()-began,
              "returncode": proc.returncode, "timed_out": timeout, "FDS_errors": errors,
              "FDS_success_marker": success, "passed": proc.returncode == 0 and success and not errors and not timeout,
              "progress_records": monitor, "warnings": [line.strip() for line in log.splitlines() if line.strip().startswith("WARNING:")]}
    save(case / "run_result.json", result)
    print(json.dumps({"event": "finished", "case": name, "passed": result["passed"], "elapsed_s": result["elapsed_s"]}), flush=True)
    return result["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["prepare", "run-all"]); a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    else:
        for case_name in CASES:
            if not run_case(case_name):
                raise SystemExit("Control failed; preserve logs and investigate")
