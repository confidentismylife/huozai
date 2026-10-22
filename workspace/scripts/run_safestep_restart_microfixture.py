"""Compare continuous, periodic-checkpoint restart, and .stop restart on a small fire."""
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import time

from prepare_safestep_restart_hrr_audit import ROOT, OUT, sha, save
from run_safestep_restart_hrr_controls import EXE

FIXTURE = """! Diagnostic fixture, not a hotel or SafeStep reproduction.
&HEAD CHID='{prefix}', TITLE='Small fire restart mechanism control' /
&TIME T_END=5.5, DT=0.02, LOCK_TIME_STEP=.TRUE. /
&MESH IJK=12,12,16, XB=0.0,2.4,0.0,2.4,0.0,3.2 /
&REAC FUEL='PROPANE', SOOT_YIELD=0.01 /
&SURF ID='BURNER', HRRPUA=200.0 /
&VENT XB=0.8,1.6,0.8,1.6,0.0,0.0, SURF_ID='BURNER' /
&VENT MB='XMIN', SURF_ID='OPEN' /
&VENT MB='XMAX', SURF_ID='OPEN' /
&VENT MB='YMIN', SURF_ID='OPEN' /
&VENT MB='YMAX', SURF_ID='OPEN' /
&VENT MB='ZMAX', SURF_ID='OPEN' /
&DEVC ID='T_PROBE', XYZ=1.2,1.2,0.6, QUANTITY='TEMPERATURE' /
&DEVC ID='FUEL_PROBE', XYZ=1.2,1.2,0.4, QUANTITY='VOLUME FRACTION', SPEC_ID='PROPANE' /
&DEVC ID='CO2_PROBE', XYZ=1.2,1.2,0.6, QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON DIOXIDE' /
&DUMP DT_HRR=0.02, DT_DEVC=0.02, DT_RESTART=3.0, DT_FLUSH=0.02,
      DT_SLCF=100.0, DT_SMOKE3D=100.0 /
{restart}
&TAIL /
"""


def run(case, prefix, inject_stop=False):
    assert not (case / "process_start.json").exists()
    env = os.environ.copy(); bindir = str(__import__("pathlib").Path(EXE).parent)
    env["PATH"] = os.pathsep.join([bindir + "/mpi", bindir, env["PATH"]])
    env["I_MPI_ROOT"] = bindir + "/mpi"; env["OMP_NUM_THREADS"] = "2"
    began = time.monotonic(); monitor = []; injection = None; timed_out = False; next_monitor = began + 5
    with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen([EXE, case.name + ".fds"], cwd=case, env=env, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        start = {"started_at": datetime.now(timezone.utc).isoformat(), "pid": proc.pid,
                 "command": [EXE, case.name + ".fds"], "hard_timeout_s": 180, "OMP_NUM_THREADS": 2}
        save(case / "process_start.json", start)
        print(json.dumps({"event": "started", "case": case.name, "pid": proc.pid}), flush=True)
        while proc.poll() is None:
            elapsed = time.monotonic() - began
            if elapsed > 180:
                print(json.dumps({"event": "hard_timeout_terminating", "case": case.name}), flush=True)
                proc.kill(); timed_out = True; break
            if inject_stop and injection is None:
                device = case / f"{prefix}_devc.csv"
                if device.exists():
                    rows = device.read_text("utf-8", errors="replace").splitlines()
                    if len(rows) > 2:
                        try:
                            observed = float(rows[-1].split(",")[0])
                        except ValueError:
                            observed = -1
                        if observed >= 2.8:
                            (case / f"{prefix}.stop").write_text("", "ascii")
                            injection = {"observed_native_time_s": observed, "elapsed_wall_s": elapsed,
                                         "method": "Empty native .stop file while solver is running"}
                            save(case / "stop_injection.json", injection)
            if time.monotonic() >= next_monitor:
                monitor.append({"elapsed_s": elapsed, "process_alive": True})
                next_monitor += 5
            time.sleep(.01)
        proc.wait(timeout=15)
    log = (case / "stderr.log").read_text("utf-8", errors="replace")
    expected = "STOP: FDS stopped by user" if inject_stop else "STOP: FDS completed successfully"
    errors = [line for line in log.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
    passed = proc.returncode == 0 and expected in log and not errors and not timed_out
    save(case / "run_result.json", {**start, "elapsed_s": time.monotonic()-began, "returncode": proc.returncode,
         "timed_out": timed_out, "passed": passed, "expected_marker": expected, "expected_marker_found": expected in log,
         "FDS_errors": errors, "progress_records": monitor, "stop_injection": injection,
         "restart_files": [p.name for p in case.glob("*.restart")]})
    print(json.dumps({"event": "finished", "case": case.name, "passed": passed, "elapsed_s": time.monotonic()-began}), flush=True)
    assert passed, f"Inspect {case.name}; do not retry in place"


def parent(name, inject_stop):
    case = OUT / "microfixture" / name; case.mkdir(parents=True, exist_ok=False)
    prefix = name
    (case / f"{name}.fds").write_text(FIXTURE.format(prefix=prefix, restart=""), "utf-8")
    run(case, prefix, inject_stop)
    assert len(list(case.glob("*.restart"))) == 1, "Need native checkpoint for paired comparison"
    return case, prefix


def resume(parent_case, prefix, name):
    case = OUT / "microfixture" / name; case.mkdir(exist_ok=False)
    hashes = {}
    for p in parent_case.iterdir():
        if p.is_file() and p.name.startswith(prefix) and not p.name.endswith(".stop"):
            shutil.copyfile(p, case / p.name); hashes[p.name] = sha(p)
    (case / f"{name}.fds").write_text(FIXTURE.format(prefix=prefix, restart="&MISC RESTART=.TRUE. /"), "utf-8")
    save(case / "derivation.json", {"parent": parent_case.relative_to(ROOT).as_posix(),
         "copied_files_sha256": hashes, "input_change_only": "RESTART=.TRUE.", "same_CHID": prefix,
         "checkpoint_bytes_modified": False})
    run(case, prefix)


if __name__ == "__main__":
    save(OUT / "frozen_microfixture_plan.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "grid": [12,12,16], "cells": 2304,
        "DT_s": .02, "T_END_s": 5.5, "periodic_restart_interval_s": 3.0,
        "stop_trigger_observed_time_s": 2.8, "cases": ["micro_continuous", "micro_periodic_resume", "micro_stop_parent", "micro_stop_resume"],
        "question": "Does native .stop suppress chemistry and disturb recovery compared with passive periodic checkpoints?",
        "bounds": "One diagnostic fixture; not hotel validation or universal solver benchmark",
        "solver_modified": False})
    continuous, continuous_prefix = parent("micro_continuous", False)
    resume(continuous, continuous_prefix, "micro_periodic_resume")
    stopped, stopped_prefix = parent("micro_stop_parent", True)
    resume(stopped, stopped_prefix, "micro_stop_resume")
