"""Run one explicitly prepared FDS diagnostic once, with logs and hard timeout.

This does not retry or generate inputs. Subsequent repaired cases use new dirs.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    if not args.case.replace("_", "").isalnum():
        raise ValueError("Simple case name required")
    config = json.loads((OUT/"frozen_diagnostic_plan.json").read_text("utf-8"))
    case = OUT/"runs"/args.case
    input_path = case/(args.case+".fds")
    assert input_path.is_file(), input_path
    assert not (case/"run_result.json").exists() and not (case/"process_start.json").exists(), "No overwrite or automatic retry"
    env = os.environ.copy()
    settings = config["process_environment"]
    env["PATH"] = os.pathsep.join(settings["prepend_PATH"]+[env["PATH"]])
    for key in ("I_MPI_ROOT", "OMP_NUM_THREADS"):
        env[key] = settings[key]
    started = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    with (case/"stdout.log").open("wb") as stdout, (case/"stderr.log").open("wb") as stderr:
        process = subprocess.Popen([config["executable"], input_path.name], cwd=case, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        record = {"pid": process.pid, "command": [config["executable"], input_path.name], "working_directory": case.as_posix(),
                  "started_at": started, "timeout_s": config["runtime_limits"]["timeout_per_case_s"],
                  "subprocess_environment_changes": settings}
        (case/"process_start.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({"event": "started", "case": args.case, "pid": process.pid}, ensure_ascii=True), flush=True)
        timed_out = False
        next_log = time.monotonic()+20
        while process.poll() is None:
            elapsed = time.monotonic()-began
            if elapsed > config["runtime_limits"]["timeout_per_case_s"]:
                print(json.dumps({"event": "hard_timeout_terminating", "case": args.case, "elapsed_s": elapsed}), flush=True)
                process.kill()
                timed_out = True
                break
            if time.monotonic() >= next_log:
                print(json.dumps({"event": "alive", "case": args.case, "elapsed_s": round(elapsed, 1),
                                  "stderr_bytes": (case/"stderr.log").stat().st_size}), flush=True)
                next_log = time.monotonic()+20
            time.sleep(.25)
        process.wait(timeout=15)
    stderr_text = (case/"stderr.log").read_text("utf-8", errors="replace")
    errors = [line.strip() for line in stderr_text.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
    result = {**record, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic()-began,
              "returncode": process.returncode, "timed_out": timed_out,
              "FDS_reported_errors": errors,
              "status": "TIMEOUT" if timed_out else "FDS_REPORTED_ERROR" if errors else "PROCESS_EXITED_SUCCESS" if process.returncode == 0 else "PROCESS_EXITED_ERROR",
              "file_sizes": {p.name: p.stat().st_size for p in case.iterdir() if p.is_file()},
              "stderr_last4000": (case/"stderr.log").read_text("utf-8", errors="replace")[-4000:]}
    (case/"run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"event": "finished", "case": args.case, "elapsed_s": round(result["elapsed_s"], 2),
                      "returncode": process.returncode, "timed_out": timed_out,
                      "stderr_tail": result["stderr_last4000"][-2200:]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
