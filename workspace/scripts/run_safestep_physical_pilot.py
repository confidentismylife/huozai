"""Run one prepared reconstruction once; monitor, log, and enforce a hard bound."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_physics_time_contract_20261016"


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--case", required=True);a = p.parse_args()
    assert a.case in ("physics_init", "hotel_literature_pilot120", "hotel_literature_pilot120_v2", "hotel_literature_pilot120_resume")
    c = json.loads((OUT/"frozen_physics_time_contract.json").read_text("utf-8"))
    case = OUT/"runs"/a.case;inp = case/(a.case+".fds")
    assert inp.is_file() and not (case/"process_start.json").exists(), "Never overwrite or retry in place"
    if a.case != "physics_init":
        init = (OUT/"runs/physics_init/stderr.log").read_text("utf-8", errors="replace")
        assert "STOP: Set-up only" in init and not re.search(r"\bERROR(?:\(|:)", init), "Initialization must pass first"
    env = os.environ.copy();bindir = str(Path(c["FDS_executable"]).parent)
    env["PATH"] = os.pathsep.join([bindir+"/mpi", bindir, env["PATH"]])
    env["I_MPI_ROOT"] = bindir+"/mpi";env["OMP_NUM_THREADS"] = c["runtime"]["OMP_NUM_THREADS"]
    began = time.monotonic();started = datetime.now(timezone.utc).isoformat();monitor = []
    with (case/"stdout.log").open("wb") as out, (case/"stderr.log").open("wb") as err:
        proc = subprocess.Popen([c["FDS_executable"], inp.name], cwd=case, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        start_record = {"command": [c["FDS_executable"], inp.name], "cwd": case.as_posix(), "pid": proc.pid,
                        "started_at": started, "hard_timeout_s": c["runtime"]["timeout_s"], "OMP_NUM_THREADS": env["OMP_NUM_THREADS"]}
        (case/"process_start.json").write_text(json.dumps(start_record, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({"event": "started", "case": a.case, "pid": proc.pid}), flush=True)
        next_record = time.monotonic()+20;timed_out = False
        while proc.poll() is None:
            elapsed = time.monotonic()-began
            if elapsed > c["runtime"]["timeout_s"]:
                print(json.dumps({"event": "hard_timeout_terminating", "elapsed_s": elapsed}), flush=True)
                proc.kill();timed_out = True;break
            if time.monotonic() >= next_record:
                text = (case/"stderr.log").read_text("utf-8", errors="replace")
                times = re.findall(r"Simulation Time:\s*([\d.Ee+\-]+)\s*s", text)
                record = {"event": "alive", "elapsed_s": round(elapsed,1), "last_reported_simulation_s": float(times[-1]) if times else None,
                          "stderr_bytes": len(text)}
                monitor.append(record);print(json.dumps(record), flush=True);next_record = time.monotonic()+20
            time.sleep(.25)
        proc.wait(timeout=15)
    text = (case/"stderr.log").read_text("utf-8", errors="replace")
    errors = [line.strip() for line in text.splitlines() if re.match(r"\s*ERROR(?:\(|:)", line)]
    success = "STOP: Set-up only" in text or "STOP: FDS completed successfully" in text
    result = {**start_record, "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic()-began,
              "returncode": proc.returncode, "timed_out": timed_out, "FDS_errors": errors,
              "FDS_success_marker": success, "passed": proc.returncode == 0 and not errors and success and not timed_out,
              "warnings": [line.strip() for line in text.splitlines() if line.strip().startswith("WARNING:")], "progress_records": monitor,
              "stderr_tail": text[-3500:]}
    (case/"run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"event": "finished", "case": a.case, "passed": result["passed"], "elapsed_s": round(result["elapsed_s"],2),
                      "returncode": proc.returncode, "warnings": result["warnings"], "stderr_tail": text[-1500:]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
