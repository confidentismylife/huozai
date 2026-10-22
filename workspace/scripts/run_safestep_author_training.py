"""Finite from-scratch runs with author code, snapshots and process monitoring."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
from fire_evac.safestep_author_training import run_training


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    cfg_path=ROOT/"schemas/safestep_author_training_20261015.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    source=ROOT/cfg["source_directory"]
    out=ROOT/cfg["output_directory"]
    if args.worker:
        arm=next(a for a in cfg["arms"] if a["id"]==args.arm)
        result=run_training(source,cfg,arm,args.seed,out/"runs"/f"{args.arm}_seed{args.seed}")
        print(json.dumps({k:v for k,v in result.items() if k!="training"},indent=2),flush=True)
        return
    if args.smoke:
        out.mkdir(exist_ok=True)
        results=[]
        for i,seed in enumerate(cfg["smoke_test"]["seeds"]):
            result=run_training(source,cfg,cfg["arms"][0],seed,out/f"preflight_v2_{i}",
                                episodes_override=cfg["smoke_test"]["episodes"])
            results.append(result)
        assert results[0]["training"]["final_weight_fingerprint"]==results[1]["training"]["final_weight_fingerprint"]
        assert results[0]["training"]["transitions"]==results[1]["training"]["transitions"]
        smoke=dict(status="PASS",episodes_per_run=cfg["smoke_test"]["episodes"],
            exact_repeated_final_weights=True,transitions=results[0]["training"]["transitions"],
            seconds_per_training_run=[r["training"]["wall_seconds"] for r in results],
            seed=cfg["smoke_test"]["seeds"][0])
        (out/"smoke_verification.json").write_text(json.dumps(smoke,indent=2),encoding="utf-8")
        print(json.dumps(smoke,indent=2));return
    assert (out/"smoke_verification.json").exists(),"Run repeated preflight before campaign"
    (out/"runs").mkdir(exist_ok=False)
    (out/"logs").mkdir(exist_ok=False)
    paths=[cfg_path,Path(__file__),ROOT/"src/fire_evac/safestep_author_training.py",
        ROOT/"src/fire_evac/safestep_author_numerics.py",*source.rglob("*")]
    frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        six_runs_locked_before_initialization=True)
    (out/"frozen_manifest.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    plan=[(arm["id"],seed) for arm in cfg["arms"] for seed in cfg["seeds"]]
    state={};status_path=out/"campaign_status.json"
    def status():
        # Single coordinator thread writes summaries after each result below.
        status_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    def launch(item):
        arm,seed=item;name=f"{arm}_seed{seed}"
        log=(out/"logs"/f"{name}.log").open("w",encoding="utf-8")
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MPLBACKEND="Agg")
        process=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--arm",arm,"--seed",str(seed)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        print(json.dumps(dict(event="started",run=name,pid=process.pid)),flush=True)
        started=time.monotonic()
        try:
            code=process.wait(timeout=cfg["per_run_timeout_seconds"])
            result=dict(exit_code=code,pid=process.pid,elapsed_s=time.monotonic()-started,
                        status="complete" if code==0 else "crashed")
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(event="hard_timeout",run=name)),flush=True)
            process.terminate();process.wait(timeout=30)
            result=dict(exit_code=process.returncode,pid=process.pid,status="timeout")
        finally:log.close()
        print(json.dumps(dict(event="finished",run=name,**result)),flush=True)
        return name,result
    with ThreadPoolExecutor(max_workers=cfg["parallel_processes"]) as pool:
        futures=[pool.submit(launch,item) for item in plan]
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            name,result=future.result();state[name]=result;status()
    print(json.dumps(dict(status="campaign_finished",runs=state),indent=2),flush=True)


if __name__=="__main__":main()
