"""Frozen three-seed Table2 training-flow reproduction and repeated preflight."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
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
from fire_evac.safestep_paper_stepwise import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--worker",action="store_true");parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    cfg_path=ROOT/"schemas/safestep_paper_stepwise_20261015.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"));out=ROOT/cfg["output_directory"]
    source=ROOT/cfg["source_directory"]
    if args.worker:
        summary=run(source,cfg,args.seed,out/"runs"/f"seed{args.seed}")
        print(json.dumps({k:v for k,v in summary.items() if k!="training"},indent=2),flush=True);return
    if args.smoke:
        out.mkdir(exist_ok=False)
        results=[]
        for i in range(cfg["smoke"]["repetitions"]):
            results.append(run(source,cfg,cfg["smoke"]["seed"],out/f"preflight_{i}",episodes=cfg["smoke"]["episodes"]))
        assert results[0]["training"]["final_policy_fingerprint"]==results[1]["training"]["final_policy_fingerprint"]
        record=dict(status="PASS",repeated_final_weights_identical=True,
            seed=cfg["smoke"]["seed"],episodes=cfg["smoke"]["episodes"],
            transitions=results[0]["training"]["transitions"],updates=results[0]["training"]["optimizer_updates"],
            wall_seconds=[r["training"]["wall_seconds"] for r in results])
        (out/"preflight_verification.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(json.dumps(record,indent=2));return
    assert (out/"preflight_verification.json").exists()
    (out/"runs").mkdir(exist_ok=False);(out/"logs").mkdir(exist_ok=False)
    dependencies=[cfg_path,Path(__file__),ROOT/"src/fire_evac/safestep_paper_stepwise.py",
        ROOT/"src/fire_evac/safestep_author_training.py",ROOT/"src/fire_evac/safestep_author_numerics.py",
        ROOT/"tests/test_safestep_paper_stepwise.py",*source.rglob("*")]
    frozen=dict(created_at=datetime.now(timezone.utc).isoformat(),locked_before_training=True,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies if p.is_file()})
    (out/"frozen_manifest.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    results={}
    def worker(seed):
        handle=(out/"logs"/f"seed{seed}.log").open("w",encoding="utf-8")
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MPLBACKEND="Agg")
        process=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--seed",str(seed)],
            cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        print(json.dumps(dict(event="started",seed=seed,pid=process.pid)),flush=True)
        started=time.monotonic()
        try:code=process.wait(timeout=cfg["hard_timeout_seconds"])
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(event="hard_timeout",seed=seed)),flush=True)
            process.terminate();process.wait(timeout=30);code=process.returncode
        handle.close()
        result=dict(exit_code=code,status="complete" if code==0 else "failed",elapsed_s=time.monotonic()-started,pid=process.pid)
        print(json.dumps(dict(event="finished",seed=seed,**result)),flush=True)
        return seed,result
    with ThreadPoolExecutor(max_workers=cfg["parallel_processes"]) as pool:
        futures=[pool.submit(worker,seed) for seed in cfg["seeds"]]
        for future in as_completed(futures):
            seed,result=future.result();results[str(seed)]=result
            (out/"campaign_status.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="finished",runs=results),indent=2),flush=True)


if __name__=="__main__":main()
