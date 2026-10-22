"""Finite six-run single-factor update-frequency experiment."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from fire_evac.safestep_frequency_ablation import run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg_path = ROOT / "schemas/safestep_frequency_ablation_20261015.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out, source = ROOT / cfg["output_directory"], ROOT / cfg["source_directory"]
    if args.worker:
        result = run(source, cfg, args.arm, args.seed, out / "runs" / f"{args.arm}_seed{args.seed}")
        print(json.dumps(result), flush=True)
        return
    if args.preflight:
        out.mkdir(exist_ok=False)
        summaries = {}
        for arm in cfg["arms"]:
            runs = []
            for i in range(cfg["preflight"]["repetitions_per_arm"]):
                runs.append(run(source, cfg, arm, cfg["preflight"]["seed"], out / f"preflight_{arm}_{i}",
                                cfg["preflight"]["control_steps"]))
            assert runs[0]["training"]["final_policy_fingerprint"] == runs[1]["training"]["final_policy_fingerprint"]
            for subdir in ("transitions", "optimizer"):
                first = sorted((out / f"preflight_{arm}_0" / subdir).glob("*.npz"))
                second = sorted((out / f"preflight_{arm}_1" / subdir).glob("*.npz"))
                assert len(first) == len(second)
                for a, b in zip(first, second):
                    with np.load(a) as x, np.load(b) as y:
                        assert set(x.files) == set(y.files)
                        for key in x.files:
                            assert np.array_equal(x[key], y[key]), (arm, subdir, key)
            summaries[arm] = runs[0]["training"]
        a, b = [out / f"preflight_{arm}_0/transitions/chunk_00000.npz" for arm in cfg["arms"]]
        with np.load(a) as x, np.load(b) as y:
            assert np.array_equal(x["records"][:65], y["records"][:65])
        assert summaries["episode_end"]["initial_policy_fingerprint"] == summaries["every_step"]["initial_policy_fingerprint"]
        result = dict(status="PASS", repeated_transitions_optimizer_journals_and_weights_identical=True,
                      paired_first65_transitions_identical=True, summaries=summaries)
        (out / "preflight_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    assert (out / "preflight_verification.json").exists()
    assert (out / "preflight_independent_verification.json").exists()
    (out / "runs").mkdir(exist_ok=False)
    (out / "logs").mkdir(exist_ok=False)
    files = [cfg_path, Path(__file__), ROOT / "src/fire_evac/safestep_frequency_ablation.py",
             ROOT / "tests/test_safestep_frequency_ablation.py", ROOT / "scripts/verify_safestep_frequency_ablation.py",
             ROOT / "src/fire_evac/safestep_author_numerics.py", ROOT / "src/fire_evac/safestep_author_training.py",
             ROOT / "src/fire_evac/safestep_paper_stepwise.py", *source.rglob("*")]
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), frozen_before_formal_training=True,
                    files={str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()})
    (out / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state = {}

    def worker(item):
        arm, seed = item
        name = f"{arm}_seed{seed}"
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
        with (out / "logs" / f"{name}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--arm", arm,
                                        "--seed", str(seed)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            started = time.monotonic()
            print(json.dumps(dict(event="started", run=name, pid=process.pid)), flush=True)
            try:
                code = process.wait(timeout=cfg["hard_timeout_seconds"])
                status = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                print(json.dumps(dict(event="hard_timeout", run=name)), flush=True)
                process.terminate()
                process.wait(timeout=30)
                code, status = process.returncode, "timeout"
        result = dict(status=status, exit_code=code, elapsed_s=time.monotonic() - started, pid=process.pid)
        print(json.dumps(dict(event="finished", run=name, **result)), flush=True)
        return name, result

    plan = [(arm, seed) for arm in cfg["arms"] for seed in cfg["seeds"]]
    with ThreadPoolExecutor(max_workers=cfg["parallel_processes"]) as pool:
        for future in as_completed([pool.submit(worker, item) for item in plan]):
            name, result = future.result()
            state[name] = result
            (out / "campaign_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="campaign_finished", runs=state), indent=2), flush=True)


if __name__ == "__main__":
    main()
