"""Repeat two predetermined worlds to verify execution-only reproducibility."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import run_safestep_robust_gate as engine
import run_safestep_robust_gate_parallel as continuation

RUN = ROOT / "outputs/safestep_robust_gate_20261018"
OUT = ROOT / "outputs/safestep_robust_execution_check_20261018"


def main():
    assert not OUT.exists()
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    chosen = [frozen["world_schedule"][0],
              next(s for s in frozen["world_schedule"] if s["stage"] == "synthetic_confirmation")]
    engine.check(frozen["input_sha256"])
    OUT.mkdir()
    engine.save(OUT / "plan.json", dict(selection="Original context and first scheduled confirmation world; no outcome-based selection",
        worlds=[s["id"] for s in chosen], scope="Repeat verification, excluded from primary matrix effects",
        engine_input_sha256=frozen["input_sha256"]))
    cfg = frozen["configuration"]
    continuation.initialize_worker(cfg)
    _, graph, times, source, distances, regions, mapping = continuation.WORKER
    results = []
    for spec in chosen:
        reference = RUN / "worlds" / spec["id"]
        destination = OUT / spec["id"]
        _, episodes, moves, reached = engine.run_world(spec, destination, graph, times,
            source, distances, regions, mapping, cfg)
        # Gzip/NPZ headers contain file timestamps; compare decoded contents instead.
        left = gzip.decompress((reference / "episodes.jsonl.gz").read_bytes())
        right = gzip.decompress((destination / "episodes.jsonl.gz").read_bytes())
        assert left == right
        assert (reference / "summary.csv").read_bytes() == (destination / "summary.csv").read_bytes()
        for filename in ("native_inputs.npz", "tables_t250.npz", "tables_t300.npz", "tables_t450.npz", "tables_t600.npz"):
            with np.load(reference / filename) as a, np.load(destination / filename) as b:
                assert a.files == b.files
                assert all(np.array_equal(a[k], b[k]) for k in a.files)
        results.append(dict(world_id=spec["id"], repeated_episodes=episodes, repeated_moves=moves,
            reached=reached, decoded_trajectory_sha256=hashlib.sha256(right).hexdigest(),
            all_actions_scores_and_native_arrays_identical=True))
    engine.check(frozen["input_sha256"])
    engine.save(OUT / "verification.json", dict(status="PASS_DETERMINISTIC_EXECUTION_REPEAT",
        worlds=results, extra_verification_episodes=sum(r["repeated_episodes"] for r in results),
        exclude_from_primary_effect_estimates=True, model_threshold_and_seed_changes=0))
    files = [p for p in OUT.rglob("*") if p.is_file()] + [Path(__file__)]
    engine.save(OUT / "completion_manifest.json", dict(status="COMPLETE",
        artifact_sha256={p.relative_to(ROOT).as_posix(): engine.sha(p) for p in files}))
    print(json.dumps(dict(status="PASS", worlds=len(results),
        extra_verification_episodes=sum(r["repeated_episodes"] for r in results))), flush=True)


if __name__ == "__main__":
    main()
