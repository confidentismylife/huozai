"""Independent workbook arithmetic and original-network checks; no fitting."""
from __future__ import annotations
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import openpyxl
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_paper_numerics_20261015"
SOURCE = OUT / "author_repository"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worksheet(path, sheet):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook[sheet].values)
    workbook.close()
    return rows[0], {float(r[0]): dict(zip(rows[0][1:],r[1:])) for r in rows[1:] if r[0] is not None}


def main():
    torch.set_num_threads(1)
    cfg = json.loads((ROOT / "schemas/safestep_author_numerics_20261015.json").read_text(encoding="utf-8"))
    run = OUT / "numeric_run"
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "frozen_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        assert digest(ROOT / name) == expected, name
    _, rates = worksheet(SOURCE / "FED rate - room24.xlsx", "Sheet1")
    _, cumulative = worksheet(SOURCE / "fire hazard zones FED - room24.xlsx", "FED")
    _, bands = worksheet(SOURCE / "fire hazard zones FED - room24.xlsx", "Sheet1")
    _, copy_rates = worksheet(SOURCE / "fire hazard zones FED - room24.xlsx", "FED instaneous")
    checked_rates = checked_bands = 0
    for time, values in rates.items():
        for region, rate in values.items():
            # Uses independently stored cumulative-FED worksheet as a cross-check.
            assert rate == (cumulative[time][region]-cumulative[time-5][region])/5
            assert rate == copy_rates[time][region]
            checked_rates += 1
    for time, values in cumulative.items():
        for region, fed in values.items():
            expected = 0 if fed == 0 else 1 if fed < .3 else 2 if fed < 1 else 3
            assert bands[time][region] == expected
            checked_bands += 1
    node = ast.parse((SOURCE / "coordinates2areanumber.py").read_text(encoding="utf-8"))
    mapping = ast.literal_eval(next(n.value for n in node.body if isinstance(n,ast.Assign)))
    records = result["fixed_routes"] + result["stored_weight_runs"] + result["all_start_diagnostic"]
    total_steps = 0
    for record in records:
        path = [tuple(x) for x in record["path"]]
        # Original exposure rule implemented with absolute workbook timestamps.
        dt = 2/.3
        fed = rates[300.0][mapping[path[0]]]*dt
        for k, xy in enumerate(path[1:],1):
            time = min(900,300+5*int(k*dt/5))
            if xy not in ((6,4),(14,2)):
                fed += rates[float(time)][mapping[xy]]*dt
            assert sum(abs(a-b) for a,b in zip(path[k-1],xy)) in (0,1)
        assert abs(record["FED"]-fed) < 1e-13
        assert record["steps"] == len(path)-1
        assert abs(record["time_s"]-dt*record["steps"]) < 1e-12
        total_steps += record["steps"]
    assert all(row["exact_rounding_match"] and row["steps_match"] and row["time_rounding_match"]
               for row in result["fixed_route_comparisons"])
    assert all(r["matches_paper_recommended_path"] for r in result["stored_weight_runs"])
    # Compile original reviewed source (not cached bytecode) to compare all Q's.
    sys.path.insert(0,str(ROOT / "src"))
    from fire_evac.safestep_author_numerics import import_reviewed_source, StoredAuthorPolicy, load_author_inputs
    maze = import_reviewed_source(SOURCE / "maze.py", "verified_original_maze")
    previous = sys.modules.get("maze")
    sys.modules["maze"] = maze
    try:
        algorithm = import_reviewed_source(SOURCE / "algorithm_DQL.py", "verified_original_algorithm")
    finally:
        if previous is None:
            del sys.modules["maze"]
        else:
            sys.modules["maze"] = previous
    inputs = load_author_inputs(SOURCE)
    original = algorithm.DQN(61,61,4)
    original.load_state_dict(torch.load(SOURCE / "maze_dql.pth",weights_only=True,map_location="cpu"))
    original.eval()
    adapter = StoredAuthorPolicy(SOURCE,inputs["empty"],maze.Action)
    max_q_difference = 0.0
    for index, cell in enumerate(inputs["empty"]):
        x = torch.zeros(61); x[index]=1
        with torch.no_grad():
            q = original(x).numpy()
        difference = float(np.max(np.abs(q-adapter.q(cell))))
        max_q_difference = max(max_q_difference,difference)
        assert difference == 0
    # Preserve all old files explicitly listed in the last completed branch.
    previous_manifest = ROOT / "outputs/safestep_exact_supervision_20261015/completion_manifest.json"
    previous_data = json.loads(previous_manifest.read_text(encoding="utf-8"))
    previous_checked=0
    for name, expected in previous_data["artifact_sha256"].items():
        assert digest(ROOT/name) == expected, name
        previous_checked+=1
    previous_frozen = json.loads((previous_manifest.parent/"frozen_manifest.json").read_text(encoding="utf-8"))
    older_checked=0
    for name, expected in previous_frozen["protected_previous_files"].items():
        assert digest(ROOT/name) == expected, name
        older_checked+=1
    verification = dict(status="PASS",created_at=datetime.now(timezone.utc).isoformat(),
        original_commit=cfg["commit"],frozen_source_files_checked=len(manifest["files"]),
        original_rate_values_checked=checked_rates,stationary_hazard_band_values_checked=checked_bands,
        independently_recomputed_rollouts=len(records),independently_recomputed_steps=total_steps,
        fixed_route_paper_FED_rounding_matches=4,stored_weight_paper_path_matches=2,
        original_network_Q_outputs_checked=61*4,max_original_network_Q_difference=max_q_difference,
        previous_exact_supervision_artifacts_unchanged=previous_checked,
        older_frozen_research_artifacts_unchanged=older_checked,
        verifier_sha256=digest(__file__),training_updates=0,LLM_requests=0,
        old_sealed_final_crowd_scenarios_accessed=False,
        reproduction_scope="Hotel fixed-weight inference and four paper-route dose results; not training, FDS or small-case 62% result")
    (run/"verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
