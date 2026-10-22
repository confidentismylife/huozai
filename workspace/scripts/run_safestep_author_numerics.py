"""Reproduce fixed-weight and fixed-route hotel results from original artifacts."""
from __future__ import annotations
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from fire_evac.safestep_author_numerics import (
    FixedActions, StoredAuthorPolicy, author_rollout, import_reviewed_source,
    load_author_inputs, sha256, shortest_to_exit)


def main():
    config_path = ROOT / "schemas/safestep_author_numerics_20261015.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / cfg["source_directory"]
    out = ROOT / cfg["output_directory"]
    out.mkdir(exist_ok=False)
    frozen = {str(p.relative_to(ROOT)): sha256(p) for p in [config_path, Path(__file__),
              ROOT / "src/fire_evac/safestep_author_numerics.py", *source.rglob("*")]
              if p.is_file()}
    (out / "frozen_manifest.json").write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        stage="locked_before_loading_weight_or_scoring", files=frozen), indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    inputs = load_author_inputs(source, cfg["start_time_s"])
    module = import_reviewed_source(source / "maze.py", "safestep_reviewed_author_maze")
    model = StoredAuthorPolicy(source, inputs["empty"], module.Action)
    results = []
    summary_rows = []
    for target in cfg["paper_targets"]:
        run = author_rollout(module, inputs, FixedActions(target["actions"], module.Action), target["start"])
        run.update(id=target["id"], method="paper_fixed_route", target=target)
        error = run["FED"] - target["FED"]
        row = dict(id=target["id"], method=run["method"], steps=run["steps"], time_s=run["time_s"],
                   paper_FED=target["FED"], reproduced_FED=run["FED"], absolute_error=error,
                   relative_error=error/target["FED"], exact_rounding_match=round(run["FED"],4)==target["FED"],
                   within_5_percent=abs(error/target["FED"]) <= .05,
                   steps_match=run["steps"]==target["steps"],
                   time_rounding_match=abs(run["time_s"]-target["time_s"])<=.051)
        results.append(run)
        summary_rows.append(row)
    policies = []
    for target in cfg["paper_targets"][::2]:
        run = author_rollout(module, inputs, model, target["start"])
        run.update(id=target["id"].replace("recommended", "stored_weights"), method="author_stored_DQN")
        run["matches_paper_recommended_path"] = run["path"] == next(r["path"] for r in results if r["id"] == target["id"])
        run["paper_FED"] = target["FED"]
        policies.append(run)
    all_start = []
    for start in inputs["empty"]:
        run = author_rollout(module, inputs, model, start)
        all_start.append({k:v for k,v in run.items() if k != "integration_trace"} | {"start":start})
    geometry = []
    for target in cfg["paper_targets"]:
        expected_exit = (14,2) if target["id"].endswith("recommended") else (6,4)
        shortest = shortest_to_exit(inputs["grid"], target["start"], expected_exit)
        geometry.append(dict(id=target["id"], shortest_steps=len(shortest)-1,
                             paper_steps=target["steps"], match=len(shortest)-1==target["steps"], path=shortest))
    dataset = dict(grid_shape=inputs["grid"].shape,
                   nonexit_movement_cells=len(inputs["empty"]), unique_hazard_regions=58,
                   rate_times=[float(inputs["rates"]["Time"][0]),float(inputs["rates"]["Time"][-1])],
                   rate_rows=len(inputs["rates"]["Time"]),
                   pattern_times=[float(inputs["pattern"]["Time"][0]),float(inputs["pattern"]["Time"][-1])],
                   pattern_rows=len(inputs["pattern"]["Time"]), start_rate_index=inputs["start_index"],
                   reused_pattern_index_absolute_time_s=float(inputs["pattern"]["Time"][inputs["start_index"]]),
                   architecture="61-61-4 ReLU", reward_exit=float(module.Maze.reward_exit),
                   no_training=True, no_LLM_requests=True)
    output = dict(created_at=datetime.now(timezone.utc).isoformat(), dataset=dataset,
                  fixed_routes=results, fixed_route_comparisons=summary_rows,
                  stored_weight_runs=policies, all_start_diagnostic=all_start,
                  geometry=geometry, all_start_success=sum(r["status"]=="WIN" for r in all_start),
                  all_start_count=len(all_start))
    (out / "results.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out / "fixed_route_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
    for name, data in [("stored_weight_paper_starts.json",policies),("all_start_diagnostic.json",all_start)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(dataset=dataset,fixed_route_comparisons=summary_rows,
        stored_weight_runs=[{k:v for k,v in r.items() if k not in ("path","integration_trace")} for r in policies],
        all_start_success=output["all_start_success"],all_start_count=len(all_start)),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
