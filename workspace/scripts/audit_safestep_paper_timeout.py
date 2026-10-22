"""Audit the three hard-timeout runs without treating checkpoints as finals.

Checkpoint selection uses only complete-journal coverage and file readability.
No policy score is read until this operational selection is frozen to JSON.
"""
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fire_evac.safestep_author_numerics import AuthorDQN,load_author_inputs,import_reviewed_source,author_rollout
from fire_evac.safestep_paper_stepwise import GreedyPolicy


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(directory):
    arrays=[];damaged=[]
    for path in sorted((directory/"transitions").glob("*.npz")):
        try:
            with np.load(path) as data:arrays.append(data["records"])
        except Exception as exc:
            damaged.append(dict(file=str(path),exception=type(exc).__name__));break
    records=np.concatenate(arrays)
    last_step=64
    for path in sorted((directory/"optimizer").glob("*.npz")):
        try:
            with np.load(path) as data:steps=data["global_step"]
            assert np.array_equal(steps,np.arange(last_step+1,last_step+1+len(steps)))
            last_step=int(steps[-1])
        except Exception as exc:
            damaged.append(dict(file=str(path),exception=type(exc).__name__));break
    rows=list(csv.DictReader((directory/"episodes.csv").open(encoding="utf-8")))
    rows=[r for r in rows if r.get("global_steps") and r.get("episode_status")]
    cap=min(len(records),last_step)
    choices=[]
    for path in sorted(directory.glob("checkpoint_*.pth"),reverse=True):
        episode=int(path.stem.split("_")[1])
        if episode>len(rows):continue
        endpoint=int(rows[episode-1]["global_steps"])
        if endpoint>cap:continue
        try:
            weights=torch.load(path,weights_only=True,map_location="cpu")
            assert all(torch.isfinite(x).all() for x in weights.values())
            model=AuthorDQN(61);model.load_state_dict(weights,strict=True)
        except Exception as exc:
            damaged.append(dict(file=str(path),exception=type(exc).__name__));continue
        choices.append((path,episode,endpoint));break
    if not choices:raise ValueError(f"No fully journalled readable checkpoint for {directory}")
    path,episode,n=choices[0]
    return path,records[:n],rows[:episode],dict(
        rule="Latest episode checkpoint with readable finite tensors and complete transition/optimizer journal coverage; selected before any policy score",
        source_directory=str(directory.relative_to(ROOT)),checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest(path),
        checkpoint_episode=episode,prefix_transitions=n,prefix_updates=n-64,
        observed_complete_CSV_episodes=len(rows),last_CSV_global_step=int(rows[-1]["global_steps"]),
        complete_transition_journal_rows=len(records),complete_optimizer_journal_global_step=last_step,
        incomplete_or_unreadable_tail_files=damaged,planned_episodes=12000,training_complete=False,
        is_interim_checkpoint=True)


def main():
    torch.set_num_threads(1)
    cfg=read(ROOT/"schemas/safestep_paper_stepwise_20261015.json")
    source=ROOT/cfg["source_directory"];run_root=ROOT/cfg["output_directory"]
    out=ROOT/"outputs/safestep_paper_stepwise_timeout_audit_20261015"
    out.mkdir(exist_ok=False)
    campaign=read(run_root/"campaign_status.json")
    assert all(v["status"]=="failed" and v["elapsed_s"]>=cfg["hard_timeout_seconds"] for v in campaign.values())
    frozen=read(run_root/"frozen_manifest.json")
    for path,h in frozen["files"].items():assert digest(ROOT/path)==h,path
    spec=importlib.util.spec_from_file_location("stepwise_verifier",ROOT/"scripts/verify_safestep_paper_stepwise.py")
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    inputs=load_author_inputs(source)
    selections=[]
    # Freeze all operational choices before any checkpoint gets evaluated.
    for seed in cfg["seeds"]:
        path,records,rows,selection=coverage(run_root/"runs"/f"seed{seed}")
        selection["seed"]=seed;selections.append(selection)
    (out/"checkpoint_selection_before_scoring.json").write_text(json.dumps(selections,indent=2),encoding="utf-8")
    reports=[]
    reference=read(ROOT/"outputs/safestep_paper_numerics_20261015/numeric_run/results.json")["stored_weight_runs"]
    for seed,selection in zip(cfg["seeds"],selections):
        directory=run_root/"runs"/f"seed{seed}"
        checkpoint,records,rows,recheck=coverage(directory)
        assert recheck["checkpoint"]==selection["checkpoint"]
        derived=dict(status="interrupted_checkpoint_prefix",episodes=selection["checkpoint_episode"],
            transitions=len(records),optimizer_updates=len(records)-64,
            coordinate_order=inputs["empty"]+list(inputs["exits"]),
            audited_update_indices=[n for n in cfg["audit_update_indices"] if n<=len(records)-64])
        metadata,records,rows,coordinates,neighbors,fer=verifier.physics(directory,cfg,inputs,
            training_override=derived,records_override=records,episodes_override=rows)
        optimizer=verifier.optimizer_journal(directory,cfg,records,rows,derived,
            update_limit=derived["optimizer_updates"],sync_log_required=False)
        contexts=verifier.context_replay(directory,cfg,records,derived)
        model=AuthorDQN(61);model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"));model.eval()
        maze=import_reviewed_source(source/"maze.py",f"timeout_maze_{seed}");maze.Maze.reward_exit=cfg["reward_exit"]
        policy=GreedyPolicy(model,inputs["empty"],maze.Action)
        traces=[]
        with torch.no_grad():greedy=[int(model(torch.eye(61)[i]).argmax()) for i in range(61)]
        for start_index,start in enumerate(inputs["empty"]):
            record=author_rollout(maze,inputs,policy,start);record["start"]=start;traces.append(record)
            state=start_index;path=[state];fed=fer[0,state]*2/.3
            for k in range(1,302):
                state=int(neighbors[state,greedy[state]]);path.append(state)
                fed+=fer[min(int(k*(2/.3)/5),len(fer)-1),state]*2/.3
                if state in (61,62):break
            assert [coordinates[x] for x in path]==record["path"]
            assert np.isclose(fed,record["FED"],rtol=0,atol=1e-11)
        paper=[]
        for expected in reference:
            record=next(r for r in traces if list(r["start"])==expected["path"][0])
            paper.append(dict(start=record["start"],status=record["status"],steps=record["steps"],FED=record["FED"],
                paper_path_match=record["path"]==[tuple(x) for x in expected["path"]]))
        all_rows=list(csv.DictReader((directory/"episodes.csv").open(encoding="utf-8")))
        all_rows=[r for r in all_rows if r.get("global_steps") and r.get("episode_status")]
        tail=all_rows[-1000:]
        report=dict(seed=seed,selection=selection,training_complete=False,planned_final_acceptance="NOT_EVALUATED",
            physics_prefix_transitions=len(records),optimizer_prefix=optimizer,exactly_replayed_Adam_updates=contexts,
            diagnostic_rollouts=61,checkpoint_success=sum(r["status"]=="WIN" for r in traces),
            checkpoint_invalid=sum(r["invalid_steps"] for r in traces),checkpoint_revisits=sum(r["revisits"] for r in traces),
            checkpoint_mean_FED=float(np.mean([r["FED"] for r in traces])),paper_start_diagnostics=paper,
            last_CSV_1000_training_episodes=dict(count=len(tail),wins=sum(r["episode_status"]=="WIN" for r in tail),
                timeouts=sum(r["episode_status"]=="LOSE" for r in tail),steps=sum(int(r["steps"]) for r in tail)))
        target=out/f"seed{seed}";target.mkdir()
        (target/"interim_diagnostic.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        (target/"checkpoint_rollout_traces.json").write_text(json.dumps(traces,indent=2),encoding="utf-8")
        reports.append(report);print(json.dumps(report),flush=True)
    old=read(ROOT/"outputs/safestep_author_training_20261015_v2/completion_manifest.json")
    for path,h in old["artifact_sha256"].items():assert digest(ROOT/path)==h,path
    summary=dict(status="INTERRUPTED_TRAINING_PREFIX_AUDIT_PASS",created_at=datetime.now(timezone.utc).isoformat(),
        planned_runs=3,completed_12000_episode_runs=0,reason="All3 met frozen3600-second process limit",
        checkpoint_selection_before_policy_scoring=True,reports=reports,
        total_verified_prefix_transitions=sum(r["physics_prefix_transitions"] for r in reports),
        total_verified_optimizer_updates=sum(r["optimizer_prefix"]["updates"] for r in reports),
        total_terminal_labels=sum(r["optimizer_prefix"]["terminal_TD_labels"] for r in reports),
        total_replay_IDs=sum(r["optimizer_prefix"]["replay_IDs"] for r in reports),
        exact_Adam_updates_replayed=sum(r["exactly_replayed_Adam_updates"] for r in reports),
        interim_rollouts=183,final_12000_episode_results_available=False,
        source_sync_list_available=False,per_update_sync_age_columns_verified=True,
        unchecked_interrupted_tail_excluded=True,old_artifacts_unchanged=len(old["artifact_sha256"]),
        old_final48_crowd_scenarios_accessed=False,auditor_sha256=digest(__file__),verifier_sha256=digest(ROOT/"scripts/verify_safestep_paper_stepwise.py"))
    (out/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'},indent=2))


if __name__=="__main__":main()
