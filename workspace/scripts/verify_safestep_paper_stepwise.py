"""Independent full journals, terminal targets and sampled Adam-step verification."""
from datetime import datetime,timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fire_evac.safestep_author_numerics import AuthorDQN,load_author_inputs
from fire_evac.safestep_author_training import tensor_fingerprint


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics(directory,cfg,inputs,training_override=None,records_override=None,episodes_override=None):
    # Overrides verify an explicitly labelled interrupted prefix without creating
    # any fictitious training_complete file or treating it as a final result.
    training=read(directory/"training_complete.json") if training_override is None else training_override
    records=(np.concatenate([np.load(p)["records"] for p in sorted((directory/"transitions").glob("*.npz"))])
             if records_override is None else records_override)
    assert len(records)==training["transitions"]
    coordinates=[tuple(x) for x in training["coordinate_order"]]
    assert coordinates==inputs["empty"]+list(inputs["exits"])
    index={xy:i for i,xy in enumerate(coordinates)}
    exits={index[xy] for xy in inputs["exits"]};dt=2/.3
    neighbors=np.zeros((63,4),dtype=int)
    for i,(x,y) in enumerate(coordinates):
        for action,(dx,dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
            neighbors[i,action]=index.get((x+dx,y+dy),i)
    assert np.array_equal(neighbors[records["state"],records["action"]],records["next_state"])
    nt=len(next(iter(inputs["streams"].values())))
    fer=np.column_stack([inputs["streams"].get(xy,[0.]*nt) for xy in coordinates])
    rates=fer[np.minimum((dt*records["step"]/5).astype(int),nt-1),records["next_state"]]
    doses=rates*dt
    boundaries=np.r_[0,np.flatnonzero(np.diff(records["episode"]))+1,len(records)]
    episodes=(list(csv.DictReader((directory/"episodes.csv").open(encoding="utf-8")))
              if episodes_override is None else episodes_override)
    assert len(episodes)==training["episodes"]==len(boundaries)-1
    for ep,(a,b),row in zip(range(1,len(episodes)+1),zip(boundaries[:-1],boundaries[1:]),episodes):
        block=records[a:b]
        assert np.all(block["episode"]==ep)
        assert np.array_equal(block["step"],np.arange(1,len(block)+1))
        assert np.array_equal(block["state"][1:],block["next_state"][:-1])
        start=inputs["empty"][random.Random(int(row["reset_seed"])).randint(0,60)]
        assert coordinates[block["state"][0]]==start==(int(row["start_x"]),int(row["start_y"]))
        initial=fer[0,block["state"][0]]*dt
        cumulative=np.cumsum(np.r_[initial,doses[a:b]])[1:]
        assert np.allclose(cumulative,block["FED"],rtol=1e-12,atol=5e-11)
        previous=np.r_[initial,block["FED"][:-1]]
        utility=np.select([previous==0,(previous>0)&(previous<.3),(previous>.3)&(previous<1),previous>1],
            [0.,-100.,-300.,-800.],default=np.nan)
        seen=set();walking=[]
        for old,new in zip(block["state"],block["next_state"]):
            if old==new:walking.append(-1.)
            else:
                walking.append(cfg["reward_exit"] if new in exits else -.5 if new in seen else -.25)
                seen.add(int(new))
        expected=.1*np.asarray(walking)+.2*utility*doses[a:b]
        assert np.allclose(expected,block["reward"],rtol=1e-12,atol=1e-10)
        status=np.where(np.isin(block["next_state"],list(exits)),0,np.where(block["step"]>300,1,2))
        assert np.array_equal(status,block["status"])
        assert np.all(block["status"][:-1]==2) and block["status"][-1] in (0,1)
        assert int(row["steps"])==len(block)
        assert abs(float(row["FED"])-block["FED"][-1])<1e-13
        assert abs(float(row["reward"])-sum(block["reward"]))<1e-8
        assert int(row["updates"])==max(0,b-cfg["batch_size"])
        assert int(row["global_steps"])==b
        expected_eps=cfg["epsilon_start"]*cfg["epsilon_decay_per_episode"]**(ep-1)
        assert np.isclose(float(row["epsilon"]),expected_eps,rtol=1e-12,atol=1e-15)
    return training,records,episodes,coordinates,neighbors,fer


def optimizer_journal(directory,cfg,records,episodes,training,update_limit=None,sync_log_required=True):
    updates=0;terminal_labels=0;max_terminal_error=0.
    by_ep_sum=np.zeros(len(episodes)+1);by_ep_count=np.zeros(len(episodes)+1,dtype=int)
    for path in sorted((directory/"optimizer").glob("*.npz")):
        if update_limit is not None and updates>=update_limit:break
        with np.load(path) as z:data={k:z[k] for k in z.files}
        if update_limit is not None:
            keep=min(len(data["ids"]),update_limit-updates)
            data={k:v[:keep] for k,v in data.items()}
        ids=data["ids"];n=len(ids)
        assert ids.shape==(n,cfg["batch_size"])
        steps=np.arange(updates+cfg["batch_size"]+1,updates+n+cfg["batch_size"]+1)
        assert np.array_equal(data["global_step"],steps)
        assert np.all(ids<steps[:,None])
        assert np.all(ids>=np.maximum(0,steps-cfg["replay_capacity"])[:,None])
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
        assert np.array_equal(data["episode"],records[steps-1]["episode"])
        sync=((steps-1)//cfg["target_sync_global_steps"])*cfg["target_sync_global_steps"]
        assert np.array_equal(data["target_last_sync_step"],sync)
        assert np.isfinite(data["q"]).all() and np.isfinite(data["targets"]).all() and np.isfinite(data["loss"]).all()
        sampled=records[ids];terminal=sampled["status"]!=2
        rewards=sampled["reward"].astype(np.float32)
        if terminal.any():
            error=float(np.max(np.abs(data["targets"][terminal]-rewards[terminal])))
            assert error==0.;max_terminal_error=max(max_terminal_error,error)
        terminal_labels+=int(terminal.sum())
        loss=np.mean((data["q"]-data["targets"])**2,axis=1,dtype=np.float32)
        assert np.allclose(loss,data["loss"],rtol=5e-6,atol=1e-6)
        by_ep_sum+=np.bincount(data["episode"],weights=data["loss"],minlength=len(by_ep_sum))
        by_ep_count+=np.bincount(data["episode"],minlength=len(by_ep_count))
        updates+=n
    assert updates==training["optimizer_updates"]==max(0,len(records)-cfg["batch_size"])
    for ep,row in enumerate(episodes,1):
        expected=by_ep_sum[ep]/by_ep_count[ep] if by_ep_count[ep] else 0.
        assert np.isclose(float(row["mean_update_loss"]),expected,rtol=1e-10,atol=1e-8)
    if (directory/"target_syncs.json").exists():
        assert read(directory/"target_syncs.json")==list(range(cfg["target_sync_global_steps"],len(records)+1,cfg["target_sync_global_steps"]))
    else:
        assert not sync_log_required
    return dict(updates=updates,replay_IDs=updates*cfg["batch_size"],terminal_TD_labels=terminal_labels,
                maximum_terminal_bootstrap=max_terminal_error)


def context_replay(directory,cfg,records,training):
    identity=torch.eye(61);checked=[]
    for path in sorted((directory/"update_contexts").glob("*.pt")):
        context=torch.load(path,weights_only=True,map_location="cpu")
        batch=records[np.asarray(context["ids"],dtype=np.int64)]
        policy=AuthorDQN(61);target=AuthorDQN(61)
        policy.load_state_dict(context["policy_before"]);target.load_state_dict(context["target"])
        optimizer=torch.optim.Adam(policy.parameters(),lr=cfg["learning_rate"])
        optimizer.load_state_dict(context["optimizer_before"])
        states=torch.from_numpy(batch["state"].astype(np.int64));actions=torch.from_numpy(batch["action"].astype(np.int64))
        q=policy(identity[states]).gather(1,actions[:,None]).squeeze(1)
        labels=torch.from_numpy(batch["reward"].astype(np.float32))
        active=torch.from_numpy(batch["status"]==2)
        with torch.no_grad():
            if active.any():
                ns=torch.from_numpy(batch["next_state"].astype(np.int64))[active]
                labels[active]+=cfg["gamma"]*target(identity[ns]).max(1).values
        assert torch.equal(q.detach(),context["q"]) and torch.equal(labels,context["labels"])
        loss=((q-labels)**2).mean()
        assert loss.item()==context["loss"]
        optimizer.zero_grad();loss.backward();optimizer.step()
        assert tensor_fingerprint(policy.state_dict())==tensor_fingerprint(context["policy_after"])
        assert all(p.grad is None for p in target.parameters())
        checked.append(context["update"])
    expected=[n for n in cfg["audit_update_indices"] if n<=training["optimizer_updates"]]
    assert checked==expected==training["audited_update_indices"]
    return len(checked)


def final_execution(directory,inputs,coordinates,neighbors,fer):
    network=AuthorDQN(61)
    network.load_state_dict(torch.load(directory/"final_model.pth",weights_only=True,map_location="cpu"))
    greedy=[];identity=torch.eye(61)
    for state in range(61):
        with torch.no_grad():greedy.append(int(network(identity[state]).argmax()))
    traces=read(directory/"evaluation_traces.json");summary=read(directory/"summary.json")
    dt=2/.3;nt=len(fer);exits={61,62};wins=0;invalid=0;revisits=0;steps=0;feds=[]
    for state,record in enumerate(traces):
        start=state;path=[state];fed=fer[0,state]*dt
        for k in range(1,302):
            state=int(neighbors[state,greedy[state]]);path.append(state)
            fed+=fer[min(int(k*dt/5),nt-1),state]*dt
            if state in exits:break
        assert [coordinates[x] for x in path]==[tuple(x) for x in record["path"]]
        assert abs(record["FED"]-fed)<1e-13
        assert record["status"]==("WIN" if state in exits else "LOSE")
        wins+=state in exits;invalid+=sum(a==b for a,b in zip(path,path[1:]));revisits+=len(path)-len(set(path));steps+=len(path)-1;feds.append(fed)
    assert len(traces)==61 and wins==summary["all_start_success"]
    assert invalid==summary["invalid_steps"] and revisits==summary["revisits"]
    assert np.isclose(np.mean(feds),summary["mean_FED"],atol=1e-13,rtol=0)
    reference=read(ROOT/"outputs/safestep_paper_numerics_20261015/numeric_run/results.json")["stored_weight_runs"]
    matching=[]
    for expected,row in zip(reference,summary["paper_starts"]):
        record=next(r for r in traces if r["start"]==expected["path"][0])
        match=record["path"]==expected["path"]
        dose_match=round(record["FED"],4)==expected["paper_FED"]
        assert row["path_match"]==match and row["FED_rounding_match"]==dose_match
        matching.append(match and dose_match)
    assert summary["both_paper_routes_match"]==all(matching)
    assert (directory/"final_model.pth").stat().st_mtime_ns<=(directory/"training_complete.json").stat().st_mtime_ns
    assert (directory/"training_complete.json").stat().st_mtime_ns<=(directory/"evaluation_traces.json").stat().st_mtime_ns
    return dict(final_rollouts=61,final_steps=steps,success=wins,both_paper_routes_match=all(matching))


def verify_run(directory,cfg,inputs):
    training,records,episodes,coordinates,neighbors,fer=physics(directory,cfg,inputs)
    result=dict(run=directory.name,episodes=training["episodes"],transitions=len(records),
        optimizer=optimizer_journal(directory,cfg,records,episodes,training),
        exactly_replayed_Adam_updates=context_replay(directory,cfg,records,training),
        execution=final_execution(directory,inputs,coordinates,neighbors,fer))
    return result


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true");args=parser.parse_args()
    cfg=read(ROOT/"schemas/safestep_paper_stepwise_20261015.json")
    out=ROOT/cfg["output_directory"];inputs=load_author_inputs(ROOT/cfg["source_directory"])
    if args.preflight:
        result=verify_run(out/"preflight_0",cfg,inputs)
        (out/"preflight_independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2));return
    frozen=read(out/"frozen_manifest.json")
    for path,h in frozen["files"].items():assert digest(ROOT/path)==h,path
    rows=[]
    for seed in cfg["seeds"]:
        directory=out/"runs"/f"seed{seed}"
        row=verify_run(directory,cfg,inputs);rows.append(row);print(json.dumps(row),flush=True)
        training=read(directory/"training_complete.json")
        assert training["episodes"]==cfg["episodes"]
        old=read(ROOT/f"outputs/safestep_author_training_20261015_v2/runs/paper_listed_parameters_seed{seed}/training_complete.json")
        assert training["initial_policy_fingerprint"]==old["initial_networks"][0]["fingerprint"]
        old_starts=list(csv.DictReader((ROOT/f"outputs/safestep_author_training_20261015_v2/runs/paper_listed_parameters_seed{seed}/episodes.csv").open(encoding="utf-8")))
        new_starts=list(csv.DictReader((directory/"episodes.csv").open(encoding="utf-8")))
        assert [r["reset_seed"] for r in old_starts]==[r["reset_seed"] for r in new_starts]
    old_manifest=read(ROOT/"outputs/safestep_author_training_20261015_v2/completion_manifest.json")
    for path,h in old_manifest["artifact_sha256"].items():assert digest(ROOT/path)==h,path
    result=dict(status="PASS",created_at=datetime.now(timezone.utc).isoformat(),runs=rows,
        total_transitions=sum(r["transitions"] for r in rows),
        total_updates=sum(r["optimizer"]["updates"] for r in rows),
        total_replay_IDs_checked=sum(r["optimizer"]["replay_IDs"] for r in rows),
        total_terminal_TD_labels_checked=sum(r["optimizer"]["terminal_TD_labels"] for r in rows),
        exactly_replayed_Adam_updates=sum(r["exactly_replayed_Adam_updates"] for r in rows),
        independent_final_rollouts=183,old_artifacts_unchanged=len(old_manifest["artifact_sha256"]),
        paired_initial_weights_and_reset_seeds=True,old_final48_crowd_scenarios_accessed=False,
        caveat="Paper-flow reconstruction under disclosed unknowns, not unique original author training, equal-compute improvement, or physical validation.",
        verifier_sha256=digest(__file__))
    (out/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
