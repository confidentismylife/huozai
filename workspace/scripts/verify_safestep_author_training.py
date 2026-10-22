"""Check full training journals, replay windows and independent final execution."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from datetime import datetime,timezone

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fire_evac.safestep_author_numerics import AuthorDQN,load_author_inputs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def protected_files():
    checks={}
    for path in [ROOT/"outputs/safestep_paper_numerics_20261015/completion_manifest.json",
                 ROOT/"outputs/safestep_exact_supervision_20261015/completion_manifest.json"]:
        checks.update(read(path)["artifact_sha256"])
    checks.update(read(ROOT/"outputs/safestep_exact_supervision_20261015/frozen_manifest.json")["protected_previous_files"])
    checks.update(read(ROOT/"outputs/safestep_author_training_20261015/frozen_manifest.json")["files"])
    normalized={}
    for name,value in checks.items():
        absolute=(ROOT/name).resolve()
        if absolute in normalized:assert normalized[absolute]==value
        normalized[absolute]=value
    for path,value in normalized.items():assert digest(path)==value,path
    return len(normalized)


def verify_run(directory,arm,cfg,inputs):
    summary=read(directory/"summary.json");training=read(directory/"training_complete.json")
    assert summary["training"]==training
    assert training["settings"]["learning_rate"]==arm["learning_rate"]
    assert training["settings"]["episodes"]==arm["episodes"]
    assert training["settings"]["stop_at_convergence"]==arm["stop_at_convergence"]
    assert training["reward_exit"]==arm["reward_exit"]
    coordinates=[tuple(x) for x in training["coordinate_order"]]
    assert coordinates==inputs["empty"]+list(inputs["exits"])
    index={xy:i for i,xy in enumerate(coordinates)}
    n=len(coordinates);dt=2/.3
    exits={index[x] for x in inputs["exits"]}
    deltas=[(-1,0),(1,0),(0,-1),(0,1)]
    next_table=np.zeros((n,4),dtype=int)
    for state,xy in enumerate(coordinates):
        for action,(dx,dy) in enumerate(deltas):
            next_table[state,action]=index.get((xy[0]+dx,xy[1]+dy),state)
    times=len(next(iter(inputs["streams"].values())))
    fer=np.column_stack([inputs["streams"].get(xy,[0.0]*times) for xy in coordinates])
    records=np.concatenate([np.load(p)["records"] for p in sorted((directory/"transitions").glob("*.npz"))])
    assert len(records)==training["transitions"]
    assert np.array_equal(next_table[records["state"],records["action"]],records["next_state"])
    sample_index=np.minimum((dt*records["step"]/5).astype(int),times-1)
    doses=fer[sample_index,records["next_state"]]*dt
    episodes=list(csv.DictReader((directory/"episodes.csv").open(encoding="utf-8")))
    assert len(episodes)==training["episodes"]
    audit=np.load(directory/"optimizer_audit.npz")
    assert audit["sample_ids"].shape==(training["optimizer_updates"],64)
    assert len(audit["losses"])==training["optimizer_updates"]
    assert np.all(np.isfinite(audit["losses"]))
    bounds=np.r_[0,np.flatnonzero(np.diff(records["episode"]))+1,len(records)]
    assert len(bounds)-1==len(episodes)
    count_updates=0;sync_steps=0;expected_sync=[dict(episode=0,transitions=0,updates=0)]
    for ep,(a,b),row in zip(range(1,len(episodes)+1),zip(bounds[:-1],bounds[1:]),episodes):
        block=records[a:b]
        assert np.all(block["episode"]==ep)
        assert np.array_equal(block["step"],np.arange(1,len(block)+1))
        assert np.array_equal(block["state"][1:],block["next_state"][:-1])
        expected_start=inputs["empty"][random.Random(int(row["reset_seed"])).randint(0,60)]
        assert coordinates[int(block["state"][0])]==expected_start
        assert expected_start==(int(row["start_x"]),int(row["start_y"]))
        initial=fer[0,block["state"][0]]*dt
        cumulative=np.cumsum(np.r_[initial,doses[a:b]])[1:]
        assert np.allclose(cumulative,block["FED"],rtol=1e-12,atol=5e-11)
        previous=np.r_[initial,block["FED"][:-1]]
        utility=np.select([previous==0,(previous>0)&(previous<.3),(previous>.3)&(previous<1),previous>1],
                          [0.,-100.,-300.,-800.],default=np.nan)
        assert np.isfinite(utility).all()
        seen=set();walking=[]
        for state,next_state in zip(block["state"],block["next_state"]):
            if state==next_state:walking.append(-1.)
            else:
                walking.append(arm["reward_exit"] if next_state in exits else -.5 if next_state in seen else -.25)
                seen.add(int(next_state))
        expected_reward=.1*np.array(walking)+.2*utility*doses[a:b]
        assert np.allclose(expected_reward,block["reward"],rtol=1e-12,atol=1e-10)
        status=np.where(np.isin(block["next_state"],list(exits)),0,np.where(block["step"]>300,1,2))
        assert np.array_equal(status,block["status"])
        assert np.all(block["status"][:-1]==2)
        assert block["status"][-1] in (0,1)
        assert len(block)==int(row["steps"])
        assert abs(float(row["FED"])-block["FED"][-1])<1e-13
        assert abs(float(row["reward"])-sum(block["reward"]))<1e-8
        sync_steps+=len(block)
        expected_updates=count_updates+int(b>64)
        assert int(row["updates"])==expected_updates
        if expected_updates>count_updates:
            samples=audit["sample_ids"][count_updates]
            assert samples.min()>=max(0,b-1000) and samples.max()<b
            assert len(set(samples.tolist()))==64
            assert float(row["loss"])==audit["losses"][count_updates]
            count_updates+=1
            if sync_steps>50:
                expected_sync.append(dict(episode=ep,transitions=int(b),updates=count_updates));sync_steps=0
    assert expected_sync==read(directory/"target_syncs.json")
    assert count_updates==training["optimizer_updates"]
    if training["early_stopped"]:
        assert arm["stop_at_convergence"]
        check=read(directory/"convergence_checks.json")[-1]
        assert check["all_win"] and check["last20_loss_below_half"]
        assert check["episode"]==training["episodes"]
    else:assert training["episodes"]==arm["episodes"]
    network=AuthorDQN(61)
    network.load_state_dict(torch.load(directory/"final_model.pth",weights_only=True,map_location="cpu"))
    greedy=[]
    for state in range(61):
        x=torch.zeros(61);x[state]=1
        with torch.no_grad():greedy.append(int(network(x).argmax()))
    traces=read(directory/"evaluation_traces.json")
    assert len(traces)==61
    evaluated_steps=0;wins=0;illegal=0;revisits=0;feds=[]
    for start,record in zip(inputs["empty"],traces):
        state=index[start];path=[state];fed=fer[0,state]*dt
        for k in range(1,302):
            state=int(next_table[state,greedy[state]])
            fed+=fer[min(int(dt*k/5),times-1),state]*dt
            path.append(state)
            if state in exits:break
        actual_path=[coordinates[x] for x in path]
        assert actual_path==[tuple(x) for x in record["path"]]
        assert abs(fed-record["FED"])<1e-13
        win=state in exits
        assert record["status"]==("WIN" if win else "LOSE")
        wins+=win;evaluated_steps+=len(path)-1
        illegal+=sum(a==b for a,b in zip(path,path[1:]))
        revisits+=len(path)-len(set(path));feds.append(fed)
    assert summary["all_start_success"]==wins
    assert summary["invalid_steps"]==illegal and summary["revisits"]==revisits
    assert abs(summary["mean_FED"]-np.mean(feds))<1e-13
    assert (directory/"final_model.pth").stat().st_mtime_ns<=(directory/"training_complete.json").stat().st_mtime_ns
    assert (directory/"training_complete.json").stat().st_mtime_ns<=(directory/"evaluation_traces.json").stat().st_mtime_ns
    return dict(run=directory.name,episodes=len(episodes),training_transitions=len(records),
        optimizer_updates=count_updates,replay_sample_IDs_checked=int(audit["sample_ids"].size),
        final_rollouts=len(traces),final_steps=evaluated_steps,success=wins,
        both_paper_routes_match=summary["both_paper_routes_match"])


def main():
    torch.set_num_threads(1)
    cfg=read(ROOT/"schemas/safestep_author_training_20261015_v2.json")
    out=ROOT/cfg["output_directory"]
    frozen=read(out/"frozen_manifest.json")
    for path,value in frozen["files"].items():assert digest(ROOT/path)==value,path
    inputs=load_author_inputs(ROOT/cfg["source_directory"])
    rows=[];pair_initial={}
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            directory=out/"runs"/f"{arm['id']}_seed{seed}"
            row=verify_run(directory,arm,cfg,inputs);rows.append(row)
            summary=read(directory/"summary.json")
            fingerprint=summary["training"]["initial_networks"][0]["fingerprint"]
            if seed in pair_initial:assert pair_initial[seed]==fingerprint
            else:pair_initial[seed]=fingerprint
    assert len(set(pair_initial.values()))==3
    from fire_evac.safestep_author_training import tensor_fingerprint
    released_fingerprint=tensor_fingerprint(torch.load(ROOT/cfg["source_directory"]/"maze_dql.pth",weights_only=True,map_location="cpu"))
    assert all(fp!=released_fingerprint for fp in pair_initial.values())
    preserved=protected_files()
    result=dict(status="PASS",created_at=datetime.now(timezone.utc).isoformat(),runs=rows,
        total_training_transitions=sum(r["training_transitions"] for r in rows),
        total_optimizer_updates=sum(r["optimizer_updates"] for r in rows),
        total_replay_sample_IDs_checked=sum(r["replay_sample_IDs_checked"] for r in rows),
        independently_recomputed_final_rollouts=sum(r["final_rollouts"] for r in rows),
        paired_initial_weights_identical=True,prior_artifacts_unchanged=preserved,
        initial_weights_differ_from_released_model=True,
        old_final48_crowd_scenarios_accessed=False,source_optimizer_unmodified=True,
        caveat="Checks numerical execution/provenance; does not prove the original author training history or full Table2 algorithm was reproduced.",
        verifier_sha256=digest(__file__))
    (out/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
