"""Frozen retrospective target decomposition and bounded common-future replication."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/gain_target_alignment_diagnostic_development_v1"
PRIOR = ROOT/"outputs/override_worth_evidence_development_v1"
STATUS = ROOT/"outputs/third_round_development_status.json"
PROTOCOL = ROOT/"reports/收益目标对齐与十二未来复核开发方案_2026-10-03.md"
sys.path.insert(0,str(ROOT/"src"))
from gain_target_diagnostic import decompose, gain_class, paired_future_interval


def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,b): Path(p).write_text(json.dumps(b,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csv_write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)


def load_states():
    old=read(PRIOR/"development_manifest.json")
    lookup={s["scenario_id"]:s for s in old["calibration_pool_records"]+old["test_records"]}
    rows=[]
    for block,folder in (("calibration","calibration_states"),("test","test_states")):
        for p in sorted((PRIOR/folder).glob("*.npz")):
            with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
            family=p.stem.split("_")[0]; scenario=lookup[family]
            rows.append({"block":block,"state_id":p.stem,"family":family,"scenario":scenario,"path":str(p),"data":data})
    assert len(rows)==108
    return rows


def selection(states):
    grouped=defaultdict(list)
    for s in states: grouped[s["family"]].append(s)
    nonneutral=[];controls=[]
    for f, ss in sorted(grouped.items()):
        gains=[float((s["data"]["evaluation_outcomes"][0,:,0]-s["data"]["evaluation_outcomes"][1,:,0]).mean()*100) for s in ss]
        if any(gain_class(g)!="neutral_or_small" for g in gains):nonneutral.append(f)
    for block in ("calibration","test"):
        eligible=[f for f,ss in grouped.items() if ss[0]["block"]==block and f not in nonneutral
            and all(int(s["data"]["proposal"])!=12 for s in ss)]
        eligible.sort(key=lambda f:(-max(float(s["data"]["predicted_gains"][int(s["data"]["proposal"])]) for s in grouped[f]),f))
        assert eligible; controls.append(eligible[0])
    assert len(nonneutral)==6 and len(controls)==2
    selected=sorted(nonneutral+controls)
    return selected, {f:("original_non_neutral" if f in nonneutral else "highest_prediction_neutral_control") for f in selected}


def check_inputs(m):
    for path,expected in m["source_input_sha256"].items():assert sha(path)==expected,path


def prepare():
    assert not OUT.exists(),"preserve prior run"
    old=read(PRIOR/"development_manifest.json"); completion=read(PRIOR/"completion_manifest.json")
    for p,h in completion["files"].items(): assert sha(ROOT/p)==h,p
    for p,h in old["source_input_sha256"].items():assert sha(p)==h,p
    states=load_states();selected,reasons=selection(states)
    oldseeds=set()
    for s in states:
        for key in ("population_seed","planning_seeds","evaluation_seeds"):
            oldseeds.update(map(int,s["data"][key].reshape(-1)))
    for p in old["source_input_sha256"]:
        if Path(p).suffix==".npz" and Path(p).parent.name in ("states","training_states","audit_states"):
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds"):
                    if key in z: oldseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    for r in old["seed_registry"]:
        oldseeds.add(r["population"]);oldseeds.update(r["planning"]+r["evaluation"])
    oldseeds|={s+9000001 for s in list(oldseeds)}
    ss=[s for s in states if s["family"] in selected]
    registry=[{"state_id":s["state_id"],"family":s["family"],"population_seed":int(s["data"]["population_seed"]),
        "future_seeds":[610000000+1000*i+j for j in range(12)]} for i,s in enumerate(ss)]
    new={j for r in registry for j in r["future_seeds"]};sensor={j+9000001 for j in new}
    assert len(new)==192 and not new.intersection(sensor) and not (new|sensor).intersection(oldseeds)
    audit=[min(s["state_id"] for s in ss if s["block"]==block) for block in ("calibration","test")]
    OUT.mkdir();(OUT/"previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes())
    sources=old["source_input_sha256"].copy()
    for p,h in completion["files"].items():
        target=OUT/"previous_status_exact_bytes.json" if (ROOT/p).resolve()==STATUS else ROOT/p
        sources[str(target.resolve())]=h
    frozen=[PROTOCOL,Path(__file__),ROOT/"scripts/gain_target_diagnostic.py",ROOT/"tests/test_gain_target_diagnostic.py",PRIOR/"completion_manifest.json"]
    for p in frozen:sources[str(p.resolve())]=sha(p)
    (OUT/"source_snapshot").mkdir()
    for p in frozen[:-1]:shutil.copy2(p,OUT/"source_snapshot"/p.name)
    m={"role":"viewed_development_target_alignment_and_targeted_replication_not_confirmation",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"source_input_sha256":sources,
        "previous_status_sha256":sha(STATUS),"prior_completion_sha256":sha(PRIOR/"completion_manifest.json"),
        "selected_families":selected,"selection_reasons":reasons,"audit_state_ids":audit,"seed_registry":registry,
        "scenario_records":{s["family"]:s["scenario"] for s in ss},"frozen_ppo_models":old["frozen_ppo_models"],
        "candidate_names":old["candidate_names"],"outcome_fields":old["outcome_fields"],"trace_fields":old["trace_fields"],
        "futures_per_state":12,"maximum_full_rollouts":800,"expected_replication_rollouts":768,"expected_audit_rollouts":4,
        "bootstrap_seed":20262302,"bootstrap_draws":2000,"new_seed_namespaces_disjoint":True,
        "all_fits":0,"new_ppo_training":0,"new_llm_calls":0,"qualification_thresholds_changed":False,
        "old_test_now_viewed_development":True,"true_source_diagnostic_only":True}
    write(OUT/"development_manifest.json",m)
    print(json.dumps({"prepared":True,"families":selected,"rollouts":772,"source_inputs":len(sources)},ensure_ascii=False),flush=True)


def collect(m,states):
    import torch
    from stable_baselines3 import PPO
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    from fire_evac.source_world_planning import PLANS,source_world_scenarios,compatible_source_worlds
    from fire_evac.ppo_reference_planning import FrozenPPOContinuation,evaluate_continuation,policy_parameter_sha256
    from run_source_world_planning import make_env,vector
    torch.set_num_threads(1)
    candidates=list(PLANS)
    for r in m["frozen_ppo_models"]:
        model=PPO.load(r["path"],device="cpu");assert policy_parameter_sha256(model)==r["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(r["training_seed"],model))
    assert [p.name for p in candidates]==m["candidate_names"]
    byid={s["state_id"]:s for s in states}
    scenarios=[DynamicConflictScenario(**s) for s in m["scenario_records"].values()]
    worlds=[make_env(scenarios),make_env(scenarios)]; actual=make_env(scenarios)
    folder=OUT/"replication_states";folder.mkdir(exist_ok=True);started=time.monotonic()
    try:
        for i,entry in enumerate(m["seed_registry"]):
            path=folder/(entry["state_id"]+".npz")
            if path.exists():continue
            s=byid[entry["state_id"]];old=s["data"]; scenario=DynamicConflictScenario(**s["scenario"])
            proposal=int(old["proposal"]); assert proposal!=12
            x,_=actual.reset(seed=entry["population_seed"],options={"scenario":scenario})
            np.testing.assert_array_equal(x,old["observation"])
            support,errors,mismatch=compatible_source_worlds(source_world_scenarios(scenario),x,0)
            assert support.all() and not mismatch
            risks=np.empty((2,2,12,len(m["outcome_fields"])));traces=[];keys=[];audits=[]
            for w,hypothesis in enumerate(source_world_scenarios(scenario)):
                wx,_=worlds[w].reset(seed=entry["population_seed"],options={"scenario":hypothesis})
                np.testing.assert_array_equal(wx,x)
                for a,plan in enumerate((12,proposal)):
                    for f,seed in enumerate(entry["future_seeds"]):
                        trace=[] if w==scenario.source_branch else None
                        risks[w,a,f]=vector(evaluate_continuation(worlds[w],wx,candidates[plan],seed,trace))
                        if trace is not None:keys.append((a,f));traces.append(np.asarray(trace,float))
            if entry["state_id"] in m["audit_state_ids"]:
                for a,plan in enumerate((12,proposal)):
                    result=vector(evaluate_continuation(actual,x,candidates[plan],entry["future_seeds"][0]))
                    np.testing.assert_array_equal(result,risks[scenario.source_branch,a,0]);audits.append(result)
            temp=path.with_suffix(".tmp.npz")
            np.savez_compressed(temp,observation=x,proposal=np.asarray(proposal),predicted_gains=old["predicted_gains"],
                population_seed=np.asarray(entry["population_seed"]),future_seeds=np.asarray(entry["future_seeds"]),supported=support,
                true_source=np.asarray(scenario.source_branch),world_risks=risks,audit_outcomes=np.asarray(audits).reshape(-1,len(m["outcome_fields"])),
                trace_keys=np.asarray(keys),trace_offsets=np.cumsum([0]+[len(t) for t in traces]),execution_trace=np.concatenate(traces),
                replication_rollouts=np.asarray(48),audit_rollouts=np.asarray(len(audits)))
            temp.replace(path)
            print(f"replication {i+1}/16 {entry['state_id']}; {time.monotonic()-started:.1f}s",flush=True)
    finally:
        actual.close()
        for env in worlds:env.close()
    for r,actor in zip(m["frozen_ppo_models"],candidates[12:]):assert policy_parameter_sha256(actor.model)==r["parameter_sha256"]


def family_stats(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r["family"]].append(r)
    cols=("predicted_gain_pp","robust_gain_pp","true_planning_gain_pp","actual_gain_pp","minimum_world_gain_pp",
        "prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp","total_error_pp")
    stats={"states":len(rows),"families":len(grouped)}
    for c in cols:
        stats[c]=float(np.mean([np.mean([r[c] for r in rs]) for rs in grouped.values()]))
        if c in ("prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp","total_error_pp"):
            stats["mean_abs_"+c]=float(np.mean([np.mean([abs(r[c]) for r in rs]) for rs in grouped.values()]))
    stats["classes"]=dict(Counter(r["actual_class"] for r in rows))
    stats["robust_benefit_any_world_harm_states"]=sum(r["robust_gain_pp"]>.1 and r["minimum_world_gain_pp"]< -1e-10 for r in rows)
    stats["different_worst_sources"]=sum(r["different_worst_sources"] for r in rows)
    return stats


def analyze(m,states):
    assert not (OUT/"analysis_manifest.json").exists(),"preserve completed analysis"
    rows=[];byid={s["state_id"]:s for s in states}
    for s in states:
        d=s["data"];p=int(d["proposal"])
        r={"block":s["block"],"state_id":s["state_id"],"family":s["family"],"conflict_type":s["scenario"]["conflict_type"],
            "proposal":m["candidate_names"][p],"proposal_index":p,"eligible":p!=12,"true_source_diagnostic":s["scenario"]["source_branch"]}
        r.update(decompose(float(d["predicted_gains"][p]),d["planning_risks"],d["evaluation_outcomes"],d["supported"],s["scenario"]["source_branch"]))
        r["actual_class"]=gain_class(r["actual_gain_pp"]);r["worth_score"]=float(d["worth_probabilities"][2]);rows.append(r)
    csv_write(OUT/"cached_decomposition.csv",rows)
    groups=[]
    for block in ("calibration","test"):
        for scope in ("all_states","eligible_proposals"):
            rs=[r for r in rows if r["block"]==block and (scope=="all_states" or r["eligible"])]
            groups.append({"block":block,"scope":scope,**family_stats(rs)})
        for cls in ("beneficial","harmful","neutral_or_small"):
            rs=[r for r in rows if r["block"]==block and r["eligible"] and r["actual_class"]==cls]
            if rs:groups.append({"block":block,"scope":cls,**family_stats(rs)})
        for kind in sorted({r["conflict_type"] for r in rows if r["block"]==block}):
            rs=[r for r in rows if r["block"]==block and r["conflict_type"]==kind and r["eligible"]]
            if rs:groups.append({"block":block,"scope":kind,**family_stats(rs)})
    rep_rows=[];rep_family=[];future_gains={};rollouts=audit_count=trace_steps=hold_blocks=neutral_positive=0
    old_margin=read(PRIOR/"development_manifest.json")["prior_margin"]
    new_margin=read(PRIOR/"frozen_conditional_gate.json")["conditional_margin"]
    for file in sorted((OUT/"replication_states").glob("*.npz")):
        s=byid[file.stem];old=s["data"]
        with np.load(file,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        risks=d["world_risks"];source=int(d["true_source"]);p=int(d["proposal"])
        decomposition=decompose(float(d["predicted_gains"][p]),risks,risks[source],d["supported"],source)
        g=(risks[source,0,:,0]-risks[source,1,:,0])*100
        original=next(r for r in rows if r["state_id"]==file.stem)
        row={"block":s["block"],"family":s["family"],"state_id":file.stem,"selection_reason":m["selection_reasons"][s["family"]],
            "proposal":m["candidate_names"][p],"original_actual_gain_pp":original["actual_gain_pp"],"original_actual_class":original["actual_class"],
            "replicated_actual_class":gain_class(float(g.mean())),"first6_actual_gain_pp":float(g[:6].mean()),"last6_actual_gain_pp":float(g[6:].mean()),
            "first6_class":gain_class(float(g[:6].mean())),"last6_class":gain_class(float(g[6:].mean())),
            "worth_score":float(old["worth_probabilities"][2]),"prior_guard_accepts":bool(float(old["predicted_gains"][p])>old_margin+.001),
            "conditional_guard_accepts":bool(float(old["predicted_gains"][p])>new_margin+.001),
            "worth_condition_passes":bool(old["worth_probabilities"][2]>=.8 and old["worth_probabilities"][0]<=.1)}
        row.update(decomposition)
        row["filtered_guard_accepts"]=row["conditional_guard_accepts"] and row["worth_condition_passes"]
        for k,name in enumerate(m["outcome_fields"][:6]):
            row["baseline_"+name]=float(risks[source,0,:,k].mean());row["proposal_"+name]=float(risks[source,1,:,k].mean())
        rep_rows.append(row);future_gains[file.stem]=g
        rollouts+=int(d["replication_rollouts"]);audit_count+=int(d["audit_rollouts"])
        trace=d["execution_trace"];trace_steps+=len(trace);hold_blocks+=int(trace[:,5].sum());neutral_positive+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
    for j,family in enumerate(m["selected_families"]):
        rs=[r for r in rep_rows if r["family"]==family];assert len(rs)==2
        gains=np.asarray([future_gains[r["state_id"]] for r in rs]);interval=paired_future_interval(gains,m["bootstrap_seed"]+j)
        vals={c:float(np.mean([r[c] for r in rs])) for c in ("predicted_gain_pp","robust_gain_pp","actual_gain_pp","minimum_world_gain_pp","original_actual_gain_pp","first6_actual_gain_pp","last6_actual_gain_pp")}
        rep_family.append({"family":family,"block":rs[0]["block"],"selection_reason":rs[0]["selection_reason"],"proposal":rs[0]["proposal"],
            **vals,"replicated_class":gain_class(vals["actual_gain_pp"]),"original_class":gain_class(vals["original_actual_gain_pp"]),
            "future_only_ci_low_pp":float(interval[0]),"future_only_ci_high_pp":float(interval[1]),
            "both_population_classes_match":rs[0]["replicated_actual_class"]==rs[1]["replicated_actual_class"],
            "both_half_classes_match":all(r["first6_class"]==r["last6_class"] for r in rs)})
    assert rollouts==768 and audit_count==4 and rollouts+audit_count<=m["maximum_full_rollouts"]
    csv_write(OUT/"replication_state_results.csv",rep_rows);csv_write(OUT/"replication_family_results.csv",rep_family)
    summary={"role":m["role"],"cached_states":len(rows),"cached_eligible_states":sum(r["eligible"] for r in rows),
        "cached_groups":groups,"targeted_families":rep_family,"replicated_states":len(rep_rows),"full_rollouts":rollouts+audit_count,
        "replication_rollouts":rollouts,"audit_rollouts":audit_count,"actual_trace_steps":trace_steps,"hold_block_steps":hold_blocks,
        "neutral_positive_release_steps":neutral_positive,"original_to_replication_state_classes":[{"original":a,"replicated":b,"count":n}
            for (a,b),n in sorted(Counter((r["original_actual_class"],r["replicated_actual_class"]) for r in rep_rows).items())],
        "different_half_class_states":sum(r["first6_class"]!=r["last6_class"] for r in rep_rows),
        "different_population_class_families":sum(not r["both_population_classes_match"] for r in rep_family),
        "replication_robust_benefit_any_world_harm_states":sum(r["robust_gain_pp"]>.1 and r["minimum_world_gain_pp"]< -1e-10 for r in rep_rows),
        "fixed_gate_diagnostics":[{"gate":key,"accepted":sum(r[key] for r in rep_rows),
            "replicated_classes":dict(Counter(r["replicated_actual_class"] for r in rep_rows if r[key]))}
            for key in ("prior_guard_accepts","conditional_guard_accepts","filtered_guard_accepts","worth_condition_passes")],
        "all_fits":0,"new_ppo_training":0,"new_llm_calls":0,"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "limits":["Retrospective decomposition of viewed development data, arithmetic not causal attribution.",
            "Eight families selected using earlier empirical outcomes; no generalization or unbiased efficacy estimate.",
            "Twelve futures give finite-sample diagnostic labels, not true expected effects or calibrated safety probabilities.",
            "Intervals only resample future IDs conditional on two fixed population starts and frozen policies; no multiplicity adjustment.",
            "No new family, behavior/map/noise shift, new LLM contribution, training or formal qualification."]}
    write(OUT/"summary.json",summary);check_inputs(m)
    write(OUT/"analysis_manifest.json",{"runner_sha256":sha(__file__),"development_manifest_sha256":sha(OUT/"development_manifest.json"),
        "summary_sha256":sha(OUT/"summary.json"),"csv_sha256":{p.name:sha(p) for p in sorted(OUT.glob("*.csv"))},
        "replication_cache_sha256":{p.name:sha(p) for p in sorted((OUT/"replication_states").glob("*.npz"))}})
    print(json.dumps({"analyzed":True,"rollouts":rollouts+audit_count,"families":rep_family},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--run",action="store_true");args=parser.parse_args()
    if args.prepare_only:prepare();return
    if not args.run:parser.error("choose --prepare-only or --run")
    m=read(OUT/"development_manifest.json");check_inputs(m);states=load_states();collect(m,states);analyze(m,states)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
