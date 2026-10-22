"""Four fixed proposal-gain fits with disjoint family banks and bounded rollouts."""
from collections import Counter,defaultdict
from dataclasses import asdict,fields
from datetime import datetime,timezone
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/proposal_gain_factorial_development_v1"
PRIOR=ROOT/"outputs/gain_target_alignment_diagnostic_development_v1"
WORTH=ROOT/"outputs/override_worth_evidence_development_v1"
STATUS=ROOT/"outputs/third_round_development_status.json"
PROTOCOL=ROOT/"reports/收益标签与轻量模型四组对照开发方案_2026-10-03.md"
sys.path.insert(0,str(ROOT/"src"))
from fire_evac.dynamic_conflicts import DynamicConflictScenario,make_domain_randomized_trust_bank
from proposal_gain_models import inputs,targets,fit,PairGainModel,calibrated_margin,accepted
from consistent_gain_policy import GainRidge
from override_worth_model import proposal,meta_features,WorthModel

NAMES=("R_linear","R_kernel","M_linear","M_kernel")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,b):Path(p).write_text(json.dumps(b,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csv_write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cls(g):return "harmful" if g < -1e-12 else "beneficial" if g > .001 else "neutral_or_small"
def signature(r):return json.dumps({k:v for k,v in r.items() if k!="scenario_id"},sort_keys=True)
def stamp():return datetime.now(timezone.utc).isoformat()


def scenario_records(value):
    out=[];keys={f.name for f in fields(DynamicConflictScenario)}
    if isinstance(value,dict):
        if {"scenario_id","source_branch","growth_rate","onset_step"}<=set(value):
            try:out.append(asdict(DynamicConflictScenario(**{k:v for k,v in value.items() if k in keys})))
            except (ValueError,TypeError):pass
        else:
            for v in value.values():out+=scenario_records(v)
    elif isinstance(value,list):
        for v in value:out+=scenario_records(v)
    return out


def prepare():
    assert not OUT.exists(),"preserve existing stage"
    old=read(PRIOR/"development_manifest.json");completion=read(PRIOR/"completion_manifest.json")
    for p,h in completion["files"].items():assert sha(ROOT/p)==h,p
    for p,h in old["source_input_sha256"].items():assert sha(p)==h,p
    bankfiles=sorted(set((ROOT/"outputs").rglob("development_manifest.json"))|set((ROOT/"outputs").rglob("*scenarios.json")))
    records=[]
    for p in bankfiles:records+=scenario_records(read(p))
    signatures={signature(r) for r in records};old_ids={r["scenario_id"] for r in records}
    banks={block:[asdict(s) for s in make_domain_randomized_trust_bank(n,seed,prefix)] for block,n,seed,prefix in
        (("train",96,20262401,"GTRN"),("calibration",24,20262402,"GCAL"),("test",36,20262403,"GTEST"))}
    newrecords=sum(banks.values(),[]);assert len({signature(r) for r in newrecords})==156
    assert not {signature(r) for r in newrecords}.intersection(signatures)
    assert not {r["scenario_id"] for r in newrecords}.intersection(old_ids)
    registry=[]
    for block,bank in banks.items():
        for record in bank:
            for r in range(2):
                i=len(registry)
                registry.append({"block":block,"state_id":record["scenario_id"]+f"_{r:02d}","family":record["scenario_id"],"rep":r,
                    "population_seed":710000000+i,"planning_seeds":[730000000+1000*i+f for f in range(6)],
                    "evaluation_seeds":[] if block=="train" else [770000000+1000*i+f for f in range(6)]})
    previousseeds=set()
    for p in old["source_input_sha256"]:
        if Path(p).suffix==".npz" and Path(p).parent.name in ("states","training_states","audit_states","calibration_states","test_states","replication_states"):
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z:previousseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    # Include the immediately preceding replicated futures (not all are inherited in its input list).
    for entry in old["seed_registry"]:previousseeds.add(entry["population_seed"]);previousseeds.update(entry["future_seeds"])
    previousseeds|={s+9000001 for s in list(previousseeds)}
    ns=[{e["population_seed"] for e in registry},{s for e in registry for s in e["planning_seeds"]},{s for e in registry for s in e["evaluation_seeds"]}]
    ns += [{s+9000001 for s in part} for part in ns.copy()]
    for i,part in enumerate(ns):
        assert not part.intersection(previousseeds)
        assert all(not part.intersection(other) for other in ns[i+1:])
    OUT.mkdir();(OUT/"previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes());sources=old["source_input_sha256"].copy()
    for p,h in completion["files"].items():
        path=OUT/"previous_status_exact_bytes.json" if (ROOT/p).resolve()==STATUS else ROOT/p;sources[str(path.resolve())]=h
    frozen=(PROTOCOL,Path(__file__),ROOT/"scripts/proposal_gain_models.py",ROOT/"tests/test_proposal_gain_models.py",PRIOR/"completion_manifest.json")
    for p in tuple(bankfiles)+frozen:sources[str(p.resolve())]=sha(p)
    (OUT/"source_snapshot").mkdir()
    for p in frozen[:-1]:shutil.copy2(p,OUT/"source_snapshot"/p.name)
    worthmanifest=read(WORTH/"development_manifest.json")
    m={"role":"new_same_generator_family_disjoint_target_by_model_development_not_confirmation","created_at_utc":stamp(),
        "source_input_sha256":sources,"previous_status_sha256":sha(STATUS),"prior_completion_sha256":sha(PRIOR/"completion_manifest.json"),
        "banks":banks,"seed_registry":registry,"scenario_de_duplication_files":[str(p) for p in bankfiles],
        "old_scenario_records_checked":len(records),"old_unique_scenario_signatures":len(signatures),
        "ridge_checkpoint":worthmanifest["ridge_checkpoint"],"worth_checkpoint":str(WORTH/"worth_model.npz"),
        "frozen_ppo_models":worthmanifest["frozen_ppo_models"],"candidate_names":worthmanifest["candidate_names"],
        "outcome_fields":worthmanifest["outcome_fields"],"trace_fields":worthmanifest["trace_fields"],
        "model_names":list(NAMES),"prior_margin":read(WORTH/"frozen_conditional_gate.json")["conditional_margin"],
        "maximum_full_rollouts":9000,"maximum_by_design":8928,"planning_futures":6,"independent_actual_futures":6,
        "new_fits":4,"new_deep_network_training":0,"new_ppo_training":0,"new_llm_calls":0,"qualification_thresholds_changed":False,
        "seed_namespaces_disjoint":True,"feature_dimension":66,"regularization":.01,"bootstrap_seed":20262404,"bootstrap_draws":2000,
        "fixed_primary":{"treatment":"M_kernel_full","control":"frozen_full","metric":"final_incapacitation"}}
    write(OUT/"development_manifest.json",m)
    print(json.dumps({"prepared":True,"bank_families":{k:len(v) for k,v in banks.items()},"source_inputs":len(sources),"maximum_rollouts":9000},ensure_ascii=False),flush=True)


def check(m):
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p


def candidates_load(m):
    import torch
    from stable_baselines3 import PPO
    from fire_evac.source_world_planning import PLANS
    from fire_evac.ppo_reference_planning import FrozenPPOContinuation,policy_parameter_sha256
    torch.set_num_threads(1);candidates=list(PLANS)
    for r in m["frozen_ppo_models"]:
        model=PPO.load(r["path"],device="cpu");assert policy_parameter_sha256(model)==r["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(r["training_seed"],model))
    assert [p.name for p in candidates]==m["candidate_names"]
    return candidates


def collect(block,m,candidates,ridge,worth):
    from fire_evac.source_world_planning import source_world_scenarios,compatible_source_worlds
    from fire_evac.ppo_reference_planning import evaluate_continuation
    from run_source_world_planning import make_env,vector
    records={r["scenario_id"]:r for r in m["banks"][block]};scenarios=[DynamicConflictScenario(**r) for r in records.values()]
    worlds=[make_env(scenarios),make_env(scenarios)];actual=make_env(scenarios);folder=OUT/(block+"_states");folder.mkdir(exist_ok=True)
    entries=[e for e in m["seed_registry"] if e["block"]==block];start=time.monotonic()
    try:
        for i,e in enumerate(entries):
            path=folder/(e["state_id"]+".npz")
            if path.exists():continue
            scenario=DynamicConflictScenario(**records[e["family"]]);x,_=actual.reset(seed=e["population_seed"],options={"scenario":scenario})
            g=ridge.predict(x);p=proposal(g);prob=np.array([0.,1.,0.]) if p==12 else worth.probabilities(meta_features(x,g,p))
            support,errors,mismatch=compatible_source_worlds(source_world_scenarios(scenario),x,0);assert support.all() and not mismatch
            risks=np.empty((2,2,6,11));cached={}
            for w,hypothesis in enumerate(source_world_scenarios(scenario)):
                wx,_=worlds[w].reset(seed=e["population_seed"],options={"scenario":hypothesis});np.testing.assert_array_equal(wx,x)
                for a,plan in enumerate((12,p)):
                    for f,seed in enumerate(e["planning_seeds"]):
                        key=(w,plan,f)
                        if key not in cached:cached[key]=vector(evaluate_continuation(worlds[w],wx,candidates[plan],seed))
                        risks[w,a,f]=cached[key]
            outcomes=np.empty((2,len(e["evaluation_seeds"]),11));executed={};traces={}
            for a,plan in enumerate((12,p)):
                for f,seed in enumerate(e["evaluation_seeds"]):
                    key=(plan,f)
                    if key not in executed:
                        trace=[];executed[key]=vector(evaluate_continuation(actual,x,candidates[plan],seed,trace));traces[key]=np.asarray(trace,float)
                    outcomes[a,f]=executed[key]
            keys=sorted(traces);trace=np.concatenate([traces[k] for k in keys]) if keys else np.empty((0,6));tmp=path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,observation=x,predicted_gains=g,proposal=np.asarray(p),worth_probabilities=prob,
                population_seed=np.asarray(e["population_seed"]),planning_seeds=np.asarray(e["planning_seeds"]),evaluation_seeds=np.asarray(e["evaluation_seeds"],int),
                supported=support,world_errors=errors,planning_risks=risks,evaluation_outcomes=outcomes,
                planning_rollouts=np.asarray(len(cached)),evaluation_rollouts=np.asarray(len(executed)),
                trace_keys=np.asarray(keys,int).reshape(-1,2),trace_offsets=np.cumsum([0]+[len(traces[k]) for k in keys]),execution_trace=trace)
            tmp.replace(path)
            if (i+1)%12==0 or i+1==len(entries):print(f"{block} {i+1}/{len(entries)} {time.monotonic()-start:.1f}s",flush=True)
    finally:
        actual.close()
        for env in worlds:env.close()


def data_rows(block,m):
    records={r["scenario_id"]:r for r in m["banks"][block]};rows=[]
    for file in sorted((OUT/(block+"_states")).glob("*.npz")):
        with np.load(file,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        family=file.stem.rsplit("_",1)[0];r={"state_id":file.stem,"family":family,"scenario":records[family],"data":d,
            "plan":int(d["proposal"]),"targets":targets(d["planning_risks"],d["supported"])}
        r["features"]=inputs(d["observation"],r["plan"]);rows.append(r)
    assert len(rows)==2*len(m["banks"][block]);return rows


def train(m):
    freeze=OUT/"frozen_models.json"
    if freeze.exists():
        f=read(freeze)
        for name,h in f["model_sha256"].items():assert sha(OUT/"models"/(name+".npz"))==h
        return {name:PairGainModel.load(OUT/"models"/(name+".npz")) for name in NAMES},f
    assert not (OUT/"models").exists(),"preserve partial fits"
    rows=[r for r in data_rows("train",m) if r["plan"]!=12];assert rows,"no training proposals"
    x=np.array([r["features"] for r in rows]);families=np.array([r["family"] for r in rows]);y=np.array([r["targets"] for r in rows])
    np.savez_compressed(OUT/"training_dataset.npz",features=x,targets=y,families=families,state_ids=np.array([r["state_id"] for r in rows]),plans=np.array([r["plan"] for r in rows]))
    (OUT/"models").mkdir();models={};info={}
    forbidden=[r["scenario_id"] for b in ("calibration","test") for r in m["banks"][b]]
    for name in NAMES:
        target,kind=name.split("_");model,fitinfo=fit(x,y[:,0 if target=="R" else 1],families,kind,forbidden)
        model.save(OUT/"models"/(name+".npz"));models[name]=model;info[name]=fitinfo
    f={"frozen_at_utc":stamp(),"manifest_sha256":sha(OUT/"development_manifest.json"),"training_dataset_sha256":sha(OUT/"training_dataset.npz"),
        "training_states":len(rows),"training_families":len(set(families)),"training_plan_indices":sorted(set(r["plan"] for r in rows)),
        "training_target_class_counts":{t:dict(Counter(cls(g) for g in y[:,j])) for j,t in enumerate(("R","M"))},
        "fit_information":info,"model_sha256":{n:sha(OUT/"models"/(n+".npz")) for n in NAMES},
        "new_fits":4,"no_calibration_or_test_in_fit":True}
    write(freeze,f);print(json.dumps({"models_frozen":True,"training_states":len(rows),"training_families":len(set(families)),"classes":f["training_target_class_counts"]},ensure_ascii=False),flush=True)
    return models,f


def calibrate(m,models,f):
    path=OUT/"frozen_gates.json"
    if path.exists():return read(path)
    rows=[r for r in data_rows("calibration",m) if r["plan"]!=12];families=np.array([r["family"] for r in rows]);gates={}
    for name in NAMES:
        j=0 if name.startswith("R_") else 1;pred=np.array([models[name].predict(r["features"]) for r in rows]);emp=np.array([r["targets"][j] for r in rows])
        margin,values,q=calibrated_margin(pred,emp,families,m["prior_margin"])
        gates[name]={"margin":margin,"family_max_optimism":values,"new_quantile":q}
    body={"frozen_at_utc":stamp(),"frozen_models_sha256":sha(OUT/"frozen_models.json"),"calibration_eligible_states":len(rows),
        "calibration_eligible_families":len(set(families)),"prior_margin":m["prior_margin"],"gates":gates,
        "actual_outcomes_used":False,"worth_threshold":.8,"harm_threshold":.1,"minimum_gain":.001,
        "unseen_training_plan_abstains":True,"margin_never_reduced":True}
    write(path,body);print(json.dumps({"gates_frozen":True,"margins_pp":{n:g["margin"]*100 for n,g in gates.items()}},ensure_ascii=False),flush=True);return body


def equal_family_mean(values,families):
    return float(np.mean([np.mean(np.asarray(values)[np.asarray(families)==f]) for f in sorted(set(families))]))


def interval(values,kinds,seed):
    values=np.asarray(values);kinds=np.asarray(kinds);rng=np.random.default_rng(seed);total=np.zeros(2000)
    for kind in sorted(set(kinds)):
        part=values[kinds==kind];total+=part[rng.integers(len(part),size=(2000,len(part)))].sum(axis=1)
    return np.quantile(total/len(values),[.025,.975])


def analyze(m,models,f,gates):
    assert not (OUT/"analysis_manifest.json").exists(),"preserve completed analysis"
    evidence=[];decisions=[];budget=Counter();quality=[];gate_summary=[];testfamilies={r["scenario_id"]:r for r in m["banks"]["test"]};byblock={}
    arms=("pure_ppo_601","point_proposal","frozen_gain","frozen_full")+tuple(n+suffix for n in NAMES for suffix in ("_gain","_full"))
    for block in ("train","calibration","test"):
        rows=data_rows(block,m);byblock[block]=rows
        for r in rows:
            d=r["data"];p=r["plan"];eligible=p!=12;actual=float((d["evaluation_outcomes"][0,:,0]-d["evaluation_outcomes"][1,:,0]).mean()) if block!="train" else None
            means=d["planning_risks"].mean(axis=2)[:,:,0];wg=means[:,0]-means[:,1]
            base={"block":block,"state_id":r["state_id"],"family":r["family"],"conflict_type":r["scenario"]["conflict_type"],"eligible":eligible,
                "proposal":m["candidate_names"][p],"proposal_index":p,"seen_plan":p in f["training_plan_indices"],
                "R_gain_pp":r["targets"][0]*100,"M_gain_pp":r["targets"][1]*100,"R_class":cls(r["targets"][0]),"M_class":cls(r["targets"][1]),
                "actual_gain_pp":None if actual is None else actual*100,"actual_class":"not_collected" if actual is None else cls(actual),
                "world0_gain_pp":wg[0]*100,"world1_gain_pp":wg[1]*100,"worth_score":float(d["worth_probabilities"][2]),
                "planning_first3_R_pp":float(targets(d["planning_risks"][:,:,:3],d["supported"])[0]*100),
                "planning_last3_R_pp":float(targets(d["planning_risks"][:,:,3:],d["supported"])[0]*100),
                "planning_first3_M_pp":float(targets(d["planning_risks"][:,:,:3],d["supported"])[1]*100),
                "planning_last3_M_pp":float(targets(d["planning_risks"][:,:,3:],d["supported"])[1]*100)}
            preds={n:float(models[n].predict(r["features"])) if eligible else 0. for n in NAMES}
            preds.update({"frozen":float(d["predicted_gains"][p]),"zero":0.})
            for name,pred in preds.items():evidence.append({**base,"model":name,"predicted_gain_pp":pred*100,"predicted_class":cls(pred)})
            if block!="train":
                decisions_for={"pure_ppo_601":False,"point_proposal":eligible,
                    "frozen_gain":eligible and accepted(preds["frozen"],m["prior_margin"],d["worth_probabilities"],full=False),
                    "frozen_full":eligible and accepted(preds["frozen"],m["prior_margin"],d["worth_probabilities"])}
                for name in NAMES:
                    for suffix in ("_gain","_full"):
                        decisions_for[name+suffix]=eligible and accepted(preds[name],gates["gates"][name]["margin"],d["worth_probabilities"],base["seen_plan"],suffix=="_full")
                for arm in arms:
                    choice=bool(decisions_for[arm]);a=1 if choice else 0;oc=d["evaluation_outcomes"][a].mean(axis=0)
                    decisions.append({"block":block,"state_id":r["state_id"],"family":r["family"],"conflict_type":base["conflict_type"],"arm":arm,
                        "override":choice,"executed_plan":m["candidate_names"][p if choice else 12],"proposal_actual_class":base["actual_class"],
                        **{metric:float(oc[j]) for j,metric in enumerate(m["outcome_fields"])}})
            budget["planning_rollouts"]+=int(d["planning_rollouts"]);budget["evaluation_rollouts"]+=int(d["evaluation_rollouts"])
            tr=d["execution_trace"];budget["actual_trace_steps"]+=len(tr);budget["hold_block_steps"]+=int(tr[:,5].sum());budget["neutral_positive_release_steps"]+=int(((tr[:,3]==2)&(tr[:,4]>0)).sum())
        for name in tuple(NAMES)+("frozen","zero"):
            erows=[e for e in evidence if e["block"]==block and e["model"]==name and e["eligible"]]
            for scope in ("all_proposals","seen_training_plans","unseen_training_plans"):
                er=[e for e in erows if scope=="all_proposals" or e["seen_plan"]==(scope=="seen_training_plans")]
                if not er:continue
                fam=[e["family"] for e in er];q={"block":block,"model":name,"scope":scope,"states":len(er),"families":len(set(fam))}
                for target in ("R","M")+(() if block=="train" else ("actual",)):
                    pred=np.array([e["predicted_gain_pp"] for e in er]);truth=np.array([e[target+"_gain_pp"] for e in er])
                    q[target+"_mae_pp"]=equal_family_mean(abs(pred-truth),fam);q[target+"_bias_pp"]=equal_family_mean(pred-truth,fam)
                q["predicted_beneficial_states"]=sum(e["predicted_class"]=="beneficial" for e in er)
                if block!="train":
                    true=sum(e["actual_class"]=="beneficial" for e in er);matched=sum(e["actual_class"]==e["predicted_class"]=="beneficial" for e in er)
                    q.update({"actual_classes":dict(Counter(e["actual_class"] for e in er)),"actual_beneficial_states":true,"beneficial_recalled":matched,
                        "beneficial_recall":matched/true if true else None,"predicted_benefit_actual_harm_states":sum(e["predicted_class"]=="beneficial" and e["actual_class"]=="harmful" for e in er)})
                quality.append(q)
    csv_write(OUT/"prediction_evidence.csv",evidence);csv_write(OUT/"execution_arm_results.csv",decisions)
    for block in ("calibration","test"):
        for arm in arms:
            chosen=[d for d in decisions if d["block"]==block and d["arm"]==arm and d["override"]]
            gate_summary.append({"block":block,"arm":arm,"overrides":len(chosen),"actual_classes":dict(Counter(d["proposal_actual_class"] for d in chosen))})
    effects=[];arm_summary=[];dtest=[d for d in decisions if d["block"]=="test"];lookup={(d["state_id"],d["arm"]):d for d in dtest}
    controls=("pure_ppo_601","frozen_full")
    for arm in arms:
        ds=[d for d in dtest if d["arm"]==arm];families=[d["family"] for d in ds]
        arm_summary.append({"arm":arm,"states":len(ds),"families":len(set(families)),**{metric:equal_family_mean([d[metric] for d in ds],families) for metric in m["outcome_fields"][:6]}})
        for control in controls:
            vals=[];kinds=[]
            for family in sorted(testfamilies):
                famds=[d for d in ds if d["family"]==family]
                vals.append(float(np.mean([d["final_incapacitation"]-lookup[(d["state_id"],control)]["final_incapacitation"] for d in famds]))*100)
                kinds.append(testfamilies[family]["conflict_type"])
            vals=np.array(vals);ci=interval(vals,kinds,m["bootstrap_seed"])
            effects.append({"treatment":arm,"control":control,"families":len(vals),"delta_incap_pp":float(vals.mean()),
                "exploratory_ci_low_pp":float(ci[0]),"exploratory_ci_high_pp":float(ci[1]),
                "improved_families":int((vals< -1e-10).sum()),"worsened_families":int((vals>1e-10).sum()),"tied_families":int((abs(vals)<=1e-10).sum())})
    csv_write(OUT/"test_paired_effects.csv",effects);csv_write(OUT/"test_arm_summary.csv",arm_summary)
    alltest=[e for e in evidence if e["block"]=="test" and e["model"]=="zero" and e["eligible"]]
    metrics={q["model"]:q for q in quality if q["block"]=="test" and q["scope"]=="all_proposals"}
    factor_differences=[]
    for target in ("R","M","actual"):
        for a,b in (("R_kernel","R_linear"),("M_kernel","M_linear"),("M_linear","R_linear"),("M_kernel","R_kernel")):
            if a in metrics and b in metrics:factor_differences.append({"target_evaluation":target,"treatment":a,"control":b,"delta_mae_pp":metrics[a][target+"_mae_pp"]-metrics[b][target+"_mae_pp"]})
    total=budget["planning_rollouts"]+budget["evaluation_rollouts"];assert total<=m["maximum_by_design"]<=9000
    summary={"role":m["role"],"new_fits":4,"new_ppo_training":0,"new_llm_calls":0,"new_deep_network_training":0,
        "family_counts":{b:len(m["banks"][b]) for b in m["banks"]},"training_support":f,
        "eligible_family_counts":{b:len({r["family"] for r in rs if r["plan"]!=12}) for b,rs in byblock.items()},
        "total_unique_full_rollouts":total,"budget":dict(budget),"prediction_quality":quality,"gate_summary":gate_summary,
        "test_paired_effects":effects,"test_arm_summary":arm_summary,"factor_mae_differences":factor_differences,
        "test_proposal_actual_classes":dict(Counter(e["actual_class"] for e in alltest)),
        "test_R_M_class_disagreements":sum(e["R_class"]!=e["M_class"] for e in alltest),
        "test_R_actual_class_disagreements":sum(e["R_class"]!=e["actual_class"] for e in alltest),
        "test_M_actual_class_disagreements":sum(e["M_class"]!=e["actual_class"] for e in alltest),
        "primary_paired_effect":next(e for e in effects if e["treatment"]=="M_kernel_full" and e["control"]=="frozen_full"),
        "margins_pp":{n:gates["gates"][n]["margin"]*100 for n in NAMES},"freeze_allowed":False,"hard_shield_qualification_passed":False,
        "limits":["Same-generator family-disjoint development comparison, not final confirmation or powered safety evaluation.",
            "Only old GainRidge nominees compared with 601; no claim about omitted superior plans or full-library optimality.",
            "Four deterministic fits differ in target and architecture; no training-seed robustness or parameter search.",
            "R and M have different semantics; six planning and actual futures are empirical labels, not expected truth or guarantees.",
            "Original classifier retained with old meta-features; unseen training plans abstain for all new models.",
            "Intervals condition on fixed policies and same generator; two population starts and futures are not independent families.",
            "No new LLM calls, PPO training, map/behavior/noise transfer or real-fire validation."]}
    write(OUT/"summary.json",summary);check(m)
    write(OUT/"analysis_manifest.json",{"runner_sha256":sha(__file__),"development_manifest_sha256":sha(OUT/"development_manifest.json"),
        "summary_sha256":sha(OUT/"summary.json"),"frozen_models_sha256":sha(OUT/"frozen_models.json"),"frozen_gates_sha256":sha(OUT/"frozen_gates.json"),
        "csv_sha256":{p.name:sha(p) for p in sorted(OUT.glob("*.csv"))},
        "cache_sha256":{str(p.relative_to(OUT)):sha(p) for b in m["banks"] for p in sorted((OUT/(b+"_states")).glob("*.npz"))}})
    print(json.dumps({"completed_analysis":True,"full_rollouts":total,"eligible_families":summary["eligible_family_counts"],"primary":summary["primary_paired_effect"]},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--run",action="store_true");args=parser.parse_args()
    if args.prepare_only:prepare();return
    if not args.run:parser.error("choose --prepare-only or --run")
    m=read(OUT/"development_manifest.json");check(m);candidates=candidates_load(m);ridge=GainRidge.load(m["ridge_checkpoint"]);worth=WorthModel.load(m["worth_checkpoint"])
    collect("train",m,candidates,ridge,worth);models,f=train(m)
    mark=OUT/"calibration_started_after_model_freeze.json"
    if not mark.exists():write(mark,{"at_utc":stamp(),"frozen_models_sha256":sha(OUT/"frozen_models.json")})
    collect("calibration",m,candidates,ridge,worth);gates=calibrate(m,models,f)
    mark=OUT/"test_started_after_gate_freeze.json"
    if not mark.exists():write(mark,{"at_utc":stamp(),"frozen_models_sha256":sha(OUT/"frozen_models.json"),"frozen_gates_sha256":sha(OUT/"frozen_gates.json")})
    collect("test",m,candidates,ridge,worth)
    from fire_evac.ppo_reference_planning import policy_parameter_sha256
    for record,actor in zip(m["frozen_ppo_models"],candidates[12:]):assert policy_parameter_sha256(actor.model)==record["parameter_sha256"]
    analyze(m,models,f,gates)


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
