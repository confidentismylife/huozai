"""Independent read-only reconstruction of four fits, calibration and outcomes."""
from collections import Counter,defaultdict
from dataclasses import asdict,fields
from datetime import datetime
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/proposal_gain_factorial_development_v1"
NAMES=("R_linear","R_kernel","M_linear","M_kernel")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
def cls(g):return "harmful" if g < -1e-12 else "beneficial" if g > .001 else "neutral_or_small"
def weights(f):
    count=Counter(f);return np.array([1/(len(count)*count[k]) for k in f])
def mean_family(v,f):return float(np.mean([np.mean(np.asarray(v)[np.asarray(f)==k]) for k in sorted(set(f))]))


def gaussian(x,z,gamma):
    # Direct differences avoid depending on the production kernel implementation.
    return np.exp(-gamma*np.sum((x[:,None,:]-z[None,:,:])**2,axis=-1))


def predict(model,x):
    tx=np.atleast_2d((x-model["mean"])/model["std"])
    if str(model["kind"])=="linear":return (tx@model["coefficients"]+model["intercept"]).reshape(-1)
    k=gaussian(tx,model["train"],float(model["gamma"]))
    centered=k-(k@model["weights"])[:,None]-model["kernel_row_mean"]+model["kernel_grand_mean"]
    return (centered@model["coefficients"]+model["intercept"]).reshape(-1)


def main():
    assert not (OUT/"verification.json").exists(),"preserve prior verification"
    m=read(OUT/"development_manifest.json");s=read(OUT/"summary.json");am=read(OUT/"analysis_manifest.json")
    f=read(OUT/"frozen_models.json");gate=read(OUT/"frozen_gates.json")
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    for key,path in (("runner_sha256",ROOT/"scripts/run_proposal_gain_factorial.py"),("development_manifest_sha256",OUT/"development_manifest.json"),
        ("summary_sha256",OUT/"summary.json"),("frozen_models_sha256",OUT/"frozen_models.json"),("frozen_gates_sha256",OUT/"frozen_gates.json")):
        assert am[key]==sha(path)
    for p,h in am["csv_sha256"].items():assert sha(OUT/p)==h
    for p,h in am["cache_sha256"].items():assert sha(OUT/p)==h
    assert f["training_dataset_sha256"]==sha(OUT/"training_dataset.npz") and f["manifest_sha256"]==sha(OUT/"development_manifest.json")
    assert gate["frozen_models_sha256"]==sha(OUT/"frozen_models.json")
    calmark=read(OUT/"calibration_started_after_model_freeze.json");testmark=read(OUT/"test_started_after_gate_freeze.json")
    assert calmark["frozen_models_sha256"]==testmark["frozen_models_sha256"]==sha(OUT/"frozen_models.json")
    assert testmark["frozen_gates_sha256"]==sha(OUT/"frozen_gates.json")
    assert datetime.fromisoformat(f["frozen_at_utc"])<=datetime.fromisoformat(calmark["at_utc"])<=datetime.fromisoformat(gate["frozen_at_utc"])<=datetime.fromisoformat(testmark["at_utc"])
    allrecords=sum(m["banks"].values(),[])
    def signature(r):return json.dumps({k:v for k,v in r.items() if k!="scenario_id"},sort_keys=True)
    assert len({signature(r) for r in allrecords})==len(allrecords)==156
    sys.path.insert(0,str(ROOT/"src"));from fire_evac.dynamic_conflicts import DynamicConflictScenario
    keys={k.name for k in fields(DynamicConflictScenario)}
    def oldrecords(value):
        rows=[]
        if isinstance(value,dict):
            if {"scenario_id","source_branch","growth_rate","onset_step"}<=set(value):
                rows.append(asdict(DynamicConflictScenario(**{k:v for k,v in value.items() if k in keys})))
            else:
                for v in value.values():rows.extend(oldrecords(v))
        elif isinstance(value,list):
            for v in value:rows.extend(oldrecords(v))
        return rows
    old=[]
    for p in m["scenario_de_duplication_files"]:old.extend(oldrecords(read(p)))
    assert len(old)==m["old_scenario_records_checked"]
    assert not {signature(r) for r in allrecords}.intersection(signature(r) for r in old)
    assert not {r["scenario_id"] for r in allrecords}.intersection(r["scenario_id"] for r in old)
    previousseeds=set()
    for p in m["source_input_sha256"]:
        if Path(p).suffix==".npz" and Path(p).parent.name in ("states","training_states","audit_states","calibration_states","test_states","replication_states"):
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z:previousseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    previousseeds|={p+9000001 for p in list(previousseeds)}
    spaces=[{e["population_seed"] for e in m["seed_registry"]},{p for e in m["seed_registry"] for p in e["planning_seeds"]},{p for e in m["seed_registry"] for p in e["evaluation_seeds"]}]
    spaces += [{p+9000001 for p in part} for part in spaces.copy()]
    for i,part in enumerate(spaces):
        assert not part.intersection(previousseeds)
        assert all(not part.intersection(other) for other in spaces[i+1:])
    dataset=load(OUT/"training_dataset.npz");tx=dataset["features"];ty=dataset["targets"];fam=dataset["families"];w=weights(fam)
    assert tx.shape==(len(fam),66) and ty.shape==(len(fam),2) and np.all(tx[:,28:33]==0)
    forbidden={r["scenario_id"] for b in ("calibration","test") for r in m["banks"][b]}
    assert not set(fam).intersection(forbidden)
    assert len(fam)==f["training_states"] and len(set(fam))==f["training_families"]
    assert sorted(set(map(int,dataset["plans"])))==f["training_plan_indices"]
    assert {k:dict(Counter(cls(v) for v in ty[:,i])) for i,k in enumerate(("R","M"))}==f["training_target_class_counts"]
    models={};equation_residuals={}
    for name in NAMES:
        path=OUT/"models"/(name+".npz");assert sha(path)==f["model_sha256"][name];model=load(path);models[name]=model
        yy=ty[:,0 if name.startswith("R_") else 1];mu=w@tx;std=np.sqrt(w@((tx-mu)**2));std=np.where(std<.001,1.,std);xx=(tx-mu)/std
        close(model["mean"],mu);close(model["std"],std);close(model["intercept"],w@yy);close(model["weights"],w)
        if name.endswith("linear"):
            residual=xx.T@(w*(predict(model,tx)-yy))+.01*model["coefficients"]
            assert np.all(model["coefficients"][28:33]==0)
        else:
            close(model["train"],xx);active=max(1,int((np.var(xx,axis=0)>1e-12).sum()));close(model["gamma"],1/(2*active))
            k=gaussian(xx,xx,float(model["gamma"]));row=k@w;grand=w@row;close(model["kernel_row_mean"],row);close(model["kernel_grand_mean"],grand)
            center=k-row[:,None]-row[None,:]+grand;root=np.sqrt(w);dual=model["coefficients"]/root
            residual=(root[:,None]*center*root[None,:]+.01*np.eye(len(fam)))@dual-root*(yy-w@yy)
        error=float(np.max(np.abs(residual)));assert error<1e-10;equation_residuals[name]=error
        close(w@np.abs(predict(model,tx)-yy),f["fit_information"][name]["training_mae"])
    ridge=load(m["ridge_checkpoint"]);worth=load(m["worth_checkpoint"]);records={r["scenario_id"]:r for r in allrecords}
    ptable=table(OUT/"prediction_evidence.csv");plook={(r["state_id"],r["model"]):r for r in ptable}
    atable=table(OUT/"execution_arm_results.csv");alook={(r["state_id"],r["arm"]):r for r in atable}
    cache={};budget=Counter();eligible=defaultdict(set)
    for i,e in enumerate(m["seed_registry"]):
        state=e["state_id"];family=e["family"];d=load(OUT/(e["block"]+"_states")/(state+".npz"));p=int(d["proposal"]);record=records[family]
        assert int(d["population_seed"])==e["population_seed"]==710000000+i
        assert e["planning_seeds"]==[730000000+1000*i+f for f in range(6)]
        assert e["evaluation_seeds"]==([] if e["block"]=="train" else [770000000+1000*i+f for f in range(6)])
        for key in ("planning_seeds","evaluation_seeds"):np.testing.assert_array_equal(d[key],e[key])
        obs=d["observation"].astype(float).copy();obs[28:33]=0;oldpred=((obs-ridge["mean"])/ridge["std"])@ridge["coefficients"]+ridge["intercept"];oldpred[12]=0
        close(oldpred,d["predicted_gains"]);winner=min(range(17),key=lambda j:(-oldpred[j],j!=12,j));winner=winner if oldpred[winner]>.001 else 12;assert winner==p
        feature=np.r_[obs,np.eye(17)[p]];prob=np.array([0.,1.,0.])
        if p!=12:
            meta=np.r_[feature,oldpred[p],oldpred[p]-np.max(np.delete(oldpred,p))]
            logit=((meta-worth["mean"])/worth["std"])@worth["coefficient"]+worth["intercept"];prob=np.exp(logit-logit.max());prob/=prob.sum();eligible[e["block"]].add(family)
        close(prob,d["worth_probabilities"])
        risks=d["planning_risks"];actual=d["evaluation_outcomes"];assert risks.shape==(2,2,6,11) and d["supported"].all()
        assert actual.shape==(2,0 if e["block"]=="train" else 6,11)
        assert np.isfinite(risks).all() and np.isfinite(actual).all()
        loss=np.array([[np.mean(risks[w,a,:,0]) for a in range(2)] for w in range(2)]);wg=loss[:,0]-loss[:,1]
        r=float(loss[:,0].max()-loss[:,1].max());mm=float(wg.min());assert mm<=r+1e-12
        aa=float(np.mean(actual[0,:,0])-np.mean(actual[1,:,0])) if e["block"]!="train" else None
        if p==12:np.testing.assert_array_equal(risks[:,0],risks[:,1]);np.testing.assert_array_equal(actual[0],actual[1]);assert r==mm==0
        pr={n:float(predict(models[n],feature)[0]) if p!=12 else 0. for n in NAMES};pr.update({"frozen":float(oldpred[p]),"zero":0.})
        for n,pred in pr.items():
            row=plook[state,n];assert row["eligible"]==str(p!=12);assert row["seen_plan"]==str(p in f["training_plan_indices"])
            for key,value in (("R_gain_pp",r*100),("M_gain_pp",mm*100),("predicted_gain_pp",pred*100),("world0_gain_pp",wg[0]*100),("world1_gain_pp",wg[1]*100)):close(float(row[key]),value)
            assert row["predicted_class"]==cls(pred) and row["R_class"]==cls(r) and row["M_class"]==cls(mm)
            if aa is None:assert row["actual_class"]=="not_collected" and row["actual_gain_pp"]==""
            else:close(float(row["actual_gain_pp"]),aa*100);assert row["actual_class"]==cls(aa)
            for suffix,idx in (("first3",slice(0,3)),("last3",slice(3,6))):
                l=risks[:,:,idx,0].mean(axis=2);close(float(row["planning_"+suffix+"_R_pp"]),(l[:,0].max()-l[:,1].max())*100);close(float(row["planning_"+suffix+"_M_pp"]),(l[:,0]-l[:,1]).min()*100)
        if e["block"]=="train" and p!=12:
            j=np.flatnonzero(dataset["state_ids"]==state);assert len(j)==1;j=int(j[0]);close(dataset["features"][j],feature);close(dataset["targets"][j],[r,mm])
        if e["block"]!="train":
            def pass_gate(pred,margin,seen=True,full=True):return bool(p!=12 and seen and pred-margin>.001 and (not full or (prob[2]>=.8 and prob[0]<=.1)))
            choices={"pure_ppo_601":False,"point_proposal":p!=12,"frozen_gain":pass_gate(pr["frozen"],m["prior_margin"],full=False),"frozen_full":pass_gate(pr["frozen"],m["prior_margin"])}
            for name in NAMES:
                for suffix in ("_gain","_full"):choices[name+suffix]=pass_gate(pr[name],gate["gates"][name]["margin"],p in f["training_plan_indices"],suffix=="_full")
            for arm,choice in choices.items():
                row=alook[state,arm];assert row["override"]==str(choice) and row["executed_plan"]==m["candidate_names"][p if choice else 12]
                oc=actual[int(choice)].mean(axis=0)
                for j,metric in enumerate(m["outcome_fields"]):close(float(row[metric]),oc[j])
            for name in NAMES:assert not choices[name+"_full"] or choices[name+"_gain"]
        keys,offsets,trace=d["trace_keys"],d["trace_offsets"],d["execution_trace"]
        expected_keys=[] if e["block"]=="train" else [(plan,future) for plan in sorted({12,p}) for future in range(6)]
        np.testing.assert_array_equal(keys,np.asarray(expected_keys,int).reshape(-1,2));assert offsets[0]==0 and offsets[-1]==len(trace)
        if len(trace):
            actual_joint=np.where(trace[:,1]==8,8,np.where(trace[:,3]==2,-1,4*trace[:,3]+trace[:,1]%4));np.testing.assert_array_equal(trace[:,2],actual_joint)
            assert np.isin(trace[:,1],range(9)).all() and np.isin(trace[:,3],(0,1,2)).all()
            for j,(plan,future) in enumerate(keys):
                tr=trace[offsets[j]:offsets[j+1]];oc=actual[0 if plan==12 else 1,future];assert len(tr)==oc[5]
                np.testing.assert_array_equal(tr[:,0],np.arange(len(tr)));close(tr[:,4].mean(),oc[6]);close((tr[:,1]==8).sum(),oc[7]);close(tr[:,5].mean(),oc[8]);close(tr[0,1],oc[9]);close(tr[0,2],oc[10])
        assert int(d["planning_rollouts"])==(12 if p==12 else 24)
        assert int(d["evaluation_rollouts"])==(0 if e["block"]=="train" else 6 if p==12 else 12)
        budget["planning_rollouts"]+=int(d["planning_rollouts"]);budget["evaluation_rollouts"]+=int(d["evaluation_rollouts"]);budget["actual_trace_steps"]+=len(trace);budget["hold_block_steps"]+=int(trace[:,5].sum());budget["neutral_positive_release_steps"]+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
        cache[state]={"block":e["block"],"family":family,"R":r,"M":mm,"A":aa,"pred":pr,"plan":p,"type":record["conflict_type"]}
    assert len(cache)==312 and len(plook)==1872 and len(alook)==1440
    assert dict(budget)==s["budget"] and budget["planning_rollouts"]+budget["evaluation_rollouts"]==s["total_unique_full_rollouts"]<=8928
    assert {b:len(eligible[b]) for b in m["banks"]}==s["eligible_family_counts"]
    calibration=[c for c in cache.values() if c["block"]=="calibration" and c["plan"]!=12]
    assert len(calibration)==gate["calibration_eligible_states"] and len(eligible["calibration"])==gate["calibration_eligible_families"]
    for name in NAMES:
        values={family:max(c["pred"][name]-c["R" if name.startswith("R_") else "M"] for c in calibration if c["family"]==family) for family in sorted(eligible["calibration"])}
        assert values.keys()==gate["gates"][name]["family_max_optimism"].keys()
        for k,value in values.items():close(value,gate["gates"][name]["family_max_optimism"][k])
        q=float(np.quantile(list(values.values()),.8,method="higher")) if values else None;margin=max(m["prior_margin"],.001,q if q is not None else .001)
        close(margin,gate["gates"][name]["margin"]);close(margin*100,s["margins_pp"][name]);assert margin>=m["prior_margin"]
    for q in s["prediction_quality"]:
        er=[r for r in ptable if r["block"]==q["block"] and r["model"]==q["model"] and r["eligible"]=="True" and
            (q["scope"]=="all_proposals" or (r["seen_plan"]=="True")== (q["scope"]=="seen_training_plans"))]
        families=[r["family"] for r in er];assert len(er)==q["states"] and len(set(families))==q["families"]
        pred=np.array([float(r["predicted_gain_pp"]) for r in er])
        for target in ("R","M")+(() if q["block"]=="train" else ("actual",)):
            truth=np.array([float(r[target+"_gain_pp"]) for r in er]);close(mean_family(abs(pred-truth),families),q[target+"_mae_pp"]);close(mean_family(pred-truth,families),q[target+"_bias_pp"])
        assert sum(r["predicted_class"]=="beneficial" for r in er)==q["predicted_beneficial_states"]
        if q["block"]!="train":
            assert dict(Counter(r["actual_class"] for r in er))==q["actual_classes"]
            true=sum(r["actual_class"]=="beneficial" for r in er);recalled=sum(r["actual_class"]==r["predicted_class"]=="beneficial" for r in er)
            assert true==q["actual_beneficial_states"] and recalled==q["beneficial_recalled"]
            assert (None if true==0 else recalled/true)==q["beneficial_recall"]
            assert sum(r["actual_class"]=="harmful" and r["predicted_class"]=="beneficial" for r in er)==q["predicted_benefit_actual_harm_states"]
    for row in s["gate_summary"]:
        chosen=[r for r in atable if r["block"]==row["block"] and r["arm"]==row["arm"] and r["override"]=="True"]
        assert len(chosen)==row["overrides"] and dict(Counter(r["proposal_actual_class"] for r in chosen))==row["actual_classes"]
    testfamilies=sorted(r["scenario_id"] for r in m["banks"]["test"]);test_types={r["scenario_id"]:r["conflict_type"] for r in m["banks"]["test"]}
    for row in s["test_arm_summary"]:
        er=[r for r in atable if r["block"]=="test" and r["arm"]==row["arm"]]
        for metric in m["outcome_fields"][:6]:close(mean_family([float(r[metric]) for r in er],[r["family"] for r in er]),row[metric])
    for effect in s["test_paired_effects"]:
        vals=[]
        for family in testfamilies:
            er=[r for r in atable if r["block"]=="test" and r["arm"]==effect["treatment"] and r["family"]==family]
            vals.append(np.mean([float(r["final_incapacitation"])-float(alook[r["state_id"],effect["control"]]["final_incapacitation"]) for r in er])*100)
        vals=np.array(vals);rng=np.random.default_rng(m["bootstrap_seed"]);boot=np.zeros(2000)
        for kind in sorted(set(test_types.values())):
            part=vals[[test_types[family]==kind for family in testfamilies]];boot+=part[rng.integers(len(part),size=(2000,len(part)))].sum(axis=1)
        ci=np.quantile(boot/len(vals),[.025,.975]);close(vals.mean(),effect["delta_incap_pp"]);close(ci,[effect["exploratory_ci_low_pp"],effect["exploratory_ci_high_pp"]])
        assert int((vals< -1e-10).sum())==effect["improved_families"] and int((vals>1e-10).sum())==effect["worsened_families"]
    assert s["primary_paired_effect"]==next(r for r in s["test_paired_effects"] if r["treatment"]=="M_kernel_full" and r["control"]=="frozen_full")
    testprops=[c for c in cache.values() if c["block"]=="test" and c["plan"]!=12]
    assert dict(Counter(cls(c["A"]) for c in testprops))==s["test_proposal_actual_classes"]
    assert sum(cls(c["R"])!=cls(c["M"]) for c in testprops)==s["test_R_M_class_disagreements"]
    assert sum(cls(c["R"])!=cls(c["A"]) for c in testprops)==s["test_R_actual_class_disagreements"]
    assert sum(cls(c["M"])!=cls(c["A"]) for c in testprops)==s["test_M_actual_class_disagreements"]
    qlookup={q["model"]:q for q in s["prediction_quality"] if q["block"]=="test" and q["scope"]=="all_proposals"}
    for delta in s["factor_mae_differences"]:close(delta["delta_mae_pp"],qlookup[delta["treatment"]][delta["target_evaluation"]+"_mae_pp"]-qlookup[delta["control"]][delta["target_evaluation"]+"_mae_pp"])
    result={"all_checks_passed":True,"source_inputs_unchanged":len(m["source_input_sha256"]),"states_verified":312,
        "prediction_rows_verified":1872,"execution_arm_rows_verified":1440,"full_rollouts_verified":s["total_unique_full_rollouts"],
        "actual_trace_steps_verified":budget["actual_trace_steps"],"fit_normal_equation_residuals":equation_residuals,
        "all_four_fits_verified_without_refitting":True,"family_banks_and_seeds_disjoint_verified":True,
        "training_only_inputs_and_calibration_only_margins_verified":True,"models_then_gates_then_test_freeze_verified":True,
        "full_gate_is_subset_of_gain_gate_verified":True,"paired_family_effects_and_bootstrap_verified":True,
        "no_new_simulations_or_fits":True,"verifier_sha256":sha(__file__),"analysis_manifest_sha256":sha(OUT/"analysis_manifest.json")}
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
