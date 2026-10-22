"""Independently check fresh banks, frozen multi-head equations and execution."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT / "outputs/five_ppo_fresh_family_student_development_v1"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-9)
def cls(g):return "beneficial" if g>.001 else "harmful" if g < -1e-12 else "neutral_or_small"
def select(g,margin=0.):
    scores=[0. if j==0 else float(g[j])-margin for j in range(5)];best=max(scores);p=min(j for j in range(5) if best-scores[j]<=1e-12)
    return p if scores[p]>.001 else 0
def pooled(r,s):
    loss=np.array([[np.mean(r[:,w,k,:,0]) for k in range(5)] for w in range(2)])
    active=loss[s];R=np.array([max(active[:,0])-max(active[:,k]) for k in range(5)])
    M=np.array([min(active[w,0]-active[w,k] for w in range(len(active))) for k in range(5)])
    R[0]=M[0]=0;return np.array([R,M]),loss
def gram(x,z,gamma):return np.exp(-gamma*np.square(x[:,None,:]-z[None,:,:]).sum(axis=2))
def prediction(model,x):
    x=np.atleast_2d(np.asarray(x,float)).copy();x[:,28:33]=0;tx=(x-model["mean"])/model["std"]
    if str(model["kind"])=="linear":v=tx@model["coefficient"]+model["intercept"]
    else:
        k=gram(tx,model["train"],float(model["gamma"]));center=k-(k@model["weights"])[:,None]-model["kernel_row_mean"][None,:]+float(model["kernel_grand_mean"])
        v=center@model["coefficient"]+model["intercept"]
    return np.c_[np.zeros(len(x)),v]
def interval(values,kinds,seed):
    rng=np.random.default_rng(seed);v=np.array(values);kinds=np.array(kinds);draw=np.zeros(2000)
    for k in sorted(set(kinds)):
        z=v[kinds==k];draw+=z[rng.integers(len(z),size=(2000,len(z)))].sum(axis=1)
    return np.quantile(draw/len(v),[.025,.975])


def main():
    assert not (OUT / "verification.json").exists()
    m,s,am,collection,frozen,gates=(read(OUT / p) for p in ("development_manifest.json","summary.json","analysis_manifest.json","collection_complete.json","frozen_models.json","frozen_gates.json"))
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    for p,h in collection["family_file_sha256"].items():assert sha(OUT / p)==h
    for p,h in am["csv_sha256"].items():assert sha(OUT / p)==h
    assert am["summary_sha256"]==sha(OUT / "summary.json") and am["collection_complete_sha256"]==sha(OUT / "collection_complete.json")
    assert am["analyzer_sha256"]==sha(ROOT / "scripts/analyze_five_ppo_fresh_student.py")
    assert collection["runner_sha256"]==sha(ROOT / "scripts/run_five_ppo_fresh_student.py")
    assert frozen["training_dataset_sha256"]==sha(OUT / "training_dataset.npz") and gates["frozen_models_sha256"]==sha(OUT / "frozen_models.json")
    ids={b:{r["scenario_id"] for r in bank} for b,bank in m["banks"].items()}
    assert [len(ids[b]) for b in ("train","calibration","test")]==[24,12,24]
    assert all(not ids[a]&ids[b] for a in ids for b in ids if a!=b)
    signatures={json.dumps({k:v for k,v in r.items() if k!="scenario_id"},sort_keys=True) for bank in m["banks"].values() for r in bank}
    assert len(signatures)==60 and not m["formal_final_bank"] and not m["qualification_thresholds_changed"]
    # De-duplication reconstructed with historical records using the preserved parser.
    sys.path.insert(0,str(ROOT / "scripts"));from run_proposal_gain_factorial import scenario_records,signature
    previous=[]
    for p in m["scenario_deduplication_files"]:previous+=scenario_records(read(p))
    assert not signatures&{signature(r) for r in previous}
    assert not set.union(*ids.values())&{r["scenario_id"] for r in previous}
    priorseeds=set()
    for p in m["source_input_sha256"]:
        if Path(p).suffix==".npz":
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","population_seeds","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z:priorseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    priorseeds|={x+9000001 for x in list(priorseeds)}
    spaces=[{x for e in m["seed_registry"] for x in e["population_seeds"]},
        {x for e in m["seed_registry"] for v in e["planning_seeds"] for x in v},
        {x for e in m["seed_registry"] for v in e["evaluation_seeds"] for x in v}]
    spaces += [{x+9000001 for x in v} for v in spaces.copy()]
    for i,v in enumerate(spaces):assert not v&priorseeds and all(not v&other for other in spaces[i+1:])
    labels={(r["family"],r["plan_index"],r["target"]):r for r in table(OUT / "pooled_candidate_labels.csv")}
    families={(r["family"],r["arm"]):r for r in table(OUT / "arm_family_results.csv")}
    populations={(r["family"],int(r["population"]),r["arm"]):r for r in table(OUT / "arm_population_results.csv")}
    opportunities={r["family"]:r for r in table(OUT / "family_library_opportunities.csv")}
    pdata={};actual={};selections={};budget=Counter()
    for i,e in enumerate(m["seed_registry"]):
        b=e["block"];f=e["family"];p=load(OUT / (b+"_planning") / (f+".npz"));pdata[f]=p
        assert e["population_seeds"]==[910000000+1000*i+j for j in range(2)]
        assert e["planning_seeds"]==[[930000000+10000*i+1000*j+k for k in range(6)] for j in range(2)]
        np.testing.assert_array_equal(p["population_seeds"],e["population_seeds"]);np.testing.assert_array_equal(p["planning_seeds"],e["planning_seeds"])
        assert p["planning_risks"].shape==(2,2,5,6,11) and p["supported"].all()
        np.testing.assert_array_equal(p["observations"][0],p["observations"][1]);np.testing.assert_array_equal(p["observation"],p["observations"][0])
        x=p["observation"].astype(float).copy();x[28:33]=0;np.testing.assert_array_equal(x,p["features"])
        targets,loss=pooled(p["planning_risks"],p["supported"]);close(targets,p["pooled_targets"]);close(loss,p["pooled_world_losses"])
        for j,t in enumerate(("R","M")):
            for plan in range(5):
                row=labels[f,str(plan),t];close(float(row["pooled_gain_pp"]),targets[j,plan]*100);assert row["class"]==cls(targets[j,plan])
        assert int(p["planning_rollouts"])==120;budget["planning_rollouts"]+=120
        marker=read(OUT / (b+"_planning") / (f+"_started.json"))
        if b=="train":assert datetime.fromisoformat(marker["started_at_utc"])<datetime.fromisoformat(frozen["frozen_at_utc"]);continue
        minimum=frozen["frozen_at_utc"] if b=="calibration" else gates["frozen_at_utc"]
        assert datetime.fromisoformat(marker["started_at_utc"])>=datetime.fromisoformat(minimum)
        ap=OUT / (b+"_actual") / (f+".npz");a=load(ap);actual[f]=a["evaluation_outcomes"]
        sp=OUT / (b+"_choices") / (f+".json");sel=read(sp);selections[f]=sel;start=read(OUT / (b+"_actual") / (f+"_started.json"))
        assert sel["planning_sha256"]==sha(OUT / (b+"_planning") / (f+".npz"))
        assert sel["models_sha256"]==sha(OUT / "frozen_models.json") and sel["gates_sha256"]==sha(OUT / "frozen_gates.json")
        assert start["selection_sha256"]==str(a["selection_sha256"])==sha(sp) and not sel["actual_results_seen"]
        assert datetime.fromisoformat(gates["frozen_at_utc"])<=datetime.fromisoformat(sel["frozen_at_utc"])<=datetime.fromisoformat(start["started_at_utc"])
        assert e["evaluation_seeds"]==[[970000000+10000*i+1000*j+k for k in range(6)] for j in range(2)]
        np.testing.assert_array_equal(a["evaluation_seeds"],e["evaluation_seeds"]);np.testing.assert_array_equal(a["population_seeds"],e["population_seeds"])
        o=a["evaluation_outcomes"];assert o.shape==(2,5,6,11)
        means=o.mean(axis=(0,2));gain=means[0,0]-means[:,0];row=opportunities[f]
        assert int(row["beneficial_candidate_count"])==int((gain>.001).sum());close(float(row["empirical_best_actual_gain_pp"]),gain.max()*100)
        assert row["any_sampled_zero601_world"]==str(bool((loss[:,0]<=1e-12).any()))
        for arm,plan in sel["choices"].items():
            fr=families[f,arm];assert int(fr["plan_index"])==plan and fr["override"]==str(plan!=0) and fr["actual_class"]==cls(gain[plan])
            close(float(fr["actual_gain_pp"]),gain[plan]*100);assert row[arm+"_captures_benefit"]==str(bool(gain[plan]>.001))
            for j,metric in enumerate(m["outcome_fields"][:6]):close(float(fr[metric]),means[plan,j])
            for population in range(2):
                pop=o[population].mean(axis=1);g=pop[0,0]-pop[plan,0];pr=populations[f,population,arm]
                assert pr["actual_class"]==cls(g);close(float(pr["actual_gain_pp"]),g*100)
                for j,metric in enumerate(m["outcome_fields"][:6]):close(float(pr[metric]),pop[plan,j])
        trace=a["execution_trace"];offsets=a["trace_offsets"];keys=a["trace_keys"]
        np.testing.assert_array_equal(keys,[(p,k,j) for p in range(2) for k in range(5) for j in range(6)])
        assert len(offsets)==61 and offsets[0]==0 and offsets[-1]==len(trace)
        assert np.isin(trace[:,1],range(9)).all() and np.isin(trace[:,3],(0,1,2)).all()
        joint=np.where(trace[:,1]==8,8,np.where(trace[:,3]==2,-1,4*trace[:,3]+trace[:,1]%4));np.testing.assert_array_equal(joint,trace[:,2])
        for j,(population,plan,future) in enumerate(keys):
            tr=trace[offsets[j]:offsets[j+1]];oc=o[population,plan,future]
            assert len(tr)==oc[5];np.testing.assert_array_equal(tr[:,0],np.arange(len(tr)))
            close(tr[:,4].mean(),oc[6]);close((tr[:,1]==8).sum(),oc[7]);close(tr[:,5].mean(),oc[8]);close(tr[0,1],oc[9]);close(tr[0,2],oc[10])
        assert int(a["actual_rollouts"])==60;budget["actual_rollouts"]+=60;budget["actual_trace_steps"]+=len(trace)
        budget["hold_block_steps"]+=int(trace[:,5].sum());budget["neutral_positive_release_steps"]+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
    train=load(OUT / "training_dataset.npz");trainids=[e["family"] for e in m["seed_registry"] if e["block"]=="train"]
    np.testing.assert_array_equal(train["families"],trainids);assert set(frozen["training_family_ids"])==ids["train"]
    close(train["observations"],[pdata[f]["features"] for f in trainids]);close(train["targets"],[pdata[f]["pooled_targets"] for f in trainids])
    x=train["observations"];w=np.full(24,1/24);mean=w@x;std=np.sqrt(w@((x-mean)**2));std=np.where(std<.001,1.,std);tx=(x-mean)/std
    models={}
    for name,h in frozen["model_sha256"].items():
        path=OUT / "models" / (name+".npz");assert sha(path)==h;model=load(path);models[name]=model
        target,kind=name.split("_");y=train["targets"][:,0 if target=="R" else 1,1:];intercept=w@y
        close(model["mean"],mean);close(model["std"],std);close(model["intercept"],intercept);close(model["weights"],w)
        if kind=="linear":
            residual=(tx.T@(w[:,None]*tx)+.01*np.eye(49))@model["coefficient"]-tx.T@(w[:,None]*(y-intercept))
        else:
            active=max(1,int((np.var(tx,axis=0)>1e-12).sum()));close(model["gamma"],1/(2*active));close(model["train"],tx)
            k=gram(tx,tx,float(model["gamma"]));row=k@w;grand=w@row;close(model["kernel_row_mean"],row);close(model["kernel_grand_mean"],grand)
            center=k-row[:,None]-row[None,:]+grand;root=np.sqrt(w)
            residual=(root[:,None]*center*root[None,:]+.01*np.eye(24))@(model["coefficient"]/root[:,None])-root[:,None]*(y-intercept)
        assert np.max(abs(residual))<1e-9
    for j,target in enumerate(("R","M")):close(frozen["mean_scores"][target],train["targets"][:,j].mean(axis=0))
    ca=[e["family"] for e in m["seed_registry"] if e["block"]=="calibration"]
    assert gates["calibration_family_ids"]==ca and not gates["actual_outcomes_used"]
    for name,g in gates["gates"].items():
        target=name[0] if name in models else name[-1];j=0 if target=="R" else 1
        residual=[]
        for f in ca:
            pred=prediction(models[name],pdata[f]["features"])[0] if name in models else np.array(frozen["mean_scores"][target])
            value=float((pred[1:]-pdata[f]["pooled_targets"][j,1:]).max());residual.append(value);close(g["family_max_candidate_optimism"][f],value)
        q=sorted(residual)[-1];close(g["empirical_quantile"],q);close(g["margin"],max(m["prior_margin"],.001,q))
    for f,sel in selections.items():
        expected={name:i for i,name in enumerate(m["candidate_names"])}
        for name in (*models,"mean_R","mean_M"):
            p=prediction(models[name],pdata[f]["features"])[0] if name in models else np.array(frozen["mean_scores"][name[-1]])
            close(p,sel["predictions"][name]);expected[name+"_point"]=select(p);expected[name+"_margin"]=select(p,gates["gates"][name]["margin"])
        for j,target in enumerate(("R","M")):
            expected["teacher_"+target+"_point"]=select(pdata[f]["pooled_targets"][j]);expected["teacher_"+target+"_margin"]=select(pdata[f]["pooled_targets"][j],m["prior_margin"])
        assert sel["choices"]==expected and len(expected)==21
    assert dict(budget)==s["budget"] and budget["planning_rollouts"]+budget["actual_rollouts"]==9360
    assert len(labels)==600 and len(families)==756 and len(populations)==1512
    kinds={r["scenario_id"]:r["conflict_type"] for bank in m["banks"].values() for r in bank}
    for row in s["all_paired_effects"]:
        fs=[e["family"] for e in m["seed_registry"] if e["block"]=="test"]
        v=np.array([float(families[f,row["treatment"]]["final_incapacitation"])-float(families[f,row["control"]]["final_incapacitation"]) for f in fs])*100
        close([row["delta_incap_pp"],row["exploratory_ci_low_pp"],row["exploratory_ci_high_pp"]],[v.mean(),*interval(v,[kinds[f] for f in fs],m["bootstrap_seed"])])
        assert row["improved_families"]==int((v < -1e-10).sum()) and row["worsened_families"]==int((v>1e-10).sum())
    for row in s["prediction_quality"]:
        fs=[e["family"] for e in m["seed_registry"] if e["block"]==row["block"]];pred=np.array([selections[f]["predictions"].get(row["model"],[0.]*5) for f in fs])[:,1:]
        A=np.array([actual[f].mean(axis=(0,2))[0,0]-actual[f].mean(axis=(0,2))[:,0] for f in fs])[:,1:]
        R=np.array([pdata[f]["pooled_targets"][0,1:] for f in fs]);M=np.array([pdata[f]["pooled_targets"][1,1:] for f in fs])
        close([row["R_MAE_pp"],row["M_MAE_pp"],row["actual_MAE_pp"]],[np.abs(pred-R).mean()*100,np.abs(pred-M).mean()*100,np.abs(pred-A).mean()*100])
        positive=pred>.001;beneficial=A>.001;assert row["actual_beneficial_candidates"]==int(beneficial.sum())
        assert row["predicted_positive_actual_harmful"]==int((positive&(A < -1e-12)).sum())
        expected=float((positive&beneficial).sum()/beneficial.sum()) if beneficial.any() else None
        assert row["actual_candidate_positive_recall"]==expected
    d=read(OUT / "miss_diagnostic.json");assert d["script_sha256"]==sha(ROOT / "scripts/diagnose_five_ppo_fresh_student.py")
    assert d["audit_csv_sha256"]==sha(OUT / "actual_beneficial_candidate_audit.csv")
    for name,h in d["plots_sha256"].items():assert sha(OUT / name)==h
    audit=table(OUT / "actual_beneficial_candidate_audit.csv");assert len(audit)==9 and len({r["family"] for r in audit})==4
    for r in audit:
        f=r["family"];k=int(r["plan_index"]);means=actual[f].mean(axis=(0,2));close(float(r["actual_gain_pp"]),(means[0,0]-means[k,0])*100)
        close(float(r["planning_R_pp"]),pdata[f]["pooled_targets"][0,k]*100);close(float(r["planning_M_pp"]),pdata[f]["pooled_targets"][1,k]*100)
        close(float(r["predicted_M_kernel_pp"]),selections[f]["predictions"]["M_kernel"][k]*100)
        tx1=(pdata[f]["features"]-models["M_kernel"]["mean"])/models["M_kernel"]["std"];dist=((models["M_kernel"]["train"]-tx1)**2).sum(axis=1);nearest=int(dist.argmin())
        assert r["nearest_training_family"]==str(train["families"][nearest]);close(float(r["nearest_training_M_label_pp"]),train["targets"][nearest,1,k]*100)
    result={"all_checks_passed":True,"source_inputs_unchanged":len(m["source_input_sha256"]),"new_families_verified":60,
        "structural_and_seed_disjointness_verified":True,"pool_before_nonlinear_targets_verified":True,
        "four_frozen_multihead_model_normal_equations_checked":True,"training_only_preprocessing_and_mean_scores_verified":True,
        "family_max_candidate_calibration_and_prior_floor_verified":True,"choices_frozen_before_actual_and_no_holdout_in_fit_verified":True,
        "planning_rollouts_verified":7200,"actual_rollouts_verified":2160,"actual_trace_steps_verified":budget["actual_trace_steps"],
        "hold_block_steps_verified":budget["hold_block_steps"],"neutral_positive_release_steps_verified":budget["neutral_positive_release_steps"],
        "candidate_target_rows_verified":600,"family_arm_rows_verified":756,"population_arm_rows_verified":1512,
        "all_paired_intervals_and_prediction_MAE_recomputed":True,"nine_candidate_miss_diagnosis_verified":True,
        "new_model_fits_in_verifier":0,"new_simulations_in_verifier":0,
        "verifier_sha256":sha(__file__),"analysis_manifest_sha256":sha(OUT / "analysis_manifest.json")}
    (OUT / "verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
