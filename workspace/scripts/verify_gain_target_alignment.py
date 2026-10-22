"""Independent read-only reconstruction; no environment, fitting or simulation."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/gain_target_alignment_diagnostic_development_v1"
PRIOR=ROOT/"outputs/override_worth_evidence_development_v1"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def cls(g):return "harmful" if g < -1e-10 else "beneficial" if g > .1 else "neutral_or_small"
def close(a,b):np.testing.assert_allclose(float(a),float(b),rtol=1e-11,atol=1e-10)


def arithmetic(predicted,risks,actual,support,true_source):
    # Explicit loops keep this independent of the production arithmetic helper.
    loss=np.array([[np.mean(risks[w,a,:,0]) for a in range(2)] for w in range(2)])
    sources=list(np.flatnonzero(support));wg=[loss[w,0]-loss[w,1] for w in range(2)]
    baseline_max=max(loss[w,0] for w in sources);proposal_max=max(loss[w,1] for w in sources)
    r=(baseline_max-proposal_max)*100;t=wg[true_source]*100
    a=(np.mean(actual[0,:,0])-np.mean(actual[1,:,0]))*100;p=predicted*100
    bw=max(sources,key=lambda w:loss[w,0]);pw=max(sources,key=lambda w:loss[w,1])
    result={"predicted_gain_pp":p,"robust_gain_pp":r,"true_planning_gain_pp":t,"actual_gain_pp":a,
        "prediction_to_robust_pp":p-r,"robust_to_true_pp":r-t,"true_to_actual_pp":t-a,"total_error_pp":p-a,
        "minimum_world_gain_pp":min(wg[w] for w in sources)*100,"world0_gain_pp":wg[0]*100,"world1_gain_pp":wg[1]*100,
        "baseline_worst_source":bw,"proposal_worst_source":pw,"different_worst_sources":bw!=pw}
    close(result["total_error_pp"],sum(result[k] for k in ("prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp")))
    assert min(wg[w] for w in sources)*100-1e-10<=r<=max(wg[w] for w in sources)*100+1e-10
    return result


def stats(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r["family"]].append(r)
    cols=("predicted_gain_pp","robust_gain_pp","true_planning_gain_pp","actual_gain_pp","minimum_world_gain_pp",
        "prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp","total_error_pp")
    result={"states":len(rows),"families":len(grouped)}
    for c in cols:
        result[c]=np.mean([np.mean([r[c] for r in rs]) for rs in grouped.values()])
        if c.endswith("_pp") and c in ("prediction_to_robust_pp","robust_to_true_pp","true_to_actual_pp","total_error_pp"):
            result["mean_abs_"+c]=np.mean([np.mean([abs(r[c]) for r in rs]) for rs in grouped.values()])
    result["classes"]=dict(Counter(r["actual_class"] for r in rows))
    result["robust_benefit_any_world_harm_states"]=sum(r["robust_gain_pp"]>.1 and r["minimum_world_gain_pp"]< -1e-10 for r in rows)
    result["different_worst_sources"]=sum(r["different_worst_sources"] for r in rows)
    return result


def main():
    assert not (OUT/"verification.json").exists(),"preserve verifier results"
    m=read(OUT/"development_manifest.json");am=read(OUT/"analysis_manifest.json");summary=read(OUT/"summary.json")
    assert m["all_fits"]==m["new_ppo_training"]==m["new_llm_calls"]==0
    assert m["old_test_now_viewed_development"] and not m["qualification_thresholds_changed"]
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    assert am["runner_sha256"]==sha(ROOT/"scripts/run_gain_target_alignment.py")
    assert am["development_manifest_sha256"]==sha(OUT/"development_manifest.json")
    assert am["summary_sha256"]==sha(OUT/"summary.json")
    for p,h in am["csv_sha256"].items():assert sha(OUT/p)==h
    for p,h in am["replication_cache_sha256"].items():assert sha(OUT/"replication_states"/p)==h
    old=read(PRIOR/"development_manifest.json");scenarios={r["scenario_id"]:r for r in old["calibration_pool_records"]+old["test_records"]}
    cached={};newrows=[];original_csv={r["state_id"]:r for r in table(OUT/"cached_decomposition.csv")}
    ridgepath=Path(old["ridge_checkpoint"])
    with np.load(ridgepath,allow_pickle=False) as z:ridge={k:z[k].copy() for k in z.files}
    oldseeds=set()
    for block,folder in (("calibration","calibration_states"),("test","test_states")):
        for file in sorted((PRIOR/folder).glob("*.npz")):
            with np.load(file,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
            family=file.stem.split("_")[0];scenario=scenarios[family];p=int(d["proposal"])
            assert d["planning_risks"].shape==(2,2,3,11) and d["evaluation_outcomes"].shape==(2,3,11)
            for key in ("population_seed","planning_seeds","evaluation_seeds"):oldseeds.update(map(int,d[key].reshape(-1)))
            x=d["observation"].astype(float).copy();x[28:33]=0
            pred=((x-ridge["mean"])/ridge["std"])@ridge["coefficients"]+ridge["intercept"];pred[12]=0
            np.testing.assert_array_equal(pred,d["predicted_gains"])
            nomination=min(range(17),key=lambda j:(-pred[j],j!=12,j))
            nomination=nomination if pred[nomination]>.001 else 12
            assert nomination==p
            ar=arithmetic(pred[p],d["planning_risks"],d["evaluation_outcomes"],d["supported"],scenario["source_branch"])
            row={"state_id":file.stem,"block":block,"family":family,"eligible":p!=12,"conflict_type":scenario["conflict_type"],
                **ar,"actual_class":cls(ar["actual_gain_pp"])}
            actualrow=original_csv[file.stem]
            for k,v in ar.items():
                if isinstance(v,(bool,np.bool_)):assert actualrow[k]==str(v)
                else:close(actualrow[k],v)
            assert actualrow["actual_class"]==row["actual_class"] and actualrow["eligible"]==str(p!=12)
            assert int(actualrow["true_source_diagnostic"])==scenario["source_branch"]
            cached[file.stem]=(d,scenario,row);newrows.append(row)
    assert len(cached)==len(original_csv)==summary["cached_states"]==108
    assert sum(r["eligible"] for r in newrows)==summary["cached_eligible_states"]==78
    for group in summary["cached_groups"]:
        block,scope=group["block"],group["scope"]
        rs=[r for r in newrows if r["block"]==block and (scope=="all_states" or r["eligible"])]
        if scope in ("beneficial","harmful","neutral_or_small"):rs=[r for r in rs if r["actual_class"]==scope]
        elif scope not in ("all_states","eligible_proposals"):rs=[r for r in rs if r["conflict_type"]==scope]
        expected=stats(rs)
        for k,v in expected.items():
            if isinstance(v,dict):assert group[k]==v
            else:close(group[k],v)
    # Independently reconstruct selection using previously observed labels only.
    families=sorted({r["family"] for r in newrows});nonneutral=[f for f in families if any(r["actual_class"]!="neutral_or_small" for r in newrows if r["family"]==f)]
    controls=[]
    for block in ("calibration","test"):
        eligible=[f for f in families if f not in nonneutral and all(r["eligible"] for r in newrows if r["family"]==f)
            and next(r for r in newrows if r["family"]==f)["block"]==block]
        controls.append(min(eligible,key=lambda f:(-max(r["predicted_gain_pp"] for r in newrows if r["family"]==f),f)))
    assert m["selected_families"]==sorted(nonneutral+controls) and len(nonneutral)==6
    for p in old["source_input_sha256"]:
        if Path(p).suffix==".npz" and Path(p).parent.name in ("states","training_states","audit_states"):
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds"):
                    if key in z:oldseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    for r in old["seed_registry"]:oldseeds.add(r["population"]);oldseeds.update(r["planning"]+r["evaluation"])
    oldseeds|={s+9000001 for s in list(oldseeds)}
    newseed=[s for e in m["seed_registry"] for s in e["future_seeds"]]
    assert len(set(newseed))==len(newseed)==192
    assert not (set(newseed)|{s+9000001 for s in newseed}).intersection(oldseeds)
    assert not set(newseed).intersection({s+9000001 for s in newseed})
    repcsv={r["state_id"]:r for r in table(OUT/"replication_state_results.csv")};rep={};steps=holds=neutral=nr=na=0
    prior_margin=old["prior_margin"];new_margin=read(PRIOR/"frozen_conditional_gate.json")["conditional_margin"]
    for i,entry in enumerate(m["seed_registry"]):
        assert entry["future_seeds"]==[610000000+1000*i+f for f in range(12)]
        state=entry["state_id"];d,scenario,original=cached[state]
        with np.load(OUT/"replication_states"/(state+".npz"),allow_pickle=False) as z:z={k:z[k].copy() for k in z.files}
        assert z["world_risks"].shape==(2,2,12,11) and np.isfinite(z["world_risks"]).all()
        assert int(z["proposal"])==int(d["proposal"])!=12 and int(z["population_seed"])==entry["population_seed"]==int(d["population_seed"])
        assert int(z["true_source"])==scenario["source_branch"]
        for key in ("observation","predicted_gains","supported"):np.testing.assert_array_equal(z[key],d[key])
        np.testing.assert_array_equal(z["future_seeds"],entry["future_seeds"])
        risks=z["world_risks"];source=int(z["true_source"]);p=int(z["proposal"])
        ar=arithmetic(float(d["predicted_gains"][p]),risks,risks[source],z["supported"],source)
        csvrow=repcsv[state]
        for k,v in ar.items():
            if isinstance(v,(bool,np.bool_)):assert csvrow[k]==str(v)
            else:close(csvrow[k],v)
        gains=(risks[source,0,:,0]-risks[source,1,:,0])*100
        assert csvrow["original_actual_class"]==original["actual_class"] and csvrow["replicated_actual_class"]==cls(float(gains.mean()))
        for k,v in (("original_actual_gain_pp",original["actual_gain_pp"]),("first6_actual_gain_pp",gains[:6].mean()),("last6_actual_gain_pp",gains[6:].mean())):close(csvrow[k],v)
        for k,name in enumerate(m["outcome_fields"][:6]):
            for a,label in enumerate(("baseline","proposal")):close(csvrow[label+"_"+name],risks[source,a,:,k].mean())
        prior=bool(d["predicted_gains"][p]>prior_margin+.001);cond=bool(d["predicted_gains"][p]>new_margin+.001)
        worth=bool(d["worth_probabilities"][2]>=.8 and d["worth_probabilities"][0]<=.1)
        for k,v in (("prior_guard_accepts",prior),("conditional_guard_accepts",cond),("worth_condition_passes",worth),("filtered_guard_accepts",cond and worth)):assert csvrow[k]==str(v)
        audit=z["audit_outcomes"]
        assert audit.shape==((2,11) if state in m["audit_state_ids"] else (0,11))
        if len(audit):np.testing.assert_array_equal(audit,risks[source,:,0,:])
        keys,offsets,trace=z["trace_keys"],z["trace_offsets"],z["execution_trace"]
        assert keys.shape==(24,2) and offsets.shape==(25,) and offsets[0]==0 and offsets[-1]==len(trace)
        np.testing.assert_array_equal(keys,[(a,f) for a in range(2) for f in range(12)])
        assert np.isin(trace[:,1],range(9)).all() and np.isin(trace[:,3],(0,1,2)).all()
        actual=np.where(trace[:,1]==8,8,np.where(trace[:,3]==2,-1,4*trace[:,3]+trace[:,1]%4))
        np.testing.assert_array_equal(trace[:,2],actual)
        for j,(a,f) in enumerate(keys):
            tr=trace[offsets[j]:offsets[j+1]];oc=risks[source,a,f]
            assert len(tr)==oc[5] and len(tr)>0
            np.testing.assert_array_equal(tr[:,0],np.arange(len(tr)))
            close(tr[:,4].mean(),oc[6]);close((tr[:,1]==8).sum(),oc[7]);close(tr[:,5].mean(),oc[8]);close(tr[0,1],oc[9]);close(tr[0,2],oc[10])
        nr+=int(z["replication_rollouts"]);na+=int(z["audit_rollouts"]);steps+=len(trace);holds+=int(trace[:,5].sum());neutral+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
        rep[state]={"family":original["family"],"gains":gains,"arithmetic":ar,"csvrow":csvrow}
    assert nr==summary["replication_rollouts"]==768 and na==summary["audit_rollouts"]==4
    assert nr+na==summary["full_rollouts"]==772<=m["maximum_full_rollouts"]
    assert steps==summary["actual_trace_steps"] and holds==summary["hold_block_steps"] and neutral==summary["neutral_positive_release_steps"]
    famcsv={r["family"]:r for r in table(OUT/"replication_family_results.csv")}
    for j,record in enumerate(summary["targeted_families"]):
        family=record["family"];rstates=sorted(s for s in rep if rep[s]["family"]==family);assert len(rstates)==2
        gains=np.array([rep[s]["gains"] for s in rstates]);means=gains.mean(axis=0)
        rng=np.random.default_rng(m["bootstrap_seed"]+j)
        boot=np.asarray([means[idx].mean() for idx in rng.integers(12,size=(2000,12))]);interval=np.quantile(boot,[.025,.975])
        for k,v in (("actual_gain_pp",gains.mean()),("future_only_ci_low_pp",interval[0]),("future_only_ci_high_pp",interval[1]),
            ("first6_actual_gain_pp",gains[:,:6].mean()),("last6_actual_gain_pp",gains[:,6:].mean())):
            close(record[k],v);close(famcsv[family][k],v)
        for k in ("predicted_gain_pp","robust_gain_pp","minimum_world_gain_pp"):
            close(record[k],np.mean([rep[s]["arithmetic"][k] for s in rstates]));close(famcsv[family][k],record[k])
        assert record["replicated_class"]==cls(gains.mean())
        assert record["both_population_classes_match"]==(cls(gains[0].mean())==cls(gains[1].mean()))
        assert record["both_half_classes_match"]==all(cls(g[:6].mean())==cls(g[6:].mean()) for g in gains)
    transitions=[{"original":a,"replicated":b,"count":n} for (a,b),n in sorted(Counter((r["original_actual_class"],r["replicated_actual_class"]) for r in repcsv.values()).items())]
    assert transitions==summary["original_to_replication_state_classes"]
    assert sum(r["first6_class"]!=r["last6_class"] for r in repcsv.values())==summary["different_half_class_states"]
    assert sum(not r["both_population_classes_match"] for r in summary["targeted_families"])==summary["different_population_class_families"]
    assert sum(r["arithmetic"]["robust_gain_pp"]>.1 and r["arithmetic"]["minimum_world_gain_pp"]< -1e-10 for r in rep.values())==summary["replication_robust_benefit_any_world_harm_states"]
    for gate in summary["fixed_gate_diagnostics"]:
        accepted=[r for r in repcsv.values() if r[gate["gate"]]=="True"]
        assert len(accepted)==gate["accepted"] and dict(Counter(r["replicated_actual_class"] for r in accepted))==gate["replicated_classes"]
    result={"all_checks_passed":True,"source_inputs_unchanged":len(m["source_input_sha256"]),"cached_states_verified":108,
        "eligible_states_verified":78,"replication_states_verified":16,"replication_rollouts_verified":nr,"true_scene_audit_rollouts_verified":na,
        "actual_trace_steps_verified":steps,"hold_block_steps_verified":holds,"neutral_positive_release_steps_verified":neutral,
        "exact_decomposition_verified":True,"robust_loss_difference_and_minimum_world_gain_separately_verified":True,
        "frozen_predictions_proposals_unchanged_verified":True,"selection_and_seed_disjointness_verified":True,
        "family_future_only_bootstrap_verified":True,"no_new_simulations_or_fits_in_verifier":True,
        "verifier_sha256":sha(__file__),"analysis_manifest_sha256":sha(OUT/"analysis_manifest.json")}
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
