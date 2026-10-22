"""Independent arithmetic, sequencing, execution and preservation checks."""
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/candidate_library_coverage_audit_development_v1"
PRIOR=ROOT/"outputs/proposal_gain_factorial_development_v1"
ARMS=("pure_ppo_601","old_proposal","frozen_full","R_point","M_point","R_margin","M_margin","oracle_source_plan","empirical_hindsight_best")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
def cls(g):return "harmful" if g < -1e-12 else "beneficial" if g>.001 else "neutral_or_small"
def select(g,margin=0.):
    values=[0. if p==12 else float(g[p])-margin for p in range(17)];best=max(values)
    tied=[p for p in range(17) if best-values[p]<=1e-12];p=min(tied,key=lambda k:(k!=12,k))
    return p if values[p]>.001 else 12


def main():
    assert not (OUT/"verification.json").exists(),"preserve prior verification"
    m,s,am=(read(OUT/p) for p in ("development_manifest.json","summary.json","analysis_manifest.json"))
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p
    assert am["runner_sha256"]==sha(ROOT/"scripts/run_candidate_library_coverage.py")
    assert am["development_manifest_sha256"]==sha(OUT/"development_manifest.json") and am["summary_sha256"]==sha(OUT/"summary.json")
    for p,h in am["csv_sha256"].items():assert sha(OUT/p)==h
    for p,h in am["state_file_sha256"].items():assert sha(OUT/"states"/p)==h
    assert m["new_fits"]==m["new_ppo_training"]==m["new_llm_calls"]==0 and not m["qualification_thresholds_changed"]
    old=read(PRIOR/"development_manifest.json");expected=[]
    for kind in sorted({r["conflict_type"] for r in old["banks"]["test"]}):
        expected+=sorted((r for r in old["banks"]["test"] if r["conflict_type"]==kind),key=lambda r:r["scenario_id"])[:2]
    assert sorted(expected,key=lambda r:r["scenario_id"])==m["selected_scenario_records"] and len(expected)==12
    familyset={r["scenario_id"] for r in expected};oldentries={e["state_id"]:e for e in old["seed_registry"] if e["block"]=="test" and e["family"] in familyset}
    assert len(oldentries)==24
    oldseeds=set()
    for p in m["source_input_sha256"]:
        path=Path(p)
        if path.suffix==".npz" and path.parent.name in ("states","training_states","audit_states","calibration_states","test_states","replication_states","train_states"):
            with np.load(path,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z:oldseeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    oldseeds|={x+9000001 for x in list(oldseeds)}
    spaces=[{seed for e in m["seed_registry"] for seed in e[key]} for key in ("planning_seeds","evaluation_seeds")]
    spaces += [{x+9000001 for x in part} for part in spaces.copy()]
    for i,part in enumerate(spaces):
        assert len(part)==144 and not part.intersection(oldseeds)
        assert all(not part.intersection(other) for other in spaces[i+1:])
    library={ (r["state_id"],int(r["plan_index"])):r for r in table(OUT/"full_library_plan_evidence.csv")}
    armcsv={ (r["state_id"],r["arm"]):r for r in table(OUT/"arm_state_results.csv")}
    statecsv={r["state_id"]:r for r in table(OUT/"state_opportunity_audit.csv")};families={r["scenario_id"]:r for r in expected}
    ridge,worth=load(m["ridge_checkpoint"]),load(m["worth_checkpoint"]);budget=Counter();state_results=[];actual_arm_results=[]
    for i,e in enumerate(m["seed_registry"]):
        state=e["state_id"];record=families[e["family"]];pp=OUT/"states"/(state+"_planning.npz");ap=OUT/"states"/(state+"_actual.npz");sp=OUT/"states"/(state+"_selection.json")
        p,a=load(pp),load(ap);selection=read(sp);start=read(OUT/"states"/(state+"_actual_started.json"))
        assert selection["planning_sha256"]==start["planning_sha256"]==sha(pp)
        assert start["selection_sha256"]==str(a["selection_sha256"])==sha(sp)
        assert not selection["independent_actual_results_seen"]
        assert datetime.fromisoformat(selection["frozen_at_utc"])<=datetime.fromisoformat(start["at_utc"])
        assert e["population_seed"]==oldentries[state]["population_seed"]==int(p["population_seed"])==int(a["population_seed"])
        assert e["planning_seeds"]==[810000000+1000*i+f for f in range(6)]
        assert e["evaluation_seeds"]==[850000000+1000*i+f for f in range(6)]
        np.testing.assert_array_equal(p["planning_seeds"],e["planning_seeds"]);np.testing.assert_array_equal(a["evaluation_seeds"],e["evaluation_seeds"])
        oldstate=load(PRIOR/"test_states"/(state+".npz"));np.testing.assert_array_equal(p["observation"],oldstate["observation"])
        obs=p["observation"].astype(float).copy();obs[28:33]=0;g=((obs-ridge["mean"])/ridge["std"])@ridge["coefficients"]+ridge["intercept"];g[12]=0
        close(g,p["predicted_gains"]);oldplan=min(range(17),key=lambda k:(-g[k],k!=12,k));oldplan=oldplan if g[oldplan]>.001 else 12
        assert oldplan==int(p["old_proposal"])==int(oldstate["proposal"])
        prob=np.array([0.,1.,0.])
        if oldplan!=12:
            meta=np.r_[obs,np.eye(17)[oldplan],g[oldplan],g[oldplan]-np.max(np.delete(g,oldplan))]
            logits=((meta-worth["mean"])/worth["std"])@worth["coefficient"]+worth["intercept"];prob=np.exp(logits-logits.max());prob/=prob.sum()
        close(prob,p["worth_probabilities"])
        risks=p["planning_risks"];out=a["evaluation_outcomes"]
        assert risks.shape==(2,17,6,11) and out.shape==(17,6,11) and np.isfinite(risks).all() and np.isfinite(out).all() and p["supported"].all()
        assert ((risks[:,:,:,0]>=0)&(risks[:,:,:,0]<=1)).all()
        losses=np.array([[np.mean(risks[w,k,:,0]) for k in range(17)] for w in range(2)])
        R=np.array([max(losses[:,12])-max(losses[:,k]) for k in range(17)])
        M=np.array([min(losses[w,12]-losses[w,k] for w in range(2)) for k in range(17)])
        T=losses[record["source_branch"],12]-losses[record["source_branch"]]
        A=np.array([out[12,:,0].mean()-out[k,:,0].mean() for k in range(17)])
        R[12]=M[12]=T[12]=A[12]=0
        close(R,p["R_scores"]);close(M,p["M_scores"]);close(T,p["diagnostic_source_scores"])
        frozen=oldplan if oldplan!=12 and g[oldplan]-m["prior_margin"]>.001 and prob[2]>=.8 and prob[0]<=.1 else 12
        choices={"pure_ppo_601":12,"old_proposal":oldplan,"frozen_full":frozen,"R_point":select(R),"M_point":select(M),
            "R_margin":select(R,m["prior_margin"]),"M_margin":select(M,m["prior_margin"]),"oracle_source_plan":select(T)}
        assert selection["choices"]==choices
        choices["empirical_hindsight_best"]=select(A)
        for k in range(17):
            row=library[state,k];first=out[12,:3,0].mean()-out[k,:3,0].mean();last=out[12,3:,0].mean()-out[k,3:,0].mean()
            for key,value in (("old_prediction_gain_pp",g[k]*100),("R_gain_pp",R[k]*100),("M_gain_pp",M[k]*100),
                ("source_planning_gain_pp",T[k]*100),("world0_gain_pp",(losses[0,12]-losses[0,k])*100),("world1_gain_pp",(losses[1,12]-losses[1,k])*100),
                ("actual_gain_pp",A[k]*100),("actual_first3_gain_pp",first*100),("actual_last3_gain_pp",last*100)):
                close(float(row[key]),value)
            assert row["actual_class"]==cls(A[k]) and row["actual_halves_same_class"]==str(cls(first)==cls(last))
            assert row["is_old_proposal"]==str(k==oldplan) and row["sub_library"]==("manual" if k<12 else "frozen_PPO")
            for j,metric in enumerate(m["outcome_fields"]):close(float(row[metric]),out[k,:,j].mean())
        for arm,k in choices.items():
            row=armcsv[state,arm];assert int(row["plan_index"])==k and row["override"]==str(k!=12)
            assert row["hindsight_selected"]==str(arm=="empirical_hindsight_best") and row["source_privileged"]==str(arm=="oracle_source_plan")
            close(float(row["actual_gain_pp"]),A[k]*100);assert row["actual_class"]==cls(A[k])
            for j,metric in enumerate(m["outcome_fields"]):close(float(row[metric]),out[k,:,j].mean())
            actual_arm_results.append({"state_id":state,"family":e["family"],"arm":arm,"plan_index":k,"actual_class":cls(A[k]),
                "final_incapacitation":float(out[k,:,0].mean())})
        sr=statecsv[state];benef=A>.001;zero=bool((losses[:,12]<=1e-12).any());best=choices["empirical_hindsight_best"]
        flags={"missed_by_old_proposal":bool(benef.any() and A[oldplan]<=.001),"R_point_captures_actual_benefit":bool(A[choices["R_point"]]>.001),
            "M_point_captures_actual_benefit":bool(A[choices["M_point"]]>.001),"oracle_captures_actual_benefit":bool(A[choices["oracle_source_plan"]]>.001),
            "any_zero_baseline_world":zero,"M_point_strict_improvement_structurally_impossible_in_sample":zero}
        for key,value in flags.items():assert sr[key]==str(value)
        if zero:assert M.max()<=1e-12 and choices["M_point"]==12
        for key,value in (("old_actual_gain_pp",A[oldplan]*100),("empirical_best_actual_gain_pp",A[best]*100),
            ("manual_max_actual_gain_pp",A[:12].max()*100),("PPO_max_actual_gain_pp",A[12:].max()*100),("max_R_gain_pp",R.max()*100),("max_M_gain_pp",M.max()*100),
            ("baseline_world0_loss",losses[0,12]),("baseline_world1_loss",losses[1,12])):close(float(sr[key]),value)
        for key,value in (("library_beneficial_plan_count",benef.sum()),("manual_beneficial_plan_count",benef[:12].sum()),("PPO_beneficial_plan_count",benef[12:].sum())):assert int(sr[key])==value
        state_results.append({"family":e["family"],"A":A,"old_plan":oldplan,"best":best,"flags":flags,"beneficial":benef,"choices":choices})
        trace=a["execution_trace"];offsets=a["trace_offsets"];keys=a["trace_keys"]
        np.testing.assert_array_equal(keys,[(k,f) for k in range(17) for f in range(6)]);assert offsets.shape==(103,) and offsets[0]==0 and offsets[-1]==len(trace)
        assert np.isin(trace[:,1],range(9)).all() and np.isin(trace[:,3],(0,1,2)).all()
        joint=np.where(trace[:,1]==8,8,np.where(trace[:,3]==2,-1,4*trace[:,3]+trace[:,1]%4));np.testing.assert_array_equal(trace[:,2],joint)
        for j,(k,future) in enumerate(keys):
            tr=trace[offsets[j]:offsets[j+1]];oc=out[k,future];assert len(tr)==oc[5]
            np.testing.assert_array_equal(tr[:,0],np.arange(len(tr)));close(tr[:,4].mean(),oc[6]);close((tr[:,1]==8).sum(),oc[7]);close(tr[:,5].mean(),oc[8]);close(tr[0,1],oc[9]);close(tr[0,2],oc[10])
        assert int(p["planning_rollouts"])==204 and int(a["actual_rollouts"])==102
        budget["planning_rollouts"]+=204;budget["actual_rollouts"]+=102;budget["actual_trace_steps"]+=len(trace);budget["hold_block_steps"]+=int(trace[:,5].sum());budget["neutral_positive_release_steps"]+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
    assert len(library)==408 and len(armcsv)==216 and len(statecsv)==24 and dict(budget)==s["budget"]
    assert budget["planning_rollouts"]==4896 and budget["actual_rollouts"]==2448 and s["total_unique_full_rollouts"]==7344
    missed=[r for r in state_results if r["flags"]["missed_by_old_proposal"]]
    expected_opportunity={"states_with_empirical_library_benefit":sum(r["beneficial"].any() for r in state_results),
        "families_with_empirical_library_benefit":len({r["family"] for r in state_results if r["beneficial"].any()}),
        "missed_by_old_states":len(missed),"missed_by_old_families":len({r["family"] for r in missed}),
        "states_with_manual_library_benefit":sum(r["beneficial"][:12].any() for r in state_results),"states_with_PPO_library_benefit":sum(r["beneficial"][12:].any() for r in state_results),
        "R_point_captured_missed_states":sum(r["flags"]["R_point_captures_actual_benefit"] for r in missed),
        "M_point_captured_missed_states":sum(r["flags"]["M_point_captures_actual_benefit"] for r in missed),
        "oracle_captured_missed_states":sum(r["flags"]["oracle_captures_actual_benefit"] for r in missed),
        "states_with_any_zero_baseline_world":sum(r["flags"]["any_zero_baseline_world"] for r in state_results),
        "zero_world_states_with_empirical_actual_opportunity":sum(r["flags"]["any_zero_baseline_world"] and r["beneficial"].any() for r in state_results)}
    assert {k:int(v) for k,v in expected_opportunity.items()}==s["opportunity_summary"]
    familycsv={r["family"]:r for r in table(OUT/"family_opportunity_audit.csv")}
    for record in s["family_opportunity"]:
        rows=[r for r in state_results if r["family"]==record["family"]];assert len(rows)==2
        close(record["old_mean_actual_gain_pp"],np.mean([r["A"][r["old_plan"]] for r in rows])*100)
        close(record["hindsight_mean_best_gain_pp"],np.mean([r["A"][r["best"]] for r in rows])*100)
        for key,flag in (("missed_states","missed_by_old_proposal"),("R_point_benefit_states","R_point_captures_actual_benefit"),("M_point_benefit_states","M_point_captures_actual_benefit"),
            ("oracle_benefit_states","oracle_captures_actual_benefit"),("zero_baseline_world_states","any_zero_baseline_world")):
            assert sum(r["flags"][flag] for r in rows)==record[key]==int(familycsv[record["family"]][key])
    armlookup={(r["state_id"],r["arm"]):r for r in actual_arm_results};family_order=sorted(families)
    for summary in s["arm_summary"]:
        ar=[r for r in actual_arm_results if r["arm"]==summary["arm"]]
        assert sum(r["plan_index"]!=12 for r in ar)==summary["overrides"] and dict(Counter(r["actual_class"] for r in ar))==summary["actual_classes"]
        assert dict(Counter("manual" if r["plan_index"]<12 else "frozen_PPO" for r in ar))==summary["chosen_sub_libraries"]
        for metric in m["outcome_fields"][:6]:close(summary[metric],np.mean([float(row[metric]) for (state,arm),row in armcsv.items() if arm==summary["arm"]]))
    for effect in s["paired_effects"]:
        assert effect["treatment"]!="empirical_hindsight_best"
        vals=[]
        for family in family_order:
            rs=[r for r in actual_arm_results if r["family"]==family and r["arm"]==effect["treatment"]]
            vals.append(np.mean([r["final_incapacitation"]-armlookup[r["state_id"],effect["control"]]["final_incapacitation"] for r in rs])*100)
        vals=np.array(vals);rng=np.random.default_rng(m["bootstrap_seed"]);boot=np.zeros(2000)
        for kind in sorted({r["conflict_type"] for r in expected}):
            part=vals[[families[f]["conflict_type"]==kind for f in family_order]];boot+=part[rng.integers(len(part),size=(2000,len(part)))].sum(axis=1)
        ci=np.quantile(boot/12,[.025,.975]);close(effect["delta_incap_pp"],vals.mean());close([effect["exploratory_ci_low_pp"],effect["exploratory_ci_high_pp"]],ci)
        assert effect["improved_families"]==int((vals< -1e-10).sum()) and effect["worsened_families"]==int((vals>1e-10).sum())
    assert s["primary_paired_effect"]==next(r for r in s["paired_effects"] if r["treatment"]=="M_point" and r["control"]=="old_proposal")
    assert dict(Counter(row["actual_class"] for row in library.values()))==s["all_plan_actual_class_counts"]
    assert sum(row["actual_halves_same_class"]=="False" for row in library.values())==s["all_plan_actual_half_class_disagreements"]
    assert dict(Counter(cls(r["A"][r["old_plan"]]) for r in state_results))==s["old_proposal_new_actual_classes"]
    result={"all_checks_passed":True,"source_inputs_unchanged":len(m["source_input_sha256"]),"planning_rollouts_verified":4896,"actual_rollouts_verified":2448,
        "plan_state_rows_verified":408,"arm_state_rows_verified":216,"population_states_verified":24,"actual_trace_steps_verified":budget["actual_trace_steps"],
        "hold_block_steps_verified":budget["hold_block_steps"],"neutral_positive_release_steps_verified":budget["neutral_positive_release_steps"],
        "selection_before_independent_actual_verified":True,"complete_library_and_sub_library_opportunities_verified":True,
        "zero_baseline_world_M_positive_impossibility_verified":True,"privileged_oracle_and_biased_hindsight_separate_verified":True,
        "paired_family_intervals_recomputed":True,"frozen_inputs_and_reused_population_new_future_streams_verified":True,
        "new_fits":0,"new_simulations_in_verifier":0,"verifier_sha256":sha(__file__),"analysis_manifest_sha256":sha(OUT/"analysis_manifest.json")}
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
