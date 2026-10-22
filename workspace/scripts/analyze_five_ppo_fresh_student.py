"""Prespecified family-level analysis of fresh five-candidate students."""
from collections import Counter
from pathlib import Path
import csv
import json
import sys
import numpy as np
from run_five_ppo_fresh_student import OUT, ROOT, MODEL_NAMES, read, write, sha, load, check


def csv_write(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def cls(g):return "beneficial" if g>.001 else "harmful" if g < -1e-12 else "neutral_or_small"
def interval(values,kinds,seed):
    v=np.array(values);kinds=np.array(kinds);rng=np.random.default_rng(seed);total=np.zeros(2000)
    for kind in sorted(set(kinds)):
        part=v[kinds==kind];total+=part[rng.integers(len(part),size=(2000,len(part)))].sum(axis=1)
    return np.quantile(total/len(v),[.025,.975])


def main():
    assert not (OUT / "analysis_manifest.json").exists()
    m=read(OUT / "development_manifest.json");check(m);complete=read(OUT / "collection_complete.json")
    for p,h in complete["family_file_sha256"].items():assert sha(OUT / p)==h
    candidate_rows=[];family_rows=[];population_rows=[];opportunities=[];quality=[];budget=Counter()
    kinds={r["scenario_id"]:r["conflict_type"] for bank in m["banks"].values() for r in bank}
    family_values={};actual_byblock={};predictions_byblock={}
    for e in m["seed_registry"]:
        p=load(OUT / (e["block"]+"_planning") / (e["family"]+".npz"));budget["planning_rollouts"]+=int(p["planning_rollouts"])
        for j,target in enumerate(("R","M")):
            for plan,name in enumerate(m["candidate_names"]):
                candidate_rows.append({"block":e["block"],"family":e["family"],"conflict_type":kinds[e["family"]],"plan":name,"plan_index":plan,
                    "target":target,"pooled_gain_pp":float(p["pooled_targets"][j,plan]*100),"class":cls(p["pooled_targets"][j,plan])})
        if e["block"]=="train":continue
        a=load(OUT / (e["block"]+"_actual") / (e["family"]+".npz"));sel=read(OUT / (e["block"]+"_choices") / (e["family"]+".json"))
        outcomes=a["evaluation_outcomes"];means=outcomes.mean(axis=(0,2));A=means[0,0]-means[:,0]
        budget["actual_rollouts"]+=int(a["actual_rollouts"]);trace=a["execution_trace"]
        budget["actual_trace_steps"]+=len(trace);budget["hold_block_steps"]+=int(trace[:,5].sum())
        budget["neutral_positive_release_steps"]+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
        key=e["block"];actual_byblock.setdefault(key,[]).append(A)
        pred={**sel["predictions"],"zero":[0.]*5};predictions_byblock.setdefault(key,[]).append((p,pred))
        common={"block":key,"family":e["family"],"conflict_type":kinds[e["family"]]}
        for arm,plan in sel["choices"].items():
            row={**common,"arm":arm,"plan_index":plan,"plan":m["candidate_names"][plan],"override":plan!=0,
                "actual_gain_pp":float(A[plan]*100),"actual_class":cls(A[plan]),
                **{metric:float(means[plan,j]) for j,metric in enumerate(m["outcome_fields"][:6])}}
            family_rows.append(row);family_values[key,e["family"],arm]=row
            for population in range(2):
                oc=outcomes[population].mean(axis=1);g=oc[0,0]-oc[plan,0]
                population_rows.append({**common,"population":population,"arm":arm,"plan_index":plan,"actual_gain_pp":float(g*100),"actual_class":cls(g),
                    **{metric:float(oc[plan,j]) for j,metric in enumerate(m["outcome_fields"][:6])}})
        beneficial=A>.001
        opportunities.append({**common,"beneficial_candidate_count":int(beneficial.sum()),"empirical_best_actual_gain_pp":float(A.max()*100),
            **{arm+"_captures_benefit":bool(A[plan]>.001) for arm,plan in sel["choices"].items()},
            "pooled_601_world_min_loss":float(p["pooled_world_losses"][:,0].min()),
            "any_sampled_zero601_world":bool((p["pooled_world_losses"][:,0]<=1e-12).any())})
    for block in ("calibration","test"):
        values=predictions_byblock[block];A=np.array(actual_byblock[block]);R=np.array([p["pooled_targets"][0] for p,_ in values]);M=np.array([p["pooled_targets"][1] for p,_ in values])
        for name in (*MODEL_NAMES,"mean_R","mean_M","zero"):
            pred=np.array([q[name] for _,q in values])
            g=pred[:,1:];actual=A[:,1:];positive=g>.001;beneficial=actual>.001;harmful=actual < -1e-12
            quality.append({"block":block,"model":name,"families":len(values),"nonbaseline_candidate_rows":4*len(values),
                "R_MAE_pp":float(np.abs(g-R[:,1:]).mean()*100),"M_MAE_pp":float(np.abs(g-M[:,1:]).mean()*100),
                "actual_MAE_pp":float(np.abs(g-actual).mean()*100),"actual_bias_pp":float((g-actual).mean()*100),
                "predicted_beneficial_candidates":int(positive.sum()),"actual_beneficial_candidates":int(beneficial.sum()),
                "actual_harmful_candidates":int(harmful.sum()),"beneficial_candidates_predicted_positive":int((beneficial&positive).sum()),
                "actual_candidate_positive_recall":float((beneficial&positive).sum()/beneficial.sum()) if beneficial.any() else None,
                "predicted_positive_precision":float((beneficial&positive).sum()/positive.sum()) if positive.any() else None,
                "predicted_positive_actual_harmful":int((positive&harmful).sum())})
    summaries=[];effects=[]
    for block in ("calibration","test"):
        families=[e["family"] for e in m["seed_registry"] if e["block"]==block]
        for arm in sorted({r["arm"] for r in family_rows}):
            rows=[family_values[block,f,arm] for f in families];pop=[r for r in population_rows if r["block"]==block and r["arm"]==arm]
            summaries.append({"block":block,"arm":arm,"families":len(rows),"override_families":sum(r["override"] for r in rows),
                "actual_family_classes":dict(Counter(r["actual_class"] for r in rows)),"actual_population_classes":dict(Counter(r["actual_class"] for r in pop)),
                **{metric:float(np.mean([r[metric] for r in rows])) for metric in m["outcome_fields"][:6]}})
        if block=="test":
            pairs=[(arm,"pure_ppo_601") for arm in sorted({r["arm"] for r in family_rows})]
            pairs += [(m["fixed_primary"]["treatment"],m["fixed_primary"]["control"]),*map(tuple,m["fixed_secondary"])]
            for treatment,control in pairs:
                values=np.array([family_values[block,f,treatment]["final_incapacitation"]-family_values[block,f,control]["final_incapacitation"] for f in families])*100
                ci=interval(values,[kinds[f] for f in families],m["bootstrap_seed"])
                effects.append({"treatment":treatment,"control":control,"families":len(families),"delta_incap_pp":float(values.mean()),
                    "exploratory_ci_low_pp":float(ci[0]),"exploratory_ci_high_pp":float(ci[1]),
                    "improved_families":int((values < -1e-10).sum()),"worsened_families":int((values>1e-10).sum()),"tied_families":int((abs(values)<=1e-10).sum()),
                    "teacher_simulator_privileged":treatment.startswith("teacher_")})
    primary=next(r for r in effects if r["treatment"]==m["fixed_primary"]["treatment"] and r["control"]==m["fixed_primary"]["control"])
    opportunity_summary=[]
    for block in ("calibration","test"):
        rows=[r for r in opportunities if r["block"]==block];opportunity_summary.append({"block":block,"families":len(rows),
            "families_with_library_benefit":sum(r["beneficial_candidate_count"]>0 for r in rows),
            "beneficial_candidate_rows":sum(r["beneficial_candidate_count"] for r in rows),
            "zero_baseline_world_families":sum(r["any_sampled_zero601_world"] for r in rows),
            "captured_library_benefit_by_arm":{arm:sum(r[arm+"_captures_benefit"] for r in rows) for arm in sorted({r["arm"] for r in family_rows})}})
    assert budget["planning_rollouts"]==7200 and budget["actual_rollouts"]==2160
    for name,rows in (("pooled_candidate_labels.csv",candidate_rows),("arm_family_results.csv",family_rows),
        ("arm_population_results.csv",population_rows),("family_library_opportunities.csv",opportunities),
        ("prediction_quality.csv",quality),("paired_effects.csv",effects)):csv_write(OUT / name,rows)
    summary={"role":m["role"],"total_unique_full_rollouts":9360,"budget":dict(budget),"new_matrix_model_fits":4,
        "new_training_mean_references":2,"new_ppo_training":0,"new_llm_calls":0,
        "training_families":24,"calibration_families":12,"test_families":24,"arms":21,
        "pooled_candidate_label_rows":len(candidate_rows),"arm_family_rows":len(family_rows),"arm_population_rows":len(population_rows),
        "primary_paired_effect":primary,"all_paired_effects":effects,"arm_summary":summaries,
        "opportunity_summary":opportunity_summary,"prediction_quality":quality,
        "margins_pp":{n:g["margin"]*100 for n,g in read(OUT / "frozen_gates.json")["gates"].items()},
        "freeze_allowed":False,"hard_shield_qualification_passed":False,
        "limits":["Fresh families under the same generator; small balanced development sample, not final bank or real-fire efficacy.",
            "Two populations pooled before nonlinear world scores, six futures each; not exact conditional expectations.",
            "Choice is once at initial observation among frozen closed-loop PPO actors, not dynamic inter-policy switching or LLM correction.",
            "R/M teacher labels and teachers retain simulator and fixed latent-population privileges; actual outcomes excluded from fitting/calibration.",
            "Twelve calibration families and empirical95% candidate-max residual quantile do not guarantee deployment safety coverage.",
            "Initial selector does not replace the old worth-classifier full gate or qualify original safety criteria.",
            "Exploratory family bootstrap intervals unadjusted for multiple comparisons; hindsight library opportunity is biased diagnosis.",
            "No new PPO/LLM training, behavior/map/noise shift or independent LLM contribution established."]}
    write(OUT / "summary.json",summary);check(m)
    write(OUT / "analysis_manifest.json",{"analyzer_sha256":sha(__file__),"collection_complete_sha256":sha(OUT / "collection_complete.json"),
        "summary_sha256":sha(OUT / "summary.json"),"csv_sha256":{p.name:sha(p) for p in sorted(OUT.glob("*.csv"))}})
    print(json.dumps({"completed_analysis":True,"rollouts":9360,"primary":primary,"opportunity":opportunity_summary,
        "test_student_arms":[r for r in summaries if r["block"]=="test" and r["arm"] in ("M_kernel_margin","M_kernel_point","R_kernel_point","mean_M_point","mean_R_point")]},ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
