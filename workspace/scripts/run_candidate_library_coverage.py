"""Freeze a viewed subset and audit all 17 plans without further training."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/candidate_library_coverage_audit_development_v1"
PRIOR=ROOT/"outputs/proposal_gain_factorial_development_v1"
STATUS=ROOT/"outputs/third_round_development_status.json"
PROTOCOL=ROOT/"reports/完整候选计划覆盖审计固定方案_2026-10-03.md"
sys.path.insert(0,str(ROOT/"src"))
from candidate_library_scores import scores,diagnostic_source_scores,choose,actual_gains
from consistent_gain_policy import GainRidge
from override_worth_model import WorthModel,proposal,meta_features
ARMS=("pure_ppo_601","old_proposal","frozen_full","R_point","M_point","R_margin","M_margin","oracle_source_plan","empirical_hindsight_best")

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,b):Path(p).write_text(json.dumps(b,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def stamp():return datetime.now(timezone.utc).isoformat()
def cls(g):return "harmful" if g< -1e-12 else "beneficial" if g>.001 else "neutral_or_small"
def csv_write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare():
    assert not OUT.exists(),"preserve existing audit"
    old=read(PRIOR/"development_manifest.json");completion=read(PRIOR/"completion_manifest.json")
    for p,h in completion["files"].items():assert sha(ROOT/p)==h,p
    for p,h in old["source_input_sha256"].items():assert sha(p)==h,p
    bank=old["banks"]["test"];selected=[]
    for kind in sorted({r["conflict_type"] for r in bank}):selected.extend(sorted((r for r in bank if r["conflict_type"]==kind),key=lambda r:r["scenario_id"])[:2])
    selected.sort(key=lambda r:r["scenario_id"]);assert len(selected)==12
    selected_ids={r["scenario_id"] for r in selected};entries=[e for e in old["seed_registry"] if e["block"]=="test" and e["family"] in selected_ids]
    assert len(entries)==24
    registry=[{"state_id":e["state_id"],"family":e["family"],"rep":e["rep"],"population_seed":e["population_seed"],
        "planning_seeds":[810000000+1000*i+f for f in range(6)],"evaluation_seeds":[850000000+1000*i+f for f in range(6)]} for i,e in enumerate(entries)]
    previous_seeds=set()
    files=set(Path(p) for p in old["source_input_sha256"])
    files|={ROOT/p for p in completion["files"]}
    for p in sorted(files):
        if p.suffix==".npz" and p.parent.name in ("states","training_states","audit_states","calibration_states","test_states","replication_states","train_states"):
            with np.load(p,allow_pickle=False) as z:
                for key in ("population_seed","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z:previous_seeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    previous_seeds|={s+9000001 for s in list(previous_seeds)}
    spaces=[{s for e in registry for s in e[key]} for key in ("planning_seeds","evaluation_seeds")]
    spaces += [{s+9000001 for s in part} for part in spaces.copy()]
    for i,part in enumerate(spaces):
        assert len(part)==144 and not part.intersection(previous_seeds)
        assert all(not part.intersection(other) for other in spaces[i+1:])
    OUT.mkdir();(OUT/"previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes());sources=old["source_input_sha256"].copy()
    for p,h in completion["files"].items():
        path=OUT/"previous_status_exact_bytes.json" if (ROOT/p).resolve()==STATUS else ROOT/p;sources[str(path.resolve())]=h
    frozen=(PROTOCOL,Path(__file__),ROOT/"scripts/candidate_library_scores.py",ROOT/"tests/test_candidate_library_scores.py",PRIOR/"completion_manifest.json")
    for p in frozen:sources[str(p.resolve())]=sha(p)
    (OUT/"source_snapshot").mkdir()
    for p in frozen[:-1]:shutil.copy2(p,OUT/"source_snapshot"/p.name)
    m={"role":"viewed_GTEST_fixed_subset_full17_library_coverage_diagnostic_not_confirmation","created_at_utc":stamp(),
        "source_input_sha256":sources,"previous_status_sha256":sha(STATUS),"prior_completion_sha256":sha(PRIOR/"completion_manifest.json"),
        "selected_scenario_records":selected,"selection_rule":"first_two_IDs_per_type_no_outcome_enrichment","original_test_bank":bank,
        "seed_registry":registry,"planning_futures":6,"actual_futures":6,"old_population_seeds_intentionally_reused":True,
        "new_future_and_sensor_namespaces_disjoint":True,"candidate_names":old["candidate_names"],"frozen_ppo_models":old["frozen_ppo_models"],
        "ridge_checkpoint":old["ridge_checkpoint"],"worth_checkpoint":old["worth_checkpoint"],"prior_margin":old["prior_margin"],
        "outcome_fields":old["outcome_fields"],"trace_fields":old["trace_fields"],"maximum_full_rollouts":7344,
        "expected_planning_rollouts":4896,"expected_actual_rollouts":2448,"new_fits":0,"new_ppo_training":0,"new_llm_calls":0,
        "qualification_thresholds_changed":False,"arms":list(ARMS),"bootstrap_seed":20262501,"bootstrap_draws":2000,
        "fixed_primary":{"treatment":"M_point","control":"old_proposal","metric":"final_incapacitation"},
        "truth_free_world_choices_are_simulator_privileged_not_deployable_student":True,
        "oracle_hidden_source_and_hindsight_selection_explicitly_diagnostic":True}
    write(OUT/"development_manifest.json",m)
    print(json.dumps({"prepared":True,"families":[r['scenario_id'] for r in selected],"source_inputs":len(sources),"budget":7344},ensure_ascii=False),flush=True)


def check(m):
    for p,h in m["source_input_sha256"].items():assert sha(p)==h,p


def collect(m):
    import torch
    from stable_baselines3 import PPO
    from fire_evac.dynamic_conflicts import DynamicConflictScenario
    from fire_evac.source_world_planning import PLANS,source_world_scenarios,compatible_source_worlds
    from fire_evac.ppo_reference_planning import FrozenPPOContinuation,evaluate_continuation,policy_parameter_sha256
    from run_source_world_planning import make_env,vector
    torch.set_num_threads(1);candidates=list(PLANS)
    for r in m["frozen_ppo_models"]:
        actor=PPO.load(r["path"],device="cpu");assert policy_parameter_sha256(actor)==r["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(r["training_seed"],actor))
    assert [p.name for p in candidates]==m["candidate_names"]
    ridge=GainRidge.load(m["ridge_checkpoint"]);worth=WorthModel.load(m["worth_checkpoint"])
    records={r["scenario_id"]:r for r in m["selected_scenario_records"]};scenarios=[DynamicConflictScenario(**r) for r in records.values()]
    actual=make_env(scenarios);worlds=[make_env(scenarios),make_env(scenarios)];folder=OUT/"states";folder.mkdir(exist_ok=True);start=time.monotonic()
    try:
        for i,e in enumerate(m["seed_registry"]):
            state=e["state_id"];planning=folder/(state+"_planning.npz");selection=folder/(state+"_selection.json");evaluated=folder/(state+"_actual.npz")
            scenario=DynamicConflictScenario(**records[e["family"]]);x,_=actual.reset(seed=e["population_seed"],options={"scenario":scenario})
            with np.load(PRIOR/"test_states"/(state+".npz"),allow_pickle=False) as z:np.testing.assert_array_equal(x,z["observation"])
            if not planning.exists():
                gains=ridge.predict(x);p=proposal(gains);prob=np.array([0.,1.,0.]) if p==12 else worth.probabilities(meta_features(x,gains,p))
                support,errors,mismatch=compatible_source_worlds(source_world_scenarios(scenario),x,0);assert support.all() and not mismatch
                risks=np.empty((2,17,6,11))
                for w,hypothesis in enumerate(source_world_scenarios(scenario)):
                    wx,_=worlds[w].reset(seed=e["population_seed"],options={"scenario":hypothesis});np.testing.assert_array_equal(wx,x)
                    for plan in range(17):
                        for f,seed in enumerate(e["planning_seeds"]):risks[w,plan,f]=vector(evaluate_continuation(worlds[w],wx,candidates[plan],seed))
                R,M=scores(risks,support);T=diagnostic_source_scores(risks,scenario.source_branch)
                frozen=bool(p!=12 and gains[p]-m["prior_margin"]>.001 and prob[2]>=.8 and prob[0]<=.1)
                choices={"pure_ppo_601":12,"old_proposal":p,"frozen_full":p if frozen else 12,
                    "R_point":choose(R),"M_point":choose(M),"R_margin":choose(R,m["prior_margin"]),"M_margin":choose(M,m["prior_margin"]),
                    "oracle_source_plan":choose(T)}
                tmp=planning.with_suffix(".tmp.npz")
                np.savez_compressed(tmp,observation=x,predicted_gains=gains,old_proposal=np.asarray(p),worth_probabilities=prob,supported=support,
                    world_errors=errors,population_seed=np.asarray(e["population_seed"]),planning_seeds=np.asarray(e["planning_seeds"]),
                    planning_risks=risks,R_scores=R,M_scores=M,diagnostic_source_scores=T,planning_rollouts=np.asarray(204))
                tmp.replace(planning)
                write(selection,{"frozen_at_utc":stamp(),"state_id":state,"planning_sha256":sha(planning),"choices":choices,
                    "independent_actual_results_seen":False,"source_truth_used_only_in_oracle_choice":True})
            assert selection.exists(),"preserve partial planning missing freeze marker"
            sel=read(selection);assert sel["planning_sha256"]==sha(planning)
            if not evaluated.exists():
                marker=folder/(state+"_actual_started.json");assert not marker.exists(),"preserve interrupted actual rather than silently repeat"
                write(marker,{"at_utc":stamp(),"selection_sha256":sha(selection),"planning_sha256":sha(planning)})
                outcomes=np.empty((17,6,11));traces=[];keys=[]
                for plan in range(17):
                    for f,seed in enumerate(e["evaluation_seeds"]):
                        trace=[];outcomes[plan,f]=vector(evaluate_continuation(actual,x,candidates[plan],seed,trace));traces.append(np.asarray(trace,float));keys.append((plan,f))
                tmp=evaluated.with_suffix(".tmp.npz")
                np.savez_compressed(tmp,evaluation_outcomes=outcomes,population_seed=np.asarray(e["population_seed"]),evaluation_seeds=np.asarray(e["evaluation_seeds"]),
                    selection_sha256=np.asarray(sha(selection)),actual_rollouts=np.asarray(102),trace_keys=np.asarray(keys),
                    trace_offsets=np.cumsum([0]+[len(t) for t in traces]),execution_trace=np.concatenate(traces))
                tmp.replace(evaluated)
            if (i+1)%2==0:print(f"full17 audit {i+1}/24 {e['family']} {time.monotonic()-start:.1f}s",flush=True)
    finally:
        actual.close()
        for env in worlds:env.close()
    for r,actor in zip(m["frozen_ppo_models"],candidates[12:]):assert policy_parameter_sha256(actor.model)==r["parameter_sha256"]


def interval(values,kinds,seed):
    values,kinds=np.asarray(values),np.asarray(kinds);rng=np.random.default_rng(seed);boot=np.zeros(2000)
    for kind in sorted(set(kinds)):
        part=values[kinds==kind];boot+=part[rng.integers(len(part),size=(2000,len(part)))].sum(axis=1)
    return np.quantile(boot/len(values),[.025,.975])


def analyze(m):
    assert not (OUT/"analysis_manifest.json").exists(),"preserve completed analysis"
    records={r["scenario_id"]:r for r in m["selected_scenario_records"]};plans=[];arm_rows=[];state_rows=[];budget=Counter()
    for e in m["seed_registry"]:
        state=e["state_id"];record=records[e["family"]]
        with np.load(OUT/"states"/(state+"_planning.npz"),allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
        with np.load(OUT/"states"/(state+"_actual.npz"),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        outcomes=a["evaluation_outcomes"];A=actual_gains(outcomes);R,M=p["R_scores"],p["M_scores"];old=int(p["old_proposal"])
        means=p["planning_risks"].mean(axis=2)[:,:,0];wg=means[:,12,None]-means
        choice=read(OUT/"states"/(state+"_selection.json"))["choices"].copy();choice["empirical_hindsight_best"]=choose(A)
        common={"state_id":state,"family":e["family"],"conflict_type":record["conflict_type"]}
        for plan in range(17):
            oc=outcomes[plan].mean(axis=0);first=float((outcomes[12,:3,0]-outcomes[plan,:3,0]).mean());last=float((outcomes[12,3:,0]-outcomes[plan,3:,0]).mean())
            plans.append({**common,"plan_index":plan,"plan":m["candidate_names"][plan],"sub_library":"manual" if plan<12 else "frozen_PPO",
                "is_old_proposal":plan==old,"old_prediction_gain_pp":float(p["predicted_gains"][plan]*100),
                "R_gain_pp":float(R[plan]*100),"M_gain_pp":float(M[plan]*100),"source_planning_gain_pp":float(p["diagnostic_source_scores"][plan]*100),
                "world0_gain_pp":float(wg[0,plan]*100),"world1_gain_pp":float(wg[1,plan]*100),
                "actual_gain_pp":float(A[plan]*100),"actual_class":cls(A[plan]),"actual_first3_gain_pp":first*100,"actual_last3_gain_pp":last*100,
                "actual_halves_same_class":cls(first)==cls(last),**{metric:float(oc[j]) for j,metric in enumerate(m["outcome_fields"])}})
        for arm in ARMS:
            plan=choice[arm];oc=outcomes[plan].mean(axis=0)
            arm_rows.append({**common,"arm":arm,"plan_index":plan,"plan":m["candidate_names"][plan],"override":plan!=12,
                "source_privileged":arm=="oracle_source_plan","hindsight_selected":arm=="empirical_hindsight_best",
                "actual_gain_pp":float(A[plan]*100),"actual_class":cls(A[plan]),
                "R_gain_pp":float(R[plan]*100),"M_gain_pp":float(M[plan]*100),
                **{metric:float(oc[j]) for j,metric in enumerate(m["outcome_fields"])}})
        best=choice["empirical_hindsight_best"];benef=A>.001;manual_best=float(A[:12].max());ppo_best=float(A[12:].max())
        zero_worlds=means[:,12]<=1e-12
        if zero_worlds.any():assert M.max()<=1e-12
        state_rows.append({**common,"old_plan_index":old,"old_plan":m["candidate_names"][old],"old_actual_gain_pp":float(A[old]*100),"old_actual_class":cls(A[old]),
            "empirical_best_index":best,"empirical_best_plan":m["candidate_names"][best],"empirical_best_actual_gain_pp":float(A[best]*100),
            "library_beneficial_plan_count":int(benef.sum()),"manual_beneficial_plan_count":int(benef[:12].sum()),"PPO_beneficial_plan_count":int(benef[12:].sum()),
            "manual_max_actual_gain_pp":manual_best*100,"PPO_max_actual_gain_pp":ppo_best*100,
            "missed_by_old_proposal":bool(benef.any() and A[old]<=.001),
            "R_point_captures_actual_benefit":bool(A[choice["R_point"]]>.001),"M_point_captures_actual_benefit":bool(A[choice["M_point"]]>.001),
            "oracle_captures_actual_benefit":bool(A[choice["oracle_source_plan"]]>.001),
            "baseline_world0_loss":float(means[0,12]),"baseline_world1_loss":float(means[1,12]),"any_zero_baseline_world":bool(zero_worlds.any()),
            "max_R_gain_pp":float(R.max()*100),"max_M_gain_pp":float(M.max()*100),"M_point_strict_improvement_structurally_impossible_in_sample":bool(zero_worlds.any())})
        budget["planning_rollouts"]+=int(p["planning_rollouts"]);budget["actual_rollouts"]+=int(a["actual_rollouts"])
        trace=a["execution_trace"];budget["actual_trace_steps"]+=len(trace);budget["hold_block_steps"]+=int(trace[:,5].sum());budget["neutral_positive_release_steps"]+=int(((trace[:,3]==2)&(trace[:,4]>0)).sum())
    csv_write(OUT/"full_library_plan_evidence.csv",plans);csv_write(OUT/"arm_state_results.csv",arm_rows);csv_write(OUT/"state_opportunity_audit.csv",state_rows)
    family_rows=[];families=sorted(records)
    for family in families:
        sr=[r for r in state_rows if r["family"]==family]
        family_rows.append({"family":family,"conflict_type":records[family]["conflict_type"],"population_states":len(sr),
            "old_mean_actual_gain_pp":float(np.mean([r["old_actual_gain_pp"] for r in sr])),
            "hindsight_mean_best_gain_pp":float(np.mean([r["empirical_best_actual_gain_pp"] for r in sr])),
            "states_with_library_benefit":sum(r["library_beneficial_plan_count"]>0 for r in sr),
            "missed_states":sum(r["missed_by_old_proposal"] for r in sr),
            "R_point_benefit_states":sum(r["R_point_captures_actual_benefit"] for r in sr),"M_point_benefit_states":sum(r["M_point_captures_actual_benefit"] for r in sr),
            "oracle_benefit_states":sum(r["oracle_captures_actual_benefit"] for r in sr),"zero_baseline_world_states":sum(r["any_zero_baseline_world"] for r in sr)})
    csv_write(OUT/"family_opportunity_audit.csv",family_rows)
    lookup={(r["state_id"],r["arm"]):r for r in arm_rows};effects=[];arm_summary=[]
    for arm in ARMS:
        ar=[r for r in arm_rows if r["arm"]==arm]
        arm_summary.append({"arm":arm,"families":12,"states":24,"overrides":sum(r["override"] for r in ar),
            "actual_classes":dict(Counter(r["actual_class"] for r in ar)),"chosen_sub_libraries":dict(Counter("manual" if r["plan_index"]<12 else "frozen_PPO" for r in ar)),
            **{metric:float(np.mean([np.mean([r[metric] for r in ar if r["family"]==f]) for f in families])) for metric in m["outcome_fields"][:6]}})
        if arm=="empirical_hindsight_best":continue  # Never supply efficacy CI for actual-data-selected choices.
        for control in ("pure_ppo_601","old_proposal","frozen_full"):
            vals=np.array([np.mean([r["final_incapacitation"]-lookup[r["state_id"],control]["final_incapacitation"] for r in ar if r["family"]==f])*100 for f in families])
            kinds=[records[f]["conflict_type"] for f in families];ci=interval(vals,kinds,m["bootstrap_seed"])
            effects.append({"treatment":arm,"control":control,"families":12,"delta_incap_pp":float(vals.mean()),
                "exploratory_ci_low_pp":float(ci[0]),"exploratory_ci_high_pp":float(ci[1]),
                "improved_families":int((vals< -1e-10).sum()),"worsened_families":int((vals>1e-10).sum()),"tied_families":int((abs(vals)<=1e-10).sum()),
                "oracle_source_privileged":arm=="oracle_source_plan"})
    csv_write(OUT/"paired_effects.csv",effects)
    assert budget["planning_rollouts"]==4896 and budget["actual_rollouts"]==2448
    missed=[r for r in state_rows if r["missed_by_old_proposal"]]
    summary={"role":m["role"],"new_fits":0,"new_ppo_training":0,"new_llm_calls":0,"selected_families":12,"population_states":24,"plans":17,
        "total_unique_full_rollouts":7344,"budget":dict(budget),"arm_summary":arm_summary,"paired_effects":effects,
        "primary_paired_effect":next(r for r in effects if r["treatment"]=="M_point" and r["control"]=="old_proposal"),
        "opportunity_summary":{"states_with_empirical_library_benefit":sum(r["library_beneficial_plan_count"]>0 for r in state_rows),
            "families_with_empirical_library_benefit":sum(r["states_with_library_benefit"]>0 for r in family_rows),
            "missed_by_old_states":len(missed),"missed_by_old_families":len({r["family"] for r in missed}),
            "states_with_manual_library_benefit":sum(r["manual_beneficial_plan_count"]>0 for r in state_rows),
            "states_with_PPO_library_benefit":sum(r["PPO_beneficial_plan_count"]>0 for r in state_rows),
            "R_point_captured_missed_states":sum(r["R_point_captures_actual_benefit"] for r in missed),
            "M_point_captured_missed_states":sum(r["M_point_captures_actual_benefit"] for r in missed),
            "oracle_captured_missed_states":sum(r["oracle_captures_actual_benefit"] for r in missed),
            "states_with_any_zero_baseline_world":sum(r["any_zero_baseline_world"] for r in state_rows),
            "zero_world_states_with_empirical_actual_opportunity":sum(r["any_zero_baseline_world"] and r["library_beneficial_plan_count"]>0 for r in state_rows)},
        "family_opportunity":family_rows,"all_plan_actual_class_counts":dict(Counter(r["actual_class"] for r in plans)),
        "all_plan_actual_half_class_disagreements":sum(not r["actual_halves_same_class"] for r in plans),
        "old_proposal_new_actual_classes":dict(Counter(r["old_actual_class"] for r in state_rows)),
        "freeze_allowed":False,"hard_shield_qualification_passed":False,
        "limits":["Viewed GTEST subset selected by first two IDs per type; development diagnosis, not natural prevalence or confirmation.",
            "Complete existing finite17 library only; neither global optimality nor certified safety bound.",
            "R/M planners do not read hidden source but use simulator and fixed latent population state; privileged offline teachers, not deployed students.",
            "Oracle planning uses true source; empirical hindsight selects with actual outcomes and has selection optimism; no hindsight efficacy interval.",
            "Six planning and actual futures are finite empirical labels; zero sampled baseline loss is not zero true expected loss.",
            "M>0.10pp demands improvement in every support world, stronger than M>=0 empirical non-deterioration.",
            "Paired intervals condition on12 selected families, fixed populations and policies; no multiplicity adjustment or transfer testing.",
            "No new training, LLM calls, threshold relaxation, formal final bank or real-fire validation."]}
    write(OUT/"summary.json",summary);check(m)
    write(OUT/"analysis_manifest.json",{"runner_sha256":sha(__file__),"development_manifest_sha256":sha(OUT/"development_manifest.json"),"summary_sha256":sha(OUT/"summary.json"),
        "csv_sha256":{p.name:sha(p) for p in sorted(OUT.glob("*.csv"))},"state_file_sha256":{p.name:sha(p) for p in sorted((OUT/"states").iterdir()) if p.is_file()}})
    print(json.dumps({"completed_analysis":True,"rollouts":7344,"opportunities":summary["opportunity_summary"],"primary":summary["primary_paired_effect"]},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");parser.add_argument("--run",action="store_true");args=parser.parse_args()
    if args.prepare_only:prepare();return
    if not args.run:parser.error("choose --prepare-only or --run")
    m=read(OUT/"development_manifest.json");check(m);collect(m);analyze(m)


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
