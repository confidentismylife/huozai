"""Freeze fresh family banks, fit four students and collect a bounded evaluation."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.dynamic_conflicts import DynamicConflictScenario, make_domain_randomized_trust_bank
from five_ppo_student import pooled_labels, fit, MultiGain, calibrate
from candidate_label_interface import observable_features, choose_candidate
from run_proposal_gain_factorial import scenario_records, signature
from finalize_set_aware_distillation import check_completion

OUT = ROOT / "outputs/five_ppo_fresh_family_student_development_v1"
PRIOR = ROOT / "outputs/ppo_candidate_label_interface_development_v1"
AUDIT = ROOT / "outputs/candidate_library_coverage_audit_development_v1"
STATUS = ROOT / "outputs/third_round_development_status.json"
PROTOCOL = ROOT / "reports/五PPO全新家庭轻量选择实验固定方案_2026-10-04.md"
MODEL_NAMES = ("R_linear", "R_kernel", "M_linear", "M_kernel")
def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def check(m):
    for p, h in m["source_input_sha256"].items(): assert sha(p) == h, p


def prepare():
    assert not OUT.exists()
    assert check_completion(PRIOR.name, STATUS) == 40
    old = read(PRIOR / "development_manifest.json"); check(old)
    bankfiles = sorted(set((ROOT / "outputs").rglob("development_manifest.json")) | set((ROOT / "outputs").rglob("*scenarios.json")))
    prior_records = []
    for p in bankfiles: prior_records += scenario_records(read(p))
    oldsignatures = {signature(r) for r in prior_records}; oldids = {r["scenario_id"] for r in prior_records}
    banks = {block: [asdict(r) for r in make_domain_randomized_trust_bank(n, seed, prefix)] for block, n, seed, prefix in
        (("train",24,2026100401,"F5TR"),("calibration",12,2026100402,"F5CA"),("test",24,2026100403,"F5TE"))}
    records = sum(banks.values(), [])
    assert len({signature(r) for r in records}) == 60 and not {signature(r) for r in records} & oldsignatures
    assert not {r["scenario_id"] for r in records} & oldids
    for b, bank in banks.items(): assert sorted(Counter(r["conflict_type"] for r in bank).values()) == [len(bank)//6]*6
    registry = []
    for block, bank in banks.items():
        for record in bank:
            i = len(registry)
            registry.append({"block":block,"family":record["scenario_id"],"population_seeds":[910000000+1000*i+p for p in range(2)],
                "planning_seeds":[[930000000+10000*i+1000*p+f for f in range(6)] for p in range(2)],
                "evaluation_seeds": [] if block == "train" else [[970000000+10000*i+1000*p+f for f in range(6)] for p in range(2)]})
    previous_seeds = set()
    for p in old["source_input_sha256"]:
        path = Path(p)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for key in ("population_seed","population_seeds","planning_seeds","evaluation_seeds","future_seeds"):
                    if key in z: previous_seeds.update(map(int, np.asarray(z[key]).reshape(-1)))
    previous_seeds |= {s+9000001 for s in list(previous_seeds)}
    spaces = [{s for e in registry for s in e["population_seeds"]},
              {s for e in registry for part in e["planning_seeds"] for s in part},
              {s for e in registry for part in e["evaluation_seeds"] for s in part}]
    spaces += [{s+9000001 for s in part} for part in spaces.copy()]
    for i, part in enumerate(spaces):
        assert not part & previous_seeds and all(not part & other for other in spaces[i+1:])
    OUT.mkdir(); (OUT / "previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes())
    sources = dict(old["source_input_sha256"])
    for p, h in read(PRIOR / "completion_manifest.json")["files"].items():
        path = ROOT / p; path = OUT / "previous_status_exact_bytes.json" if path.resolve() == STATUS else path
        sources[str(path.resolve())] = h
    frozen = (PROTOCOL, Path(__file__), ROOT / "scripts/five_ppo_student.py", ROOT / "tests/test_five_ppo_student.py", PRIOR / "completion_manifest.json")
    for p in tuple(bankfiles)+frozen: sources[str(p.resolve())] = sha(p)
    (OUT / "source_snapshot").mkdir()
    for p in frozen[:-1]: shutil.copyfile(p, OUT / "source_snapshot" / p.name)
    audit = read(AUDIT / "development_manifest.json")
    m = {"role":"fresh_family_five_PPO_pooled_observable_student_development_not_final_qualification","created_at_utc":stamp(),
        "source_input_sha256":sources,"previous_status_sha256":sha(STATUS),"prior_completion_sha256":sha(PRIOR / "completion_manifest.json"),
        "banks":banks,"seed_registry":registry,"scenario_deduplication_files":[str(p) for p in bankfiles],
        "historical_scenario_records_checked":len(prior_records),"historical_unique_signatures":len(oldsignatures),
        "new_banks_structurally_disjoint":True,"seed_namespaces_disjoint":True,
        "frozen_ppo_models":audit["frozen_ppo_models"],"candidate_names":[f"pure_ppo_{i}" for i in range(601,606)],
        "outcome_fields":audit["outcome_fields"],"trace_fields":audit["trace_fields"],"prior_margin":audit["prior_margin"],
        "population_replicates":2,"planning_futures":6,"independent_actual_futures":6,"feature_dimension":49,
        "candidate_conditioning":"four separate relative-gain heads; baseline601 is exact0",
        "pool_before_nonlinear_world_scores":True,"new_label_estimand_explicit":True,"regularization":.01,
        "model_names":list(MODEL_NAMES),"new_matrix_model_fits":4,"new_training_mean_references":2,
        "new_ppo_training":0,"new_llm_calls":0,"maximum_full_rollouts":9360,"expected_planning_rollouts":7200,"expected_actual_rollouts":2160,
        "calibration_quantile":.95,"calibration_candidate_maximum":True,"margin_never_reduced":True,
        "qualification_thresholds_changed":False,"formal_final_bank":False,
        "fixed_primary":{"treatment":"M_kernel_margin","control":"mean_M_margin","metric":"final_incapacitation"},
        "fixed_secondary":[["M_kernel_point","mean_M_point"],["R_kernel_point","mean_R_point"]],
        "bootstrap_seed":2026100404,"bootstrap_draws":2000,"minimum_gain":.001}
    write(OUT / "development_manifest.json",m)
    print(json.dumps({"prepared":True,"families":{b:len(v) for b,v in banks.items()},"source_inputs":len(sources),"rollout_cap":9360},ensure_ascii=False),flush=True)


def family_data(block, m):
    entries = [e for e in m["seed_registry"] if e["block"] == block]
    data = [load(OUT / (block+"_planning") / (e["family"]+".npz")) for e in entries]
    return entries, data


def collect_planning(block, m, candidates):
    from fire_evac.source_world_planning import source_world_scenarios, compatible_source_worlds
    from fire_evac.ppo_reference_planning import evaluate_continuation
    from run_source_world_planning import make_env, vector
    records = {r["scenario_id"]:r for r in m["banks"][block]}; scenarios = [DynamicConflictScenario(**r) for r in records.values()]
    worlds = [make_env(scenarios),make_env(scenarios)]; actual = make_env(scenarios)
    folder = OUT / (block+"_planning"); folder.mkdir(exist_ok=True); started = time.monotonic()
    entries = [e for e in m["seed_registry"] if e["block"] == block]
    try:
        for i,e in enumerate(entries):
            path = folder / (e["family"]+".npz")
            if path.exists(): continue
            marker = folder / (e["family"]+"_started.json"); assert not marker.exists(), "preserve interrupted simulations"
            write(marker,{"started_at_utc":stamp(),"family":e["family"],"planned_rollouts":120})
            scenario = DynamicConflictScenario(**records[e["family"]]); risks = np.empty((2,2,5,6,11)); observations=[]; errors=[]; supports=[]
            for population in range(2):
                x,_ = actual.reset(seed=e["population_seeds"][population], options={"scenario":scenario}); observations.append(x.copy())
                support, err, mismatch = compatible_source_worlds(source_world_scenarios(scenario),x,0); assert support.all() and not mismatch
                supports.append(support); errors.append(err)
                for w,hypothesis in enumerate(source_world_scenarios(scenario)):
                    wx,_ = worlds[w].reset(seed=e["population_seeds"][population], options={"scenario":hypothesis}); np.testing.assert_array_equal(wx,x)
                    for plan in range(5):
                        for f,seed in enumerate(e["planning_seeds"][population]): risks[population,w,plan,f]=vector(evaluate_continuation(worlds[w],wx,candidates[plan],seed))
            np.testing.assert_array_equal(observations[0],observations[1]); np.testing.assert_array_equal(supports[0],supports[1])
            labels, means = pooled_labels(risks,supports[0]); tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,observations=np.array(observations),observation=observations[0],features=observable_features(observations[0]),
                planning_risks=risks,pooled_targets=labels,pooled_world_losses=means,supported=supports[0],world_errors=np.array(errors),
                population_seeds=np.array(e["population_seeds"]),planning_seeds=np.array(e["planning_seeds"]),planning_rollouts=np.array(120))
            tmp.replace(path)
            print(f"planning {block} {i+1}/{len(entries)} {time.monotonic()-started:.1f}s",flush=True)
    finally:
        actual.close()
        for env in worlds: env.close()


def train(m):
    path = OUT / "frozen_models.json"
    if path.exists():
        f=read(path)
        for name,h in f["model_sha256"].items(): assert sha(OUT / "models" / (name+".npz"))==h
        return {name:MultiGain.load(OUT / "models" / (name+".npz")) for name in MODEL_NAMES},f
    assert not (OUT / "models").exists()
    entries,data=family_data("train",m); families=[e["family"] for e in entries]
    observations=np.array([d["features"] for d in data]); targets=np.array([d["pooled_targets"] for d in data])
    np.savez_compressed(OUT / "training_dataset.npz",observations=observations,targets=targets,families=np.array(families))
    (OUT / "models").mkdir(); models={}; information={}
    heldout=[r["scenario_id"] for b in ("calibration","test") for r in m["banks"][b]]
    for name in MODEL_NAMES:
        target,kind=name.split("_"); model,info=fit(observations,targets[:,0 if target=="R" else 1],families,kind,families,heldout)
        model.save(OUT / "models" / (name+".npz")); models[name]=model; information[name]=info
    frozen={"frozen_at_utc":stamp(),"training_dataset_sha256":sha(OUT / "training_dataset.npz"),"manifest_sha256":sha(OUT / "development_manifest.json"),
        "training_family_ids":families,"fit_information":information,"model_sha256":{n:sha(OUT / "models" / (n+".npz")) for n in MODEL_NAMES},
        "mean_scores":{"R":targets[:,0].mean(axis=0).tolist(),"M":targets[:,1].mean(axis=0).tolist()},
        "new_matrix_model_fits":4,"new_training_mean_references":2,"no_calibration_test_or_actual_in_fit":True}
    write(path,frozen); print(json.dumps({"models_frozen":True,"training_families":24,"new_matrix_fits":4},ensure_ascii=False),flush=True)
    return models,frozen


def freeze_gates(m, models, frozen):
    path=OUT / "frozen_gates.json"
    if path.exists(): return read(path)
    entries,data=family_data("calibration",m); families=[e["family"] for e in entries]; x=np.array([d["features"] for d in data]); targets=np.array([d["pooled_targets"] for d in data])
    gates={}
    for name in (*MODEL_NAMES,"mean_R","mean_M"):
        target=name[0] if name in MODEL_NAMES else name[-1]; j=0 if target=="R" else 1
        prediction=models[name].predict(x) if name in MODEL_NAMES else np.tile(frozen["mean_scores"][target],(len(x),1))
        margin,residuals,q=calibrate(prediction,targets[:,j],families,m["prior_margin"])
        gates[name]={"margin":margin,"family_max_candidate_optimism":residuals,"empirical_quantile":q}
    body={"frozen_at_utc":stamp(),"frozen_models_sha256":sha(OUT / "frozen_models.json"),"calibration_family_ids":families,
        "gates":gates,"prior_margin":m["prior_margin"],"quantile":.95,"method":"higher","actual_outcomes_used":False,
        "old_nominee_classifier_not_applied_to_new_candidates":True,"margin_reference_not_replacement_for_old_full_gate":True}
    write(path,body); print(json.dumps({"gates_frozen":True,"margins_pp":{n:g["margin"]*100 for n,g in gates.items()}},ensure_ascii=False),flush=True)
    return body


def freeze_choices(block,m,models,frozen,gates):
    entries,data=family_data(block,m); folder=OUT / (block+"_choices"); folder.mkdir(exist_ok=True)
    for e,d in zip(entries,data):
        path=folder / (e["family"]+".json")
        if path.exists(): continue
        predictions={name:models[name].predict(d["features"]) for name in MODEL_NAMES}
        predictions.update({"mean_"+target:np.array(frozen["mean_scores"][target]) for target in ("R","M")})
        choices={name:i for i,name in enumerate(m["candidate_names"])}
        for name,g in predictions.items():
            choices[name+"_point"]=choose_candidate(g); choices[name+"_margin"]=choose_candidate(g,gates["gates"][name]["margin"])
        for j,target in enumerate(("R","M")):
            choices["teacher_"+target+"_point"]=choose_candidate(d["pooled_targets"][j])
            choices["teacher_"+target+"_margin"]=choose_candidate(d["pooled_targets"][j],m["prior_margin"])
        assert len(choices)==21
        write(path,{"frozen_at_utc":stamp(),"family":e["family"],"planning_sha256":sha(OUT / (block+"_planning") / (e["family"]+".npz")),
            "models_sha256":sha(OUT / "frozen_models.json"),"gates_sha256":sha(OUT / "frozen_gates.json"),
            "predictions":{n:g.tolist() for n,g in predictions.items()},"choices":choices,"actual_results_seen":False,
            "teacher_has_simulator_and_fixed_population_privilege":True,"student_true_source_used":False})


def collect_actual(block,m,candidates):
    from fire_evac.ppo_reference_planning import evaluate_continuation
    from run_source_world_planning import make_env,vector
    records={r["scenario_id"]:r for r in m["banks"][block]}; env=make_env([DynamicConflictScenario(**r) for r in records.values()])
    folder=OUT / (block+"_actual"); folder.mkdir(exist_ok=True); entries=[e for e in m["seed_registry"] if e["block"]==block]; started=time.monotonic()
    try:
        for i,e in enumerate(entries):
            path=folder / (e["family"]+".npz")
            if path.exists(): continue
            choice=OUT / (block+"_choices") / (e["family"]+".json"); selection=read(choice)
            planning=OUT / (block+"_planning") / (e["family"]+".npz"); assert selection["planning_sha256"]==sha(planning)
            marker=folder / (e["family"]+"_started.json"); assert not marker.exists(),"preserve interrupted actual"
            write(marker,{"started_at_utc":stamp(),"selection_sha256":sha(choice),"gates_sha256":sha(OUT / "frozen_gates.json")})
            outcomes=np.empty((2,5,6,11)); traces=[]; keys=[]; scenario=DynamicConflictScenario(**records[e["family"]]); p=load(planning)
            for population in range(2):
                x,_=env.reset(seed=e["population_seeds"][population],options={"scenario":scenario}); np.testing.assert_array_equal(x,p["observations"][population])
                for plan in range(5):
                    for f,seed in enumerate(e["evaluation_seeds"][population]):
                        trace=[]; outcomes[population,plan,f]=vector(evaluate_continuation(env,x,candidates[plan],seed,trace)); traces.append(np.array(trace,float)); keys.append((population,plan,f))
            tmp=path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,evaluation_outcomes=outcomes,population_seeds=np.array(e["population_seeds"]),evaluation_seeds=np.array(e["evaluation_seeds"]),
                selection_sha256=np.array(sha(choice)),actual_rollouts=np.array(60),trace_keys=np.array(keys),trace_offsets=np.cumsum([0]+[len(t) for t in traces]),execution_trace=np.concatenate(traces))
            tmp.replace(path); print(f"actual {block} {i+1}/{len(entries)} {time.monotonic()-started:.1f}s",flush=True)
    finally: env.close()


def run(m):
    import torch
    from stable_baselines3 import PPO
    from fire_evac.ppo_reference_planning import FrozenPPOContinuation,policy_parameter_sha256
    check(m); torch.set_num_threads(1); candidates=[]
    for record in m["frozen_ppo_models"]:
        model=PPO.load(record["path"],device="cpu"); assert policy_parameter_sha256(model)==record["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(record["training_seed"],model))
    assert [c.name for c in candidates]==m["candidate_names"]
    collect_planning("train",m,candidates); models,frozen=train(m)
    collect_planning("calibration",m,candidates); gates=freeze_gates(m,models,frozen)
    freeze_choices("calibration",m,models,frozen,gates); collect_actual("calibration",m,candidates)
    collect_planning("test",m,candidates); freeze_choices("test",m,models,frozen,gates); collect_actual("test",m,candidates)
    for record,actor in zip(m["frozen_ppo_models"],candidates): assert policy_parameter_sha256(actor.model)==record["parameter_sha256"]
    check(m)
    write(OUT / "collection_complete.json",{"completed_at_utc":stamp(),"planning_rollouts":7200,"actual_rollouts":2160,"total_full_rollouts":9360,
        "new_matrix_model_fits":4,"new_training_mean_references":2,"new_ppo_training":0,"new_llm_calls":0,
        "frozen_actor_parameters_unchanged":True,"runner_sha256":sha(__file__),
        "family_file_sha256":{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.parent.name.endswith(("_planning","_choices","_actual"))}})
    print(json.dumps({"collection_completed":True,"rollouts":9360,"new_matrix_fits":4},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare-only",action="store_true"); parser.add_argument("--run",action="store_true"); args=parser.parse_args()
    if args.prepare_only: prepare()
    elif args.run: run(read(OUT / "development_manifest.json"))
    else: parser.error("choose --prepare-only or --run")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
