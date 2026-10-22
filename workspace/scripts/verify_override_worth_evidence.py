"""Independent arithmetic and model replay; no new fits or full rollouts."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import sys

import numpy as np
from run_consistent_gain_crossfit import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/override_worth_evidence_development_v1"
PRIOR = ROOT/"outputs/consistent_gain_crossfit_development_v1"
SOURCE = ROOT/"outputs/baseline_relative_plan_distillation_development_v1"


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)


def point(g):
    assert np.isfinite(g).all() and g[12] == 0
    p = min(range(17), key=lambda j: (-g[j], j != 12, j))
    return p if g[p] > .001 else 12


def class_label(g):
    return 0 if g < -1e-12 else 2 if g > .001 else 1


def meta(obs, gain, p):
    obs = obs.astype(float).copy(); obs[28:33] = 0
    onehot = np.zeros(17); onehot[p] = 1
    return np.r_[obs, onehot, gain[p], gain[p]-np.max(np.delete(gain, p))]


def ridge_predict(model, obs):
    x = obs.astype(float).copy(); x[28:33] = 0
    g = ((x-model["mean"])/model["std"])@model["coefficients"]+model["intercept"]
    g[12] = 0
    return g


def probabilities(model, x):
    logits = ((x-model["mean"])/model["std"])@model["coefficient"]+model["intercept"]
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits); return exp/exp.sum(axis=-1, keepdims=True)


def guarded(g, margin):
    score = g-margin; score[12] = 0
    p = min(range(17), key=lambda j: (-score[j], j != 12, j))
    return p if score[p] > .001 else 12


def bootstrap(delta, kinds):
    rng, sample = np.random.default_rng(20262203), np.zeros(2000)
    kinds = np.asarray(kinds)
    for kind in sorted(set(kinds)):
        part = delta[kinds == kind]; sample += part[rng.integers(len(part), size=(2000, len(part)))].sum(axis=1)
    return np.quantile(sample/len(delta), [.025, .975])


def main():
    m, f, gate, summary, analysis = (read(OUT/p) for p in ("development_manifest.json", "frozen_classifier.json", "frozen_conditional_gate.json", "summary.json", "analysis_manifest.json"))
    for path, expected in m["source_input_sha256"].items(): assert sha(path) == expected
    assert f["manifest_sha256"] == sha(OUT/"development_manifest.json") and f["model_sha256"] == sha(OUT/"worth_model.npz")
    assert f["training_dataset_sha256"] == sha(OUT/"worth_training_dataset.npz")
    assert analysis["runner_sha256"] == sha(ROOT/"scripts/run_override_worth_evidence.py")
    assert analysis["summary_sha256"] == sha(OUT/"summary.json")
    assert analysis["development_manifest_sha256"] == sha(OUT/"development_manifest.json")
    assert analysis["frozen_classifier_sha256"] == sha(OUT/"frozen_classifier.json")
    assert analysis["frozen_gate_sha256"] == sha(OUT/"frozen_conditional_gate.json")
    for path, expected in analysis["csv_sha256"].items(): assert sha(OUT/path) == expected
    for path, expected in analysis["cache_sha256"].items(): assert sha(OUT/path) == expected
    marker = read(OUT/"test_started_after_freeze.json")
    assert marker["classifier_sha256"] == sha(OUT/"frozen_classifier.json") and marker["gate_sha256"] == sha(OUT/"frozen_conditional_gate.json")
    load = lambda path: {k: v.copy() for k, v in np.load(path, allow_pickle=False).items()}
    model, ridge, data = load(OUT/"worth_model.npz"), load(m["ridge_checkpoint"]), load(OUT/"worth_training_dataset.npz")
    old, oof = load(SOURCE/"training_dataset.npz"), load(PRIOR/"crossfit_predictions.npz")
    lookup = {state: i for i, state in enumerate(old["state_ids"])}
    reconstructed = []
    for j in np.flatnonzero(oof["steps"] == 0):
        p = point(oof["predictions"][j])
        if p != 12:
            i = lookup[oof["state_ids"][j]]; assert old["large_train_mask"][i]
            reconstructed.append((oof["state_ids"][j], old["families"][i], p, oof["actual_primary_gains"][j,p], meta(old["observations"][i], oof["predictions"][j], p)))
    np.testing.assert_array_equal(data["state_ids"], [r[0] for r in reconstructed]); close(data["features"], np.asarray([r[4] for r in reconstructed]))
    np.testing.assert_array_equal(data["plans"], [r[2] for r in reconstructed]); close(data["empirical_gains"], [r[3] for r in reconstructed])
    np.testing.assert_array_equal(data["labels"], [class_label(r[3]) for r in reconstructed])
    np.testing.assert_array_equal(data["families"], [r[1] for r in reconstructed])
    counts = Counter(data["families"]); assert len(counts) == 33 and len(reconstructed) == 66
    assert f["fit"]["class_counts"] == {str(c): int((data["labels"] == c).sum()) for c in range(3)}
    assert f["fit"]["training_family_ids"] == sorted(counts)
    w = np.asarray([1/(len(counts)*counts[family]) for family in data["families"]])
    mean = (data["features"]*w[:,None]).sum(axis=0)
    sd = np.sqrt(((data["features"]-mean)**2*w[:,None]).sum(axis=0)); sd = np.where(sd < .001, 1., sd)
    close(model["mean"], mean); close(model["std"], sd)
    p = probabilities(model, data["features"]); residual = (p-np.eye(3)[data["labels"]])*w[:,None]
    gradient = ((data["features"]-mean)/sd).T@residual+.01*model["coefficient"]
    max_gradient = max(float(abs(gradient).max()), float(abs(residual.sum(axis=0)).max()))
    assert max_gradient < 1e-5; close(max_gradient, f["fit"]["max_abs_gradient"])
    loss = (-np.log(p[np.arange(len(p)), data["labels"]])*w).sum()+.005*(model["coefficient"]**2).sum()
    close(loss, f["fit"]["loss"])
    selection = read(OUT/"calibration_selection.json")
    assert selection["frozen_classifier_sha256"] == sha(OUT/"frozen_classifier.json") and gate["selection_sha256"] == sha(OUT/"calibration_selection.json")
    tally, eligible = Counter(), Counter()
    for j, row in enumerate(selection["records"]):
        record = m["calibration_pool_records"][j]
        assert row["pool_index"] == j and row["family"] == record["scenario_id"] and row["conflict_type"] == record["conflict_type"]
        if row["proposal"] != 12:
            assert row["proposal_gain_pp"] > .10; eligible[row["conflict_type"]] += 1
        selected = row["proposal"] != 12 and tally[row["conflict_type"]] < 6
        assert selected == row["selected"]
        if selected: tally[row["conflict_type"]] += 1
    assert dict(tally) == selection["selected_families_by_type"] and dict(eligible) == selection["eligible_families_by_type"]
    assert sum(tally.values()) == selection["selected_families"] == 30
    new_scenarios = m["calibration_pool_records"]+m["test_records"]
    signature = lambda s: json.dumps({k:v for k,v in s.items() if k != "scenario_id"},sort_keys=True)
    oldm = read(SOURCE/"development_manifest.json")
    previous = oldm["old_scenario_records"]+oldm["training_scenario_records"]+oldm["audit_scenario_records"]+oldm["excluded_previous_audit_records"]
    previous += read(ROOT/"outputs/set_aware_distillation_development_v1/development_manifest.json")["audit_scenario_records"]
    previous += read(ROOT/"outputs/teacher_factorial_pilot_development_v1/development_manifest.json")["scenario_records"]
    for folder in ("confirmatory_joint_control", "guarded_confirmatory_v2"):
        for name in ("train_scenarios.json", "final_scenarios.json"): previous += read(ROOT/"outputs"/folder/name)
    assert len(set(map(signature, new_scenarios))) == 204 and not set(map(signature,new_scenarios)).intersection(map(signature,previous))
    previous_seeds = set()
    for path in m["source_input_sha256"]:
        if Path(path).suffix == ".npz" and Path(path).parent.name in ("states", "training_states", "audit_states"):
            with np.load(path, allow_pickle=False) as z:
                for key in ("population_seed", "planning_seeds", "evaluation_seeds"):
                    if key in z: previous_seeds.update(map(int,np.asarray(z[key]).reshape(-1)))
    previous_seeds |= {s+9000001 for s in list(previous_seeds)}
    registry = {(r["block"],r["family"],r["rep"]):r for r in m["seed_registry"]}
    namespaces = [{r["population"] for r in registry.values()},{s for r in registry.values() for s in r["planning"]},{s for r in registry.values() for s in r["evaluation"]}]
    namespaces += [{s+9000001 for s in ns} for ns in namespaces.copy()]
    for i,ns in enumerate(namespaces):
        assert not ns.intersection(previous_seeds) and all(not ns.intersection(other) for other in namespaces[i+1:])
    evidence = rows(OUT/"proposal_evidence.csv"); eindex = {r["state_id"]:r for r in evidence}
    episodes = rows(OUT/"evaluation_episode_results.csv"); epindex = {(r["state_id"],r["arm"],int(r["future_seed"])):r for r in episodes}
    actuals = {arm:defaultdict(list) for arm in m["arms"]}; family_errors = defaultdict(list); budget = Counter()
    rejected_opportunities = {}; predicted_confident_harm = []
    for block, folder, expected_states in (("calibration","calibration_states",60),("test","test_states",48)):
        files = sorted((OUT/folder).glob("*.npz")); assert len(files) == expected_states
        for path in files:
            z = load(path); family, rep = path.stem.split("_"); entry = registry[(block,family,int(rep))]
            assert int(z["population_seed"]) == entry["population"]
            np.testing.assert_array_equal(z["planning_seeds"],entry["planning"]); np.testing.assert_array_equal(z["evaluation_seeds"],entry["evaluation"])
            gain = ridge_predict(ridge,z["observation"]); p = point(gain); close(gain,z["predicted_gains"]); assert p == int(z["proposal"])
            probs = np.array([0.,1.,0.]) if p == 12 else probabilities(model,meta(z["observation"],gain,p))
            close(probs,z["worth_probabilities"])
            prior = guarded(gain,m["prior_margin"]); conditional = guarded(gain,gate["conditional_margin"])
            passed = bool(probs[2] >= .8 and probs[0] <= .1); filtered = conditional if passed else 12
            assert conditional in (12,p) and filtered in (12,conditional) and (conditional == 12 or prior != 12)
            for array in (z["planning_risks"],z["evaluation_outcomes"]): close(array[..., [0,2,3]].sum(axis=-1),1.)
            if p == 12:
                np.testing.assert_array_equal(z["evaluation_outcomes"][0],z["evaluation_outcomes"][1]); np.testing.assert_array_equal(z["planning_risks"][:,0],z["planning_risks"][:,1])
            primary = z["planning_risks"][z["supported"]].mean(axis=2)[:,:,0].max(axis=0)
            plan_gain, actual_gain = float(primary[0]-primary[1]),float((z["evaluation_outcomes"][0,:,0]-z["evaluation_outcomes"][1,:,0]).mean())
            row = eindex[path.stem]
            close([float(row[k]) for k in ("predicted_gain_pp","harm_score","neutral_score","worth_score","planning_gain_pp","actual_gain_pp")],
                  [gain[p]*100,*probs,plan_gain*100,actual_gain*100])
            assert int(row["actual_class"]) == class_label(actual_gain) and int(row["planning_class"]) == class_label(plan_gain)
            for key, flag in (("prior_override",prior!=12),("conditional_override",conditional!=12),("filtered_override",filtered!=12),("score_filter_passed",passed)):
                assert (row[key] == "True") == flag
            if block == "calibration": family_errors[family].append(gain[p]-plan_gain)
            if p != 12 and passed and class_label(actual_gain) == 0: predicted_confident_harm.append(path.stem)
            choices = [12,p,prior,conditional,filtered]
            for arm, choice in zip(m["arms"],choices):
                outcome = z["evaluation_outcomes"][0 if choice == 12 else 1]
                if block == "test": actuals[arm][family].extend(outcome)
                for j,seed in enumerate(z["evaluation_seeds"]):
                    erow = epindex[(path.stem,arm,int(seed))]
                    assert erow["chosen_plan"] == m["candidate_names"][choice]
                    close([float(erow[f]) for f in m["outcome_fields"]],outcome[j])
            keys, offsets, trace = z["trace_keys"],z["trace_offsets"],z["execution_trace"]
            np.testing.assert_array_equal(keys,sorted({(plan,future) for plan in (12,p) for future in range(3)}))
            assert offsets[0] == 0 and offsets[-1] == len(trace) and np.all(np.diff(offsets)>0)
            assert np.isin(trace[:,1],np.arange(9)).all() and np.isin(trace[:,3],(0,1,2)).all()
            actual_joint = np.where(trace[:,1]==8,8,np.where(trace[:,3]==2,-1,4*trace[:,3]+trace[:,1]%4))
            np.testing.assert_array_equal(trace[:,2],actual_joint)
            for j,(plan,future) in enumerate(keys):
                t = trace[offsets[j]:offsets[j+1]]; outcome = z["evaluation_outcomes"][0 if plan == 12 else 1,future]
                np.testing.assert_array_equal(t[:,0],np.arange(len(t)))
                close([len(t),t[:,4].mean(),(t[:,1]==8).sum(),t[:,5].mean(),t[0,1],t[0,2]],outcome[5:])
            assert int(z["planning_rollouts"]) == (6 if p == 12 else 12) and int(z["evaluation_rollouts"]) == len(keys)
            budget["planning_rollouts"] += int(z["planning_rollouts"]); budget["evaluation_rollouts"] += len(keys)
            budget["actual_trace_steps"] += len(trace); budget["neutral_positive_release_steps"] += int(((trace[:,3]==2)&(trace[:,4]>0)).sum()); budget["hold_blocked_steps"] += int(trace[:,5].sum())
    maxima = {family:float(max(values)) for family,values in family_errors.items()}
    for family,value in maxima.items(): close(value,gate["family_max_nominee_optimism"][family])
    q = float(np.quantile(list(maxima.values()),.8,method="higher")); close(q,gate["new_nominee_error_quantile"])
    close(max(m["prior_margin"],.001,q),gate["conditional_margin"])
    for filename, expected in gate["calibration_cache_sha256"].items(): assert sha(OUT/"calibration_states"/filename) == expected
    kinds_map = {r["scenario_id"]:r["conflict_type"] for r in m["test_records"]}; ids = sorted(kinds_map); kinds = [kinds_map[f] for f in ids]
    matrices = {arm:np.asarray([np.mean(actuals[arm][family],axis=0) for family in ids]) for arm in m["arms"]}
    for row in summary["test_arm_summary"]: close([row[f] for f in m["outcome_fields"][:9]],matrices[row["arm"]].mean(axis=0)[:9])
    for row in summary["test_paired_effects"]:
        delta = (matrices[row["treatment"]][:,0]-matrices[row["control"]][:,0])*100
        close([row["delta_incap_pp"],row["exploratory_ci_low_pp"],row["exploratory_ci_high_pp"]],[delta.mean(),*bootstrap(delta,kinds)])
        assert [row[k] for k in ("improved_families","worsened_families","tied_families")] == [int((delta < -1e-12).sum()),int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum())]
    for group in summary["gate_summary"]:
        rs = [r for r in evidence if r["block"] == group["block"]]
        flag = {"point_proposal":None,"prior_consistent_guard":"prior_override","conditional_consistent_guard":"conditional_override","worth_filtered_guard":"filtered_override"}[group["arm"]]
        eligible = [r for r in rs if int(r["proposal_index"]) != 12] if flag is None else [r for r in rs if r[flag]=="True"]
        assert len(eligible) == group["overrides"]
        assert [group[k] for k in ("actual_harmful","actual_neutral_or_small","actual_beneficial")] == [sum(int(r["actual_class"])==c for r in eligible) for c in range(3)]
    for quality in summary["classifier_quality"]:
        rs = [r for r in evidence if r["block"] == quality["block"] and int(r["proposal_index"]) != 12]
        p = np.asarray([[float(r[k]) for k in ("harm_score","neutral_score","worth_score")] for r in rs]); y = np.asarray([int(r["actual_class"]) for r in rs]); freq = np.asarray(f["class_frequency_reference"])
        confusion = np.zeros((3,3),int)
        for true,pred in zip(y,p.argmax(axis=1)): confusion[true,pred] += 1
        np.testing.assert_array_equal(confusion,quality["confusion_true_by_predicted"])
        close([quality[k] for k in ("brier_model","brier_frequency_reference","log_loss_model","log_loss_frequency_reference")],
              [((p-np.eye(3)[y])**2).sum(axis=1).mean(),((freq-np.eye(3)[y])**2).sum(axis=1).mean(),-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean(),-np.log(freq[y]).mean()])
    assert dict(budget) == summary["budget"] and budget["planning_rollouts"]+budget["evaluation_rollouts"] == 1674
    supplements = {}
    for block in ("calibration","test"):
        rs = [r for r in evidence if r["block"] == block and int(r["proposal_index"]) != 12]
        supplements[block] = {"score_filter_passed_actual_classes": {str(c):sum(r["score_filter_passed"]=="True" and int(r["actual_class"])==c for r in rs) for c in range(3)},
            "beneficial_proposal_families": sorted({r["family"] for r in rs if int(r["actual_class"])==2}),
            "harmful_proposal_families": sorted({r["family"] for r in rs if int(r["actual_class"])==0}),
            "high_worth_actual_harm_states": [r["state_id"] for r in rs if r["score_filter_passed"]=="True" and int(r["actual_class"])==0],
            "missed_beneficial_by_prior_guard": sum(int(r["actual_class"])==2 and r["prior_override"]!="True" for r in rs),
            "missed_beneficial_by_conditional_guard": sum(int(r["actual_class"])==2 and r["conditional_override"]!="True" for r in rs),
            "missed_beneficial_by_filtered_guard": sum(int(r["actual_class"])==2 and r["filtered_override"]!="True" for r in rs)}
    result = {"all_checks_passed": True,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_inputs_unchanged":len(m["source_input_sha256"]),"new_full_rollouts_in_verification":0,"new_fits_in_verification":0,"new_llm_calls":0,
        "classifier_gradient_and_training_only_normalization_checked":True,"model_max_abs_gradient":max_gradient,
        "screening_order_recomputed_from_saved_records":True,"screening_selected_predictions_verified_from_caches":True,
        "all204_scenarios_and_seed_namespaces_disjoint_checked":True,"calibration_families":30,"natural_test_families":24,
        "population_states_verified":108,"planning_rollouts_verified":budget["planning_rollouts"],"evaluation_rollouts_verified":budget["evaluation_rollouts"],
        "actual_trace_steps_verified":budget["actual_trace_steps"],"all_five_arm_outcomes_and_family_bootstrap_recomputed":True,
        "worth_and_planning_label_quality_recomputed":True,"actual_supplement":supplements,
        "no_calibration_proposals_in_safe_exit_congested_pool":True,
        "verifier_sha256":sha(__file__),"analysis_manifest_sha256":sha(OUT/"analysis_manifest.json")}
    write(OUT/"verification.json",result)
    print(json.dumps({k:result[k] for k in ("all_checks_passed","source_inputs_unchanged","population_states_verified","actual_trace_steps_verified","actual_supplement")},ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
