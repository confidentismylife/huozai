"""Fixed proposal enrichment, one worth fit, fresh bounded development evidence."""
from collections import Counter, defaultdict
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
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from fire_evac.dynamic_conflicts import DynamicConflictScenario, make_domain_randomized_trust_bank
from fire_evac.source_world_planning import PLANS, OUTCOME_FIELDS, source_world_scenarios, compatible_source_worlds
from fire_evac.ppo_reference_planning import FrozenPPOContinuation, evaluate_continuation, policy_parameter_sha256, TRACE_FIELDS
from run_source_world_planning import make_env, vector
from run_consistent_gain_crossfit import sha, read, write, write_csv
from consistent_gain_policy import GainRidge, decide
from override_worth_model import proposal, meta_features, labels, filtered_choice, fit_worth, WorthModel

OUT = ROOT/"outputs/override_worth_evidence_development_v1"
PRIOR = ROOT/"outputs/consistent_gain_crossfit_development_v1"
SOURCE = ROOT/"outputs/baseline_relative_plan_distillation_development_v1"
STATUS = ROOT/"outputs/third_round_development_status.json"
PROTOCOL = ROOT/"reports/覆盖价值预测与新增条件校准开发方案_2026-10-03.md"
ARMS = ("pure_ppo_601", "point_proposal", "prior_consistent_guard", "conditional_consistent_guard", "worth_filtered_guard")


def signature(s):
    return json.dumps({k: v for k, v in s.items() if k != "scenario_id"}, sort_keys=True)


def seed_entry(block, j, r, family):
    i = 2*j+r
    a, b, c = (410000000, 430000000, 470000000) if block == "calibration" else (510000000, 530000000, 570000000)
    return {"block": block, "family": family, "rep": r, "population": a+i,
        "planning": [b+1000*i+f for f in range(3)], "evaluation": [c+1000*i+f for f in range(3)]}


def prepare():
    assert not OUT.exists(), "preserve existing outputs"
    previous = read(PRIOR/"completion_manifest.json")
    for file, expected in previous["files"].items():
        assert sha(ROOT/file) == expected
    sources = read(PRIOR/"development_manifest.json")["source_input_sha256"].copy()
    for file, expected in sources.items(): assert sha(file) == expected
    old = read(SOURCE/"development_manifest.json")
    old_scenarios = old["old_scenario_records"] + old["training_scenario_records"] + old["audit_scenario_records"] + old["excluded_previous_audit_records"]
    old_scenarios += read(ROOT/"outputs/teacher_factorial_pilot_development_v1/development_manifest.json")["scenario_records"]
    old_scenarios += read(ROOT/"outputs/set_aware_distillation_development_v1/development_manifest.json")["audit_scenario_records"]
    for folder in ("confirmatory_joint_control", "guarded_confirmatory_v2"):
        for file in ("train_scenarios.json", "final_scenarios.json"):
            old_scenarios += read(ROOT/"outputs"/folder/file)
    pool = [asdict(s) for s in make_domain_randomized_trust_bank(180, 20262201, "WCALP")]
    test = [asdict(s) for s in make_domain_randomized_trust_bank(24, 20262202, "WTEST")]
    assert len({signature(s) for s in pool+test}) == 204
    assert not {signature(s) for s in pool+test}.intersection(map(signature, old_scenarios))
    registry = [seed_entry(block, j, r, s["scenario_id"]) for block, bank in (("calibration", pool), ("test", test)) for j, s in enumerate(bank) for r in range(2)]
    old_seeds = set()
    for file in sources:
        if Path(file).suffix == ".npz" and Path(file).parent.name in ("states", "training_states", "audit_states"):
            with np.load(file, allow_pickle=False) as z:
                for key in ("population_seed", "planning_seeds", "evaluation_seeds"):
                    if key in z: old_seeds.update(map(int, np.asarray(z[key]).reshape(-1)))
    old_seeds |= {seed+9000001 for seed in list(old_seeds)}
    namespaces = [{r["population"] for r in registry}, {s for r in registry for s in r["planning"]}, {s for r in registry for s in r["evaluation"]}]
    namespaces += [{s+9000001 for s in ns} for ns in namespaces.copy()]
    for i, ns in enumerate(namespaces):
        assert not ns.intersection(old_seeds)
        assert all(not ns.intersection(other) for other in namespaces[i+1:])
    OUT.mkdir(); (OUT/"previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes())
    for file, expected in previous["files"].items():
        p = OUT/"previous_status_exact_bytes.json" if (ROOT/file).resolve() == STATUS else ROOT/file
        sources[str(p.resolve())] = expected
    for file in (Path(__file__), PROTOCOL, ROOT/"scripts/override_worth_model.py", ROOT/"tests/test_override_worth_model.py", PRIOR/"completion_manifest.json"):
        sources[str(file.resolve())] = sha(file)
    body = {"role": "fresh_same_generator_override_worth_development_evidence_not_final", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_allowed": False, "source_input_sha256": sources, "prior_completion_sha256": sha(PRIOR/"completion_manifest.json"),
        "previous_status_sha256": sha(STATUS), "calibration_pool_records": pool, "test_records": test,
        "calibration_pool_seed": 20262201, "test_seed": 20262202, "analysis_seed": 20262203,
        "seed_registry": registry, "new_seed_namespaces_disjoint": True, "calibration_maximum_per_type": 6,
        "calibration_maximum_families": 36, "test_families": 24, "maximum_full_rollouts": 2160,
        "ridge_checkpoint": str(PRIOR/"models/final.npz"), "prior_margin": read(PRIOR/"frozen_models.json")["additive_margin"],
        "worth_training_contract": {"states": 66, "families": 33, "class_counts": {"0": 9, "1": 42, "2": 15},
            "features": 68, "regularization": .01, "maximum_iterations": 300, "minimum_proposal_gain": .001,
            "worth_threshold": .8, "harm_threshold": .1, "margin_quantile": .8, "margin_method": "higher"},
        "arms": list(ARMS), "candidate_names": old["candidate_names"], "frozen_ppo_models": old["frozen_ppo_models"],
        "outcome_fields": list(OUTCOME_FIELDS), "trace_fields": list(TRACE_FIELDS), "new_lightweight_classifier_fits": 1,
        "new_ppo_training": 0, "new_llm_calls": 0, "new_deep_network_training": 0}
    write(OUT/"development_manifest.json", body)
    (OUT/"source_snapshot").mkdir()
    for file in (Path(__file__), PROTOCOL, ROOT/"scripts/override_worth_model.py", ROOT/"tests/test_override_worth_model.py"):
        shutil.copy2(file, OUT/"source_snapshot"/file.name)
    print("Prepared: one linear classifier, at most 36 enriched calibration +24 natural test families, 2160 full-rollout cap.", flush=True)


def check_sources(m):
    for file, expected in m["source_input_sha256"].items(): assert sha(file) == expected, file


def train(m):
    if (OUT/"frozen_classifier.json").exists():
        f = read(OUT/"frozen_classifier.json"); assert sha(OUT/"worth_model.npz") == f["model_sha256"]
        assert f["manifest_sha256"] == sha(OUT/"development_manifest.json")
        return WorthModel.load(OUT/"worth_model.npz"), f
    assert not (OUT/"worth_model.npz").exists(), "preserve partial training rather than silently repeat"
    with np.load(SOURCE/"training_dataset.npz", allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
    with np.load(PRIOR/"crossfit_predictions.npz", allow_pickle=False) as z: o = {k: z[k].copy() for k in z.files}
    lookup = {s: i for i, s in enumerate(d["state_ids"])}
    data = []
    for j in np.flatnonzero(o["steps"] == 0):
        p = proposal(o["predictions"][j])
        if p != 12:
            i = lookup[o["state_ids"][j]]; assert d["large_train_mask"][i]
            data.append((i, j, p, meta_features(d["observations"][i], o["predictions"][j], p)))
    x = np.asarray([r[3] for r in data]); families = np.asarray([d["families"][r[0]] for r in data])
    empirical = np.asarray([o["actual_primary_gains"][j, p] for _, j, p, _ in data]); y = labels(empirical)
    np.savez_compressed(OUT/"worth_training_dataset.npz", features=x, labels=y, families=families,
        state_ids=np.asarray([d["state_ids"][i] for i, _, _, _ in data]), plans=np.asarray([p for _, _, p, _ in data]), empirical_gains=empirical)
    model, info = fit_worth(x, y, families)
    assert info["states"] == 66 and info["families"] == 33 and info["class_counts"] == {"0": 9, "1": 42, "2": 15}
    model.save(OUT/"worth_model.npz")
    f = {"manifest_sha256": sha(OUT/"development_manifest.json"), "model_sha256": sha(OUT/"worth_model.npz"),
        "training_dataset_sha256": sha(OUT/"worth_training_dataset.npz"), "fit": info,
        "frozen_before_new_scenario_observations": True, "new_calibration_or_test_used_in_fit": False,
        "class_frequency_reference": [info["class_counts"][str(c)]/66 for c in range(3)]}
    write(OUT/"frozen_classifier.json", f)
    print(f"Classifier frozen: {info['states']} states/{info['families']} families; {info['iterations']} iterations, max gradient {info['max_abs_gradient']:.2g}", flush=True)
    return model, f


def scores(ridge, model, observation):
    gain = ridge.predict(observation); p = proposal(gain)
    prob = np.asarray([0., 1., 0.]) if p == 12 else model.probabilities(meta_features(observation, gain, p))
    return gain, p, prob


def select_calibration(m, ridge, model, frozen):
    path = OUT/"calibration_selection.json"
    if path.exists(): return read(path)
    env = make_env([DynamicConflictScenario(**s) for s in m["calibration_pool_records"]])
    selected, eligible, rows = Counter(), Counter(), []
    try:
        for j, record in enumerate(m["calibration_pool_records"]):
            scenario = DynamicConflictScenario(**record)
            entry = seed_entry("calibration", j, 0, scenario.scenario_id)
            observation, _ = env.reset(seed=entry["population"], options={"scenario": scenario})
            g, p, prob = scores(ridge, model, observation)
            kind = record["conflict_type"]
            if p != 12: eligible[kind] += 1
            choose = p != 12 and selected[kind] < 6
            if choose: selected[kind] += 1
            rows.append({"family": scenario.scenario_id, "pool_index": j, "conflict_type": kind, "proposal": p,
                "proposal_gain_pp": float(g[p]*100), "worth_class_scores": prob.tolist(), "selected": choose})
    finally: env.close()
    body = {"frozen_classifier_sha256": sha(OUT/"frozen_classifier.json"), "selection_before_new_rollouts": True,
        "pool_initial_resets": 180, "eligible_families_by_type": dict(eligible), "selected_families_by_type": dict(selected),
        "selected_families": sum(selected.values()), "records": rows}
    write(path, body)
    print(f"Calibration proposals: {sum(eligible.values())}/180 eligible, selected {sum(selected.values())}/36; by type {dict(selected)}", flush=True)
    return body


def collect(block, records, m, ridge, model, candidates):
    all_scenarios = [DynamicConflictScenario(**r) for _, r in records]
    if not all_scenarios: return
    actual, worlds = make_env(all_scenarios), [make_env(all_scenarios), make_env(all_scenarios)]
    folder = OUT/("calibration_states" if block == "calibration" else "test_states"); folder.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        for n, (j, record) in enumerate(records):
            scenario = DynamicConflictScenario(**record)
            for r in range(2):
                entry = seed_entry(block, j, r, scenario.scenario_id); path = folder/f"{scenario.scenario_id}_{r:02d}.npz"
                observation, _ = actual.reset(seed=entry["population"], options={"scenario": scenario})
                gain, p, prob = scores(ridge, model, observation)
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        np.testing.assert_array_equal(z["observation"], observation); np.testing.assert_array_equal(z["predicted_gains"], gain)
                        np.testing.assert_array_equal(z["worth_probabilities"], prob); assert int(z["proposal"]) == p
                    continue
                supported, errors, mismatch = compatible_source_worlds(source_world_scenarios(scenario), observation, 0)
                assert supported.all() and not mismatch
                plans = [12, p]; risks = np.empty((2, 2, 3, len(OUTCOME_FIELDS))); evaluated = {}; planning_count = 0
                for w, hypothesis in enumerate(source_world_scenarios(scenario)):
                    ox, _ = worlds[w].reset(seed=entry["population"], options={"scenario": hypothesis})
                    for a, plan in enumerate(plans):
                        for f, seed in enumerate(entry["planning"]):
                            key = (w, plan, f)
                            if key not in evaluated:
                                evaluated[key] = vector(evaluate_continuation(worlds[w], ox, candidates[plan], seed)); planning_count += 1
                            risks[w, a, f] = evaluated[key]
                outcomes = np.empty((2, 3, len(OUTCOME_FIELDS))); evaluated, traces = {}, {}
                for a, plan in enumerate(plans):
                    for f, seed in enumerate(entry["evaluation"]):
                        key = (plan, f)
                        if key not in evaluated:
                            trace = []
                            evaluated[key] = vector(evaluate_continuation(actual, observation, candidates[plan], seed, trace)); traces[key] = np.asarray(trace, float)
                        outcomes[a, f] = evaluated[key]
                keys = sorted(evaluated); tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, observation=observation, predicted_gains=gain, proposal=np.asarray(p), worth_probabilities=prob,
                    population_seed=np.asarray(entry["population"]), planning_seeds=np.asarray(entry["planning"]), evaluation_seeds=np.asarray(entry["evaluation"]),
                    supported=supported, world_errors=errors, planning_risks=risks, evaluation_outcomes=outcomes,
                    planning_rollouts=np.asarray(planning_count), evaluation_rollouts=np.asarray(len(evaluated)),
                    trace_keys=np.asarray(keys), trace_offsets=np.cumsum([0]+[len(traces[k]) for k in keys]), execution_trace=np.concatenate([traces[k] for k in keys]))
                tmp.replace(path)
            print(f"{block} {n+1}/{len(records)} {scenario.scenario_id}; {time.monotonic()-started:.1f}s", flush=True)
    finally:
        actual.close()
        for env in worlds: env.close()


def calibrate(m, selection):
    path = OUT/"frozen_conditional_gate.json"
    if path.exists(): return read(path)
    family_scores = defaultdict(list)
    for file in sorted((OUT/"calibration_states").glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            means = z["planning_risks"][z["supported"]].mean(axis=2)[:, :, 0].max(axis=0)
            empirical = means[0]-means[1]
            family_scores[file.stem.split("_")[0]].append(float(z["predicted_gains"][int(z["proposal"])]-empirical))
    worst = {family: max(v) for family, v in family_scores.items()}
    quantile = float(np.quantile(list(worst.values()), .8, method="higher")) if worst else None
    margin = max(m["prior_margin"], .001, quantile if quantile is not None else .001)
    gate = {"prior_margin": m["prior_margin"], "conditional_margin": margin, "new_nominee_error_quantile": quantile,
        "calibration_families": len(worst), "family_max_nominee_optimism": worst, "margin_never_reduced": True,
        "worth_threshold": .8, "harm_threshold": .1, "frozen_before_test_observations": True,
        "selection_sha256": sha(OUT/"calibration_selection.json"), "classifier_sha256": sha(OUT/"frozen_classifier.json"),
        "calibration_cache_sha256": {file.name: sha(file) for file in sorted((OUT/"calibration_states").glob("*.npz"))},
        "no_safety_probability_or_coverage_guarantee": True}
    write(path, gate)
    print(f"Conditional gate frozen: {len(worst)} families; margin {margin*100:.4f} pp (prior {m['prior_margin']*100:.4f})", flush=True)
    return gate


def interval(values, kinds):
    rng, sample = np.random.default_rng(20262203), np.zeros(2000)
    values, kinds = np.asarray(values), np.asarray(kinds)
    for kind in sorted(set(kinds)):
        part = values[kinds == kind]; sample += part[rng.integers(len(part), size=(2000, len(part)))].sum(axis=1)
    return np.quantile(sample/len(values), [.025, .975]).tolist()


def analyze(m, gate, classifier):
    rows, episodes, arms_summary, quality = [], [], [], []
    budget = Counter(); family_values = {arm: defaultdict(list) for arm in ARMS}
    bank_lookup = {s["scenario_id"]: s for s in m["calibration_pool_records"]+m["test_records"]}
    all_cache = []
    for block, folder in (("calibration", "calibration_states"), ("test", "test_states")):
        for path in sorted((OUT/folder).glob("*.npz")):
            all_cache.append(path)
            family = path.stem.split("_")[0]; record = bank_lookup[family]
            with np.load(path, allow_pickle=False) as z:
                p = int(z["proposal"]); prior = decide(z["predicted_gains"], m["prior_margin"])["execution"]
                conditional = decide(z["predicted_gains"], gate["conditional_margin"])["execution"]
                filtered = filtered_choice(conditional, z["worth_probabilities"])
                assert conditional in (12, p) and filtered in (12, conditional)
                assert conditional == 12 or prior != 12
                choices = [12, p, prior, conditional, filtered]
                np.testing.assert_allclose(z["planning_risks"][..., [0, 2, 3]].sum(axis=-1), 1, atol=1e-12, rtol=0)
                np.testing.assert_allclose(z["evaluation_outcomes"][..., [0, 2, 3]].sum(axis=-1), 1, atol=1e-12, rtol=0)
                base, proposed = z["evaluation_outcomes"]
                actual_gain = float((base[:, 0]-proposed[:, 0]).mean())
                means = z["planning_risks"][z["supported"]].mean(axis=2)[:, :, 0].max(axis=0)
                planning_gain = float(means[0]-means[1])
                rows.append({"block": block, "state_id": path.stem, "family": family, "conflict_type": record["conflict_type"],
                    "proposal": m["candidate_names"][p], "proposal_index": p, "predicted_gain_pp": float(z["predicted_gains"][p]*100),
                    "harm_score": float(z["worth_probabilities"][0]), "neutral_score": float(z["worth_probabilities"][1]), "worth_score": float(z["worth_probabilities"][2]),
                    "planning_gain_pp": planning_gain*100, "actual_gain_pp": actual_gain*100,
                    "planning_class": int(labels(planning_gain)), "actual_class": int(labels(actual_gain)),
                    "prior_override": prior != 12, "conditional_override": conditional != 12, "filtered_override": filtered != 12,
                    "score_filter_passed": bool(z["worth_probabilities"][2] >= .8 and z["worth_probabilities"][0] <= .1)})
                for arm, chosen in zip(ARMS, choices):
                    values = base if chosen == 12 else proposed
                    if block == "test": family_values[arm][family].extend(values)
                    for f, seed in enumerate(z["evaluation_seeds"]):
                        episodes.append({"block": block, "state_id": path.stem, "family": family, "arm": arm, "chosen_plan": m["candidate_names"][chosen],
                            "future_seed": int(seed), **dict(zip(OUTCOME_FIELDS, values[f]))})
                trace, offsets, keys = z["execution_trace"], z["trace_offsets"], z["trace_keys"]
                for j, (plan, future) in enumerate(keys):
                    t = trace[offsets[j]:offsets[j+1]]; o = z["evaluation_outcomes"][0 if plan == 12 else 1, future]
                    np.testing.assert_allclose([len(t), t[:, 4].mean(), (t[:, 1] == 8).sum(), t[:, 5].mean(), t[0, 1], t[0, 2]], o[5:], atol=1e-12, rtol=0)
                budget["planning_rollouts"] += int(z["planning_rollouts"]); budget["evaluation_rollouts"] += int(z["evaluation_rollouts"])
                budget["actual_trace_steps"] += len(trace); budget["neutral_positive_release_steps"] += int(((trace[:, 3] == 2)&(trace[:, 4] > 0)).sum())
                budget["hold_blocked_steps"] += int(trace[:, 5].sum())
    write_csv(OUT/"proposal_evidence.csv", rows); write_csv(OUT/"evaluation_episode_results.csv", episodes)
    for block in ("calibration", "test"):
        eligible = [r for r in rows if r["block"] == block and r["proposal_index"] != 12]
        if eligible:
            p = np.asarray([[r[k] for k in ("harm_score", "neutral_score", "worth_score")] for r in eligible]); y = np.asarray([r["actual_class"] for r in eligible])
            prior = np.asarray(classifier["class_frequency_reference"]); matrix = np.zeros((3, 3), int)
            for truth, predicted in zip(y, p.argmax(axis=1)): matrix[truth, predicted] += 1
            quality.append({"block": block, "eligible_states": len(eligible), "eligible_families": len({r['family'] for r in eligible}),
                "actual_class_counts": {str(c): int((y == c).sum()) for c in range(3)}, "planning_class_counts": dict(Counter(str(r["planning_class"]) for r in eligible)),
                "brier_model": float(((p-np.eye(3)[y])**2).sum(axis=1).mean()), "brier_frequency_reference": float(((prior-np.eye(3)[y])**2).sum(axis=1).mean()),
                "log_loss_model": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
                "log_loss_frequency_reference": float(-np.log(prior[y]).mean()), "confusion_true_by_predicted": matrix.tolist(),
                "score_filter_passed_states": sum(r["score_filter_passed"] for r in eligible),
                "planning_actual_class_disagreements": sum(r["actual_class"] != r["planning_class"] for r in eligible)})
        else: quality.append({"block": block, "eligible_states": 0, "eligible_families": 0, "evidence_insufficient": True})
    ids = sorted(family_values[ARMS[0]]); assert len(ids) == 24
    kinds = [bank_lookup[f]["conflict_type"] for f in ids]
    matrices = {arm: np.asarray([np.mean(family_values[arm][f], axis=0) for f in ids]) for arm in ARMS}
    effects = []
    for arm in ARMS:
        armrows = [r for r in episodes if r["block"] == "test" and r["arm"] == arm]
        arms_summary.append({"arm": arm, **{field: float(matrices[arm][:, i].mean()) for i, field in enumerate(OUTCOME_FIELDS[:9])}})
        for control in ("pure_ppo_601", "prior_consistent_guard"):
            delta = (matrices[arm][:, 0]-matrices[control][:, 0])*100; lo, hi = interval(delta, kinds)
            effects.append({"treatment": arm, "control": control, "delta_incap_pp": float(delta.mean()), "exploratory_ci_low_pp": lo, "exploratory_ci_high_pp": hi,
                "improved_families": int((delta < -1e-12).sum()), "worsened_families": int((delta > 1e-12).sum()), "tied_families": int((abs(delta) <= 1e-12).sum())})
    gate_rows = []
    for block in ("calibration", "test"):
        rs = [r for r in rows if r["block"] == block]
        for name, flag in (("point_proposal", "proposal_index"), ("prior_consistent_guard", "prior_override"), ("conditional_consistent_guard", "conditional_override"), ("worth_filtered_guard", "filtered_override")):
            chosen = [r for r in rs if r[flag] != 12] if flag == "proposal_index" else [r for r in rs if r[flag]]
            gate_rows.append({"block": block, "arm": name, "states": len(rs), "overrides": len(chosen),
                "actual_beneficial": sum(r["actual_class"] == 2 for r in chosen), "actual_harmful": sum(r["actual_class"] == 0 for r in chosen),
                "actual_neutral_or_small": sum(r["actual_class"] == 1 for r in chosen)})
    write_csv(OUT/"arm_summary.csv", arms_summary); write_csv(OUT/"paired_effects.csv", effects); write_csv(OUT/"gate_summary.csv", gate_rows)
    total = budget["planning_rollouts"]+budget["evaluation_rollouts"]; assert total <= 2160
    summary = {"role": m["role"], "freeze_allowed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_selected_families": read(OUT/"calibration_selection.json")["selected_families"], "test_families": 24,
        "one_new_lightweight_classifier_fit": True, "new_ppo_training": 0, "new_llm_calls": 0, "new_deep_network_training": 0,
        "total_unique_full_rollouts": total, "budget": dict(budget), "prior_margin_pp": m["prior_margin"]*100,
        "conditional_margin_pp": gate["conditional_margin"]*100, "classifier_quality": quality, "gate_summary": gate_rows,
        "test_arm_summary": arms_summary, "test_paired_effects": effects,
        "primary_paired_effect": next(r for r in effects if r["treatment"] == "worth_filtered_guard" and r["control"] == "prior_consistent_guard"),
        "actual_efficacy_advantage_established": False, "hard_shield_qualification_passed": False,
        "limits": ["New same-generator development evidence, not powered confirmation or final testing.",
            "Calibration pool enriched using observable proposals; not natural population prevalence or calibrated deployment risk.",
            "Linear classifier trained with OOF planning labels, evaluated on distinct actual future labels; finite sampling/world differences.",
            "Three planning and actual futures give empirical labels, not known true expected utility or safety probabilities.",
            "Two correlated population starts per family; exploratory intervals conditional on fixed models without multiplicity adjustment.",
            "No behavior/map/noise shift, full17 teacher search, dynamic meta-reselection, fair LLM contribution or real-fire validation."]}
    write(OUT/"summary.json", summary); check_sources(m)
    write(OUT/"analysis_manifest.json", {"runner_sha256": sha(__file__), "development_manifest_sha256": sha(OUT/"development_manifest.json"),
        "frozen_classifier_sha256": sha(OUT/"frozen_classifier.json"), "frozen_gate_sha256": sha(OUT/"frozen_conditional_gate.json"),
        "summary_sha256": sha(OUT/"summary.json"), "csv_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*.csv"))},
        "cache_sha256": {str(p.relative_to(OUT)): sha(p) for p in all_cache}})
    print(json.dumps({"completed": True, "calibration_families": summary["calibration_selected_families"], "full_rollouts": total,
        "primary": summary["primary_paired_effect"], "gates": gate_rows}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only: prepare(); return
    if not args.run: parser.error("choose --prepare-only or --run")
    m = read(OUT/"development_manifest.json"); check_sources(m); torch.set_num_threads(1)
    ridge = GainRidge.load(m["ridge_checkpoint"]); model, frozen = train(m)
    selection = select_calibration(m, ridge, model, frozen)
    candidates = list(PLANS)
    for r in m["frozen_ppo_models"]:
        actor = PPO.load(r["path"], device="cpu"); assert policy_parameter_sha256(actor) == r["parameter_sha256"]
        candidates.append(FrozenPPOContinuation(r["training_seed"], actor))
    chosen = {r["pool_index"] for r in selection["records"] if r["selected"]}
    collect("calibration", [(j, r) for j, r in enumerate(m["calibration_pool_records"]) if j in chosen], m, ridge, model, candidates)
    gate = calibrate(m, selection)
    write(OUT/"test_started_after_freeze.json", {"classifier_sha256": sha(OUT/"frozen_classifier.json"), "gate_sha256": sha(OUT/"frozen_conditional_gate.json")})
    collect("test", list(enumerate(m["test_records"])), m, ridge, model, candidates)
    for record, actor in zip(m["frozen_ppo_models"], candidates[12:]): assert policy_parameter_sha256(actor.model) == record["parameter_sha256"]
    analyze(m, gate, frozen)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
