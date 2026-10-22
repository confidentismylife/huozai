"""Independent read-only arithmetic/model/trace verification; no simulations."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/set_aware_distillation_development_v1"
SOURCE = ROOT / "outputs/baseline_relative_plan_distillation_development_v1"
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.relative_plan_ranking import ObservableScoreStudent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, tolerance=1e-10):
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=0)


def family_weights(families):
    counts = Counter(families)
    return np.asarray([1 / (len(counts) * counts[f]) for f in families], np.float32)


def lexical(scores):
    return min(range(17), key=lambda p: tuple(scores[p]) + (p,))


def nomination(gains, probabilities):
    return min(range(17), key=lambda p: (-probabilities[p],) + tuple(-gains[p]) + (p,))


def interval(values, kinds):
    rng = np.random.default_rng(20262002)
    samples = np.zeros(2000)
    values, kinds = np.asarray(values), np.asarray(kinds)
    for kind in sorted(set(kinds)):
        part = values[kinds == kind]
        indexes = rng.integers(len(part), size=(2000, len(part)))
        samples += part[indexes].sum(axis=1)
    return np.quantile(samples / len(values), [.025, .975])


class ReloadedModel:
    def __init__(self, record):
        assert sha(OUT / record["checkpoint"]) == record["sha256"]
        saved = torch.load(OUT / record["checkpoint"], weights_only=True, map_location="cpu")
        assert saved["contract"] == "observable_set_gain_17x7_v1"
        assert saved["objective"] == record["objective"] and saved["seed"] == record["seed"]
        assert saved["selected_epoch"] == record["selected_epoch"]
        torch.manual_seed(record["seed"])
        self.layers = nn.Sequential(nn.Linear(49, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 119))
        initial = {"layers." + name: value for name, value in self.layers.state_dict().items()}
        initial_hash = hashlib.sha256(b"".join(name.encode() + value.detach().numpy().tobytes()
            for name, value in sorted(initial.items()))).hexdigest()
        assert initial_hash == record["initial_network_sha256"]
        self.layers.load_state_dict({name.removeprefix("layers."): value for name, value in saved["state_dict"].items()})
        self.layers.eval()
        self.mean, self.std, self.scale = (saved[k].numpy() for k in ("mean", "std", "scale"))

    def predict(self, observation):
        x = np.asarray(observation, np.float32).copy()
        x[28:33] = 0
        with torch.inference_mode():
            raw = self.layers(torch.as_tensor((x - self.mean) / self.std).unsqueeze(0)).reshape(1, 17, 7)
            gains = (raw[..., :6] - raw[:, 12:13, :6])[0].numpy() * self.scale
            probabilities = torch.softmax(raw[..., 6], dim=1)[0].numpy()
        np.testing.assert_array_equal(gains[12], np.zeros(6))
        return gains, probabilities


def main():
    torch.set_num_threads(1)
    manifest, frozen, summary, analysis = (read(OUT / p) for p in
        ("development_manifest.json", "frozen_students.json", "summary.json", "analysis_manifest.json"))
    for file, expected in manifest["source_input_sha256"].items():
        assert sha(file) == expected, file
    assert frozen["manifest_sha256"] == sha(OUT / "development_manifest.json")
    assert frozen["derived_dataset_sha256"] == sha(OUT / "derived_training_targets.npz")
    assert analysis["analysis_code_sha256"] == sha(ROOT / "scripts/analyze_set_aware_distillation.py")
    assert analysis["summary_sha256"] == sha(OUT / "summary.json")
    assert analysis["integrity_sha256"] == sha(OUT / "integrity_audit.json")
    assert analysis["development_manifest_sha256"] == sha(OUT / "development_manifest.json")
    assert analysis["frozen_students_sha256"] == sha(OUT / "frozen_students.json")
    for file, expected in analysis["output_csv_sha256"].items():
        assert sha(OUT / file) == expected
    for file, expected in analysis["audit_cache_sha256"].items():
        assert sha(OUT / "audit_states" / file) == expected
    marker = read(OUT / "audit_started_after_freeze.json")
    assert marker["frozen_students_sha256"] == sha(OUT / "frozen_students.json")
    assert marker["audit_started_after_freeze"] and frozen["frozen_before_new_audit"]
    assert not frozen["new_audit_used_for_training_or_margin"]
    with np.load(SOURCE / "training_dataset.npz", allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    with np.load(OUT / "derived_training_targets.npz", allow_pickle=False) as z:
        derived = {k: z[k].copy() for k in z.files}
    for k, value in d.items():
        np.testing.assert_array_equal(value, derived[k])
    old = read(SOURCE / "development_manifest.json")
    old_records = {s["scenario_id"]: s for s in old["old_scenario_records"]}
    calibration = set(d["families"][d["calibration_mask"]])
    checkpoint, margin = [], []
    for kind in sorted({old_records[f]["conflict_type"] for f in calibration}):
        ids = sorted(f for f in calibration if old_records[f]["conflict_type"] == kind)
        assert len(ids) == 2
        checkpoint.append(ids[0]); margin.append(ids[1])
    assert checkpoint == manifest["checkpoint_family_ids"] and margin == manifest["margin_family_ids"]
    tm, cm, mm = d["large_train_mask"], np.isin(d["families"], checkpoint), np.isin(d["families"], margin)
    assert int(tm.sum()) == 592 and len(set(d["families"][tm])) == 84
    assert np.all(tm | cm | mm) and not np.any(tm & cm | tm & mm | cm & mm)
    assert sorted(set(d["families"][tm])) == manifest["training_family_ids"]
    np.testing.assert_array_equal(cm, derived["checkpoint_mask"])
    np.testing.assert_array_equal(mm, derived["margin_mask"])
    y = d["targets"].astype(np.float32)
    gains = y[:, 12:13] - y
    acceptable = y[..., 0] - y[..., 0].min(axis=1, keepdims=True) <= .001 + 1e-12
    np.testing.assert_array_equal(gains, derived["signed_gains"])
    np.testing.assert_array_equal(acceptable, derived["acceptable_sets"])
    float64_sets = d["targets"][..., 0] - d["targets"][..., 0].min(axis=1, keepdims=True) <= .001 + 1e-12
    lexical_rounding_changes = sum(lexical(a) != lexical(b) for a, b in zip(y, d["targets"]))
    w = family_weights(d["families"][tm])
    x = d["observations"].astype(np.float32).copy(); x[:, 28:33] = 0
    mean = (x[tm] * w[:, None]).sum(axis=0)
    std = np.sqrt(((x[tm] - mean)**2 * w[:, None]).sum(axis=0))
    std = np.where(std < 1e-3, 1, std).astype(np.float32)
    ymean = (gains[tm].mean(axis=1) * w[:, None]).sum(axis=0)
    scale = np.maximum(np.sqrt((((gains[tm] - ymean)**2).mean(axis=1) * w[:, None]).sum(axis=0)), 1e-3).astype(np.float32)
    models = [ReloadedModel(r) for r in frozen["models"]]
    assert len(models) == 10
    curves = rows(OUT / "training_curves.csv")
    for model, record in zip(models, frozen["models"]):
        np.testing.assert_array_equal(model.mean, mean)
        np.testing.assert_array_equal(model.std, std)
        np.testing.assert_array_equal(model.scale, scale)
        assert set(record["fit_families"]) == set(d["families"][tm | cm])
        assert not set(record["fit_families"]).intersection(margin)
        cr = [r for r in curves if r["arm"] == record["arm"]]
        best, epoch = float("inf"), 0
        for r in cr:
            regret = float(r["checkpoint_primary_regret_pp"])
            selected = regret < best - 1e-10
            assert selected == (r["selected_checkpoint"] == "True")
            if selected:
                best, epoch = regret, int(r["epoch"])
        assert epoch == record["selected_epoch"] and int(cr[-1]["epoch"]) == record["actual_epochs"]
        assert record["actual_epochs"] <= 300
        if record["actual_epochs"] < 300:
            assert record["actual_epochs"] == epoch + 80
        idx = np.flatnonzero(cm & (d["steps"] == 0))
        chosen = np.asarray([nomination(*model.predict(d["observations"][i])) for i in idx])
        costs = d["targets"][idx, :, 0]
        regret = ((costs[np.arange(len(idx)), chosen] - costs.min(axis=1)) * family_weights(d["families"][idx])).sum() * 100
        close(regret, record["best_checkpoint_primary_regret_pp"])
        close(best, regret)
    for seed in manifest["student_seeds"]:
        pair = [r for r in frozen["models"] if r["seed"] == seed]
        assert len(pair) == 2 and pair[0]["initial_network_sha256"] == pair[1]["initial_network_sha256"]
    for n, objective in enumerate(("single", "set")):
        idx = np.flatnonzero(mm & (d["steps"] == 0))
        pred = np.asarray([[m.predict(d["observations"][i])[0] for i in idx] for m in models[5*n:5*n+5]])
        mu, sd = pred[..., 0].astype(float).mean(axis=0), pred[..., 0].astype(float).std(axis=0, ddof=1)
        residual = (mu - gains[idx, :, 0]) / np.maximum(sd, .001)
        scores = {f: float(residual[d["families"][idx] == f].max()) for f in sorted(margin)}
        stored = frozen["margins"][objective]
        for f in margin:
            close(scores[f], stored["family_max_optimistic_residuals"][f])
        close(max(1., np.quantile(list(scores.values()), .8, method="higher")), stored["k"])
    signature = lambda s: json.dumps({k: v for k, v in s.items() if k != "scenario_id"}, sort_keys=True)
    previous = old["old_scenario_records"] + old["training_scenario_records"] + old["audit_scenario_records"] + old["excluded_previous_audit_records"]
    previous += read(ROOT / "outputs/teacher_factorial_pilot_development_v1/development_manifest.json")["scenario_records"]
    for folder in ("confirmatory_joint_control", "guarded_confirmatory_v2"):
        for file in ("train_scenarios.json", "final_scenarios.json"):
            previous += read(ROOT / "outputs" / folder / file)
    assert not {signature(s) for s in previous}.intersection(map(signature, manifest["audit_scenario_records"]))
    assert Counter(s["conflict_type"] for s in manifest["audit_scenario_records"]) == dict.fromkeys(sorted({s["conflict_type"] for s in manifest["audit_scenario_records"]}), 4)
    prior_audit = ROOT / "outputs/observable_teacher_label_audit_development_v1"
    prior_seeds = {int(r["population_seed"]) for r in rows(prior_audit / "state_primary_diagnostics.csv")}
    preservation = read(prior_audit / "diagnostic_manifest.json")["source_input_sha256"]
    for file in preservation:
        if Path(file).suffix == ".npz" and Path(file).parent.name in ("states", "training_states", "audit_states"):
            with np.load(file, allow_pickle=False) as z:
                for k in ("planning_seeds", "evaluation_seeds"):
                    if k in z:
                        prior_seeds.update(map(int, z[k]))
    prior_seeds |= {s + 9000001 for s in list(prior_seeds)}
    registry = {(r["family"], r["rep"]): r for r in manifest["seed_registry"]}
    namespaces = [{r["population"] for r in registry.values()}, {s for r in registry.values() for s in r["planning"]}, {s for r in registry.values() for s in r["evaluation"]}]
    namespaces += [{s + 9000001 for s in ns} for ns in namespaces.copy()]
    for i, ns in enumerate(namespaces):
        assert not ns.intersection(prior_seeds)
        assert all(not ns.intersection(other) for other in namespaces[i+1:])
    legacy = [ObservableScoreStudent.load(SOURCE / r["checkpoint"]) for r in manifest["legacy_large_rank_records"]]
    arms, names = manifest["arms"], manifest["candidate_names"]
    selections = {(r["state_id"], r["arm"]): r for r in rows(OUT / "state_selection.csv")}
    episodes = {(r["state_id"], r["arm"], int(r["future_seed"])): r for r in rows(OUT / "evaluation_episode_results.csv")}
    gates_csv = {(r["state_id"], r["objective"]): r for r in rows(OUT / "gate_diagnostics.csv")}
    aggregate = {arm: defaultdict(list) for arm in arms}
    gate_values = defaultdict(list)
    plan_total = execution_total = trace_steps = neutral = hold = 0
    files = sorted((OUT / "audit_states").glob("*.npz")); assert len(files) == 48
    for path in files:
        family, rep, _ = path.stem.split("_")
        entry = registry[(family, int(rep))]
        with np.load(path, allow_pickle=False) as z:
            assert int(z["population_seed"]) == entry["population"]
            np.testing.assert_array_equal(z["planning_seeds"], entry["planning"])
            np.testing.assert_array_equal(z["evaluation_seeds"], entry["evaluation"])
            reloaded = [m.predict(z["observation"]) for m in models]
            np.testing.assert_array_equal(z["gains"], np.asarray([p[0] for p in reloaded]))
            np.testing.assert_array_equal(z["probabilities"], np.asarray([p[1] for p in reloaded]))
            np.testing.assert_array_equal(z["legacy_priorities"], np.asarray([m.priorities(z["observation"]) for m in legacy]))
            expected = []
            means = z["risks"][z["supported"]].mean(axis=2)
            worst, average = means.max(axis=0), means.mean(axis=0)
            assert z["supported"].all() and np.isfinite(z["risks"]).all()
            close(z["risks"][..., [0, 2, 3]].sum(axis=-1), 1.)
            for n, objective in enumerate(("single", "set")):
                gs = z["gains"][5*n:5*n+5].astype(float)
                probs = z["probabilities"][5*n:5*n+5].astype(float)
                mu, sd = gs.mean(axis=0), gs[..., 0].std(axis=0, ddof=1)
                nominee = nomination(mu, probs.mean(axis=0))
                k = frozen["margins"][objective]["k"]
                lower = mu[nominee, 0] - k * max(sd[nominee], .001)
                guarded = nominee if nominee != 12 and lower > .001 else 12
                expected += [nominee, guarded]
                row = gates_csv[(path.stem, objective)]
                assert row["nominee"] == names[nominee] and row["guarded_plan"] == names[guarded]
                close([float(row[c]) for c in ("nominee_mean_gain_pp", "nominee_model_std_pp", "nominee_lower_score_pp")], [mu[nominee, 0]*100, sd[nominee]*100, lower*100])
                close(float(row["nominee_cached_planning_gain_pp"]), (worst[12, 0] - worst[nominee, 0])*100)
                base = z["evaluation_outcomes"][arms.index("pure_ppo_601"), :, 0]
                evdelta = (z["evaluation_outcomes"][2*n, :, 0] - base).mean()*100
                close(float(row["nominee_independent_eval_delta_vs601_pp"]), evdelta)
                category = "baseline" if nominee == 12 else "allowed" if guarded != 12 else "blocked"
                gate_values[objective].append((family, category, nominee, guarded, evdelta))
            expected += [nomination(g, p) for g, p in reloaded] + list(range(12, 17))
            teacher = min(range(17), key=lambda p: (worst[p, 0], average[p, 0], worst[p, 1], average[p, 1], worst[p, 3], worst[p, 4], p))
            expected += [teacher, lexical(z["legacy_priorities"].mean(axis=0)), 0]
            np.testing.assert_array_equal(expected, z["chosen_indices"])
            close(z["evaluation_outcomes"][..., [0, 2, 3]].sum(axis=-1), 1.)
            for a, (arm, plan) in enumerate(zip(arms, expected)):
                assert selections[(path.stem, arm)]["selected_plan"] == names[plan]
                for b, other in enumerate(expected):
                    if other == plan:
                        np.testing.assert_array_equal(z["evaluation_outcomes"][a], z["evaluation_outcomes"][b])
                aggregate[arm][family].extend(z["evaluation_outcomes"][a])
                for f, seed in enumerate(z["evaluation_seeds"]):
                    close([float(episodes[(path.stem, arm, int(seed))][field]) for field in manifest["outcome_fields"]], z["evaluation_outcomes"][a, f])
            keys, offsets, trace = z["trace_keys"], z["trace_offsets"], z["execution_trace"]
            np.testing.assert_array_equal(keys, sorted({(p, f) for p in expected for f in range(3)}))
            assert len(keys) == int(z["evaluation_rollouts"]) and offsets[0] == 0 and offsets[-1] == len(trace)
            assert np.all(np.diff(offsets) > 0) and trace.shape[1] == 6
            assert np.isin(trace[:, 1], np.arange(9)).all() and np.isin(trace[:, 2], np.arange(-1, 9)).all()
            assert np.isin(trace[:, 3], (0, 1, 2)).all() and np.isin(trace[:, 5], (0, 1)).all()
            assert np.all((trace[:, 4] >= 0) & (trace[:, 4] <= 1))
            actual = np.where(trace[:, 1] == 8, 8, np.where(trace[:, 3] == 2, -1, 4*trace[:, 3] + trace[:, 1]%4))
            np.testing.assert_array_equal(trace[:, 2], actual)
            for i, (plan, f) in enumerate(keys):
                t = trace[offsets[i]:offsets[i+1]]
                outcome = z["evaluation_outcomes"][expected.index(plan), f]
                assert len(t) == outcome[5]
                np.testing.assert_array_equal(t[:, 0], np.arange(len(t)))
                close([t[:, 4].mean(), (t[:, 1] == 8).sum(), t[:, 5].mean(), t[0, 1], t[0, 2]], outcome[6:])
            plan_total += int(z["planning_rollouts"]); execution_total += len(keys)
            trace_steps += len(trace); neutral += int(((trace[:, 3] == 2) & (trace[:, 4] > 0)).sum()); hold += int(trace[:, 5].sum())
    family_ids = sorted(aggregate[arms[0]])
    kinds_map = {r["scenario_id"]: r["conflict_type"] for r in manifest["audit_scenario_records"]}
    kinds = [kinds_map[f] for f in family_ids]
    matrix = {arm: np.asarray([np.mean(aggregate[arm][f], axis=0) for f in family_ids]) for arm in arms}
    for row in summary["arm_summary"]:
        close([row[f] for f in manifest["outcome_fields"][:9]], matrix[row["arm"]].mean(axis=0)[:9])
        assert row["selected_601_states"] == sum(r["selected_plan"] == names[12] for r in selections.values() if r["arm"] == row["arm"])
    for row in summary["all_paired_effects"]:
        delta = (matrix[row["treatment"]][:, 0] - matrix["pure_ppo_601"][:, 0])*100
        close([row["delta_incap_pp"], row["exploratory_ci_low_pp"], row["exploratory_ci_high_pp"]], [delta.mean(), *interval(delta, kinds)])
        assert [row[f] for f in ("improved_families", "worsened_families", "tied_families")] == [int((delta < -1e-12).sum()), int((delta > 1e-12).sum()), int((np.abs(delta) <= 1e-12).sum())]
    for row in summary["factorial_effects"]:
        idx, multiplier = {"incap_pp": (0, 100), "fed": (1, 1), "remaining_pp": (3, 100)}[row["metric"]]
        delta = (matrix[row["treatment"]][:, idx] - matrix[row["control"]][:, idx])*multiplier
        close([row["delta"], row["exploratory_ci_low"], row["exploratory_ci_high"]], [delta.mean(), *interval(delta, kinds)])
    interaction = ((matrix["set_guarded"][:, 0] - matrix["set_ungated"][:, 0]) - (matrix["single_guarded"][:, 0] - matrix["single_ungated"][:, 0]))*100
    close(list(summary["incap_interaction"].values()), [interaction.mean(), *interval(interaction, kinds)])
    gate_descriptions = {}
    for group in summary["gate_summary"]:
        objective = group["objective"]; g = gate_values[objective]
        assert group["nonbaseline_nomination_states"] == sum(v[2] != 12 for v in g)
        assert group["allowed_override_states"] == sum(v[1] == "allowed" for v in g)
        assert group["blocked_nomination_states"] == sum(v[1] == "blocked" for v in g)
        assert group["nominee_counts"] == dict(Counter(names[v[2]] for v in g))
        assert group["guarded_plan_counts"] == dict(Counter(names[v[3]] for v in g))
        descriptions = {}
        for category in ("blocked", "allowed", "baseline"):
            grouped = defaultdict(list)
            for f, cat, _, _, delta in g:
                grouped[f].append(delta if category == cat else 0.)
            close(group[f"{category}_ungated_delta_contribution_pp"], np.mean([np.mean(v) for v in grouped.values()]))
            selected = [v[4] for v in g if v[1] == category]
            descriptions[category] = {"states": len(selected), "actual_beneficial_states": sum(v < -1e-12 for v in selected),
                "actual_harmful_states": sum(v > 1e-12 for v in selected), "actual_tied_states": sum(abs(v) <= 1e-12 for v in selected)}
        gate_descriptions[objective] = descriptions
    assert plan_total == 4896 == summary["planning_rollouts"]
    assert execution_total == summary["unique_evaluation_rollouts"] and plan_total + execution_total == summary["total_unique_full_rollouts"]
    assert trace_steps == summary["actual_execution_trace_steps"] and neutral == summary["neutral_positive_release_trace_steps"] and hold == summary["hold_blocked_trace_steps"]
    set_sizes = acceptable.sum(axis=1)
    by_type = {kind: {arm: {"delta_vs601_pp": float(((matrix[arm] - matrix["pure_ppo_601"])[np.asarray(kinds) == kind, 0]*100).mean()),
                               "incap_percent": float(matrix[arm][np.asarray(kinds) == kind, 0].mean()*100)}
        for arm in ("single_ungated", "single_guarded", "set_ungated", "set_guarded", "teacher_robust3", "legacy_large_rank_ensemble", "live_graded")}
        for kind in sorted(set(kinds))}
    result = {"role": "independent_read_only_set_aware_pilot_verification", "all_checks_passed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "families_checked": 24, "states_checked": 48,
        "model_checkpoints_reloaded": 10, "model_normalization_training_only_verified": True,
        "checkpoint_and_margin_family_separation_verified": True, "paired_initial_networks_verified": True,
        "margins_recomputed_from_all17_plans_and_six_families": True, "seed_namespaces_and_scenarios_disjoint_verified": True,
        "new_planning_rollouts_verified": plan_total, "unique_execution_rollouts_verified": execution_total,
        "actual_actuator_trace_steps_verified": trace_steps, "neutral_positive_release_steps": neutral, "hold_blocked_steps": hold,
        "all22_decisions_and_duplicate_outcomes_verified": True, "primary_and_factorial_bootstrap_recomputed": True,
        "source_inputs_unchanged": len(manifest["source_input_sha256"]), "new_simulations": 0, "new_training": 0, "new_llm_calls": 0,
        "float32_vs_float64_lexical_label_changes": int(lexical_rounding_changes),
        "float32_vs_float64_acceptable_set_membership_changes": int((acceptable != float64_sets).sum()),
        "training_acceptable_set_size_distribution": dict(Counter(map(int, set_sizes[tm]))),
        "training_mean_acceptable_set_size": float(set_sizes[tm].mean()),
        "gate_actual_outcome_descriptions": gate_descriptions, "descriptive_effects_by_conflict_type": by_type,
        "verifier_sha256": sha(__file__), "analysis_manifest_sha256": sha(OUT / "analysis_manifest.json")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else v) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("all_checks_passed", "source_inputs_unchanged", "states_checked", "unique_execution_rollouts_verified", "actual_actuator_trace_steps_verified", "float32_vs_float64_lexical_label_changes", "float32_vs_float64_acceptable_set_membership_changes")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
