"""Independently check equations, held-out folds and every cached choice."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/consistent_gain_crossfit_development_v1"
PRIOR = ROOT / "outputs/set_aware_distillation_development_v1"
SOURCE = ROOT / "outputs/baseline_relative_plan_distillation_development_v1"
from verify_set_aware_distillation import ReloadedModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)


def best(scores):
    assert scores.shape == (17,) and scores[12] == 0 and np.isfinite(scores).all()
    return min(range(17), key=lambda p: (-scores[p], p != 12, p))


def fixed_rule(prediction, penalty):
    lower = prediction.astype(float) - penalty
    lower[12] = 0
    nominee = best(lower)
    execution = nominee if lower[nominee] > .001 else 12
    return best(prediction), execution, lower


def features(x):
    x = np.asarray(x, float).copy(); x[..., 28:33] = 0
    return x


def predict(model, x):
    pred = ((features(x) - model["mean"]) / model["std"]) @ model["coefficients"] + model["intercept"]
    pred[..., 12] = 0.
    return pred


def recompute_margin(pred, actual, families):
    nominated = np.asarray([best(row) for row in pred])
    optimism = pred[np.arange(len(pred)), nominated] - actual[np.arange(len(pred)), nominated]
    scores = {f: float(optimism[families == f].max()) for f in sorted(set(families))}
    return max(.001, np.quantile(list(scores.values()), .8, method="higher")), scores, nominated


def main():
    torch.set_num_threads(1)
    m, f, s, a = (read(OUT / path) for path in ("development_manifest.json", "frozen_models.json", "summary.json", "analysis_manifest.json"))
    for file, expected in m["source_input_sha256"].items():
        assert sha(file) == expected
    assert f["development_manifest_sha256"] == sha(OUT / "development_manifest.json")
    assert f["crossfit_predictions_sha256"] == sha(OUT / "crossfit_predictions.npz")
    assert a["runner_sha256"] == sha(ROOT / "scripts/run_consistent_gain_crossfit.py")
    assert a["frozen_models_sha256"] == sha(OUT / "frozen_models.json")
    assert a["manifest_sha256"] == sha(OUT / "development_manifest.json")
    for file, expected in a["files"].items():
        assert sha(OUT / file) == expected
    assert read(OUT / "diagnostic_started_after_freeze.json")["frozen_models_sha256"] == sha(OUT / "frozen_models.json")
    with np.load(SOURCE / "training_dataset.npz", allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    tm = d["large_train_mask"]
    x, families = d["observations"][tm], d["families"][tm]
    actual = d["targets"][tm, 12:13, 0] - d["targets"][tm, :, 0]
    train_ids = sorted(set(families)); assert len(train_ids) == 84 and len(x) == 592
    rng, reconstructed_folds = np.random.default_rng(20262101), {}
    for kind in sorted({m["kind_by_family"][family] for family in train_ids}):
        ids = sorted(family for family in train_ids if m["kind_by_family"][family] == kind)
        assert len(ids) == 14
        for j, family in enumerate(np.asarray(ids)[rng.permutation(14)]):
            reconstructed_folds[str(family)] = j % 4
    assert reconstructed_folds == m["fold_by_training_family"]
    folds = np.asarray([reconstructed_folds[family] for family in families])
    models, oof = {}, np.empty_like(actual)
    for record in f["models"]:
        assert sha(OUT / record["path"]) == record["sha256"]
        with np.load(OUT / record["path"], allow_pickle=False) as z:
            model = {k: z[k].copy() for k in z.files}
        models[record["name"]] = model
        fit_mask = np.isin(families, record["fit_family_ids"])
        assert not set(record["fit_family_ids"]).intersection(record["heldout_family_ids"])
        assert not set(record["fit_family_ids"]).intersection(m["checkpoint_family_ids"] + m["margin_family_ids"])
        assert int(fit_mask.sum()) == record["fit_states"]
        if record["name"] != "final":
            fold = int(record["name"].split("_")[1])
            np.testing.assert_array_equal(fit_mask, folds != fold)
            assert sorted(set(families[~fit_mask])) == record["heldout_family_ids"]
            assert int((~fit_mask).sum()) == record["heldout_states"]
            oof[~fit_mask] = predict(model, x[~fit_mask])
        else:
            assert fit_mask.all()
        fitx, fity, fitfamilies = features(x[fit_mask]), actual[fit_mask], families[fit_mask]
        counts = Counter(fitfamilies)
        w = np.asarray([1/(len(counts)*counts[family]) for family in fitfamilies])
        mean = (fitx * w[:, None]).sum(axis=0)
        sd = np.sqrt(((fitx - mean)**2*w[:, None]).sum(axis=0))
        sd = np.where(sd < .001, 1., sd)
        intercept = (fity*w[:, None]).sum(axis=0)
        close(model["mean"], mean); close(model["std"], sd); close(model["intercept"], intercept)
        standardized = (fitx - mean)/sd
        lhs = (standardized.T @ (standardized*w[:, None]) + .01*np.eye(49)) @ model["coefficients"]
        rhs = standardized.T @ ((fity-intercept)*w[:, None])
        close(lhs, rhs)  # normal-equation residual, no additional fitting
        np.testing.assert_array_equal(model["coefficients"][:, 12], np.zeros(49))
        assert model["intercept"][12] == 0
    assert len(models) == 5
    with np.load(OUT / "crossfit_predictions.npz", allow_pickle=False) as z:
        close(oof, z["predictions"]); close(actual, z["actual_primary_gains"])
        np.testing.assert_array_equal(z["state_ids"], d["state_ids"][tm])
        np.testing.assert_array_equal(z["families"], families)
        np.testing.assert_array_equal(z["steps"], d["steps"][tm])
        np.testing.assert_array_equal(z["fold_indices"], folds)
    initial = d["steps"][tm] == 0
    margin_mask = np.isin(d["families"], m["margin_family_ids"]) & (d["steps"] == 0)
    margin_pred = predict(models["final"], d["observations"][margin_mask])
    margin_actual = d["targets"][margin_mask, 12:13, 0] - d["targets"][margin_mask, :, 0]
    for label, pred, true, names in (("oof", oof[initial], actual[initial], families[initial]),
                                   ("independent", margin_pred, margin_actual, d["families"][margin_mask])):
        calculated, scores, nominees = recompute_margin(pred, true, names)
        close(calculated, f[label + "_margin"])
        stored = f[label + "_margin_details"]
        assert set(scores) == set(stored["family_max_nominee_optimism"])
        for family, value in scores.items():
            close(value, stored["family_max_nominee_optimism"][family])
        np.testing.assert_array_equal(nominees, stored["nominees"])
    close(f["additive_margin"], max(.001, f["oof_margin"], f["independent_margin"]))
    records = rows(OUT / "diagnostic_records.csv")
    by_key = {(r["block"], r["state_id"], r["arm"]): r for r in records}
    assert len(records) == len(by_key) == s["diagnostic_record_count"] == 1392
    with np.load(OUT / "diagnostic_vectors.npz", allow_pickle=False) as z:
        vectors = {k: z[k].copy() for k in z.files}
    oldf = read(PRIOR / "frozen_students.json")
    old_models = [ReloadedModel(r) for r in oldf["models"]]
    dindex = {state: i for i, state in enumerate(d["state_ids"])}
    oof_index = {state: i for i, state in enumerate(d["state_ids"][tm])}
    available_counts = defaultdict(int)
    coherent_checked = negative_nominee_old = 0
    for i, (block, state, family) in enumerate(zip(vectors["blocks"], vectors["state_ids"], vectors["families"])):
        gains = probabilities = None; available = None
        if block == "viewed_sgdev_initial":
            with np.load(PRIOR / "audit_states" / f"{state}.npz", allow_pickle=False) as z:
                primary = z["risks"][z["supported"]].mean(axis=2)[:, :, 0].max(axis=0)
                true = primary[12] - primary; obs = z["observation"].copy()
                gains, probabilities = z["gains"].copy(), z["probabilities"].copy()
                available = set(z["chosen_indices"].tolist())
            loaded = [model.predict(obs) for model in old_models]
            close(gains, np.asarray([p[0] for p in loaded])); close(probabilities, np.asarray([p[1] for p in loaded]))
            pred = predict(models["final"], obs)
        elif block == "old_checkpoint_initial":
            j = dindex[state]; assert d["steps"][j] == 0 and family in m["checkpoint_family_ids"]
            true = d["targets"][j, 12, 0] - d["targets"][j, :, 0]
            loaded = [model.predict(d["observations"][j]) for model in old_models]
            gains, probabilities = np.asarray([p[0] for p in loaded]), np.asarray([p[1] for p in loaded])
            pred = predict(models["final"], d["observations"][j])
        else:
            assert block == "train_oof_initial"
            j = oof_index[state]; assert initial[j]
            true, pred = actual[j], oof[j]
        close(true, vectors["actual_primary_gains"][i]); close(pred, vectors["ridge_predictions"][i])
        raw, executed, lower = fixed_rule(pred, f["additive_margin"])
        expected = {"pure_ppo_601": (12, 0., 0., True), "cached_teacher_primary": (best(true), true.max()*100, true.max()*100, False),
            "ridge_gain_ungated": (raw, pred[raw]*100, np.nan, False),
            "ridge_gain_consistent": (executed, pred[executed]*100, lower[executed]*100, True)}
        if gains is not None:
            for n, objective in enumerate(("single", "set")):
                gs, ps = gains[5*n:5*n+5].astype(float), probabilities[5*n:5*n+5].astype(float)
                mu = gs.mean(axis=0)
                original = min(range(17), key=lambda p: (-ps.mean(axis=0)[p],) + tuple(-mu[p]) + (p,))
                k = oldf["margins"][objective]["k"]
                penalty = k*np.maximum(gs[..., 0].std(axis=0, ddof=1), .001)
                old_score = mu[original, 0] - penalty[original]
                old_exec = original if original != 12 and old_score > .001 else 12
                raw, executed, lower = fixed_rule(mu[:, 0], penalty)
                expected.update({f"{objective}_class_ungated": (original, mu[original, 0]*100, np.nan, False),
                    f"{objective}_class_guarded": (old_exec, mu[old_exec, 0]*100, old_score*100 if old_exec != 12 else 0., False),
                    f"{objective}_gain_ungated": (raw, mu[raw, 0]*100, np.nan, False),
                    f"{objective}_gain_consistent": (executed, mu[executed, 0]*100, lower[executed]*100, True)})
        for arm, (selected, mu, score, unified) in expected.items():
            row = by_key[(block, state, arm)]
            assert int(row["selected_index"]) == selected and row["selected_plan"] == m["candidate_names"][selected]
            close([float(row[c]) for c in ("predicted_primary_gain_pp", "same_score_for_nomination_and_override_pp", "cached_planning_gain_pp", "cached_primary_regret_pp")],
                [mu, score, true[selected]*100, (true.max()-true[selected])*100])
            assert (row["selected_601"] == "True") == (selected == 12)
            assert (row["cached_beneficial_override"] == "True") == (selected != 12 and true[selected]*100 > 1e-10)
            assert (row["cached_harmful_override"] == "True") == (selected != 12 and true[selected]*100 < -1e-10)
            assert (row["cached_tied_override"] == "True") == (selected != 12 and abs(true[selected]*100) <= 1e-10)
            if unified:
                coherent_checked += 1
                assert selected == 12 or score > .10
            if "class_ungated" in arm and selected != 12 and mu < 0:
                negative_nominee_old += 1
            if available is not None:
                assert (row["actual_evaluation_cached"] == "True") == (selected in available)
                available_counts[arm] += selected in available
            else:
                assert row["actual_evaluation_cached"] == "not_applicable"
    for row in s["summary_rows"]:
        source = [r for r in records if r["block"] == row["block"] and r["arm"] == row["arm"]]
        families = sorted({r["family"] for r in source})
        assert len(families) == row["families"] and len(source) == row["states"]
        for field, metric in (("cached_primary_gain_pp_family_equal", "cached_planning_gain_pp"), ("cached_primary_regret_pp_family_equal", "cached_primary_regret_pp")):
            close(row[field], np.mean([np.mean([float(r[metric]) for r in source if r["family"] == family]) for family in families]))
        for field, flag in (("selected_601_states", "selected_601"), ("beneficial_override_states", "cached_beneficial_override"),
                            ("harmful_override_states", "cached_harmful_override"), ("tied_override_states", "cached_tied_override")):
            assert row[field] == sum(r[flag] == "True" for r in source)
        assert row["negative_predicted_gain_nonbaseline_states"] == sum(int(r["selected_index"]) != 12 and float(r["predicted_primary_gain_pp"]) < -1e-10 for r in source)
        if row["block"] == "viewed_sgdev_initial":
            assert row["actual_execution_cache_available_states"] == available_counts[row["arm"]]
    ridge_initial = [r for r in records if r["block"] == "train_oof_initial" and r["arm"] == "ridge_gain_consistent"]
    harmful_families = sorted({r["family"] for r in ridge_initial if r["cached_harmful_override"] == "True"})
    result = {"all_checks_passed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_inputs_unchanged": len(m["source_input_sha256"]), "five_model_normal_equations_checked_without_refitting": True,
        "all_training_only_normalization_and_family_folds_checked": True, "model_checkpoints_checked": 5,
        "oof_states_checked": 592, "diagnostic_rows_checked": len(records), "coherent_decisions_checked": coherent_checked,
        "old_class_negative_gain_nomination_rows": negative_nominee_old,
        "oof_consistent_harmful_families": harmful_families,
        "independent_margin_families_with_nonbaseline_nomination": len(set(d["families"][margin_mask][np.asarray(f["independent_margin_details"]["nominees"]) != 12])),
        "old_models_reloaded": 10, "all_margins_and_choices_recomputed": True,
        "actual_outcome_cache_availability_only_no_biased_subset_effect": True,
        "new_fits_in_verifier": 0, "new_simulations": 0, "new_llm_calls": 0,
        "verifier_sha256": sha(__file__), "analysis_manifest_sha256": sha(OUT / "analysis_manifest.json")}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
