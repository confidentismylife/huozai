"""Frozen cached diagnostic: gain-consistent selection and five tiny ridge fits."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consistent_gain_policy import BASE, ALPHA, FLOOR, MIN_GAIN, winner, decide, ensemble_scores, family_margin, fit_gain, GainRidge
from set_aware_plan_student import SetGainStudent, ensemble_decision

OUT = ROOT / "outputs/consistent_gain_crossfit_development_v1"
SOURCE = ROOT / "outputs/baseline_relative_plan_distillation_development_v1"
PRIOR = ROOT / "outputs/set_aware_distillation_development_v1"
PROTOCOL = ROOT / "reports/统一提名与覆盖收益交叉拟合开发方案_2026-10-03.md"
STATUS = ROOT / "outputs/third_round_development_status.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist() if isinstance(v, np.ndarray) else v) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def dataset():
    with np.load(SOURCE / "training_dataset.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def prepare():
    assert not OUT.exists(), "preserve existing output"
    prior = read(PRIOR / "completion_manifest.json")
    for file, expected in prior["files"].items():
        assert sha(ROOT / file) == expected
    source_hashes = read(PRIOR / "development_manifest.json")["source_input_sha256"].copy()
    for file, expected in source_hashes.items():
        assert sha(file) == expected
    d, old = dataset(), read(SOURCE / "development_manifest.json")
    kinds = {r["scenario_id"]: r["conflict_type"] for r in old["old_scenario_records"] + old["training_scenario_records"]}
    train_families = sorted(set(d["families"][d["large_train_mask"]]))
    rng, fold_map = np.random.default_rng(20262101), {}
    for kind in sorted({kinds[f] for f in train_families}):
        ids = sorted(f for f in train_families if kinds[f] == kind)
        assert len(ids) == 14
        for i, family in enumerate(np.asarray(ids)[rng.permutation(14)]):
            fold_map[str(family)] = i % 4
    OUT.mkdir()
    (OUT / "previous_status_exact_bytes.json").write_bytes(STATUS.read_bytes())
    previous_status_sha = sha(STATUS)
    for file, expected in prior["files"].items():
        effective = OUT / "previous_status_exact_bytes.json" if (ROOT / file).resolve() == STATUS else ROOT / file
        source_hashes[str(effective.resolve())] = expected
    for file in (Path(__file__), ROOT / "scripts/consistent_gain_policy.py", ROOT / "tests/test_consistent_gain_policy.py", PROTOCOL):
        source_hashes[str(file.resolve())] = sha(file)
    p = read(PRIOR / "development_manifest.json")
    body = {"role": "cached_consistent_gain_crossfit_development_diagnostic_not_control_efficacy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_allowed": False,
        "source_input_sha256": source_hashes, "prior_completion_sha256": sha(PRIOR / "completion_manifest.json"),
        "previous_live_status_sha256": previous_status_sha, "fold_seed": 20262101, "fold_count": 4,
        "fold_by_training_family": fold_map, "kind_by_family": kinds,
        "checkpoint_family_ids": p["checkpoint_family_ids"], "margin_family_ids": p["margin_family_ids"],
        "candidate_names": p["candidate_names"], "fixed_ridge_alpha": ALPHA,
        "minimum_override_score": MIN_GAIN, "minimum_additive_margin": FLOOR,
        "margin_quantile": .8, "margin_quantile_method": "higher",
        "lightweight_linear_fits_budget": 5, "new_neural_training": 0, "new_ppo_training": 0, "new_llm_calls": 0, "new_simulations": 0,
        "oof_guard_uses_oof_margin_is_reuse_calibration_diagnostic": True,
        "prior_sgdev_already_viewed_not_fresh_validation": True,
        "sgdev_actual_evaluation_rule": "availability_counts_only_no_subset_effects_or_missing_fill"}
    write(OUT / "development_manifest.json", body)
    (OUT / "source_snapshot").mkdir()
    for file in (Path(__file__), ROOT / "scripts/consistent_gain_policy.py", ROOT / "tests/test_consistent_gain_policy.py", PROTOCOL):
        shutil.copy2(file, OUT / "source_snapshot" / file.name)
    print("Prepared: 5 small linear fits, 0 new simulations, frozen cached-diagnostic contract.", flush=True)


def check_sources(m):
    for file, expected in m["source_input_sha256"].items():
        assert sha(file) == expected, file


def fit_models(m, d):
    assert not (OUT / "frozen_models.json").exists(), "refuse duplicate fitting"
    tm = d["large_train_mask"]
    x, families, steps = d["observations"][tm], d["families"][tm], d["steps"][tm]
    targets = d["targets"][tm, 12:13, 0] - d["targets"][tm, :, 0]
    folds = np.asarray([m["fold_by_training_family"][f] for f in families])
    forbidden = m["checkpoint_family_ids"] + m["margin_family_ids"]
    predictions = np.empty_like(targets)
    records = []
    (OUT / "models").mkdir()
    for fold in range(4):
        heldout = sorted(set(families[folds == fold]))
        fit_mask = folds != fold
        model = fit_gain(x[fit_mask], targets[fit_mask], families[fit_mask], forbidden + heldout)
        predictions[~fit_mask] = model.predict(x[~fit_mask])
        path = OUT / "models" / f"fold_{fold}.npz"; model.save(path)
        records.append({"name": f"fold_{fold}", "path": f"models/{path.name}", "sha256": sha(path),
            "fit_family_ids": sorted(set(families[fit_mask])), "heldout_family_ids": heldout,
            "fit_states": int(fit_mask.sum()), "heldout_states": int((~fit_mask).sum())})
        print(f"Fitted fold {fold}: {len(heldout)} held-out families, {int((~fit_mask).sum())} OOF states", flush=True)
    final = fit_gain(x, targets, families, forbidden)
    path = OUT / "models/final.npz"; final.save(path)
    records.append({"name": "final", "path": "models/final.npz", "sha256": sha(path),
        "fit_family_ids": sorted(set(families)), "heldout_family_ids": forbidden, "fit_states": len(x), "heldout_states": 0})
    initial = steps == 0
    oof_margin, oof_info = family_margin(predictions[initial], targets[initial], families[initial])
    mm = np.isin(d["families"], m["margin_family_ids"]) & (d["steps"] == 0)
    margin_predictions = final.predict(d["observations"][mm])
    margin_actual = d["targets"][mm, 12:13, 0] - d["targets"][mm, :, 0]
    cal_margin, cal_info = family_margin(margin_predictions, margin_actual, d["families"][mm])
    margin = max(FLOOR, oof_margin, cal_margin)
    np.savez_compressed(OUT / "crossfit_predictions.npz", state_ids=d["state_ids"][tm], families=families, steps=steps,
        fold_indices=folds, predictions=predictions, actual_primary_gains=targets,
        margin_state_ids=d["state_ids"][mm], margin_predictions=margin_predictions, margin_actual=margin_actual)
    frozen = {"development_manifest_sha256": sha(OUT / "development_manifest.json"), "models": records,
        "crossfit_predictions_sha256": sha(OUT / "crossfit_predictions.npz"), "additive_margin": margin,
        "oof_margin": oof_margin, "independent_margin": cal_margin, "oof_margin_details": oof_info,
        "independent_margin_details": cal_info, "models_and_margin_frozen_before_diagnostic_scoring": True,
        "checkpoint_or_sgdev_used_to_fit_or_choose_margin": False}
    write(OUT / "frozen_models.json", frozen)
    print(f"Final fit frozen: margin {margin*100:.4f} pp; OOF {oof_margin*100:.4f}; six-family {cal_margin*100:.4f}", flush=True)
    return final, frozen, predictions


def diagnostic(m, d, final, frozen, oof):
    write(OUT / "diagnostic_started_after_freeze.json", {"frozen_models_sha256": sha(OUT / "frozen_models.json"), "new_simulations": 0})
    oldf = read(PRIOR / "frozen_students.json")
    old_models = [SetGainStudent.load(PRIOR / r["checkpoint"]) for r in oldf["models"]]
    result_rows = []
    vector_rows = []
    tm = d["large_train_mask"]
    oof_map = {s: pred for s, pred in zip(d["state_ids"][tm], oof)}
    def add(block, state_id, family, actual, prediction, old_gains=None, old_probs=None, available=None):
        actual = np.asarray(actual, float)
        decision = decide(prediction, frozen["additive_margin"])
        choices = {"pure_ppo_601": (BASE, 0., 0., True), "cached_teacher_primary": (winner(actual), actual.max()*100, actual.max()*100, False),
            "ridge_gain_ungated": (decision["raw_gain_choice"], float(prediction[decision["raw_gain_choice"]]*100), np.nan, False),
            "ridge_gain_consistent": (decision["execution"], float(prediction[decision["execution"]]*100), decision["selected_score_pp"], True)}
        if old_gains is not None:
            for i, objective in enumerate(("single", "set")):
                gs, ps = old_gains[5*i:5*i+5], old_probs[5*i:5*i+5]
                k = oldf["margins"][objective]["k"]
                original, guarded, info = ensemble_decision(gs, ps, k)
                unified = ensemble_scores(gs, k)
                mean = gs.astype(float)[..., 0].mean(axis=0)
                choices[f"{objective}_class_ungated"] = original, float(mean[original]*100), np.nan, False
                choices[f"{objective}_class_guarded"] = guarded, float(mean[guarded]*100), info["nominee_lower_score_pp"] if guarded != BASE else 0., False
                choices[f"{objective}_gain_ungated"] = unified["raw_gain_choice"], float(mean[unified["raw_gain_choice"]]*100), np.nan, False
                choices[f"{objective}_gain_consistent"] = unified["execution"], float(mean[unified["execution"]]*100), unified["selected_score_pp"], True
        for arm, (choice, mu, score, unified) in choices.items():
            if unified and choice != BASE:
                assert score > MIN_GAIN*100
            gain = float(actual[choice]*100)
            result_rows.append({"block": block, "state_id": str(state_id), "family": str(family), "arm": arm,
                "selected_index": int(choice), "selected_plan": m["candidate_names"][choice], "selected_601": choice == BASE,
                "predicted_primary_gain_pp": mu, "same_score_for_nomination_and_override_pp": score,
                "cached_planning_gain_pp": gain, "cached_primary_regret_pp": float((actual.max()-actual[choice])*100),
                "cached_beneficial_override": choice != BASE and gain > 1e-10,
                "cached_harmful_override": choice != BASE and gain < -1e-10,
                "cached_tied_override": choice != BASE and abs(gain) <= 1e-10,
                "actual_evaluation_cached": "not_applicable" if available is None else choice in available})
        vector_rows.append((block, state_id, family, actual.copy(), np.asarray(prediction).copy()))
    for i in np.flatnonzero(tm & (d["steps"] == 0)):
        actual = d["targets"][i, 12, 0] - d["targets"][i, :, 0]
        add("train_oof_initial", d["state_ids"][i], d["families"][i], actual, oof_map[d["state_ids"][i]])
    cp = np.isin(d["families"], m["checkpoint_family_ids"]) & (d["steps"] == 0)
    for i in np.flatnonzero(cp):
        predictions = [model.predictions(d["observations"][i]) for model in old_models]
        add("old_checkpoint_initial", d["state_ids"][i], d["families"][i], d["targets"][i, 12, 0]-d["targets"][i, :, 0],
            final.predict(d["observations"][i]), np.asarray([p[0] for p in predictions]), np.asarray([p[1] for p in predictions]))
    for path in sorted((PRIOR / "audit_states").glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            primary = z["risks"][z["supported"]].mean(axis=2)[:, :, 0].max(axis=0)
            add("viewed_sgdev_initial", path.stem, path.stem.split("_")[0], primary[12]-primary,
                final.predict(z["observation"]), z["gains"], z["probabilities"], set(z["chosen_indices"].tolist()))
    write_csv(OUT / "diagnostic_records.csv", result_rows)
    np.savez_compressed(OUT / "diagnostic_vectors.npz", blocks=np.asarray([r[0] for r in vector_rows]),
        state_ids=np.asarray([r[1] for r in vector_rows]), families=np.asarray([r[2] for r in vector_rows]),
        actual_primary_gains=np.asarray([r[3] for r in vector_rows]), ridge_predictions=np.asarray([r[4] for r in vector_rows]))
    summary_rows = []
    for block in sorted({r["block"] for r in result_rows}):
        for arm in sorted({r["arm"] for r in result_rows if r["block"] == block}):
            rows = [r for r in result_rows if r["block"] == block and r["arm"] == arm]
            families = sorted({r["family"] for r in rows})
            equal_mean = lambda key: float(np.mean([np.mean([r[key] for r in rows if r["family"] == f]) for f in families]))
            summary_rows.append({"block": block, "arm": arm, "families": len(families), "states": len(rows),
                "cached_primary_gain_pp_family_equal": equal_mean("cached_planning_gain_pp"),
                "cached_primary_regret_pp_family_equal": equal_mean("cached_primary_regret_pp"),
                "selected_601_states": sum(r["selected_601"] for r in rows),
                "beneficial_override_states": sum(r["cached_beneficial_override"] for r in rows),
                "harmful_override_states": sum(r["cached_harmful_override"] for r in rows),
                "tied_override_states": sum(r["cached_tied_override"] for r in rows),
                "negative_predicted_gain_nonbaseline_states": sum(r["selected_index"] != BASE and r["predicted_primary_gain_pp"] < -1e-10 for r in rows),
                "actual_execution_cache_available_states": sum(r["actual_evaluation_cached"] is True for r in rows) if block == "viewed_sgdev_initial" else "not_applicable"})
    write_csv(OUT / "diagnostic_summary.csv", summary_rows)
    result = {"role": m["role"], "freeze_allowed": False, "new_lightweight_linear_fits": 5,
        "new_neural_training": 0, "new_ppo_training": 0, "new_llm_calls": 0, "new_simulations": 0,
        "train_families": 84, "oof_states": 592, "oof_initial_states": 168,
        "checkpoint_families": 6, "viewed_sgdev_families": 24, "additive_margin_pp": frozen["additive_margin"]*100,
        "oof_margin_pp": frozen["oof_margin"]*100, "independent_margin_pp": frozen["independent_margin"]*100,
        "diagnostic_record_count": len(result_rows), "summary_rows": summary_rows,
        "actual_evacuations_improvement_established": False, "llm_contribution_established": False,
        "limitations": ["All source development labels and SGDEV results were previously viewed.",
            "OOF predictions exclude target family, but common gate margin reuses OOF residuals: guarded OOF is calibration reuse.",
            "Ridge-only primary regression changes capacity/objective; not a fair new neural-objective comparison.",
            "Fixed old gain-head ablation isolates selection given predictions, not learning or uncertainty guarantees.",
            "Cached planning primary is worst-source mean, not independent actual evacuation outcome.",
            "Missing actual selected-plan evaluations counted only; no selectively covered subset effect estimated.",
            "Six previously viewed margin families and empirical quantiles provide no safety coverage guarantee.",
            "No new map, behavior, noise, fire-physics validation, dynamic meta-reselection, PPO or LLM ablation."]}
    write(OUT / "summary.json", result)
    check_sources(m)
    for r in frozen["models"]:
        assert sha(OUT / r["path"]) == r["sha256"]
    write(OUT / "analysis_manifest.json", {"runner_sha256": sha(__file__), "manifest_sha256": sha(OUT / "development_manifest.json"),
        "frozen_models_sha256": sha(OUT / "frozen_models.json"),
        "files": {p.name: sha(p) for p in (OUT / "summary.json", OUT / "diagnostic_records.csv", OUT / "diagnostic_summary.csv", OUT / "diagnostic_vectors.npz")}})
    print(json.dumps({"completed": True, "linear_fits": 5, "new_simulations": 0, "margin_pp": result["additive_margin_pp"],
        "key_rows": [r for r in summary_rows if r["arm"] in ("ridge_gain_ungated", "ridge_gain_consistent", "single_gain_consistent", "set_gain_consistent")]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(); return
    if not args.run:
        parser.error("choose --prepare-only or --run")
    m = read(OUT / "development_manifest.json"); check_sources(m)
    d = dataset()
    final, frozen, oof = fit_models(m, d)
    diagnostic(m, d, final, frozen, oof)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
