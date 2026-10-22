"""Post-evaluation descriptive effects only; no fitting, gating changes or runs."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/set_aware_distillation_development_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def rows(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def interval(values, kinds):
    rng = np.random.default_rng(20262002)
    samples = np.zeros(2000)
    kinds, values = np.asarray(kinds), np.asarray(values)
    for kind in sorted(set(kinds)):
        part = values[kinds == kind]
        samples += part[rng.integers(len(part), size=(2000, len(part)))].sum(axis=1)
    return np.quantile(samples / len(values), [.025, .975]).tolist()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    m, s, v = read("development_manifest.json"), read("summary.json"), read("verification.json")
    assert v["all_checks_passed"]
    ids = sorted(r["scenario_id"] for r in m["audit_scenario_records"])
    kindmap = {r["scenario_id"]: r["conflict_type"] for r in m["audit_scenario_records"]}
    kinds = [kindmap[f] for f in ids]
    delta_map = {(r["arm"], r["family"]): float(r["delta_incap_vs601_pp"]) for r in rows("family_paired_deltas.csv")}
    teacher_benefit = -np.asarray([delta_map[("teacher_robust3", f)] for f in ids])
    retention = []
    for arm in ("single_ungated", "single_guarded", "set_ungated", "set_guarded", "legacy_large_rank_ensemble"):
        benefit = -np.asarray([delta_map[(arm, f)] for f in ids])
        contrast = benefit - .5 * teacher_benefit
        lo, hi = interval(contrast, kinds)
        nonzero = [(f, float(-b)) for f, b in zip(ids, benefit) if abs(b) > 1e-12]
        retention.append({"arm": arm, "benefit_pp": float(benefit.mean()), "teacher_benefit_pp": float(teacher_benefit.mean()),
            "descriptive_point_retention_percent": float(100*benefit.mean()/teacher_benefit.mean()),
            "benefit_minus_half_teacher_pp": float(contrast.mean()), "exploratory_ci_low_pp": lo, "exploratory_ci_high_pp": hi,
            "nonzero_family_effects": nonzero, "fifty_percent_retention_established": False})
    gate_ranges = {}
    gates = rows("gate_diagnostics.csv")
    for objective in ("single", "set"):
        rs = [r for r in gates if r["objective"] == objective and r["nominee"] != "pure_ppo_601"]
        gate_ranges[objective] = {"nonbaseline_states": len(rs), "ranges_pp": {field: [min(float(r[field]) for r in rs), max(float(r[field]) for r in rs)]
            for field in ("nominee_mean_gain_pp", "nominee_model_std_pp", "nominee_lower_score_pp", "nominee_cached_planning_gain_pp", "nominee_independent_eval_delta_vs601_pp")}}
    fingerprints = set()
    family_fingerprints = defaultdict(set)
    for p in (OUT / "audit_states").glob("*.npz"):
        with np.load(p, allow_pickle=False) as z:
            x = z["observation"].astype(np.float32).copy(); x[28:33] = 0
            fingerprints.add(x.tobytes()); family_fingerprints[p.stem.split("_")[0]].add(x.tobytes())
    with np.load(OUT / "derived_training_targets.npz", allow_pickle=False) as z:
        sizes, tm, families = z["acceptable_sets"].sum(axis=1), z["large_train_mask"], z["families"]
        size_description = {"training_states": int(tm.sum()), "all17_acceptable_states": int((sizes[tm] == 17).sum()),
            "all17_acceptable_state_percent": float((sizes[tm] == 17).mean()*100),
            "all17_acceptable_family_equal_percent": float(np.mean([np.mean(sizes[tm & (families == f)] == 17) for f in sorted(set(families[tm]))])*100),
            "singleton_states": int((sizes[tm] == 1).sum()), "singleton_state_percent": float((sizes[tm] == 1).mean()*100),
            "mean_set_size_family_equal": float(np.mean([sizes[tm & (families == f)].mean() for f in sorted(set(families[tm]))]))}
    seedpairs = []
    effects = {r["treatment"]: r for r in s["all_paired_effects"]}
    for seed in m["student_seeds"]:
        difference = np.asarray([delta_map[(f"set_{seed}", f)] - delta_map[(f"single_{seed}", f)] for f in ids])
        lo, hi = interval(difference, kinds)
        seedpairs.append({"seed": seed, "single_delta_vs601_pp": effects[f"single_{seed}"]["delta_incap_pp"],
            "set_delta_vs601_pp": effects[f"set_{seed}"]["delta_incap_pp"], "set_minus_single_pp": float(difference.mean()),
            "exploratory_ci_low_pp": lo, "exploratory_ci_high_pp": hi})
    result = {"role": "posthoc_descriptive_quantification_not_new_qualification", "teacher_retention": retention,
        "gate_nominee_score_ranges": gate_ranges, "unique_initial_observable_vectors": len(fingerprints),
        "all_families_two_populations_share_observation": all(len(v) == 1 for v in family_fingerprints.values()),
        "acceptable_set_description": size_description, "paired_initialization_diagnostics": seedpairs,
        "incap_interaction": s["incap_interaction"], "new_training": 0, "new_simulations": 0, "new_llm_calls": 0,
        "empirical_gate_unchanged": True, "hard_shield_thresholds_unchanged": True,
        "source_sha256": {name: sha(OUT / name) for name in ("summary.json", "verification.json", "family_paired_deltas.csv", "gate_diagnostics.csv", "derived_training_targets.npz")},
        "script_sha256": sha(__file__)}
    (OUT / "descriptive_quantification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
