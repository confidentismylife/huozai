"""Prospective sample-size/cost sensitivity using existing development outcomes.

No evacuation simulation, training, LLM call, or change to previous outputs.
The paired t-model calculations are planning references, not retrospective power
or an exact power calculation for the existing percentile-bootstrap analyses.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
import statsmodels
from scipy.stats import t
from statsmodels.stats.power import TTestPower

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/research_value_quantification_20261003/budget_power"
BANKS = (
    ("ppo_extended_world_planning_development_v1", "extended_minimax"),
    ("observable_plan_distillation_development_v1", "teacher_minimax"),
    ("baseline_relative_plan_distillation_development_v1", "teacher_minimax"),
    ("teacher_factorial_pilot_development_v1", "worst3"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs, descriptions, vectors = {}, [], {}
    for bank, teacher in BANKS:
        path = ROOT / "outputs" / bank / "evaluation_episode_results.csv"
        inputs[str(path)] = digest(path)
        rows = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["step"]) == 0 and row["arm"] in (teacher, "pure_ppo_601"):
                    rows[row["source_id"], row["arm"]].append(float(row["final_incapacitation"]))
        families = sorted({family for family, arm in rows})
        gains = np.asarray([100 * (np.mean(rows[f, "pure_ppo_601"])
                                   - np.mean(rows[f, teacher])) for f in families])
        vectors[bank] = gains
        descriptions.append({"bank": bank, "teacher": teacher, "families": len(families),
                             "gain_vs_601_pp": float(gains.mean()),
                             "paired_family_sd_pp": float(gains.std(ddof=1)),
                             "unit": "percentage_points_gain_positive"})

    # Future independent family means, fixed models, two-sided paired t model.
    # Each variance estimate is a separate sensitivity case, not pooled evidence.
    sample_rows, mde_rows = [], []
    for item in descriptions:
        sd = item["paired_family_sd_pp"]
        for effect in (0.1, 0.2, 0.25, 0.5):
            for alpha, number in ((0.05, 1), (0.0125, 4)):
                n = math.ceil(TTestPower().solve_power(effect_size=effect / sd,
                              alpha=alpha, power=.8, alternative="two-sided"))
                sample_rows.append({"variance_source_bank": item["bank"], "assumed_sd_pp": sd,
                                    "target_gain_pp": effect, "alpha_per_test": alpha,
                                    "primary_comparisons": number, "target_power": .8,
                                    "model": "paired_t_reference_independent_family_means",
                                    "required_families": n,
                                    "six_type_balanced_families": math.ceil(n / 6) * 6})
        for n in (24, 48, 96, 108, 192, 492):
            effect = TTestPower().solve_power(effect_size=None, nobs=n, alpha=.05,
                                            power=.8, alternative="two-sided") * sd
            mde_rows.append({"variance_source_bank": item["bank"], "assumed_sd_pp": sd,
                             "independent_families": n, "mde_pp": float(effect),
                             "alpha": .05, "target_power": .8,
                             "model": "paired_t_reference_independent_family_means"})

    pilot_path = ROOT / "outputs/teacher_factorial_pilot_development_v1/summary.json"
    inputs[str(pilot_path)] = digest(pilot_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    seconds_per_rollout = pilot["elapsed_seconds_this_invocation"] / pilot["total_unique_full_rollouts"]
    cost_rows = []
    for n in (24, 72, 108, 192, 408, 492):
        # Same 2 populations, 2 worlds, 17 candidates and 12 planning futures;
        # at most 11 logical execution arms and 6 evaluation futures.
        planning = n * 2 * 2 * 17 * 12
        evaluation_upper = n * 2 * 11 * 6
        empirical_count = pilot["total_unique_full_rollouts"] * n / pilot["families"]
        cost_rows.append({"families": n, "planning_rollouts": planning,
                          "unique_evaluation_rollout_upper_bound": evaluation_upper,
                          "full_rollout_upper_bound": planning + evaluation_upper,
                          "empirical_scaled_rollout_count_not_measured": empirical_count,
                          "empirical_scaled_minutes_not_measured": pilot["elapsed_seconds_this_invocation"] * n / pilot["families"] / 60,
                          "upper_bound_scaled_minutes_not_measured": (planning + evaluation_upper) * seconds_per_rollout / 60})
    gains = vectors["teacher_factorial_pilot_development_v1"]
    mean, sd, n = float(gains.mean()), float(gains.std(ddof=1)), len(gains)
    half = float(t.ppf(.975, n - 1) * sd / math.sqrt(n))
    result = {"role": "post_hoc_descriptive_cost_and_prospective_sensitivity_not_new_control_experiment",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "new_evacuation_rollouts": 0, "new_training_runs": 0, "new_llm_calls": 0,
              "software_versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                                    "statsmodels": statsmodels.__version__},
              "input_sha256": inputs, "bank_specific_variances_not_pooled": descriptions,
              "future_sample_size_sensitivity": sample_rows,
              "future_mde_sensitivity": mde_rows, "future_cost_scaling": cost_rows,
              "per_start_planning_cost": {"worlds": 2, "candidates": 17,
                                           "three_futures_full_rollouts": 102,
                                           "twelve_futures_full_rollouts": 408,
                                           "planning_cost_ratio": 4},
              "pilot_family_t_interval_method_sensitivity_only": {
                  "gain_pp": mean, "ci_low_pp": mean - half, "ci_high_pp": mean + half,
                  "role": "post_hoc_alternative_method_diagnostic_original_bootstrap_preserved"},
              "total_full_rollouts_across_four_listed_runs_only": sum(
                  json.loads((ROOT / "outputs" / bank / "summary.json").read_text(encoding="utf-8"))["total_unique_full_rollouts"]
                  for bank, teacher in BANKS),
              "limitations": [
                  "Target effects are decision-sensitivity examples, not certified fire safety standards.",
                  "No observed/post-hoc power is calculated from the estimated pilot mean.",
                  "Family t-model sample sizes are approximate planning references for that model, not exact power for percentile bootstrap.",
                  "Sparse, skewed pilot differences and uncertain variances require simulation of the eventual planned analysis before confirmatory sample-size freezing.",
                  "Variance sources are distinct development banks; no meta-analysis or universal effect is claimed.",
                  "Fixed-model calculations do not cover PPO training-seed variation, new maps, behavior or observation-noise shifts.",
                  "Time scaling assumes the same hardware, simulator, implementation and approximate trajectory lengths; it is not a measured deployment latency or monetary cost.",
                  "The four-run total includes only listed runs, not the whole project; planning trajectories are not independent validation observations.",
              ]}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("sample_size_sensitivity.csv", sample_rows),
                           ("mde_sensitivity.csv", mde_rows), ("cost_sensitivity.csv", cost_rows)):
        with (DESTINATION / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result["code_sha256"] = digest(__file__)
    (DESTINATION / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert all(digest(path) == expected for path, expected in inputs.items())
    print(json.dumps({"completed": True, "source_banks": descriptions,
                      "sample_examples": [r for r in sample_rows if r["target_gain_pp"] == .2 and r["alpha_per_test"] == .05],
                      "t_interval_sensitivity": result["pilot_family_t_interval_method_sensitivity_only"],
                      "no_new_control_runs": True}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
