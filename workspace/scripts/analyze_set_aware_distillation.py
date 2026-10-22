"""All-arm family-paired analysis of the frozen small set-aware pilot."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_crossed_teacher_rng_diagnostic import digest, read_json, write_json, write_csv
from set_aware_plan_student import ensemble_decision, OBJECTIVES


def stratified_interval(values, kinds, seed=20262002):
    values, kinds = np.asarray(values, float), np.asarray(kinds)
    rng = np.random.default_rng(seed)
    samples = np.zeros(2000)
    for kind in sorted(set(kinds)):
        part = values[kinds == kind]
        samples += part[rng.integers(0, len(part), size=(2000, len(part)))].sum(axis=1)
    samples /= len(values)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def analyze(output, new_planning=0, new_evaluation=0, elapsed=0):
    manifest = read_json(output / "development_manifest.json")
    frozen = read_json(output / "frozen_students.json")
    arms, names = manifest["arms"], manifest["candidate_names"]
    scenario_map = {r["scenario_id"]: r for r in manifest["audit_scenario_records"]}
    episodes, choices, gates, traces_summary = [], [], [], []
    family_metrics = {arm: defaultdict(list) for arm in arms}
    planning_total = evaluation_total = trace_steps = neutral_steps = blocked_steps = 0
    for path in sorted((output / "audit_states").glob("*.npz")):
        family = path.stem.split("_")[0]
        scenario = scenario_map[family]
        with np.load(path, allow_pickle=False) as z:
            common = {"state_id": path.stem, "source_id": family, "conflict_type": scenario["conflict_type"],
                      "population_seed": int(z["population_seed"]), "step": 0}
            planning_total += int(z["planning_rollouts"])
            evaluation_total += int(z["evaluation_rollouts"])
            sums = z["evaluation_outcomes"][:, :, [0, 2, 3]].sum(axis=2)
            np.testing.assert_allclose(sums, 1, atol=1e-12, rtol=0)
            for a, arm in enumerate(arms):
                selected = int(z["chosen_indices"][a])
                if selected == 12:
                    np.testing.assert_array_equal(z["evaluation_outcomes"][a], z["evaluation_outcomes"][arms.index("pure_ppo_601")])
                choices.append({**common, "arm": arm, "selected_plan": names[selected], "selected_601": selected == 12})
                for f, seed in enumerate(z["evaluation_seeds"]):
                    o = z["evaluation_outcomes"][a, f]
                    episodes.append({**common, "arm": arm, "selected_plan": names[selected], "future_seed": int(seed),
                                     **dict(zip(manifest["outcome_fields"], o))})
                    family_metrics[arm][family].append(o.copy())
            for i, objective in enumerate(OBJECTIVES):
                lo, hi = 5 * i, 5 * i + 5
                nominee, guarded, info = ensemble_decision(z["gains"][lo:hi], z["probabilities"][lo:hi], frozen["margins"][objective]["k"])
                assert nominee == z["chosen_indices"][2*i] and guarded == z["chosen_indices"][2*i+1]
                planning_means = z["risks"][z["supported"]].mean(axis=2)
                primary = planning_means[:, :, 0].max(axis=0)
                true_gain = (primary[12] - primary[nominee]) * 100
                nominee_eval = z["evaluation_outcomes"][2*i, :, 0]
                base_eval = z["evaluation_outcomes"][arms.index("pure_ppo_601"), :, 0]
                gates.append({**common, "objective": objective, "nominee": names[nominee], "guarded_plan": names[guarded], **info,
                    "nominee_cached_planning_gain_pp": float(true_gain),
                    "nominee_independent_eval_delta_vs601_pp": float((nominee_eval - base_eval).mean() * 100)})
            trace, keys, offsets = z["execution_trace"], z["trace_keys"], z["trace_offsets"]
            assert len(keys) == int(z["evaluation_rollouts"]) and offsets[-1] == len(trace)
            for i, (plan, future) in enumerate(keys):
                t = trace[offsets[i]:offsets[i+1]]
                a = next(a for a, selected in enumerate(z["chosen_indices"]) if selected == plan)
                o = z["evaluation_outcomes"][a, future]
                assert len(t) == o[5]
                np.testing.assert_array_equal(t[:, 0], np.arange(len(t)))
                np.testing.assert_allclose([t[:, 4].mean(), (t[:, 1] == 8).sum(), t[:, 5].mean(), t[0, 1], t[0, 2]], o[6:11], atol=1e-12, rtol=0)
            trace_steps += len(trace)
            neutral_steps += int(((trace[:, 3] == 2) & (trace[:, 4] > 0)).sum())
            blocked_steps += int(trace[:, 5].sum())
            traces_summary.append({**common, "unique_evaluation_rollouts": len(keys), "trace_steps": len(trace),
                "neutral_positive_release_steps": int(((trace[:, 3] == 2) & (trace[:, 4] > 0)).sum()),
                "hold_blocked_steps": int(trace[:, 5].sum())})
    assert len(choices) == 48 * 22 and planning_total == 4896 and evaluation_total <= 2448
    family_ids = sorted(scenario_map)
    kinds = [scenario_map[f]["conflict_type"] for f in family_ids]
    matrices = {arm: np.asarray([np.mean(family_metrics[arm][family], axis=0) for family in family_ids]) for arm in arms}
    arm_rows, effect_rows, family_deltas = [], [], []
    for arm in arms:
        matrix = matrices[arm]
        row = {"arm": arm, "families": 24, "initial_states": 48, "logical_evaluation_records": 144,
               **{name: float(matrix[:, i].mean()) for i, name in enumerate(manifest["outcome_fields"][:9])}}
        selected = [r for r in choices if r["arm"] == arm]
        row["selected_601_states"] = sum(r["selected_601"] for r in selected)
        arm_rows.append(row)
        delta = (matrix[:, 0] - matrices["pure_ppo_601"][:, 0]) * 100
        low, high = stratified_interval(delta, kinds)
        effect_rows.append({"treatment": arm, "control": "pure_ppo_601", "families": 24,
            "delta_incap_pp": float(delta.mean()), "exploratory_ci_low_pp": low, "exploratory_ci_high_pp": high,
            "improved_families": int((delta < -1e-12).sum()), "worsened_families": int((delta > 1e-12).sum()),
            "tied_families": int((np.abs(delta) <= 1e-12).sum())})
        for family, kind, value in zip(family_ids, kinds, delta):
            family_deltas.append({"family": family, "conflict_type": kind, "arm": arm, "delta_incap_vs601_pp": float(value)})
    factor_rows = []
    for treatment, control in (("set_ungated", "single_ungated"), ("set_guarded", "single_guarded"),
                               ("single_guarded", "single_ungated"), ("set_guarded", "set_ungated")):
        for metric, i, scale in (("incap_pp", 0, 100), ("fed", 1, 1), ("remaining_pp", 3, 100)):
            delta = (matrices[treatment][:, i] - matrices[control][:, i]) * scale
            low, high = stratified_interval(delta, kinds)
            factor_rows.append({"treatment": treatment, "control": control, "metric": metric,
                               "delta": float(delta.mean()), "exploratory_ci_low": low, "exploratory_ci_high": high})
    interaction = ((matrices["set_guarded"][:, 0] - matrices["set_ungated"][:, 0]) -
                   (matrices["single_guarded"][:, 0] - matrices["single_ungated"][:, 0])) * 100
    interaction_low, interaction_high = stratified_interval(interaction, kinds)
    gate_summary = []
    for objective in OBJECTIVES:
        rows = [r for r in gates if r["objective"] == objective]
        group = {"objective": objective, "states": 48, "k": frozen["margins"][objective]["k"],
                 "nonbaseline_nomination_states": sum(r["nominee"] != "pure_ppo_601" for r in rows),
                 "allowed_override_states": sum(r["allowed_override"] for r in rows),
                 "blocked_nomination_states": sum(r["blocked_nomination"] for r in rows)}
        # Conditional contributions use zero outside category and family-equal
        # means; these are descriptions, not deployment rules fitted to outcomes.
        for category in ("blocked", "allowed", "baseline"):
            def included(r):
                return r["blocked_nomination"] if category == "blocked" else r["allowed_override"] if category == "allowed" else r["nominee"] == "pure_ppo_601"
            contributions = defaultdict(list)
            for row in rows:
                contributions[row["source_id"]].append(row["nominee_independent_eval_delta_vs601_pp"] if included(row) else 0)
            group[f"{category}_ungated_delta_contribution_pp"] = float(np.mean([np.mean(v) for v in contributions.values()]))
        group["nominee_counts"] = dict(Counter(r["nominee"] for r in rows))
        group["guarded_plan_counts"] = dict(Counter(r["guarded_plan"] for r in rows))
        gate_summary.append(group)
    for name, rows in (("evaluation_episode_results.csv", episodes), ("state_selection.csv", choices), ("gate_diagnostics.csv", gates),
                       ("execution_trace_summary.csv", traces_summary), ("arm_summary.csv", arm_rows), ("paired_effects.csv", effect_rows),
                       ("family_paired_deltas.csv", family_deltas), ("factorial_effects.csv", factor_rows)):
        write_csv(output / name, rows)
    changed = [path for path, expected in manifest["source_input_sha256"].items() if digest(path) != expected]
    assert not changed
    for record in frozen["models"]:
        assert digest(output / record["checkpoint"]) == record["sha256"]
    primary = next(r for r in effect_rows if r["treatment"] == "set_guarded")
    summary = {"role": manifest["role"], "freeze_allowed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "families": 24, "initial_states": 48, "execution_arms": 22, "logical_execution_records": len(episodes),
        "new_supervised_training_runs": 10, "new_ppo_training_runs": 0, "new_llm_calls": 0,
        "planning_rollouts": planning_total, "unique_evaluation_rollouts": evaluation_total,
        "total_unique_full_rollouts": planning_total + evaluation_total,
        "new_planning_rollouts_this_invocation": new_planning, "new_evaluation_rollouts_this_invocation": new_evaluation,
        "elapsed_seconds_audit_this_invocation": elapsed, "actual_execution_trace_steps": trace_steps,
        "neutral_positive_release_trace_steps": neutral_steps, "hold_blocked_trace_steps": blocked_steps,
        "primary_paired_effect": primary, "gate_summary": gate_summary, "arm_summary": arm_rows,
        "all_paired_effects": effect_rows, "factorial_effects": factor_rows,
        "incap_interaction": {"delta_pp": float(interaction.mean()), "exploratory_ci_low_pp": interaction_low, "exploratory_ci_high_pp": interaction_high},
        "limitations": ["24 new same-generator development families, not power-backed confirmation or final testing.",
            "Calibration families were previously viewed; fixed new checkpoint/margin split does not make them blind.",
            "Five initializations model disagreement, not complete latent-population uncertainty.",
            "Six-family empirical all-plan margin is not calibrated safety probability, confidence bound, or hard shield.",
            "Unadjusted exploratory family-stratified bootstrap conditional on frozen controllers.",
            "One initial meta-plan nomination followed by full closed-loop candidate, not continual meta-plan reselection.",
            "New objectives share architecture/information/cap budgets; legacy ensemble differs in architecture and selection.",
            "Privileged planning labels and teacher include hidden initialization and simulator futures.",
            "Set near-ties need not preserve secondary FED or efficiency; those outcomes are reported separately.",
            "No behavior/map/noise shift, real fire verification, fair LLM-content ablation or formal PPO training."]}
    write_json(output / "summary.json", summary)
    write_json(output / "integrity_audit.json", {"all_checks_passed": True, "original_inputs_unchanged": len(manifest["source_input_sha256"]),
        "frozen_supervised_models_unchanged": 10, "population_conservation": True, "selected601_outcomes_equal_baseline": True,
        "actual_trace_steps_checked": trace_steps, "duplicate_execution_plans_deduplicated": True, "changed_original_inputs": []})
    write_json(output / "analysis_manifest.json", {"analysis_code_sha256": digest(__file__), "development_manifest_sha256": digest(output / "development_manifest.json"),
        "frozen_students_sha256": digest(output / "frozen_students.json"), "summary_sha256": digest(output / "summary.json"),
        "integrity_sha256": digest(output / "integrity_audit.json"), "output_csv_sha256": {p.name: digest(p) for p in sorted(output.glob("*.csv"))},
        "audit_cache_sha256": {p.name: digest(p) for p in sorted((output / "audit_states").glob("*.npz"))}})
    write_json(output / "progress.json", {"phase": "complete", "families_completed": 24, "planning_rollouts": planning_total,
        "unique_evaluation_rollouts": evaluation_total, "supervised_models": 10})
    print({"completed": True, "primary_paired_effect": primary, "gate_summary": gate_summary,
           "total_unique_full_rollouts": planning_total + evaluation_total}, flush=True)
