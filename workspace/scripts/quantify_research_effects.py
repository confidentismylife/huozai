"""Read-only effect-size audit of saved development results; no simulator imports.

Positive 'reduction' means fewer synthetic incapacitations. This does not
estimate lives saved, deployment safety, or an LLM causal contribution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research_value_quantification_20261003/effectsize_agent"
INPUT_HASHES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative: str):
    path = ROOT / relative
    INPUT_HASHES[relative] = digest(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(relative: str):
    path = ROOT / relative
    INPUT_HASHES[relative] = digest(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ci(values, seed=20261803):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return np.quantile(sampled, [0.025, 0.975]).tolist()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - width, center + width]


def changes(arm, base):
    result = {}
    for metric in ("final_incapacitation", "final_fed", "additional_congestion", "steps"):
        b, t = float(base[metric]), float(arm[metric])
        result[metric] = {"baseline": b, "treatment": t, "difference": t-b,
                          "relative_change_percent": (t-b)/b*100 if b else None}
    result["incapacitation_reduction_pp"] = (float(base["final_incapacitation"])-float(arm["final_incapacitation"]))*100
    return result


def extra_group_effect(episodes, member_arms, arm, seed):
    """Additional descriptive paired CI for a mean of fixed supervised seeds."""
    start_rows = [r for r in episodes if int(r["step"]) == 0]
    families = sorted({r["source_id"] for r in start_rows})
    differences = []
    for family in families:
        subset = [r for r in start_rows if r["source_id"] == family]
        treatment = [float(r["final_incapacitation"]) for r in subset if r["arm"] in member_arms]
        baseline = [float(r["final_incapacitation"]) for r in subset if r["arm"] == "pure_ppo_601"]
        differences.append((np.mean(treatment)-np.mean(baseline))*100)
    interval = ci(differences, seed)
    return {"subset":"initial","arm":arm,"control":"pure_ppo_601","families":len(families),
            "delta_pp":float(np.mean(differences)),"exploratory_ci_low_pp":interval[0],"exploratory_ci_high_pp":interval[1],
            "improved_families":int((np.array(differences)<-1e-12).sum()),
            "worsened_families":int((np.array(differences)>1e-12).sum()),
            "tied_families":int((np.abs(differences)<=1e-12).sum()),
            "additional_posthoc_summary":True,"bootstrap_seed":seed,"resamples":2000}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_dir = "outputs/teacher_factorial_pilot_development_v1"
    pilot = read_json(f"{pilot_dir}/summary.json")
    arms = {r["arm"]: r for r in read_csv(f"{pilot_dir}/arm_summary.csv")}
    family_rows = read_csv(f"{pilot_dir}/family_paired_deltas.csv")
    records = [r for r in family_rows if r["treatment"] == "worst3" and r["control"] == "pure_ppo_601" and r["metric"] == "final_incapacitation"]
    records.sort(key=lambda r:r["source_id"])
    assert len(records) == 24
    deltas = np.array([float(r["delta"]) for r in records])
    reductions = -deltas
    baseline = float(arms["pure_ppo_601"]["final_incapacitation"])
    treatment = float(arms["worst3"]["final_incapacitation"])
    original = next(r for r in pilot["paired_effects"] if r["treatment"] == "worst3" and r["control"] == "pure_ppo_601" and r["metric"] == "final_incapacitation")
    recomputed = ci(deltas)
    np.testing.assert_allclose(deltas.mean(), original["delta"], atol=1e-14)
    np.testing.assert_allclose(recomputed, [original["exploratory_ci_low"], original["exploratory_ci_high"]], atol=1e-14)
    order = np.argsort(-reductions, kind="stable")
    positive = int((reductions > 1e-12).sum())
    reduction_ci = [-recomputed[1], -recomputed[0]]
    concentration = [{"rank": i+1, "source_id": records[idx]["source_id"], "family_reduction_pp": float(reductions[idx]),
                      "fraction_of_total_net_reduction": float(reductions[idx]/reductions.sum()),
                      "cumulative_fraction": float(reductions[order[:i+1]].sum()/reductions.sum())}
                     for i,idx in enumerate(order[:positive])]
    delete = []
    for k in range(1, positive+1):
        kept = np.delete(reductions, order[:k])
        delete.append({"largest_gain_families_deleted": k, "deleted_ids": [records[idx]["source_id"] for idx in order[:k]],
                       "remaining_families": len(kept), "mean_reduction_pp": float(kept.mean()),
                       "sensitivity_exploratory_ci_pp": ci(kept),
                       "warning": "post_hoc sensitivity, not an alternative unbiased primary estimate"})
    ptdev = {"families":24, "baseline_incapacitation_percent":baseline*100,
             "teacher_incapacitation_percent":treatment*100,
             "absolute_reduction_pp":float(reductions.mean()),
             "relative_reduction_percent":(baseline-treatment)/baseline*100,
             "normalized_difference_per_1000_simulated_people":(baseline-treatment)*1000,
             "exploratory_reduction_ci_pp":reduction_ci,
             "exploratory_per_1000_ci":[x*10 for x in reduction_ci],
             "relative_effect_interval":None,
             "relative_effect_interval_note":"absolute bootstrap interval is not converted into a relative-effect CI by holding random baseline fixed",
             "improved_families":positive, "improvement_fraction":positive/24,
             "improvement_wilson_95_ci":wilson(positive,24),
             "wilson_note":"binomial descriptive interval; 6 types x 4 selected families, not a probability guarantee for deployment population",
             "worsened_families":int((reductions < -1e-12).sum()),
             "top_gain_families":concentration,
             "top1_fraction":concentration[0]["cumulative_fraction"],
             "top2_fraction":concentration[1]["cumulative_fraction"],
             "progressive_leave_largest_gain_out":delete,
             "all_arms_vs_601":{arm:changes(row,arms["pure_ppo_601"]) for arm,row in arms.items()},
             "factor_comparisons": [r for r in pilot["paired_effects"] if r["control"] != "pure_ppo_601"],
             "factorial_interaction":pilot["factorial_interaction"],
             "planning_budget_rollouts_by_rule":{"budget3_per_start":2*17*3,"budget12_per_start":2*17*12,
                                                "ratio12_to3":4,"increment_per_start":2*17*9},
             "bootstrap":{"unit":"scenario_family", "resamples":2000,"seed":20261803,
                          "unadjusted_exploratory":True,"original_result_reproduced":True}}
    conflict_rows = read_csv(f"{pilot_dir}/conflict_summary.csv")
    kinds = sorted({r["conflict_type"] for r in conflict_rows})
    ptdev["conflict_descriptions"] = []
    for kind in kinds:
        subset = {r["arm"]:r for r in conflict_rows if r["conflict_type"] == kind}
        b,t = float(subset["pure_ppo_601"]["final_incapacitation"]),float(subset["worst3"]["final_incapacitation"])
        ptdev["conflict_descriptions"].append({"conflict_type":kind,"families":4,"baseline_percent":b*100,
                                               "teacher_percent":t*100,"reduction_pp":(b-t)*100,
                                               "fraction_of_total_gain":(b-t)*4/(reductions.sum()/100)})
    selections = read_csv(f"{pilot_dir}/state_selection.csv")
    ptdev["worst12_vs_worst3_changed_choices"] = sum(r["worst12"] != r["worst3"] for r in selections)
    ptdev["selection_count"] = len(selections)

    obs_dir = "outputs/observable_plan_distillation_development_v1"
    obs = read_json(f"{obs_dir}/summary.json")
    obs_arms = {r["arm"]:r for r in obs["initial_summary"]}
    obs_effects = read_csv(f"{obs_dir}/scenario_paired_effects.csv")
    obs_episodes = read_csv(f"{obs_dir}/evaluation_episode_results.csv")
    obs_base = obs_arms["pure_ppo_601"]
    obs_teacher_reduction = changes(obs_arms["teacher_minimax"], obs_base)["incapacitation_reduction_pp"]
    obs_groups = {}
    for arm in ("five_full_students_expectation", "five_forecast_students_expectation", "teacher_minimax"):
        item = changes(obs_arms[arm],obs_base)
        found = [r for r in obs_effects if r["subset"] == "initial" and r["arm"] == arm and r["control"] == "pure_ppo_601"]
        item["saved_or_additional_exploratory_effect"] = found[0] if found else extra_group_effect(obs_episodes,{f"forecast_student_{seed}" for seed in range(801,806)},arm,20261602)
        if arm != "teacher_minimax":
            item["ratio_of_student_point_reduction_to_teacher_point_reduction_percent"] = item["incapacitation_reduction_pp"] / obs_teacher_reduction * 100
            item["ratio_note"] = "descriptive ratio only; student intervals cross zero, not evidence of preserved gain"
        obs_groups[arm] = item
    obs_output = {"families":24,"subset":"initial_only","groups_vs601":obs_groups,
                  "student_group_expectations_note":"five supervised initialization seeds; virtual mean performance is not a single deployed ensemble and not five new PPO training seeds"}

    expanded_dir = "outputs/baseline_relative_plan_distillation_development_v1"
    expanded = read_json(f"{expanded_dir}/summary.json")
    exp_rows = read_csv(f"{expanded_dir}/subset_summary.csv")
    exp_arms = {r["arm"]:r for r in exp_rows if r["subset"] == "initial"}
    exp_effects = read_csv(f"{expanded_dir}/scenario_paired_effects.csv")
    exp_episodes = read_csv(f"{expanded_dir}/evaluation_episode_results.csv")
    exp_groups = {}
    for arm in ("five_small_mse_expectation", "five_small_rank_expectation", "five_large_mse_expectation", "five_large_rank_expectation", "five_historical_mse_expectation", "teacher_minimax"):
        item = changes(exp_arms[arm],exp_arms["pure_ppo_601"])
        found = [r for r in exp_effects if r["subset"] == "initial" and r["arm"] == arm and r["control"] == "pure_ppo_601"]
        if not found and arm == "teacher_minimax":
            found = [r for r in expanded["initial_paired_effects"] if r["arm"] == arm and r["control"] == "pure_ppo_601"]
        members = {f"{arm.removeprefix('five_').removesuffix('_expectation')}_{seed}" for seed in range(801,806)}
        item["saved_or_additional_exploratory_effect"] = found[0] if found else extra_group_effect(exp_episodes,members,arm,20261703)
        exp_groups[arm] = item
    exp_output = {"families":36,"subset":"initial_only","groups_vs601":exp_groups,
                  "training_size_change":{"small_families":24,"large_families":84,"family_ratio":84/24,"small_rows":168,"large_rows":592,"row_ratio":592/168},
                  "note":"teacher gain is near zero with interval crossing zero; retention ratios would be unstable and misleading, so omitted"}

    history = []
    manual_dir = "outputs/source_world_planning_development_v1"
    manual = read_json(f"{manual_dir}/summary.json")
    manual_arms = {r["arm"]:r for r in manual["initial_summary"]}
    manual_effect = next(r for r in manual["initial_scenario_effects"] if r["arm"] == "minimax_world_planner")
    history.append({"bank":"reused manual-planner development36", "source":f"{manual_dir}/summary.json", "subset":"initial",
                    "families":36,"teacher":"minimax_world_planner","control":"live_graded", "library":"12 manual plans",
                    "teacher_percent":manual_arms["minimax_world_planner"]["final_incapacitation"]*100,
                    "control_percent":manual_arms["live_graded"]["final_incapacitation"]*100,
                    "reduction_pp":-manual_effect["scenario_equal_weight_delta_pp"],
                    "reduction_ci_pp":[-manual_effect["scenario_bootstrap_ci_high_pp"],-manual_effect["scenario_bootstrap_ci_low_pp"]],
                    "comparison_note":"weak rule control; cannot compare this gain with strong-PPO gain as an improvement trajectory"})
    context_dir = "outputs/source_world_planning_frozen_ppo_context_v1"
    context = read_json(f"{context_dir}/summary.json")
    context601 = next(r for r in context["per_seed"] if r["training_seed"]==601)
    history.append({"bank":"posthoc frozen PPO context36","source":f"{context_dir}/summary.json","subset":"initial","families":36,
                    "teacher":"manual minimax","control":"pure_ppo_601","library":"12 manual plans",
                    "teacher_percent":manual_arms["minimax_world_planner"]["final_incapacitation"]*100,
                    "control_percent":context601["incapacitation_rate"]*100,
                    "reduction_pp":context601["ppo_minus_minimax_delta_pp"],"reduction_ci_pp":None,
                    "comparison_note":"posthoc context; no paired interval supplied, not fair training ablation"})
    extended_dir = "outputs/ppo_extended_world_planning_development_v1"
    extended = read_json(f"{extended_dir}/summary.json")
    extended_arms = {r["arm"]:r for r in extended["initial_summary"]}
    extended_effect = next(r for r in extended["initial_scenario_effects"] if r["arm"]=="extended_minimax" and r["control"]=="pure_ppo_601")
    history.append({"bank":"reused PPO-extended development36","source":f"{extended_dir}/summary.json","subset":"initial","families":36,
                    "teacher":"extended_minimax","control":"pure_ppo_601","library":"12 manual + five frozen PPO",
                    "teacher_percent":extended_arms["extended_minimax"]["family_equal_final_incapacitation"]*100,
                    "control_percent":extended_arms["pure_ppo_601"]["family_equal_final_incapacitation"]*100,
                    "reduction_pp":-extended_effect["scenario_equal_delta_pp"],
                    "reduction_ci_pp":[-extended_effect["exploratory_ci_high_pp"],-extended_effect["exploratory_ci_low_pp"]]})
    for bank,source,arm_rows,effects,teacher in (
        ("new observable-distillation development24",f"{obs_dir}/summary.json",obs_arms,obs["initial_paired_effects"],"teacher_minimax"),
        ("new baseline-relative development36",f"{expanded_dir}/summary.json",exp_arms,expanded["initial_paired_effects"],"teacher_minimax")):
        e = next(r for r in effects if r["arm"]==teacher and r["control"]=="pure_ppo_601")
        history.append({"bank":bank,"source":source,"subset":"initial","families":e["families"],"teacher":teacher,"control":"pure_ppo_601",
                        "library":"12 manual + five frozen PPO","teacher_percent":float(arm_rows[teacher]["final_incapacitation"])*100,
                        "control_percent":float(arm_rows["pure_ppo_601"]["final_incapacitation"])*100,"reduction_pp":-e["delta_pp"],
                        "reduction_ci_pp":[-e["exploratory_ci_high_pp"],-e["exploratory_ci_low_pp"]]})
    history.append({"bank":"new PTDEV factorial development24","source":f"{pilot_dir}/summary.json","subset":"initial","families":24,
                    "teacher":"worst3","control":"pure_ppo_601","library":"12 manual + five frozen PPO",
                    "teacher_percent":treatment*100,"control_percent":baseline*100,"reduction_pp":float(reductions.mean()),
                    "reduction_ci_pp":reduction_ci})
    diagnosis = read_json("outputs/plan_teacher_diagnosis_development_v1/summary.json")
    duplicate_diag = next(r for r in diagnosis["aggregates"] if r["subset"]=="initial" and r["teacher"]=="teacher_minimax")
    separate_allstates = {"observable_summary_rows":[r for r in read_csv(f"{obs_dir}/subset_summary.csv") if r["subset"]=="all_states" and r["arm"] in ("pure_ppo_601","teacher_minimax")],
                          "expanded_summary_rows":[r for r in exp_rows if r["subset"]=="all_states" and r["arm"] in ("pure_ppo_601","teacher_minimax")],
                          "warning":"counterfactual continuation starts are dependent and differ in event time; not pooled with initial-state trials"}
    result = {"created_utc":datetime.now(timezone.utc).isoformat(),"mode":"read_only_saved_result_quantification",
              "new_simulations":0,"new_training_runs":0,"new_llm_calls":0,"freeze_allowed":False,
              "ptdev24":ptdev,"observable_distillation24":obs_output,"expanded_distillation36":exp_output,
              "historical_teacher_comparisons_initial_only":history,
              "diagnosis_same_prau36_not_another_bank":{"reduction_pp":-duplicate_diag["delta_vs_601_pp"],
                    "diagnostic_reduction_ci_pp":[-duplicate_diag["exploratory_ci_high_pp"],-duplicate_diag["exploratory_ci_low_pp"]],
                    "note":"different bootstrap seed for same saved data; not additional evidence"},
              "separate_all_state_descriptions":separate_allstates,
              "limitations":["All are viewed development data and unadjusted exploratory intervals, not a final test.",
                  "Fixed PPO601 is one strong frozen controller; no claim of averaging treatment across independent PPO retraining seeds.",
                  "Privileged teacher has copied individual simulator state and larger search budget; this is not equal-information/equal-cost LLM vs PPO evidence.",
                  "Per-1000 is normalization of simulator incapacitation rates; not an estimate of real people saved or validated toxicity.",
                  "FED, congestion and steps are synthetic simulator metrics; changes need not imply validated real hazard reduction.",
                  "Do not pool banks as meta-analysis: different candidate libraries, scenario parameters, futures and posthoc research decisions.",
                  "Wilson uses a binomial description despite balanced conflict quotas; not calibrated population generalization.",
                  "Gain concentration/deletion analyses are posthoc sensitivity, not instructions to delete difficult cases.",
                  "No probability of eventual project success or paper acceptance is estimated."],
              "source_file_sha256":INPUT_HASHES}
    for relative,hash_value in INPUT_HASHES.items():
        assert digest(ROOT/relative)==hash_value
    result["integrity"]={"all_read_source_files_unchanged":True,"read_source_files":len(INPUT_HASHES),"code_sha256":digest(Path(__file__))}
    (OUT/"effect_sizes.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    md=["# 当前研究已观察效果量化（开发证据）","","本文件仅读既有结果；没有新增仿真、训练或LLM调用。正降低表示模拟失能更少。", "",
        f"PTDEV24：基线 {baseline*100:.6f}% → 教师 {treatment*100:.6f}%；绝对降低 {reductions.mean():.6f} 个百分点，相对降低 {ptdev['relative_reduction_percent']:.3f}%。",
        f"归一化每1000模拟人员差 {ptdev['normalized_difference_per_1000_simulated_people']:.4f}；绝对降低探索性95%区间 [{reduction_ci[0]:.6f}, {reduction_ci[1]:.6f}] 个百分点。此数不是现实救人数。",
        f"6/24家庭改善，描述性Wilson95%区间 [{ptdev['improvement_wilson_95_ci'][0]*100:.2f}%, {ptdev['improvement_wilson_95_ci'][1]*100:.2f}%]；不是部署总体保证。",
        f"最大1/2家庭占总净收益 {ptdev['top1_fraction']*100:.2f}% / {ptdev['top2_fraction']*100:.2f}%。","",
        "|逐删最大收益家庭数|剩余家庭平均降低（百分点）|","|---:|---:|"]
    md += [f"|{r['largest_gain_families_deleted']}|{r['mean_reduction_pp']:.6f}|" for r in delete]
    md += ["","上述删除仅做敏感性分析，不能据此清理不利场景。","","|历史开发银行与控制|教师失能率%|控制失能率%|降低百分点|探索性降低区间|","|---|---:|---:|---:|---|"]
    for r in history:
        interval_text="未提供" if r["reduction_ci_pp"] is None else f"[{r['reduction_ci_pp'][0]:.6f}, {r['reduction_ci_pp'][1]:.6f}]"
        md.append(f"|{r['bank']} / {r['control']}|{r['teacher_percent']:.6f}|{r['control_percent']:.6f}|{r['reduction_pp']:.6f}|{interval_text}|")
    md += ["","银行、库、未来流与研发决策不同，不合并成元分析。全状态结果单列于JSON，没有与初始起点混合。","","|初版可观测学生组|相对601降低百分点|学生/教师点收益比|","|---|---:|---:|"]
    for arm in ("five_full_students_expectation","five_forecast_students_expectation"):
        r=obs_groups[arm]
        md.append(f"|{arm}|{r['incapacitation_reduction_pp']:.6f}|{r['ratio_of_student_point_reduction_to_teacher_point_reduction_percent']:.2f}%|")
    md += ["","上述比率仅作描述；学生区间跨零，不能称已保留教师收益。","","|扩训练36学生/教师|相对601降低百分点|","|---|---:|"]
    md += [f"|{arm}|{r['incapacitation_reduction_pp']:.6f}|" for arm,r in exp_groups.items()]
    md += ["","扩训练组教师点收益接近零且区间跨零，不计算容易误导的学生/教师保留比。","","完整副指标、因素比较、来源哈希和局限见同目录 effect_sizes.json。"]
    md += ["","|PTDEV主臂相对601|失能降低百分点|FED变化%|额外拥堵变化%|步数变化%|","|---|---:|---:|---:|---:|"]
    for arm in ("worst3","worst12","mean3","mean12"):
        r=ptdev["all_arms_vs_601"][arm]
        md.append(f"|{arm}|{r['incapacitation_reduction_pp']:.6f}|{r['final_fed']['relative_change_percent']:.3f}|{r['additional_congestion']['relative_change_percent']:.3f}|{r['steps']['relative_change_percent']:.3f}|")
    md += ["", "FED、副拥堵和步数不是实测物理伤害指标。3→12未来需4倍规划预算，worst主失能相同但7/48初始起点改选；这不是所有动作完全相同。",
           "当前clear与forecast_error各4家庭，两个臂主失能都为0；不能拿失能零差证明已经解决预测错误。"]
    (OUT/"effect_sizes.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"ptdev":{k:ptdev[k] for k in ("absolute_reduction_pp","relative_reduction_percent","normalized_difference_per_1000_simulated_people","exploratory_reduction_ci_pp","improvement_wilson_95_ci","top1_fraction","top2_fraction","worst12_vs_worst3_changed_choices")},
                      "progressive_deletion":delete,"obs_retention":obs_groups,
                      "expanded_reductions":{arm:r["incapacitation_reduction_pp"] for arm,r in exp_groups.items()},
                      "integrity":result["integrity"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
