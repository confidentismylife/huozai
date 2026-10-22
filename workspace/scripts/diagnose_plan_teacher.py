"""Read-only post-hoc teacher diagnosis from frozen planning/execution caches."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from fire_evac.source_world_planning import select_reference_plan

TEACHERS = (("teacher_minimax", "robust"), ("teacher_point", "point"), ("teacher_oracle", "oracle"))
BASE = "pure_ppo_601"
EPS = 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty diagnostic table: {path}")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def family_average(rows, field):
    values = {}
    for row in rows:
        values.setdefault(row["source_id"], []).append(float(row[field]))
    return {key: float(np.mean(values[key])) for key in sorted(values)}


def interval(values, seed=20261801):
    values = np.asarray(values, np.float64)
    assert len(values) and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def main(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("diagnostics must be outside the original experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "development_manifest.json").read_text(encoding="utf-8"))
    original_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "frozen_students.json").read_text(encoding="utf-8"))
    names, arms, fields = manifest["candidate_names"], manifest["arms"], manifest["outcome_fields"]
    assert len(names) == 17 and len(arms) == 38 and BASE in names
    base_index, base_arm = names.index(BASE), arms.index(BASE)
    scenarios = {s["scenario_id"]: s for s in manifest["audit_scenario_records"]}
    selections = read_csv(source / "state_selection.csv")
    assert len(selections) == 248 and len(scenarios) == 36
    csv_lookup = {}
    for row in read_csv(source / "evaluation_episode_results.csv"):
        key = row["state_id"], row["arm"], int(row["future_seed"])
        assert key not in csv_lookup
        csv_lookup[key] = row
    caches = sorted((source / "audit_states").glob("*.npz"))
    assert len(caches) == len(selections)
    selection_lookup = {row["state_id"]: row for row in selections}
    selector_path = ROOT / "src/fire_evac/source_world_planning.py"
    assert digest(selector_path) == manifest["source_hashes"]["src/fire_evac/source_world_planning.py"]
    inputs = caches + [source / name for name in (
        "development_manifest.json", "frozen_students.json", "summary.json", "run_summary.json",
        "state_selection.csv", "evaluation_episode_results.csv", "integrity_audit.json")]
    inputs += [ROOT / "reports/反事实教师误选诊断开发方案.md", selector_path]
    for model in manifest["frozen_ppo_models"]:
        path = Path(model["path"])
        assert digest(path) == model["checkpoint_sha256"]
        inputs.append(path)
    for model in frozen["models"]:
        path = source / model["checkpoint"]
        assert digest(path) == model["sha256"]
        inputs.append(path)
    before = {str(path): digest(path) for path in sorted(set(inputs))}
    diagnostic_manifest = {
        "role": "post_hoc_existing_development_cache_teacher_diagnosis_not_confirmatory",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source), "source_input_sha256": before,
        "diagnostic_code_sha256": digest(__file__),
        "major_subset": "step_0_family_equal",
        "bootstrap_seed": 20261801, "bootstrap_repetitions": 2000,
        "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0,
    }
    json_write(output / "diagnostic_manifest.json", diagnostic_manifest)
    rows, candidate_rows, loo_rows, trace_rows, family_rows = [], [], [], [], []
    csv_checks = duplicate_checks = trace_checks = 0
    planning_seeds, evaluation_seeds = set(), set()
    for path in caches:
        state_id = path.stem
        meta = selection_lookup[state_id]
        scenario = scenarios[meta["source_id"]]
        true_source, forecast_source = scenario["source_branch"], scenario["forecast_source_branch"]
        with np.load(path, allow_pickle=False) as cached:
            risks, support = cached["risks"], cached["supported"]
            chosen, outcomes = cached["chosen_indices"], cached["evaluation_outcomes"]
            assert risks.shape == (2, 17, 3, 11) and outcomes.shape == (38, 3, 11)
            assert support[true_source] and support.sum() == int(meta["supported_world_count"])
            assert int(cached["population_seed"]) == int(meta["population_seed"])
            planning_seeds.update(map(int, cached["planning_seeds"]))
            evaluation_seeds.update(map(int, cached["evaluation_seeds"]))
            world_means = risks[support].mean(axis=2)
            true_means = risks[true_source].mean(axis=1)
            worst_means, average_means = world_means.max(axis=0), world_means.mean(axis=0)
            eval_by_candidate = {}
            for a, arm in enumerate(arms):
                p = int(chosen[a])
                assert names[p] == meta[arm]
                if p in eval_by_candidate:
                    np.testing.assert_array_equal(eval_by_candidate[p], outcomes[a])
                    duplicate_checks += 1
                eval_by_candidate[p] = outcomes[a].copy()
                for f, seed in enumerate(cached["evaluation_seeds"]):
                    original = csv_lookup[(state_id, arm, int(seed))]
                    assert original["selected_plan"] == names[p]
                    np.testing.assert_allclose(
                        [float(original[field]) for field in fields], outcomes[a, f], atol=1e-13, rtol=0)
                    assert abs(outcomes[a, f, 0] + outcomes[a, f, 2] + outcomes[a, f, 3] - 1) <= 1e-12
                    csv_checks += 1
            base_eval = outcomes[base_arm].mean(axis=0)
            common = {
                "state_id": state_id, "source_id": meta["source_id"], "conflict_type": meta["conflict_type"],
                "repetition": int(meta["repetition"]), "step": int(meta["step"]),
                "supported_world_count": int(support.sum()), "forecast_wrong": true_source != forecast_source,
                "true_source": true_source, "forecast_source": forecast_source,
            }
            for p, name in enumerate(names):
                value = eval_by_candidate.get(p)
                candidate_rows.append({**common, "plan": name, "plan_index": p,
                    "true_planning_incap": float(true_means[p, 0]),
                    "worst_planning_incap": float(worst_means[p, 0]),
                    "average_world_planning_incap": float(average_means[p, 0]),
                    "true_planning_fed": float(true_means[p, 1]),
                    "independent_evaluation_available": value is not None,
                    "independent_eval_incap": "" if value is None else float(value[:, 0].mean()),
                    "independent_eval_fed": "" if value is None else float(value[:, 1].mean()),
                    "independent_eval_delta_vs_601_pp": "" if value is None else float((value[:, 0].mean()-base_eval[0])*100)})
            teacher_indices = {}
            for arm, mode in TEACHERS:
                p = select_reference_plan(risks, support, forecast_source, mode,
                                          true_source=true_source if mode == "oracle" else None)
                teacher_indices[arm] = p
                assert int(chosen[arms.index(arm)]) == p
            for arm, mode in TEACHERS:
                p = teacher_indices[arm]
                evaluation = outcomes[arms.index(arm)].mean(axis=0)
                planning_delta = float((true_means[p, 0]-true_means[base_index, 0])*100)
                delta = float((evaluation[0]-base_eval[0])*100)
                planned_worst_delta = float((worst_means[p, 0]-worst_means[base_index, 0])*100)
                if p == base_index:
                    np.testing.assert_array_equal(outcomes[arms.index(arm)], outcomes[base_arm])
                if abs(delta) <= EPS:
                    category = "heldout_incap_tie"
                elif delta < 0:
                    category = "heldout_improvement"
                elif planning_delta > EPS:
                    category = "planned_true_world_cost_and_heldout_harm"
                else:
                    category = "planning_nonworse_but_heldout_harm"
                other_source = 1-true_source
                other_delta = float((risks[other_source,p,:,0].mean()-risks[other_source,base_index,:,0].mean())*100) if support[other_source] else ""
                best_five = min(range(12,17), key=lambda k: (eval_by_candidate[k][:,0].mean(), eval_by_candidate[k][:,1].mean(),k))
                primary_ties = int(np.isclose(worst_means[:,0], worst_means[p,0], atol=1e-12, rtol=0).sum())
                rows.append({**common, "teacher": arm, "selected_plan": names[p], "same_plan_as_601": p == base_index,
                    "category": category, "primary_planning_tie_count": primary_ties,
                    "true_planning_delta_vs_601_pp": planning_delta,
                    "other_planning_delta_vs_601_pp": other_delta,
                    "worst_planning_delta_vs_601_pp": planned_worst_delta,
                    "eval_delta_vs_601_pp": delta, "relative_planning_to_eval_shift_pp": delta-planning_delta,
                    "teacher_eval_incap": float(evaluation[0]), "baseline_eval_incap": float(base_eval[0]),
                    "teacher_eval_fed": float(evaluation[1]), "baseline_eval_fed": float(base_eval[1]),
                    "eval_fed_delta_vs_601": float(evaluation[1]-base_eval[1]),
                    "teacher_mean_release": float(evaluation[6]), "baseline_mean_release": float(base_eval[6]),
                    "teacher_wait_requests": float(evaluation[7]), "baseline_wait_requests": float(base_eval[7]),
                    "teacher_hold_block_rate": float(evaluation[8]), "baseline_hold_block_rate": float(base_eval[8]),
                    "posthoc_best_five_plan": names[best_five],
                    "posthoc_best_five_eval_incap": float(eval_by_candidate[best_five][:,0].mean()),
                    "teacher_delta_vs_posthoc_best_five_pp": float((evaluation[0]-eval_by_candidate[best_five][:,0].mean())*100)})
                for omitted in range(3):
                    kept = [f for f in range(3) if f != omitted]
                    index = select_reference_plan(risks[:,:,kept,:], support, forecast_source, mode,
                                                  true_source=true_source if mode == "oracle" else None)
                    omitted_delta = (risks[true_source,index,omitted,0]-risks[true_source,base_index,omitted,0])*100
                    value = eval_by_candidate.get(index)
                    loo_rows.append({**common, "teacher": arm, "omitted_future_index": omitted,
                        "three_future_plan": names[p], "two_future_plan": names[index],
                        "choice_changed": index != p,
                        "planning_leftout_true_delta_vs_601_pp": float(omitted_delta),
                        "independent_evaluation_available": value is not None,
                        "independent_eval_delta_vs_601_pp": "" if value is None else float((value[:,0].mean()-base_eval[0])*100)})
            for p in sorted(eval_by_candidate):
                traces = []
                for f in range(3):
                    locations = np.flatnonzero(np.all(cached["trace_keys"] == (p,f),axis=1))
                    assert len(locations) == 1
                    k = int(locations[0])
                    trace = cached["execution_trace"][cached["trace_offsets"][k]:cached["trace_offsets"][k+1]]
                    assert len(trace) == int(eval_by_candidate[p][f,5])
                    assert trace[0,1] == eval_by_candidate[p][f,9] and trace[0,2] == eval_by_candidate[p][f,10]
                    assert np.all((trace[:,2] == 8) == (trace[:,4] == 0))
                    assert np.all(trace[trace[:,2] == -1,4] > 0)
                    trace_checks += len(trace)
                    traces.append(trace)
                combined = np.concatenate(traces)
                trace_rows.append({**common, "plan": names[p], "trace_steps": len(combined),
                    "mean_actual_release": float(combined[:,4].mean()),
                    "actual_wait_steps": int((combined[:,2] == 8).sum()),
                    "neutral_positive_release_steps": int((combined[:,2] == -1).sum()),
                    "blocked_steps": int(combined[:,5].sum())})
    assert planning_seeds.isdisjoint(evaluation_seeds)
    comparisons, aggregates = [], []
    subset_predicates = [
        ("initial", lambda r:r["step"] == 0),
        ("all_states_descriptive", lambda r:True),
        ("forecast_wrong_initial", lambda r:r["step"] == 0 and r["forecast_wrong"]),
        ("forecast_correct_initial", lambda r:r["step"] == 0 and not r["forecast_wrong"]),
    ] + [("conflict_initial:"+kind, lambda r,k=kind:r["step"] == 0 and r["conflict_type"] == k)
         for kind in sorted({r["conflict_type"] for r in rows})]
    for subset, predicate in subset_predicates:
        for arm, _ in TEACHERS:
            selected = [r for r in rows if r["teacher"] == arm and predicate(r)]
            if not selected:
                continue
            means = family_average(selected, "eval_delta_vs_601_pp")
            interval_low,interval_high = interval(list(means.values()))
            fields_to_average = (
                "teacher_eval_incap", "baseline_eval_incap", "true_planning_delta_vs_601_pp",
                "worst_planning_delta_vs_601_pp", "relative_planning_to_eval_shift_pp",
                "eval_fed_delta_vs_601", "teacher_mean_release", "baseline_mean_release",
                "teacher_wait_requests", "baseline_wait_requests", "teacher_hold_block_rate",
                "posthoc_best_five_eval_incap", "teacher_delta_vs_posthoc_best_five_pp",
            )
            entry = {"subset": subset, "teacher": arm, "families": len(means), "states": len(selected),
                     "delta_vs_601_pp": float(np.mean(list(means.values()))),
                     "exploratory_ci_low_pp":interval_low,"exploratory_ci_high_pp":interval_high,
                     "improved_families":sum(v < -EPS for v in means.values()),
                     "worsened_families":sum(v > EPS for v in means.values()),
                     "tied_families":sum(abs(v) <= EPS for v in means.values()),
                     "same_plan_as_601_states":sum(r["same_plan_as_601"] for r in selected),
                     "selected_plan_counts":dict(Counter(r["selected_plan"] for r in selected)),
                     "state_category_counts":dict(Counter(r["category"] for r in selected))}
            entry.update({field:float(np.mean(list(family_average(selected,field).values()))) for field in fields_to_average})
            aggregates.append(entry)
            if subset == "initial":
                for family,value in means.items():
                    family_states = [r for r in selected if r["source_id"] == family]
                    family_rows.append({"source_id":family,"teacher":arm,"conflict_type":family_states[0]["conflict_type"],
                        "forecast_wrong":family_states[0]["forecast_wrong"],"states":len(family_states),
                        "delta_vs_601_pp":value,
                        "true_planning_delta_vs_601_pp":float(np.mean([r["true_planning_delta_vs_601_pp"] for r in family_states])),
                        "relative_planning_to_eval_shift_pp":float(np.mean([r["relative_planning_to_eval_shift_pp"] for r in family_states])),
                        "selected_plans":";".join(r["selected_plan"] for r in family_states)})
        for control in ("teacher_point","teacher_oracle"):
            treatment = [r for r in rows if r["teacher"] == "teacher_minimax" and predicate(r)]
            control_rows = {r["state_id"]:r for r in rows if r["teacher"] == control and predicate(r)}
            deltas = [{**r,"paired_delta_pp":(r["teacher_eval_incap"]-control_rows[r["state_id"]]["teacher_eval_incap"])*100}
                      for r in treatment]
            if deltas:
                values = list(family_average(deltas,"paired_delta_pp").values())
                comparisons.append({"subset":subset,"treatment":"teacher_minimax","control":control,
                    "families":len(values),"delta_pp":float(np.mean(values)),
                    "exploratory_ci_low_pp":interval(values)[0],"exploratory_ci_high_pp":interval(values)[1]})
    minimax_initial = next(r for r in aggregates if r["subset"] == "initial" and r["teacher"] == "teacher_minimax")
    assert abs(minimax_initial["delta_vs_601_pp"]-original_summary["teacher_vs_strong_fixed_initial"]["delta_pp"]) < 1e-12
    loo_summary = []
    for arm,_ in TEACHERS:
        for subset,predicate in subset_predicates[:2]:
            values = [r for r in loo_rows if r["teacher"] == arm and predicate(r)]
            selected_states = {r["state_id"] for r in values}
            changed_states = {r["state_id"] for r in values if r["choice_changed"]}
            loo_summary.append({"teacher":arm,"subset":subset,"states":len(selected_states),
                "any_leave_one_future_choice_change_states":len(changed_states),
                "fraction_states_changed":len(changed_states)/len(selected_states),
                "leave_one_out_selections":len(values),
                "changed_selections":sum(r["choice_changed"] for r in values),
                "missing_independent_eval_selections":sum(not r["independent_evaluation_available"] for r in values)})
    write_csv(output/"state_teacher_diagnostics.csv",rows)
    write_csv(output/"candidate_planning_vs_evaluation.csv",candidate_rows)
    write_csv(output/"leave_one_future_out.csv",loo_rows)
    write_csv(output/"execution_diagnostics.csv",trace_rows)
    write_csv(output/"initial_family_teacher_diagnostics.csv",family_rows)
    write_csv(output/"teacher_paired_effects.csv",comparisons)
    changed = [path for path,value in before.items() if digest(path) != value]
    assert not changed
    integrity = {"all_checks_passed":True,"input_files_hashed_and_unchanged":len(before),
        "states_checked":len(caches),"source_csv_outcome_records_checked":csv_checks,
        "duplicate_candidate_equalities_checked":duplicate_checks,
        "unique_execution_trace_steps_checked":trace_checks,
        "all_three_teacher_choices_reconstructed":True,"same_plan_results_identical":True,
        "planning_and_evaluation_seeds_disjoint":True,"population_conservation_checked":True,
        "actual_wait_and_neutral_positive_release_separated":True,
        "new_simulations":0,"new_training_runs":0,"new_llm_calls":0,"changed_original_inputs":changed}
    json_write(output/"integrity_audit.json",integrity)
    summary = {"role":diagnostic_manifest["role"],"freeze_allowed":False,
        "new_simulations":0,"new_training_runs":0,"new_llm_calls":0,
        "state_count":len(caches),"initial_state_count":72,"family_count":36,
        "aggregates":aggregates,"teacher_comparisons":comparisons,"leave_one_out":loo_summary,
        "limitations":["already viewed development data and post-hoc analysis",
            "three planning and three evaluation futures do not establish population risks",
            "unknown individual states still privileged to the offline teacher",
            "unevaluated candidate outcomes remain missing",
            "posthoc best-five uses evaluation answers and is not deployable or unbiased",
            "bootstrap conditions on frozen models, no retraining or physical uncertainty",
            "leave-one-out labels are dependent and are not new independent experiments"]}
    json_write(output/"summary.json",summary)
    report(args,summary,family_rows,integrity)
    json_write(output/"analysis_manifest.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256":digest(output/"diagnostic_manifest.json"),
        "diagnostic_code_sha256":digest(__file__),
        "report_sha256":digest(args.report),"summary_sha256":digest(output/"summary.json"),
        "output_sha256":{p.name:digest(p) for p in output.glob("*.csv")}})
    print(json.dumps({"output":str(output),"report":str(args.report),"initial_minimax":minimax_initial,
                      "integrity":integrity},ensure_ascii=False))


def report(args,summary,families,integrity):
    names = {"teacher_minimax":"最坏世界教师","teacher_point":"单世界教师","teacher_oracle":"真实火源有限库oracle"}
    lines = ["# 反事实教师误选诊断开发报告","",
        "日期：2026年10月3日。本轮只读取PRAU36已查看开发数据，没有新模拟、训练或LLM调用。本报告是事后机制分析，不能替代新的独立验证。","",
        "## 1. 为什么先查教师",
        "上一轮学生未稳定超过601，而离线教师自己的收益也未稳定复现。先把教师选择错误拆开，避免把全部问题归为学生没有学好。","",
        "## 2. 数据与核查",
        f"36家庭、248可达状态，其中72初始人口起点；每状态17候选、每相容火源世界3个规划未来，原38执行组的3个检验未来。重建三个教师选择，核对{integrity['source_csv_outcome_records_checked']:,}条原CSV结果及{integrity['unique_execution_trace_steps_checked']:,}个执行步，{integrity['input_files_hashed_and_unchanged']}份输入哈希均未改变。",
        "初始起点以家庭内两人口、各三未来平均，随后家庭等权。后续起点由随机/反应式历史获得，不代表每步重新选择的连续在线教师。",
        "规划世界是真火源或另一火源的条件假设；它们使用同一人口和历史复制。判断现实选择时只看真实火源下的独立检验，最坏世界评分不能当真实安全概率。","",
        "## 3. 初始主结果",
        "| 教师 | 失能率 | 相对601差（百分点） | 探索性95%家庭区间 | 改善/变差/持平家庭 |",
        "|---|---:|---:|---:|---:|"]
    for value in summary["aggregates"]:
        if value["subset"] == "initial":
            lines.append(f"| {names[value['teacher']]} | {value['teacher_eval_incap']:.4%} | {value['delta_vs_601_pp']:+.5f} | [{value['exploratory_ci_low_pp']:+.5f}, {value['exploratory_ci_high_pp']:+.5f}] | {value['improved_families']}/{value['worsened_families']}/{value['tied_families']} |")
    lines += ["","上述区间条件于当前候选和固定模型，所有比较为探索性，未校正多重比较。真实火源oracle只知真火源，并仍用三个规划未来选择有限库；不是用检验答案选择的全局最优。",
        "","## 4. 按冲突分析最坏世界教师",
        "| 冲突 | 家庭 | 教师失能率 | 601失能率 | 检验差（百分点） | 真世界规划差（百分点） | 规划→检验相对偏移（百分点） |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for value in summary["aggregates"]:
        if value["teacher"] == "teacher_minimax" and value["subset"].startswith("conflict_initial:"):
            lines.append(f"| {value['subset'].split(':')[1]} | {value['families']} | {value['teacher_eval_incap']:.4%} | {value['baseline_eval_incap']:.4%} | {value['delta_vs_601_pp']:+.5f} | {value['true_planning_delta_vs_601_pp']:+.5f} | {value['relative_planning_to_eval_shift_pp']:+.5f} |")
    lines += ["","规划→检验相对偏移＝（教师−601）独立检验差−（教师−601）真实火源规划差；它同时含有限未来估计和挑选候选后的选择偏差。不能直接解释为模型真实风险的偏差，更不能把两个未来集合的随机差异当确定因果证明。",
        "","## 5. 具体变差家庭",
        "| 家庭 | 冲突 | 两人口选中计划 | 检验差（百分点） | 真世界规划差（百分点） |",
        "|---|---|---|---:|---:|"]
    harmful = sorted([r for r in families if r["teacher"] == "teacher_minimax" and r["delta_vs_601_pp"] > EPS],
                     key=lambda r:-r["delta_vs_601_pp"])
    for value in harmful:
        lines.append(f"| {value['source_id']} | {value['conflict_type']} | {value['selected_plans']} | {value['delta_vs_601_pp']:+.5f} | {value['true_planning_delta_vs_601_pp']:+.5f} |")
    initial = next(r for r in summary["aggregates"] if r["subset"] == "initial" and r["teacher"] == "teacher_minimax")
    lines += ["","初始72起点的选择次数："+json.dumps(initial["selected_plan_counts"],ensure_ascii=False)+"。",
        "逐起点分类："+json.dumps(initial["state_category_counts"],ensure_ascii=False)+"。这些计数不是独立场景数。",
        "‘规划已接受真实世界代价’表示真世界规划失能差>0且检验也变差；‘规划估计不差但检验变差’表示该规划差≤0而检验差>0。二者描述不同误选形态，不足以唯一识别根因。",
        "","## 6. 最坏规则与规划采样稳定性",
        "| 比较（初始） | 差（百分点） | 探索性95%区间 |",
        "|---|---:|---:|"]
    for value in summary["teacher_comparisons"]:
        if value["subset"] == "initial":
            lines.append(f"| 最坏世界−{names[value['control']]} | {value['delta_pp']:+.5f} | [{value['exploratory_ci_low_pp']:+.5f}, {value['exploratory_ci_high_pp']:+.5f}] |")
    lines += ["","| 三未来选择器 | 初始起点 | 至少一次删未来后变计划 | 占比 | 无检验缓存的删未来选择 |",
        "|---|---:|---:|---:|---:|"]
    for value in summary["leave_one_out"]:
        if value["subset"] == "initial":
            lines.append(f"| {names[value['teacher']]} | {value['states']} | {value['any_leave_one_future_choice_change_states']} | {value['fraction_states_changed']:.2%} | {value['missing_independent_eval_selections']} |")
    lines += ["","删除一个规划未来只检验小样本选择敏感性。它不是新模拟，三个删除结果相关。没有检验缓存的计划保留缺失，没有根据已有答案补造成绩。",
        "","## 7. 候选库还有多少改善空间",
        f"事后按独立检验答案在五固定PPO中挑最好的一个，初始家庭等权失能率为{initial['posthoc_best_five_eval_incap']:.4%}。与教师的平均差为{initial['teacher_delta_vs_posthoc_best_five_pp']:+.5f}个百分点。",
        "这个数使用了检验答案选计划，必然有乐观选择，不能当部署策略或无偏效应。它只说明有限库内存在部分可利用差异；未知候选未执行结果保持为空，未宣称完整17库的检验最优。五PPO只从已有固定模型中选择，未新增训练。",
        "","## 8. 能确认什么，不能确认什么",
        "可以直接确认：缓存选择是否一致、哪些家庭改善/变差、规划是否预见真世界失能代价、检验是否翻转、选择对删除一个未来是否敏感、以及实际执行是否与请求不同。",
        "不能直接确认：增加未来数一定解决问题、最坏世界必然无效、某隐藏输入是唯一原因、学生已解决教师问题、或LLM提供了独立收益。本轮没有LLM内容消融。",
        "","## 9. 下一步",
        "根据这里的规划—检验翻转与最坏情况代价证据，先在新开发场景冻结一个小型配对验证：固定17候选和五PPO，比较规划未来数与世界聚合方式。新开发验证应有独立检验未来，并限制总预算；先验证教师，暂不追加学生/PPO训练。具体因素与预算另存方案，不能根据PRAU成绩拟合例外规则后称最终验证。",
        "硬护栏原95%/10%/5%/95%门槛保持不变；正式第三轮及新最终银行仍未启动。",
        "","## 10. 文件与复现",
        f"[全部诊断汇总]({(args.output/'summary.json').resolve().as_posix()})；"
        f"[逐起点诊断]({(args.output/'state_teacher_diagnostics.csv').resolve().as_posix()})；"
        f"[逐家庭诊断]({(args.output/'initial_family_teacher_diagnostics.csv').resolve().as_posix()})；"
        f"[候选规划与检验]({(args.output/'candidate_planning_vs_evaluation.csv').resolve().as_posix()})；"
        f"[删除未来诊断]({(args.output/'leave_one_future_out.csv').resolve().as_posix()})；"
        f"[完整性核查]({(args.output/'integrity_audit.json').resolve().as_posix()})。",
        "",
        "复现只运行scripts/diagnose_plan_teacher.py；它只读原缓存并写新诊断输出，不产生疏散模拟。",
    ]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/baseline_relative_plan_distillation_development_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/plan_teacher_diagnosis_development_v1")
    parser.add_argument("--report",type=Path,default=ROOT/"reports/反事实教师误选诊断开发报告.md")
    main(parser.parse_args())
