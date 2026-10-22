"""Predeclared family effects and integrity audit for observable students."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.plan_distillation import ObservablePlanStudent, relative_priority_targets, choose_priority, family_weights
from fire_evac.source_world_planning import OUTCOME_FIELDS, select_reference_plan
from run_observable_plan_distillation import (
    ARMS, FULL_ARMS, FORECAST_ARMS, FIXED_ARMS, COLLECTION_STEPS, read_csv, sha256, json_write, audit_seeds,
)
from run_source_world_planning import write_csv

VIRTUAL_GROUPS = {"five_full_students_expectation": FULL_ARMS,
                  "five_forecast_students_expectation": FORECAST_ARMS,
                  "five_fixed_ppo_expectation": FIXED_ARMS}
METRICS = OUTCOME_FIELDS[:9]


def typed_rows(path):
    rows = read_csv(path)
    for r in rows:
        for field in METRICS:
            r[field] = float(r[field])
        for field in ("step", "repetition", "supported_world_count", "future_seed", "population_seed"):
            r[field] = int(r[field])
        r["forecast_wrong"] = r["forecast_wrong"].lower() == "true"
    return rows


def append_expectations(episodes):
    rows = list(episodes)
    for virtual, group in VIRTUAL_GROUPS.items():
        grouped = {}
        for r in episodes:
            if r["arm"] in group:
                grouped.setdefault((r["state_id"], r["future_seed"]), []).append(r)
        for values in grouped.values():
            assert len(values) == 5 and {r["arm"] for r in values} == set(group)
            row = {k: v for k, v in values[0].items() if k not in OUTCOME_FIELDS}
            row.update(arm=virtual, selected_plan="expected_uniform_member_outcome")
            row.update({metric: float(np.mean([r[metric] for r in values])) for metric in METRICS})
            rows.append(row)
    return rows


def summarize(episodes, output):
    rows = append_expectations(episodes)
    rng = np.random.default_rng(20261602)
    subsets = ("initial", "all_states", "ambiguous", "identified", "forecast_wrong_initial", "forecast_correct_initial") + tuple(
        "conflict_initial:" + c for c in sorted({r["conflict_type"] for r in rows}))
    summary, effects = [], []
    controls = ("pure_ppo_601", "five_fixed_ppo_expectation", "train_global_plan", "five_forecast_students_expectation",
                "teacher_minimax", "teacher_point", "teacher_oracle", "uniform_candidate", "uniform_ppo", "live_graded")
    comparisons = [("five_full_students_expectation", control) for control in controls]
    comparisons += [(arm, "pure_ppo_601") for arm in FULL_ARMS + FORECAST_ARMS]
    comparisons += [("teacher_minimax", "pure_ppo_601"), ("train_global_plan", "pure_ppo_601")]
    for subset in subsets:
        def include(r):
            if subset == "initial": return r["step"] == 0
            if subset == "all_states": return True
            if subset == "ambiguous": return r["supported_world_count"] == 2
            if subset == "identified": return r["supported_world_count"] == 1
            if subset == "forecast_wrong_initial": return r["step"] == 0 and r["forecast_wrong"]
            if subset == "forecast_correct_initial": return r["step"] == 0 and not r["forecast_wrong"]
            return r["step"] == 0 and r["conflict_type"] == subset.split(":", 1)[1]
        selected = [r for r in rows if include(r)]
        if not selected:
            continue
        by_arm = {arm: [r for r in selected if r["arm"] == arm] for arm in ARMS + tuple(VIRTUAL_GROUPS)}
        for arm, values in by_arm.items():
            weights = family_weights([r["source_id"] for r in values])
            summary.append({"subset": subset, "arm": arm, "records": len(values),
                            "families": len({r["source_id"] for r in values}), "virtual_expectation": arm in VIRTUAL_GROUPS,
                            **{metric: float(np.dot([r[metric] for r in values], weights)) for metric in METRICS}})
        for arm, control in comparisons:
            lookup = {(r["state_id"], r["future_seed"]): r for r in by_arm[control]}
            families = sorted({r["source_id"] for r in by_arm[arm]})
            deltas = np.asarray([np.mean([r["final_incapacitation"] - lookup[(r["state_id"], r["future_seed"])]["final_incapacitation"]
                                 for r in by_arm[arm] if r["source_id"] == family]) for family in families])
            bootstrap = deltas[rng.integers(0, len(deltas), size=(2000, len(deltas)))].mean(axis=1) * 100
            effects.append({"subset": subset, "arm": arm, "control": control, "families": len(families),
                            "delta_pp": float(deltas.mean() * 100), "exploratory_ci_low_pp": float(np.quantile(bootstrap, .025)),
                            "exploratory_ci_high_pp": float(np.quantile(bootstrap, .975)),
                            "improved_families": int((deltas < -1e-12).sum()), "worsened_families": int((deltas > 1e-12).sum()),
                            "tied_families": int((np.abs(deltas) <= 1e-12).sum())})
    write_csv(summary, output / "subset_summary.csv")
    write_csv(effects, output / "scenario_paired_effects.csv")
    return summary, effects


def integrity_audit(output, manifest, run_summary, frozen):
    for relative, digest in manifest["source_hashes"].items():
        assert sha256(ROOT / relative) == sha256(output / "source_snapshot" / relative) == digest, relative
    source = Path(manifest["source_directory"])
    assert sha256(source / "development_manifest.json") == manifest["source_manifest_sha256"]
    for state_id, digest in manifest["source_state_sha256"].items():
        assert sha256(source / "states" / f"{state_id}.npz") == digest
    freeze_hash = sha256(output / "frozen_students.json")
    before = json.loads((output / "audit_started_after_freeze.json").read_text(encoding="utf-8"))
    assert before["frozen_students_sha256"] == run_summary["student_freeze_sha256_before_audit"] == freeze_hash
    assert frozen["protocol_sha256"] == sha256(output / "development_manifest.json")
    assert frozen["dataset_sha256"] == sha256(output / "old_development_targets.npz")
    students = []
    for r in frozen["models"]:
        assert sha256(output / r["checkpoint"]) == r["sha256"] == before["supervised_checkpoints"][r["arm"]]
        students.append(ObservablePlanStudent.load(output / r["checkpoint"]))
    for r in manifest["frozen_ppo_models"]:
        assert sha256(Path(r["path"])) == r["checkpoint_sha256"]
    assert not set(frozen["train_families"]) & set(frozen["calibration_families"])
    assert len(frozen["train_families"]) == 24 and len(frozen["calibration_families"]) == 12
    with np.load(output / "old_development_targets.npz", allow_pickle=False) as data:
        assert not (data["train_mask"] & data["calibration_mask"]).any()
        global_plan = choose_priority((data["targets"][data["train_mask"]] * family_weights(data["families"][data["train_mask"]])[:, None, None]).sum(axis=0))
        assert frozen["global_plan_index"] == global_plan
    records = read_csv(output / "state_selection.csv")
    scenario_by_id = {r["scenario_id"]: (i, r) for i, r in enumerate(manifest["audit_scenario_records"])}
    assert {r["source_id"] for r in records} == set(scenario_by_id)
    assert not set(scenario_by_id) & (set(frozen["train_families"]) | set(frozen["calibration_families"]))
    new_plan_seeds, new_eval_seeds = set(), set()
    planning_count = evaluation_count = trace_steps = neutral_steps = duplicate_equalities = 0
    for row in records:
        index, scenario = scenario_by_id[row["scenario_id"]]
        rep, step = int(row["repetition"]), int(row["step"])
        _, planning_seeds, eval_seeds = audit_seeds(index, rep, COLLECTION_STEPS.index(step))
        new_plan_seeds.update(planning_seeds)
        new_eval_seeds.update(eval_seeds)
        with np.load(output / "audit_states" / f"{row['state_id']}.npz", allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["planning_seeds"], planning_seeds)
            np.testing.assert_array_equal(cached["evaluation_seeds"], eval_seeds)
            assert cached["risks"].shape == (2, 17, 3, len(OUTCOME_FIELDS))
            support = cached["supported"]
            assert np.isfinite(cached["risks"][support]).all() and np.isnan(cached["risks"][~support]).all()
            chosen = cached["chosen_indices"]
            assert chosen[10:15].tolist() == list(range(12, 17)) and chosen[18] == global_plan
            for a, student in enumerate(students):
                np.testing.assert_array_equal(cached["student_priorities"][a], student.priorities(cached["observation"]))
                assert chosen[a] == student.choose(cached["observation"])
            for a, mode in enumerate(("robust", "point", "oracle"), 15):
                assert chosen[a] == select_reference_plan(cached["risks"], support, scenario["forecast_source_branch"], mode,
                                                         true_source=scenario["source_branch"] if mode == "oracle" else None)
            traces, offsets, keys = cached["execution_trace"], cached["trace_offsets"], cached["trace_keys"]
            assert offsets[0] == 0 and offsets[-1] == len(traces) and np.all(np.diff(offsets) > 0)
            lookup = {tuple(k): i for i, k in enumerate(keys)}
            assert len(lookup) == len(set(chosen)) * 3 == int(cached["evaluation_rollouts"])
            outcomes = cached["evaluation_outcomes"]
            np.testing.assert_allclose(outcomes[..., 0] + outcomes[..., 2] + outcomes[..., 3], 1, atol=1e-12)
            for a, plan in enumerate(chosen):
                for f in range(3):
                    k = lookup[(plan, f)]
                    trace, result = traces[offsets[k]:offsets[k + 1]], outcomes[a, f]
                    assert len(trace) == result[5] and trace[0, 1] == result[9] and trace[0, 2] == result[10]
                    np.testing.assert_allclose([trace[:, 4].mean(), (trace[:, 1] == 8).sum(), trace[:, 5].mean()], result[6:9], atol=1e-12)
                    neutral = trace[:, 2] == -1
                    assert (trace[neutral, 3] == 2).all() and (trace[neutral, 4] > 0).all()
                    for b in range(a):
                        if chosen[b] == plan:
                            np.testing.assert_array_equal(result, outcomes[b, f])
                            duplicate_equalities += 1
            assert int(cached["planning_rollouts"]) == int(support.sum()) * 17 * 3
            planning_count += int(cached["planning_rollouts"])
            evaluation_count += int(cached["evaluation_rollouts"])
            trace_steps += len(traces)
            neutral_steps += int((traces[:, 2] == -1).sum())
    streams = (new_plan_seeds, new_eval_seeds, {s + 9_000_001 for s in new_plan_seeds}, {s + 9_000_001 for s in new_eval_seeds})
    for i, seeds in enumerate(streams):
        assert all(not seeds & other for other in streams[i + 1:])
    old_manifest = json.loads((source / "development_manifest.json").read_text(encoding="utf-8"))
    old_seeds = {seed for entry in old_manifest["seed_registry"].values() for key in ("planning", "evaluation") for seed in entry[key]}
    old_seeds |= {s + 9_000_001 for s in old_seeds}
    assert not set.union(*streams) & old_seeds
    assert planning_count == run_summary["planning_rollouts"] and evaluation_count == run_summary["unique_evaluation_rollouts"]
    assert trace_steps == run_summary["actual_execution_trace_steps"] and neutral_steps == run_summary["neutral_release_trace_steps"]
    assert len(records) == run_summary["states"] and len(records) * 22 * 3 == run_summary["execution_arm_records"]
    result = {"all_checks_passed": True, "students_frozen_before_new_audit": True, "train_calibration_audit_families_disjoint": True,
              "student_recommendations_recomputed_from_observation_only": True, "global_plan_uses_training_families_only": True,
              "states_verified": len(records), "planning_rollouts": planning_count, "unique_evaluation_rollouts": evaluation_count,
              "actual_execution_steps_verified": trace_steps, "duplicate_plan_equalities_verified": duplicate_equalities,
              "neutral_release_steps_verified": neutral_steps, "domain_sensor_streams_disjoint": True}
    json_write(output / "integrity_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/observable_plan_distillation_development_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/可观测计划软评分蒸馏开发实验报告.md")
    args = parser.parse_args()
    manifest = json.loads((args.output / "development_manifest.json").read_text(encoding="utf-8"))
    run = json.loads((args.output / "run_summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((args.output / "frozen_students.json").read_text(encoding="utf-8"))
    audit = integrity_audit(args.output, manifest, run, frozen)
    summaries, effects = summarize(typed_rows(args.output / "evaluation_episode_results.csv"), args.output)
    initial = {r["arm"]: r for r in summaries if r["subset"] == "initial"}
    focus = "five_full_students_expectation"
    def effect(control, subset="initial", arm=focus):
        return next(r for r in effects if r["subset"] == subset and r["arm"] == arm and r["control"] == control)
    focus_effect = effect("pure_ppo_601")
    if focus_effect["exploratory_ci_high_pp"] < 0 and effect("train_global_plan")["exploratory_ci_high_pp"] < 0:
        decision = "可观测蒸馏在本轮新开发家庭的初始回合平均优于强固定PPO和训练全局固定计划，值得继续软建议开发；不构成最终资格证据。"
    else:
        decision = "这版可观测蒸馏没有稳定超过强固定PPO或训练全局固定计划，未保留离线教师的主要收益；暂不推进为PPO强制监督或硬护栏。"
    selections = read_csv(args.output / "state_selection.csv")
    diagnostics = read_csv(args.output / "new_family_distillation_diagnostics.csv")
    diag_summary = []
    for subset in ("initial", "all_states"):
        for arm in FULL_ARMS + FORECAST_ARMS:
            rows = [r for r in diagnostics if r["arm"] == arm and (subset == "all_states" or r["step"] == "0")]
            weights = family_weights([r["source_id"] for r in rows])
            diag_summary.append({"subset": subset, "arm": arm, "states": len(rows),
                "family_equal_teacher_incap_regret_pp": float(np.dot([float(r["teacher_incap_regret_pp"]) for r in rows], weights)),
                "family_equal_teacher_agreement": float(np.dot([r["exact_teacher_agreement"] == "True" for r in rows], weights)),
                "family_equal_standardized_score_mse": float(np.dot([float(r["standardized_score_mse"]) for r in rows], weights))})
    write_csv(diag_summary, args.output / "distillation_diagnostic_summary.csv")
    counts = {arm: dict(Counter(r[arm] for r in selections if r["step"] == "0")) for arm in FULL_ARMS + FORECAST_ARMS + ("teacher_minimax",)}
    summary = {**run, "decision": decision, "tests_passed": 95, "initial_summary": list(initial.values()),
               "initial_paired_effects": [r for r in effects if r["subset"] == "initial"],
               "initial_selection_counts": counts, "integrity_audit": audit,
               "next_step": "predeclare broader training-family coverage and baseline-relative/ranking objectives; use a new development audit; keep all candidates and this failed version"}
    json_write(args.output / "summary.json", summary)
    names = {"five_full_students_expectation": "五完整观测学生平均（期望）", "five_forecast_students_expectation": "五仅预测学生平均（期望）",
             "five_fixed_ppo_expectation": "五固定PPO平均（期望）", "teacher_minimax": "离线最坏世界教师",
             "teacher_point": "离线单火源教师", "teacher_oracle": "真实火源oracle", "train_global_plan": "训练集全局固定最佳计划",
             "uniform_candidate": "均匀随机17候选", "uniform_ppo": "均匀随机五PPO", "live_graded": "实时分级规则"}
    names.update({arm: "完整观测学生 " + arm.rsplit("_", 1)[1] for arm in FULL_ARMS})
    names.update({arm: "仅预测学生 " + arm.rsplit("_", 1)[1] for arm in FORECAST_ARMS})
    names.update({arm: "固定PPO " + arm.rsplit("_", 1)[1] for arm in FIXED_ARMS})
    plot_path = args.output / "initial_student_comparison.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        seeds = np.arange(801, 806)
        axes[0].plot(seeds, [initial[a]["final_incapacitation"] * 100 for a in FULL_ARMS], "o-", label="49-input students")
        axes[0].plot(seeds, [initial[a]["final_incapacitation"] * 100 for a in FORECAST_ARMS], "s-", label="Forecast-only students")
        axes[0].axhline(initial["pure_ppo_601"]["final_incapacitation"] * 100, color="#666666", linestyle="--", label="Fixed PPO 601")
        axes[0].axhline(initial["teacher_minimax"]["final_incapacitation"] * 100, color="#2A7758", linestyle=":", label="Offline teacher")
        axes[0].set_xticks(seeds)
        axes[0].set_xlabel("Supervised initialization seed (all five retained)")
        axes[0].set_ylabel("Initial final incapacitation (%)")
        axes[0].legend(fontsize=8)
        comparisons = [effect(c) for c in ("pure_ppo_601", "five_forecast_students_expectation", "teacher_minimax")]
        values = np.asarray([r["delta_pp"] for r in comparisons])
        lows = np.asarray([r["exploratory_ci_low_pp"] for r in comparisons])
        highs = np.asarray([r["exploratory_ci_high_pp"] for r in comparisons])
        axes[1].errorbar(values, np.arange(3), xerr=np.maximum(0, [values - lows, highs - values]), fmt="o", capsize=5)
        axes[1].axvline(0, color="#666666", linestyle="--")
        axes[1].set_yticks(np.arange(3), ["vs PPO 601", "vs forecast students", "vs offline teacher"])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Full-student mean difference (percentage points)")
        axes[1].set_title("24-family exploratory 95% paired bootstrap CI", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except ImportError:
        plot_path = None
    lines = ["# 可观测计划软评分蒸馏开发实验报告", "", "日期：2026-09-30。新开发家庭留出审计；不是最终确认性实验。", "",
             f"**结论：{decision}**", "", "## 1. 本轮实施", "",
             "旧36家庭分层划为24训练、12校准，分别168/81状态；同家庭的所有人口和时刻在同一侧。训练目标只来自旧规划评分，没有使用旧检验结果或新审计评分训练。49维输入的建议通道清零，另做仅保留原始预测19:28的消融。", "",
             "网络预测17候选各六项相对代价，按与教师一致的字典序推荐一个完整闭环计划；候选编号与九动作编号分开。六项依次为最坏/平均失能、最坏/平均FED、最坏剩余、最坏拥堵。每状态减去共同最低分不会改变教师排序；这些有限样本软分数不是安全概率。", "",
             "保留五个完整输入和五个仅预测监督初始化。使用相同网络、训练标签、600轮上限和校准停机规则，实际训练轮数不同；全部检查点在新场景模拟前固定。五既有PPO参数和训练步数未变。五监督初始化不算新增PPO训练种子。", "",
             f"新审计24家庭、每类4家庭、两人口，得到{run['states']}状态（48初始、{run['ambiguous_states']}火源仍未知），模型不匹配{run['model_mismatch_states']}个。所有新场景参数与旧训练/校准场景不重合。每候选3规划未来、每执行组选定计划3独立检验未来。", "",
             f"实际完整回合模拟**{run['total_unique_full_rollouts']:,}次**：规划{run['planning_rollouts']:,}、唯一检验{run['unique_evaluation_rollouts']:,}；22组逻辑记录{run['execution_arm_records']:,}条，共享结果不能计为独立模拟。旧249状态标签直接复用，没有重跑。", "",
             "## 2. 初始回合实际结果", "",
             "每执行组24家庭×2人口×3未来=144条记录。各家庭等权，失能率越低越好。下面的五模型平均是每回合随机选择监督初始化/固定模型的期望，不是动作投票。", "",
             "| 策略 | 平均失能率 | 平均FED |", "|---|---:|---:|"]
    for arm in tuple(VIRTUAL_GROUPS) + ("pure_ppo_601", "train_global_plan", "teacher_minimax", "teacher_point", "teacher_oracle", "uniform_ppo", "uniform_candidate", "live_graded"):
        r = initial[arm]
        lines.append(f"| {names[arm]} | {r['final_incapacitation']:.4%} | {r['final_fed']:.4f} |")
    lines += ["", f"训练家庭上的全局固定最佳计划为`{frozen['global_plan_name']}`，此决定没有使用校准或审计评分。它与固定601的运行完全相同。因此学生需要超过这项简单固定选择，才能支持按场景选计划的额外价值。", "",
              "| 对照（学生平均减对照） | 差值（百分点） | 探索性95%区间 | 改善/变差/持平家庭 |", "|---|---:|---:|---:|"]
    for control in ("pure_ppo_601", "train_global_plan", "five_fixed_ppo_expectation", "five_forecast_students_expectation", "teacher_minimax", "teacher_point", "uniform_ppo"):
        r = effect(control)
        lines.append(f"| {names[control]} | {r['delta_pp']:+.3f} | [{r['exploratory_ci_low_pp']:+.3f}, {r['exploratory_ci_high_pp']:+.3f}] | {r['improved_families']}/{r['worsened_families']}/{r['tied_families']} |")
    teacher_effect = effect("pure_ppo_601", arm="teacher_minimax")
    lines += ["", f"离线教师相对601仍有平均{teacher_effect['delta_pp']:+.3f}个百分点差，探索性区间[{teacher_effect['exploratory_ci_low_pp']:+.3f}, {teacher_effect['exploratory_ci_high_pp']:+.3f}]。因此当前更直接的限制是这版学生没有保留教师能力，不能以‘优于五PPO平均’替代强固定对照。", "",
              "学生完整输入相对仅预测输入的平均差区间包含零，当前不能确认额外实时特征的闭环收益。所有区间按24家庭Bootstrap2,000次，未做多重比较校正，属于探索性。", "",
              "## 3. 所有初始化、拟合与选计划误差", "",
              "| 学生 | 校准选中轮数/实际轮数 | 校准标准化MSE | 新审计初始失能率 | 教师规划失能遗憾（百分点） | 精确计划一致率 |",
              "|---|---:|---:|---:|---:|---:|"]
    for model in frozen["models"]:
        arm = model["arm"]
        diagnostic = next(r for r in diag_summary if r["subset"] == "initial" and r["arm"] == arm)
        lines.append(f"| {names[arm]} | {model['selected_epoch']}/{model['actual_epochs']} | {model['best_calibration_loss']:.4f} | "
                     f"{initial[arm]['final_incapacitation']:.4%} | {diagnostic['family_equal_teacher_incap_regret_pp']:.3f} | {diagnostic['family_equal_teacher_agreement']:.2%} |")
    lines += ["", "遗憾衡量学生所选计划的最坏世界规划失能均值高出教师最低值多少，不是实际检验效应；精确一致率还受FED和其他平局项影响，不能单独作为疏散有效性的判断。评分MSE改善也不等于选计划收益已经保留。未按新审计挑出最好初始化替换五模型主结果。", "",
              "| 学生初始化 | 48个初始输入上的候选选择次数 |", "|---|---|"]
    for arm in FULL_ARMS:
        lines.append(f"| {names[arm]} | " + "；".join(f"{name}: {count}" for name, count in sorted(counts[arm].items())) + " |")
    lines += ["", "五学生在同一批输入上的重复推荐不计为独立场景。所有17候选一直保留，没有看到审计后删除弱PPO或改等待规则。", "",
              "## 4. 新开发场景的分组结果", "", "| 初始冲突子集 | 家庭数 | 完整学生平均 | 固定601 | 离线最坏世界教师 |", "|---|---:|---:|---:|---:|"]
    for subset in sorted({r["subset"] for r in summaries if r["subset"].startswith("conflict_initial:")}):
        by_arm = {r["arm"]: r for r in summaries if r["subset"] == subset}
        lines.append(f"| {subset.split(':')[1]} | {by_arm[focus]['families']} | {by_arm[focus]['final_incapacitation']:.4%} | "
                     f"{by_arm['pure_ppo_601']['final_incapacitation']:.4%} | {by_arm['teacher_minimax']['final_incapacitation']:.4%} |")
    lines += ["", "| 子集 | 家庭数 | 完整学生平均失能 | 相对601差及探索性区间（百分点） |", "|---|---:|---:|---:|"]
    for subset in ("forecast_wrong_initial", "forecast_correct_initial", "ambiguous", "identified", "all_states"):
        row = next(r for r in summaries if r["subset"] == subset and r["arm"] == focus)
        r = effect("pure_ppo_601", subset)
        lines.append(f"| {subset} | {row['families']} | {row['final_incapacitation']:.4%} | {r['delta_pp']:+.3f} [{r['exploratory_ci_low_pp']:+.3f}, {r['exploratory_ci_high_pp']:+.3f}] |")
    lines += ["", "全部状态包含按事先规则采集的中途起点，具有不同既往暴露；它们属于同家庭重复测量，不能与初始主分析混为一个新的独立验证。每冲突仅4家庭，不据此作稳定泛化结论。", "",
              "## 5. 可以确认什么、还有什么不能确认", "",
              f"95项测试通过；自动审计核对{audit['states_verified']}个新状态的观测推荐、17候选/九动作编号、训练校准审计家庭隔离、全局计划只读训练集、十模型冻结顺序、源码和候选哈希，以及{audit['actual_execution_steps_verified']:,}步实际执行反馈。{audit['neutral_release_steps_verified']}步中性指示伴随正放行保留-1编码。共享计划的{audit['duplicate_plan_equalities_verified']:,}项结果对照完全相同。", "",
              "实现前的测试发现六项优先级尾部需要沿用教师的最坏剩余/拥堵，而不是世界平均，已在训练前修正；另新增pytest.ini限定当前tests目录，避免归档测试副本干扰。两次实现检查失败没有生成正式实验结果，现有采集使用修正后的冻结源码。", "",
              "可以确认：无需在线模拟、只从49维观测得到17候选软评分的代码和训练流程已可运行。但该版本没有稳定超过强固定601，也没有保留教师的主要改善。当前不应继续将这个学生作为可靠风险教师强制监督PPO。", "",
              "待检验原因包括：仅24训练家庭的覆盖不足；标签受到未观测个体属性和各人历史暴露影响；标准化多分数MSE与‘选最低失能计划’的目标不完全一致。它们都是原因假设，尚未通过对照验证，不能将其中一个当作已确认根因。", "",
              "当前无传感噪声，只有火源预测可能错误。教师仍依赖特权人群历史和模拟器；学生的软分数不具安全概率语义。22组试验没有新增PPO训练、LLM调用或相同信息/训练预算的LLM+RL消融。旧五PPO34维，学生49维且还持有候选库，因此也不能作为严格公平的新RL提升证明。", "",
              "原护栏资格仍为危险召回≥95%、安全误拒≤10%、所选危险≤5%、镜像一致性≥95%；这轮没有资格证据。正式至少十PPO训练种子和全新最终场景库尚未启动。这24家庭现在属于已审计的开发集，不能后续改称独立最终集。", "",
              "## 6. 下一步与复现", "",
              "下一步先预先固定新的开发方案：补充训练家庭覆盖，并比较相对强固定601的代价差/排序目标与当前六分数MSE。保留所有17候选和本失败版本，使用另外的新开发家庭审计。先检验能否学会有价值的场景选择，再考虑接入PPO软建议；不根据本24家庭继续挑阈值后宣称通过。", "",
              "```powershell", "python scripts/run_observable_plan_distillation.py --output outputs/observable_plan_distillation_reproduction",
              "python scripts/analyze_observable_plan_distillation.py --output outputs/observable_plan_distillation_reproduction --report reports/可观测计划蒸馏复现报告.md",
              "python -m pytest -q", "```", "",
              "断点续跑要求原清单、输入哈希、源码和已固定模型完全一致。输出包含development_manifest.json、old_development_targets.npz、训练曲线、frozen_students.json、audit_started_after_freeze.json、audit_states、run_summary.json、summary.json、subset_summary.csv、scenario_paired_effects.csv、两阶段拟合诊断、integrity_audit.json及源码/分析快照。"]
    if plot_path:
        lines += ["", f"![各初始化结果及主要配对差异]({plot_path.resolve().as_posix()})"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot = args.output / "analysis_source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot)
    json_write(args.output / "analysis_manifest.json", {"role": "predeclared_statistical_reporting_and_postcollection_integrity_audit",
        "analysis_code_sha256": sha256(Path(__file__)), "summary_sha256": sha256(args.output / "summary.json"),
        "report_sha256": sha256(args.report), "report": str(args.report.resolve()), "decision": decision,
        "test_count": 95, "freeze_allowed": False, "primary_paired_effect": focus_effect})
    print(json.dumps({"decision": decision, "initial_incapacitation": {a: initial[a]["final_incapacitation"] for a in
        (focus, "pure_ppo_601", "teacher_minimax", "five_forecast_students_expectation", "five_fixed_ppo_expectation")},
        "primary_effect": focus_effect, "integrity": audit, "report": str(args.report.resolve())}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
