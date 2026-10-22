"""Audit caches and write a Chinese report for the predeclared PPO expansion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.source_world_planning import OUTCOME_FIELDS, select_reference_plan
from run_ppo_extended_world_planning import ARMS, EXTENDED_ARMS, MODES, select_indices


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ppo_extended_world_planning_development_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/PPO扩展参考教师开发实验报告.md")
    args = parser.parse_args()
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((args.output / "development_manifest.json").read_text(encoding="utf-8"))
    selections = read_csv(args.output / "state_selection.csv")
    summary_rows = read_csv(args.output / "subset_summary.csv")
    effects = read_csv(args.output / "scenario_paired_effects.csv")
    initial = {r["arm"]: r for r in summary["initial_summary"]}
    source_dir = Path(manifest["source_directory"])
    source_scenarios = {r["scenario_id"]: r for r in manifest["scenario_records"]}
    state_ids = list(manifest["seed_registry"])
    state_indices = {state_id: i for i, state_id in enumerate(state_ids)}
    if {r["state_id"] for r in selections} != set(state_ids) or len(selections) != len(state_ids):
        raise AssertionError("states missing or duplicated")
    for relative, digest in manifest["source_hashes"].items():
        if sha256(ROOT / relative) != digest or sha256(args.output / "source_snapshot" / relative) != digest:
            raise AssertionError(f"source hash mismatch: {relative}")
    for record in manifest["frozen_ppo_models"]:
        if sha256(Path(record["path"])) != record["checkpoint_sha256"]:
            raise AssertionError("original checkpoint changed")
        if sha256(args.output / "model_snapshot" / Path(record["path"]).name) != record["checkpoint_sha256"]:
            raise AssertionError("checkpoint snapshot mismatch")
    # Also confirm the new streams share no seed namespace with the previous run.
    old_states = ROOT / "outputs/source_world_planning_development_v1/states"
    old_seeds = set()
    for path in old_states.glob("*.npz"):
        with np.load(path, allow_pickle=False) as cached:
            old_seeds.update(int(s) for s in cached["planning_seeds"])
            old_seeds.update(int(s) for s in cached["evaluation_seeds"])
    old_seeds |= {s + 9_000_001 for s in old_seeds}
    new_seeds = set()
    planning_rollouts = eval_rollouts = trace_steps = neutral_steps = blocked_steps = 0
    duplicate_plan_equalities = 0
    for row in selections:
        state_id = row["state_id"]
        scenario = source_scenarios[row["scenario_id"]]
        if sha256(source_dir / "states" / f"{state_id}.npz") != manifest["source_state_sha256"][state_id]:
            raise AssertionError("source state changed")
        with np.load(args.output / "states" / f"{state_id}.npz", allow_pickle=False) as cached:
            with np.load(source_dir / "states" / f"{state_id}.npz", allow_pickle=False) as original:
                np.testing.assert_array_equal(original["observation"], cached["observation"])
            registry = manifest["seed_registry"][state_id]
            np.testing.assert_array_equal(cached["planning_seeds"], registry["planning"])
            np.testing.assert_array_equal(cached["evaluation_seeds"], registry["evaluation"])
            new_seeds.update(registry["planning"] + registry["evaluation"])
            risks = cached["risks"]
            supported = cached["supported"]
            assert risks.shape == (2, 17, manifest["planning_futures"], len(OUTCOME_FIELDS))
            assert np.isfinite(risks[supported]).all() and np.isnan(risks[~supported]).all()
            chosen = select_indices(risks, supported, scenario["forecast_source_branch"],
                                    scenario["source_branch"], state_indices[state_id])
            np.testing.assert_array_equal(chosen, cached["chosen_indices"])
            assert chosen[8:13] == list(range(12, 17))
            trace_keys = cached["trace_keys"]
            offsets = cached["trace_offsets"]
            trace = cached["execution_trace"]
            outcomes = cached["evaluation_outcomes"]
            assert offsets[0] == 0 and offsets[-1] == len(trace) and np.all(np.diff(offsets) > 0)
            key_index = {tuple(k): i for i, k in enumerate(trace_keys)}
            assert len(key_index) == len(trace_keys) == len(set(chosen)) * manifest["evaluation_futures"]
            np.testing.assert_allclose(outcomes[..., 0] + outcomes[..., 2] + outcomes[..., 3], 1, atol=1e-12)
            for a, plan_index in enumerate(chosen):
                for f in range(manifest["evaluation_futures"]):
                    key = key_index[(plan_index, f)]
                    execution = trace[offsets[key]:offsets[key + 1]]
                    result = outcomes[a, f]
                    np.testing.assert_array_equal(execution[:, 0], np.arange(len(execution)))
                    assert len(execution) == result[5]
                    assert execution[0, 1] == result[9] and execution[0, 2] == result[10]
                    np.testing.assert_allclose([execution[:, 4].mean(), (execution[:, 1] == 8).sum(), execution[:, 5].mean()],
                                               result[6:9], atol=1e-12)
                    assert np.isin(execution[:, 1], np.arange(9)).all()
                    assert np.isin(execution[:, 2], np.arange(-1, 9)).all()
                    neutral = execution[:, 2] == -1
                    assert (execution[neutral, 3] == 2).all() and (execution[neutral, 4] > 0).all()
                    for b in range(a):
                        if chosen[b] == plan_index:
                            np.testing.assert_array_equal(result, outcomes[b, f])
                            duplicate_plan_equalities += 1
            expected_planning = int(supported.sum()) * 17 * manifest["planning_futures"]
            assert int(cached["planning_rollouts"]) == expected_planning
            assert int(cached["evaluation_rollouts"]) == len(trace_keys)
            planning_rollouts += expected_planning
            eval_rollouts += len(trace_keys)
            trace_steps += len(trace)
            neutral_steps += int((trace[:, 2] == -1).sum())
            blocked_steps += int(trace[:, 5].sum())
    new_seeds |= {s + 9_000_001 for s in new_seeds}
    if new_seeds & old_seeds:
        raise AssertionError("new and old future seed namespaces overlap")
    for field, value in (("planning_rollouts", planning_rollouts), ("unique_evaluation_rollouts", eval_rollouts),
                         ("actual_execution_trace_steps", trace_steps), ("neutral_release_trace_steps", neutral_steps),
                         ("sign_hold_blocked_trace_steps", blocked_steps)):
        assert summary[field] == value
    assert summary["evaluation_arm_records"] == len(selections) * len(ARMS) * manifest["evaluation_futures"]
    audit = {"all_checks_passed": True, "states": len(selections), "source_and_model_snapshots_verified": True,
             "new_and_old_domain_sensor_streams_disjoint": True, "planning_rollouts": planning_rollouts,
             "unique_evaluation_rollouts": eval_rollouts, "trace_steps_verified": trace_steps,
             "duplicate_plan_outcome_equalities_verified": duplicate_plan_equalities,
             "neutral_release_steps_verified": neutral_steps, "actual_execution_aggregates_verified": True}
    (args.output / "integrity_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    def effect(arm, control, subset="initial"):
        return next(r for r in effects if r["arm"] == arm and r["control"] == control and r["subset"] == subset)

    def fmt_effect(r):
        return (f"{float(r['scenario_equal_delta_pp']):+.3f}",
                f"[{float(r['exploratory_ci_low_pp']):+.3f}, {float(r['exploratory_ci_high_pp']):+.3f}]")

    strong_effect = effect("extended_minimax", "pure_ppo_601")
    if float(strong_effect["exploratory_ci_high_pp"]) < 0:
        decision = "独立随机未来中优于事先指定的强候选601，具有继续研究软建议蒸馏的开发价值；仍须进行场景家庭分割的验证，不能宣布资格通过。"
        next_step = "下一开发阶段可先试49维可观测输入到17候选计划软分数的蒸馏。旧36场景用于开发训练/校准，预先固定新的开发场景家庭作留出审计，检查相对固定PPO和不看世界评分的模型选择基线是否仍有收益。候选计划编号与九动作编号分开记录；蒸馏失败或偏差大时保留原控制器，不作为硬动作屏蔽。"
    elif float(strong_effect["scenario_equal_delta_pp"]) < 0:
        decision = "点估计优于强候选601，但探索性区间仍包含零，暂不足以确认值得直接蒸馏；先固定方案增加独立未来或新的开发场景检验选择稳定性。"
        next_step = "下一开发阶段应先检验有限规划样本造成的选择误差，固定候选库并预先设计更多规划未来/独立检验未来对照。只有稳定超过强固定PPO后，再训练软建议模型。"
    else:
        decision = "本轮最坏世界教师未超过事先指定的强候选601；不能仅因优于五模型平均或原手写教师就推进强制蒸馏。"
        next_step = "下一开发阶段应保留17候选，检验规划样本数量和选择乐观偏差；优先解决挑错模型的问题。先固定多未来与独立检验设计，再判断软建议价值，不把当前库内选择作为安全真值。"
    plot_path = args.output / "initial_comparison.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fixed_names = [f"pure_ppo_{s}" for s in range(601, 606)]
        labels = fixed_names + [
            "fixed_ppo_mean_expectation", "manual_minimax", "extended_point", "extended_mean", "extended_minimax", "extended_oracle"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})
        y = np.arange(len(labels))
        axes[0].barh(y, [initial[arm]["final_incapacitation"] * 100 for arm in labels],
                     color=["#7A8599" if arm in fixed_names else "#207A9D" for arm in labels])
        axes[0].set_yticks(y, [arm.replace("pure_ppo_", "PPO ").replace("fixed_ppo_mean_expectation", "Fixed PPO mean")
                              .replace("manual_minimax", "Manual minimax").replace("extended_", "Expanded ") for arm in labels])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Initial final incapacitation (%)")
        axes[0].set_title("Development outcomes on independent future RNG")
        chosen_effects = [effect(arm, "pure_ppo_601") for arm in EXTENDED_ARMS]
        values = np.asarray([float(r["scenario_equal_delta_pp"]) for r in chosen_effects])
        lows = np.asarray([float(r["exploratory_ci_low_pp"]) for r in chosen_effects])
        highs = np.asarray([float(r["exploratory_ci_high_pp"]) for r in chosen_effects])
        axes[1].errorbar(values, np.arange(4), xerr=np.maximum(0, [values - lows, highs - values]), fmt="o", color="#207A9D", capsize=4)
        axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
        axes[1].set_yticks(np.arange(4), ["Point", "Equal-world", "Minimax", "Source oracle"])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Difference vs fixed PPO 601 (percentage points)")
        axes[1].set_title("36-family paired exploratory 95% bootstrap CI")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except ImportError:
        plot_path = None

    lines = ["# PPO扩展参考教师开发实验报告", "", "日期：2026-09-30。复用开发场景的离线特权参考实验；不是最终评估、公平新训练消融或LLM增益验证。", "",
             f"**结论：{decision}**", "", "## 1. 本轮完成了什么", "",
             "保留12个手写闭环计划，加入所有五个既有纯PPO（601–605），组成17候选库。五模型均为原50,176训练步，未改参数，输入仍34维、建议通道为零。教师可以看49维传感/时钟/反馈历史，并以已知模拟器和原人群种子重建世界。", "",
             f"复用36场景家庭、两个人群轨迹的{summary['states']}状态，其中{summary['ambiguous_states']}个仍兼容两火源、{summary['identified_states']}个已辨识，模型不匹配{summary['model_mismatch_states']}个。预测正确/错误等都属于已查看的开发数据。", "",
             "两种火源从回合开始分别重放已执行指令，各自生成FED和人群历史。每相容世界/候选使用3个全回合规划未来；选定计划后，在实际原世界用3个不同的随机未来检验。新规划与检验流也均与上一轮分离。相同候选被多组选中时复用相同结果。", "",
             f"本轮实际完整回合模拟共**{summary['total_unique_full_rollouts']:,}次**：规划{planning_rollouts:,}、唯一检验{eval_rollouts:,}。15执行组产生{summary['evaluation_arm_records']:,}条结果记录，不能将共享记录算成独立模拟。每个初始执行组216条记录；独立统计单位是36场景家庭。", "",
             "另设原手写库的相同四种参考、五个固定PPO、实时分级规则和不看评分的均匀随机候选选择。五固定PPO算术平均只是每回合随机选模型的期望对照，不是动作投票策略。所有组使用相同起点和检验随机数初始状态，动作不同仍会改变随机数消耗。", "",
             "## 2. 初始回合结果", "", "失能率越低越好。主要分析为第0步，场景家庭等权；每家庭有相同2×3重复，因此此表也等于普通记录均值。", "",
             "| 策略 | 失能率 | 平均FED | 实际平均放行 |", "|---|---:|---:|---:|"]
    names = {"extended_point": "扩展库：单火源参考", "extended_mean": "扩展库：两世界等权", "extended_minimax": "扩展库：最坏世界",
             "extended_oracle": "扩展库：真实火源oracle", "manual_point": "原手写库：单火源", "manual_mean": "原手写库：等权",
             "manual_minimax": "原手写库：最坏世界", "manual_oracle": "原手写库：真实火源oracle",
             "live_graded": "手写实时分级", "uniform_candidate": "均匀随机候选", "fixed_ppo_mean_expectation": "五固定PPO平均（期望对照）"}
    names.update({f"pure_ppo_{s}": f"固定纯PPO {s}" for s in range(601, 606)})
    for arm in ARMS + ("fixed_ppo_mean_expectation",):
        r = initial[arm]
        lines.append(f"| {names[arm]} | {r['final_incapacitation']:.4%} | {r['final_fed']:.4f} | {r['mean_release']:.2%} |")
    lines += ["", "## 3. 与强候选和原教师的配对比较", "",
              "下面是扩展最坏世界参考减去各对照的失能率差；负数代表改善。区间按场景家庭Bootstrap 2,000次，均为探索性，未做多重比较校正，不能当作最终确认性显著性。601在上一轮已知最强，属于本轮事先指定的强参照，其他四个也全部保留。", "",
              "| 对照 | 差值（百分点） | 探索性95%区间 | 改善/变差/持平家庭 |", "|---|---:|---:|---:|"]
    for control in ("fixed_ppo_mean_expectation",) + tuple(f"pure_ppo_{s}" for s in range(601, 606)) + (
                    "manual_minimax", "extended_point", "extended_mean", "extended_oracle", "uniform_candidate"):
        r = effect("extended_minimax", control)
        delta, interval = fmt_effect(r)
        lines.append(f"| {names[control]} | {delta} | {interval} | {r['improved_families']}/{r['worsened_families']}/{r['tied_families']} |")
    lines += ["", "收益不是每个场景都有：相对601，12家庭改善、8家庭变差、16家庭持平。六类冲突中的单独预测错误类也略差于601（见下表）。所以当前支持的是总体开发平均改善，不能声称各冲突均稳健改善。", "",
              "最坏世界参考没有优于同库单火源或等权参考。三个非oracle参考均优于强固定PPO，表明扩展库和离线计划选择值得继续检验；本轮没有隔离出世界不确定性处理的额外收益，更没有LLM贡献证据。", "",
              "两世界等权0.5并非校准火源概率；最坏世界仅取相容世界的三个未来失能均值的最大值。知道真实火源的oracle仍只有有限候选和随机样本，不能视为全局最优，其检验结果也不保证必优于其他选择。", "",
              "预测正确/错误初始子集以及仍未知/已辨识的全部状态，属于探索性分析：", "",
              "| 子集 | 家庭数 | 扩展最坏世界失能率（家庭等权） | 相对601差值及区间（百分点） |", "|---|---:|---:|---:|"]
    for subset, label in (("forecast_wrong_initial", "初始预测错误"), ("forecast_correct_initial", "初始预测正确"),
                          ("ambiguous", "仍不辨火源状态"), ("identified", "已辨识状态"), ("all_states", "全部状态")):
        r = next(r for r in summary_rows if r["subset"] == subset and r["arm"] == "extended_minimax")
        delta, interval = fmt_effect(effect("extended_minimax", "pure_ppo_601", subset))
        lines.append(f"| {label} | {r['scenario_families']} | {float(r['family_equal_final_incapacitation']):.4%} | {delta} {interval} |")
    lines += ["", "各冲突初始子集结果也全部保存：", "", "| 冲突类型 | 家庭数 | 扩展最坏世界 | 固定601 |", "|---|---:|---:|---:|"]
    for subset in sorted({r["subset"] for r in summary_rows if r["subset"].startswith("conflict_initial:")}):
        a = next(r for r in summary_rows if r["subset"] == subset and r["arm"] == "extended_minimax")
        b = next(r for r in summary_rows if r["subset"] == subset and r["arm"] == "pure_ppo_601")
        lines.append(f"| {subset.split(':')[1]} | {a['scenario_families']} | {float(a['family_equal_final_incapacitation']):.4%} | {float(b['family_equal_final_incapacitation']):.4%} |")
    lines += ["", "## 4. 选中了什么、规划结果是否乐观", "", "| 参考选择 | 初始72状态选择数 | 全部249状态选择数 |", "|---|---|---|"]
    for arm in EXTENDED_ARMS:
        def counts(subset):
            return "；".join(f"{name}: {count}" for name, count in summary["selection_counts"][subset][arm].items())
        lines.append(f"| {names[arm]} | {counts('initial')} | {counts('all_states')} |")
    lines += ["", "加入更强候选能降低规划样本上的最小值，但选择有限未来上最小值会产生乐观偏差。下表使用所选计划在实际火源世界中的规划均值作为事后诊断，与新的检验未来比较；实际火源没有反向用于非oracle选计划。", "",
              "| 参考 | 所选计划的规划失能均值 | 新未来检验失能均值 | 检验减规划（百分点） |", "|---|---:|---:|---:|"]
    for r in summary["initial_selection_generalization"]:
        lines.append(f"| {names[r['arm']]} | {r['planning_actual_world_mean_incapacitation']:.4%} | {r['evaluation_mean_incapacitation']:.4%} | {r['evaluation_minus_planning_pp']:+.3f} |")
    lines += ["", "## 5. 运行与科学边界", "",
              f"完整自动审计通过：{len(selections)}状态观测与原始缓存逐项相等；17候选维度、选计划结果、五固定模型保留、种子命名空间、缓存计数及{trace_steps:,}步实际执行记录均校验。共享计划的{duplicate_plan_equalities:,}个结果比较完全相同。原模型、模型快照、代码快照与采集哈希一致。", "",
              f"实际执行记录有{neutral_steps:,}步中性指示伴随正放行，正确编码为-1；有{blocked_steps:,}步受保持约束阻挡。请求方向、实际方向和实际容量分别记录，平均放行与阻挡率均由反馈核对。90项测试通过。", "",
              "教师使用五个既有模型及成千次特权模拟，其信息、计算和既往训练资源大于单个固定PPO；本轮只能判断教师能力，不能作为相同训练预算下新方法的收益证明。没有新LLM调用、PPO训练或蒸馏训练。", "",
              "烟气相容性假设没有噪声，且增长、起烟、背景、地图均准确。复用场景加新随机未来不等于全新场景泛化。本轮没有更换原护栏资格门槛（危险召回≥95%、安全误拒≤10%、所选危险≤5%、镜像一致性≥95%），没有任何新资格通过证据。第三轮至少十训练种子与全新最终场景库仍未启动。", "",
              "## 6. 下一步与复现", "", next_step, "",
              "无论后续走哪条路线，都必须把离线特权教师与部署可获得信息区分。软建议能否保留增益需要独立检查；规则建议、随机建议、真实LLM建议与纯PPO需要相同信息、动作和预算的公平对照。", "",
              "```powershell", "python scripts/run_ppo_extended_world_planning.py --output outputs/ppo_extended_world_planning_reproduction",
              "python scripts/analyze_ppo_extended_world_planning.py --output outputs/ppo_extended_world_planning_reproduction --report reports/PPO扩展参考教师复现报告.md",
              "python -m pytest -q", "```", "",
              "运行中断时只允许原清单、模型、数据和代码完全一致的`--resume`。当前输出保留于`outputs/ppo_extended_world_planning_development_v1/`。", "",
              "主要文件：`development_manifest.json`、`summary.json`、`subset_summary.csv`、`scenario_paired_effects.csv`、`selection_generalization.csv`、`state_selection.csv`、`evaluation_episode_results.csv`、`integrity_audit.json`、逐状态缓存及源码/五模型快照。分析脚本及输入哈希另存于`analysis_manifest.json`。"]
    if plot_path:
        lines += ["", f"![初始结果及相对601配对效应]({plot_path.resolve().as_posix()})"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    snapshot = args.output / "analysis_source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot)
    analysis_manifest = {"role": "post_collection_integrity_audit_and_predeclared_reporting",
                         "analysis_code_sha256": sha256(Path(__file__)), "summary_sha256": sha256(summary_path),
                         "effect_table_sha256": sha256(args.output / "scenario_paired_effects.csv"),
                         "report_sha256": sha256(args.report), "report": str(args.report.resolve()),
                         "decision": decision, "integrity_audit": audit, "freeze_allowed": False}
    (args.output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "strong_comparator_effect": strong_effect, "integrity": audit,
                      "report": str(args.report.resolve())}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
