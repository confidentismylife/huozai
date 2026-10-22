"""Report frozen synthetic confirmation, including adverse results and audit.

No new policy execution, fitting, threshold selection or new world generation.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/safestep_robust_gate_20261018"
AUDIT = ROOT / "outputs/safestep_robust_gate_audit_20261018"
SEMANTICS = ROOT / "outputs/safestep_gate_plan_semantics_20261018"
EXECUTION_CHECK = ROOT / "outputs/safestep_robust_execution_check_20261018"
SUMMARY_CHECK = ROOT / "outputs/safestep_robust_summary_check_20261018"
OUT = ROOT / "outputs/safestep_robust_gate_analysis_20261018"
REPORT = ROOT / "reports/SafeStep预测门控合成压力测试_完整结果与研究判断_2026-10-18.md"
PAPER = ROOT / "reports/SafeStep历史预测与信任门控_论文实验小节初稿_2026-10-18.md"
LABELS = {"snapshot_hold": "快照", "trend30": "30秒趋势", "half30": "半斜率趋势",
          "ensemble_mean": "三模型均值", "consensus_gate": "一致性门控",
          "switch_margin_gate": "切换门槛门控", "full_future_reference": "完整未来参照",
          "consensus_gate_low": "门控阈值2%（开发）", "consensus_gate_high": "门控阈值10%（开发）"}
FAMILIES = {"source": "原始危险率", "accelerating_local": "局部加速增长", "transient_local": "局部升降脉冲"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    assert not p.exists(), p
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(p, rows):
    assert rows and not p.exists()
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_rows(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    strings = {"world_id", "stage", "family", "sensor_id", "method"}
    for row in rows:
        for k, v in row.items():
            if k not in strings:
                row[k] = None if v == "" else float(v)
    return rows


def aggregate(rows):
    reached = int(sum(r["reached"] for r in rows))
    result = dict(n=int(sum(r["n"] for r in rows)), reached=reached,
        mean_FED=sum(r["mean_FED"] * r["reached"] for r in rows if r["mean_FED"] is not None) / reached if reached else None,
        maximum_FED=max(r["maximum_FED"] for r in rows if r["maximum_FED"] is not None),
        mean_travel_s=sum(r["mean_travel_s"] * r["reached"] for r in rows if r["mean_travel_s"] is not None) / reached if reached else None,
        immediate_reversals=int(sum(r["immediate_reversals"] for r in rows)))
    for comparator in ("snapshot", "ensemble", "half"):
        for relation in ("lower", "equal", "higher"):
            field = f"{relation}_vs_{comparator}"
            result[field] = int(sum(r[field] for r in rows))
        denom = sum(r[f"sum_comparator_FED_{comparator}"] for r in rows)
        diff = sum(r[f"sum_paired_difference_{comparator}"] for r in rows)
        result[f"relative_reduction_vs_{comparator}"] = -diff / denom if denom else None
        result[f"total_paired_difference_vs_{comparator}"] = diff
    result["opportunities"] = int(sum(r["opportunities"] for r in rows))
    result["accepted"] = int(sum(r["accepted"] for r in rows))
    for decision in ("accepted", "rejected"):
        for outcome in ("helpful", "harmful", "neutral", "failure"):
            field = f"{decision}_{outcome}"
            result[field] = int(sum(r[field] for r in rows))
    return result


def grouped(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys, group), **aggregate(part)) for group, part in sorted(groups.items())]


def verify_manifest(path):
    info = json.loads(path.read_text("utf-8"))
    for rel, digest in info["artifact_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    return info


def figure(summary, arms):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    primary = ["trend30", "half30", "ensemble_mean", "consensus_gate", "switch_margin_gate"]
    confirmation = [r for r in arms if r["stage"] == "synthetic_confirmation"]
    arm_ids = sorted({(r["family"], r["sensor_id"]) for r in confirmation})
    lookup = {(r["family"], r["sensor_id"], r["method"]): r for r in confirmation}
    data = np.asarray([[100 * lookup[f, s, m]["relative_reduction_vs_snapshot"] for f, s in arm_ids] for m in primary])
    bound = max(.1, float(np.max(np.abs(data))))
    with plt.rc_context({"font.family": "Microsoft YaHei", "font.size": 9, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
        mesh = ax.imshow(data, cmap="BrBG", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
        ax.set_xticks(range(len(arm_ids)), [FAMILIES[f] + "\n" + s for f, s in arm_ids], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(primary)), [LABELS[m] for m in primary])
        for i in range(len(primary)):
            for j in range(len(arm_ids)):
                ax.text(j, i, f"{data[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(data[i,j]) > .65 * bound else "black")
        fig.colorbar(mesh, ax=ax, label="相对快照的平均FED降低／%（负值为恶化）")
        ax.set_title("冻结合成确认种子：同一原始火灾上施加数学扰动；每格为60个种子的描述性汇总")
        fig.savefig(OUT / "confirmation_arm_effects.png", dpi=180)
        fig.savefig(OUT / "confirmation_arm_effects.svg")
        plt.close(fig)
        confirmed = {r["method"]: r for r in summary if r["stage"] == "synthetic_confirmation"}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        x = np.arange(len(primary))
        axes[0].bar(x - .18, [confirmed[m]["lower_vs_snapshot"] for m in primary], width=.36,
                    label="FED降低", color="#0072B2")
        axes[0].bar(x + .18, [confirmed[m]["higher_vs_snapshot"] for m in primary], width=.36,
                    label="FED增加", color="#D55E00", hatch="//")
        axes[0].set_xticks(x, [LABELS[m] for m in primary], rotation=20)
        axes[0].set_ylabel("起点—时刻—合成条件组合数")
        axes[0].legend()
        categories = ["accepted_helpful", "accepted_harmful", "rejected_helpful", "rejected_harmful"]
        names = ["接受有益", "接受有害", "误拒有益", "拦住有害"]
        xx = np.arange(4)
        for offset, method, color, hatch in [(-.18, "consensus_gate", "#009E73", ""),
                                             (.18, "switch_margin_gate", "#CC79A7", "//")]:
            axes[1].bar(xx + offset, [confirmed[method][c] for c in categories], width=.36,
                        color=color, hatch=hatch, label=LABELS[method])
        axes[1].set_xticks(xx, names)
        axes[1].set_ylabel("各门控轨迹中访问到的建议机会数")
        axes[1].legend()
        for ax in axes:
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("收益、伤害和误拒同时呈现；建议机会重叠，不是独立火灾样本")
        fig.savefig(OUT / "confirmation_benefit_harm_and_gate.png", dpi=180)
        fig.savefig(OUT / "confirmation_benefit_harm_and_gate.svg")
        plt.close(fig)


def make_report(run, audit, cfg, summary, arms, departures, mixtures, breakpoints):
    parallel_start = json.loads((RUN / "parallel_started.json").read_text("utf-8"))
    repeat = json.loads((EXECUTION_CHECK / "verification.json").read_text("utf-8"))
    summary_check = json.loads((SUMMARY_CHECK / "results.json").read_text("utf-8"))
    confirm = {r["method"]: r for r in summary if r["stage"] == "synthetic_confirmation"}
    gate, ensemble, half = confirm["consensus_gate"], confirm["ensemble_mean"], confirm["half30"]
    switch = confirm["switch_margin_gate"]
    method_order = cfg["primary_methods"]
    table = "\n".join(f"| {LABELS[m]} | {confirm[m]['reached']}/{confirm[m]['n']} | {confirm[m]['mean_FED']:.6f} | "
        f"{100*confirm[m]['relative_reduction_vs_snapshot']:+.3f}% | "
        f"{confirm[m]['lower_vs_snapshot']}/{confirm[m]['equal_vs_snapshot']}/{confirm[m]['higher_vs_snapshot']} | "
        f"{confirm[m]['mean_travel_s']:.2f} | {confirm[m]['immediate_reversals']} |" for m in method_order)
    gtable = "\n".join(f"| {LABELS[m]} | {confirm[m]['opportunities']} | {confirm[m]['accepted_helpful']} | "
        f"{confirm[m]['accepted_harmful']} | {confirm[m]['rejected_helpful']} | {confirm[m]['rejected_harmful']} |"
        for m in ("consensus_gate", "switch_margin_gate"))
    devtable = "\n".join(f"| {LABELS[r['method']]} | {100*r['relative_reduction_vs_snapshot']:+.3f}% | "
        f"{r['lower_vs_snapshot']}/{r['equal_vs_snapshot']}/{r['higher_vs_snapshot']} |"
        for r in summary if r["stage"] == "development")
    armtext = "\n".join(f"| {FAMILIES[r['family']]} | {r['sensor_id']} | {LABELS[r['method']]} | "
        f"{100*r['relative_reduction_vs_snapshot']:+.3f}% | {r['higher_vs_snapshot']} |"
        for r in arms if r["stage"] == "synthetic_confirmation" and r["method"] in ("half30", "ensemble_mean", "consensus_gate", "switch_margin_gate"))
    gate_claim = ("一致性门控的平均FED低于直接执行集成建议" if gate["relative_reduction_vs_ensemble"] > 0
                  else "一致性门控没有降低相对直接执行集成建议的平均FED")
    switch_break = next(r for r in breakpoints if r["method"] == "switch_margin_gate")
    pulse_arms = [r for r in arms if r["stage"] == "synthetic_confirmation" and r["family"] == "transient_local"]
    pulse_harm = {m: sum(r["relative_reduction_vs_snapshot"] < 0 for r in pulse_arms if r["method"] == m)
                  for m in ("consensus_gate", "switch_margin_gate")}
    mixture_table = "\n".join(f"| {r['overall_transient_weight']*100:.2f}% | {LABELS[r['method']]} | "
        f"{r['relative_reduction_vs_snapshot']*100:+.4f}% |" for r in mixtures)
    semantics = json.loads((SEMANTICS / "analysis_results.json").read_text("utf-8"))
    with (SEMANTICS / "all_distinct_plan_states.csv").open(encoding="utf-8-sig", newline="") as f:
        semantic_rows = list(csv.DictReader(f))
    reversed_labels = sum({r["fixed_plan_label"], r["one_action_label"]} == {"helpful", "harmful"}
                          for r in semantic_rows)
    content = f'''# SafeStep预测门控合成压力测试：完整结果与研究判断

日期：2026-10-18。状态：预定矩阵与独立核查完成。**本轮为同一个原始危险率场上的合成压力测试；新随机种子不等于新物理火灾。**

## 先看主要发现

**门控减轻了预测造成的损失，但没有建立稳定超越快照的收益。** 固定5%一致性门控相对快照的平均FED降低为{100*gate['relative_reduction_vs_snapshot']:+.3f}%（负值即更差）；切换门槛版本为{100*switch['relative_reduction_vs_snapshot']:+.3f}%。后者的微小汇总优势还依赖合成条件的权重。

{gate_claim}，相对差异按FED降低计算为{100*gate['relative_reduction_vs_ensemble']:+.3f}%。与简单半斜率趋势相比，一致性门控的相对降低为{100*gate['relative_reduction_vs_half']:+.3f}%。这表明门控有一定止损作用，不能将其直接解释为预测本身产生净收益。

合成确认组中，直接集成建议相对快照有{ensemble['higher_vs_snapshot']}个FED增加的组合，一致性门控为{gate['higher_vs_snapshot']}个，切换门槛门控为{switch['higher_vs_snapshot']}个。改善组合分别为{ensemble['lower_vs_snapshot']}、{gate['lower_vs_snapshot']}、{switch['lower_vs_snapshot']}。这是预定合成分布上的描述性结果，不是部署安全证明。

## 实际执行了什么

总共{run['worlds']}个条件：原始干净条件1个；开发4个种子×14个条件=56个；合成确认60个新种子×14个条件=840个。每个条件有4个出发时刻、61个起点。

正式完成矩阵包含{run['episodes']:,}条方法轨迹、{run['executed_moves']:,}个动作，{run['reached']:,}条到达网格出口。开发组另含两个预先指定的门槛敏感性方法，所以总数不是“所有条件×同一个方法数”。执行方式调整时另保留{len(parallel_start['preserved_partial_attempts'])}个未完成尝试，其局部结果不纳入这些数字。没有LLM/API调用、RL训练或FDS运行；48个封存最终人群场景未读取。

## 方法与门控的含义

所有在线方法获得相同地图、测量历史、采集与送达时间戳，使用同样的45步截止时间和合法动作。动态规划每次只执行第一步，随后依据新观测重算。

三模型包括30秒历史趋势、其半斜率版本、60秒历史趋势，均最多外推60秒再保持。集成方法对三个预测的区间剂量取均值，生成一条候选完整计划。门控将**同一候选计划**与**同一快照计划**放到三个预测下分别计分，要求最小相对收益达到5%才接受不同动作。这样避免“提名的计划”和“评价收益的计划”不一致。

切换门槛版本初始用5%；若候选出口与上次所选计划出口相同，用1%；若不同，用10%。它是工程启发式，会被检验是否也造成误拒或伤害。模型间分歧不等于校准过的置信区间。

## 合成确认组完整主结果

下表按固定矩阵内所有起点、出发时刻、种子和条件等权；相对降低使用“配对差额总和／对应基线FED总和”，不是每个百分比的平均。先检查到达数，再比较共同到达的FED。

| 方法 | 到达／总数 | 平均FED | 相对快照降低（负值为恶化） | 更低／相同／更高 | 平均行走秒数 | 立即折返次数 |
|---|---:|---:|---:|---:|---:|---:|
{table}

完整未来参照拥有实际未来，只衡量给定目标和45步时域内的空间。它不能作为在线公平竞争者。

![各条件下的平均FED变化]({(OUT/'confirmation_arm_effects.png').as_posix()})

## 门控究竟拦住了什么，又漏掉了什么

每次访问到候选动作与快照动作不同时，离线评分器计算“执行一步候选动作后恢复快照”的最终剂量，再与直接执行快照比较。真实未来只进入这个离线诊断，不参与门控。

| 方法 | 建议机会 | 接受有益 | 接受有害 | 误拒有益 | 拦住有害 |
|---|---:|---:|---:|---:|---:|
{gtable}

这些是各门控自身轨迹中访问到的机会，轨迹分布不完全一样；一个回合可能有多个重叠机会。不能把这些机会当作独立样本或把反事实差额相加当作总控制收益。完整回合差异以前表为准。

## 同一提案的三种执行语义

另按冻结顺序每10个确认条件抽取1个，共{semantics['sampled_worlds']}个条件，检查两个主门控访问到的全部不同建议状态，共{semantics['distinct_proposal_states']}个。选定状态按(条件、出发时刻、步数、位置)去重；这是机制审计，未改动方法或门槛。

对每个状态分别计分：完整执行固定候选计划对完整快照计划；只改一步后恢复快照；从该状态持续采用集成预测规划。前两种标签在{semantics['fixed_vs_one_action_label_disagreements']}个状态不一致，其中{reversed_labels}个出现“有益”和“有害”直接反转。

预测模型一致认为相对收益至少5%的{semantics['agreement_score_at_least_5pct_states']}个状态中，实际固定候选计划有{semantics['high_score_but_actual_fixed_plan_harmful']}个有害；“只改一步后恢复快照”有{semantics['high_score_but_one_action_harmful']}个有害。因此，多模型同意与执行后有益要分别验证。门控已经统一了提名与预测计分的计划，但固定计划收益仍不能直接充当滚动执行的收益预测。

该补充完成{semantics['extra_fixed_plan_ground_truth_scores']}项固定路线的反事实积分，没有新增闭环策略实验；不能把它们加到主轨迹数里。逐状态数据和分数区间表见[执行语义审计]({(SEMANTICS/'score_bins_and_execution_semantics.csv').as_posix()})。

![改善、伤害与误拒]({(OUT/'confirmation_benefit_harm_and_gate.png').as_posix()})

## 合成条件如何产生

两类危险率扰动都从原始场出发，在随机位置增加两个局部危险率变化：一类在120秒内加速升高后保持，另一类先升后降。空间范围由网格距离控制；振幅从事先固定的0.003～0.02 FER/秒抽取。它们没有经过燃烧或烟气输运方程验证。

每类分别配干净观测、10%噪声、30%噪声、15秒延迟以及两种噪声与延迟组合。另有原始危险率配两种噪声加延迟条件。噪声是按区域与采样时刻产生的均值为1的乘性对数正态噪声，同一对照条件使用相同测量。

延迟保留采集时间。决策时只能读取已经送达的行，不会把旧数据标成当前数据，也不会用未来行插值。所有参数和种子在公开地图上生成新合成压力条件前冻结；开发后没有选择新的模型或门槛。

这些观测仍是假定全图各区FER可获得的抽象输入。乘性噪声使原本为零的观测始终为零，且未模拟仪器检出限、漂移或相关误差。脉冲只在原始危险率上添加非负项，不能覆盖所有火灾发展方式；压力测试结论只对这些明确条件成立。

## 分条件检查

两种门控在六种“先升后降”的脉冲观测条件中，分别有{pulse_harm['consensus_gate']}/6和{pulse_harm['switch_margin_gate']}/6的平均FED高于快照；其中包含干净观测。因此，失效不能只归因于传感器噪声。三个模型共享同一批历史且主要采用相似的线性外推，赞同票数不能作为相互独立的可靠性证据。

| 危险率条件 | 观测条件 | 方法 | 相对快照降低 | 有害组合数 |
|---|---|---|---:|---:|
{armtext}

## 合成条件权重的事后敏感性

以下仅用既有缓存重新加权，不增加实验，也不把这些权重视作现实火灾发生概率。保持原始场条件的总权重为1/7，其余权重在“加速增长”和“先升后降”两类之间变化。主结果中脉冲类权重为6/14=42.86%。

切换门槛版本的汇总平均优势在脉冲类权重约{switch_break['overall_transient_break_even_weight']*100:.2f}%时消失。这说明约0.013%的平均降低容易被评价条件的构成改变，不能认定为稳定改进。

| 脉冲类总体权重 | 方法 | 相对快照降低 |
|---|---|---:|
{mixture_table}

## 开发组与门槛敏感性

开发结果包含2%和10%的固定门槛。主门槛5%及切换门槛没有据此改动；以下不与合成确认组混合。

| 方法 | 相对快照降低 | 更低／相同／更高 |
|---|---:|---:|
{devtable}

## 独立核查与执行记录

19项实现检查通过。独立审计重算全部{audit['episodes']:,}条已缓存轨迹、{audit['moves']:,}个动作和{audit['gate_decisions_checked']:,}次门控机会。它按原生采样断点另做梯形积分，最大整回合FED差额{audit['maximum_FED_replay_error']:.3g}；同时核对动作合法性、首次出口终止、接受门槛、误拒／有害标签和折返统计。另有{audit['independent_forward_reference_checks']}项正向最短路核对。

另从原始轨迹重新构建全部{summary_check['summary_rows_checked']:,}行逐条件汇总，平均剂量、配对改善／有害计数和门控混淆计数均吻合；最大数值差额{summary_check['maximum_numeric_difference']:.3g}。这个检查用于防止轨迹正确但报告汇总错误。

原计划单场景串行执行，运行中因时间预算改为两个CPU进程计算不同场景；完成结果继续复用，中断的局部尝试保留。该执行调整有独立记录，算法、阈值、生成器和种子哈希保持不变。并发时的求解耗时受资源竞争影响，本报告不据此作方法速度排名。门控还需要更多规划计算，不声称计算预算相同。

另选原始对照和冻结顺序的第一个确认条件，重复核查{repeat['extra_verification_episodes']:,}条轨迹。解压后的逐轨迹文本、汇总数据，以及所有原生输入和策略数组均完全一致。这些重复执行单列，不增加主效果评估样本数。

从原始计算开始至完成约{run['elapsed_wall_s']/60:.1f}分钟，代码实现与汇总另计。所有原先冻结文件均核对哈希；未改写旧结果。

## 研究判断

本轮能回答：在明确指定的数学扰动、噪声与延迟下，多模型一致性和切换门槛的收益—伤害权衡如何。它不能补齐作者的物理输入，也没有验证LLM或真人动态行为。

**本轮决定：保留快照为主要在线基线，把门控作为止损及失败分析结果；暂停扩大训练、继续搜索门槛或叠加LLM。** 三模型一致仍可能对应错误的固定计划，且已有门控误拒大量有益机会；仅修正“计划评价对象一致”或限制切换并不足以保证净收益。下一阶段若继续预测方向，应先提供可验证的新信息或经独立资料校准的预测不确定性，并检验相对快照的实质收益。现阶段完成复现／审计章节比继续同源调参更有证据基础。

## 复查入口

- [冻结方案]({(RUN/'frozen_plan.json').as_posix()})
- [执行方式调整]({(RUN/'execution_amendment.json').as_posix()})
- [主结果汇总]({(OUT/'stage_summary.csv').as_posix()})
- [分条件汇总]({(OUT/'arm_summary.csv').as_posix()})
- [分出发时刻汇总]({(OUT/'departure_summary.csv').as_posix()})
- [独立审计总记录]({(AUDIT/'audit_results.json').as_posix()})
- [全部逐条件原始汇总]({(RUN/'all_world_departure_summary.csv').as_posix()})
'''
    REPORT.write_text(content, encoding="utf-8")


def make_paper_section(run, audit, summary, arms, breakpoints):
    confirm = {r["method"]: r for r in summary if r["stage"] == "synthetic_confirmation"}
    gate, half = confirm["consensus_gate"], confirm["half30"]
    switch = confirm["switch_margin_gate"]
    methods = ["snapshot_hold", "trend30", "half30", "ensemble_mean", "consensus_gate",
               "switch_margin_gate", "full_future_reference"]
    table = "\n".join(f"| {LABELS[m]} | {confirm[m]['mean_FED']:.6f} | "
        f"{100*confirm[m]['relative_reduction_vs_snapshot']:+.3f}% | "
        f"{confirm[m]['lower_vs_snapshot']}/{confirm[m]['equal_vs_snapshot']}/{confirm[m]['higher_vs_snapshot']} | "
        f"{confirm[m]['mean_travel_s']:.2f} |" for m in methods)
    if gate["relative_reduction_vs_snapshot"] < -1e-10:
        interpretation = "一致性门控虽较直接集成与半斜率预测减轻损失，但平均FED仍高于快照基线，尚未兑现相对最简单在线规划的净收益。"
    elif gate["mean_FED"] >= half["mean_FED"] - 1e-10:
        interpretation = "一致性门控未在确认矩阵的平均FED上超过简单半斜率预测，因而当前证据不足以支持继续增加此门控的复杂度。"
    elif gate["higher_vs_snapshot"] > half["higher_vs_snapshot"]:
        interpretation = "一致性门控降低了相对半斜率预测的平均FED，但相对快照的有害组合数更多，说明平均改善伴随额外局部损失，未形成一致优势。"
    else:
        interpretation = "一致性门控在本确认矩阵中同时降低了相对半斜率预测的平均FED，并未增加相对快照的有害组合数；这一结果支持有限的合成可行性，后续重点应转向独立物理条件验证。"
    switch_break = next(r for r in breakpoints if r["method"] == "switch_margin_gate")
    methods_text = r'''# 历史危险率预测与建议门控的探索性评价

中文论文实验小节初稿，2026-10-18。用途：学位论文或研究报告中的探索性扩展，投稿目标未指定。本节承接已有公开危险率复现与五例偏差审计；所述新增条件由数学扰动生成，不作为原论文物理火灾的复现或独立真实火灾验证。

## 1 研究问题与方法

既有路线审计表明，出发时的危险率快照可能低估较长路线的后续累计暴露。然而，掌握真实未来信息的参照存在收益，并不意味着在线预测能够兑现这些收益。本节进一步考察：在相同历史观测下，简单趋势预测是否改善路线选择；当预测模型对候选方案持相近判断时，门控能否减少有害改道，并保留有益改道。

沿用酒店网格、两个出口和单代理运动模型，每步移动2 m，耗时约6.6667 s，最多45个合法四邻域动作。代理到达出口格即终止，无等待、拥堵和剂量阈值失能规则。实际FED按出发格驻留期间的分段线性FER积分累计。在线方法只接收已送达的原生5 s采样行；对于采集时刻为\(t_j\)、延迟为\(\delta\)的测量，只有\(t_j+\delta\le t\)时可在决策时刻\(t\)使用。

以最新已观测FER为锚点，分别采用30 s窗口的线性趋势、该趋势的半斜率版本以及60 s窗口的线性趋势。设\(t_o\)为最新采集时刻、\(b_{m,i}\)为历史窗口内区域\(i\)的最小二乘斜率，则预测表示为

\[
\widehat r_{m,i}(u)=\max\{0,r_i(t_o)+g_m b_{m,i}\min(u-t_o,60)\},\quad u\ge t.
\]

三模型的\((W_m,g_m)\)分别为\((30,1)\)、\((30,0.5)\)和\((60,1)\)。外推60 s后保持预测率，负预测率截为零。积分时显式处理零点与截断时刻。上述参数在新条件计分前固定，没有通过确认结果拟合。

快照方法将最新已送达危险率保持不变。集成方法对三个预测模型的区间剂量取均值，采用同一时间展开动态规划生成候选计划\(P_E\)。每次仅执行第一步，然后在剩余\(45-k\)步时域内更新计划。为避免提名与收益评价对象不一致，门控始终比较同一候选计划\(P_E\)和同一快照计划\(P_0\)，定义

\[
s=\min_m\frac{\widehat F_m(P_0)-\widehat F_m(P_E)}{\max(\widehat F_m(P_0),10^{-12})}.
\]

当候选首动作与基线不同且\(s\ge0.05\)时接受候选首动作，否则执行快照首动作。另设置切换门槛版本：初始门槛为5%，若候选出口与上一步所选计划出口相同则为1%，不同则为10%。这些门槛为事先固定的工程设定，模型间一致性没有被解释为校准的置信区间。

## 2 数据、对照与评价

原始输入为已固定版本的SafeStep酒店地图与FER工作表。合成压力条件在同一原始危险率场中增加两个随机局部变化，分别模拟加速增长或先升后降的脉冲。空间权重由网格距离确定，振幅为0.003～0.02 FER/s；这些场不经过烟气输运方程求解。观测条件包括无噪声、10%或30%变异系数的乘性对数正态噪声，以及0或15 s延迟，另对原始场设置两种噪声加延迟条件，共14个条件组合。

设置4个开发种子和60个另行冻结的合成确认种子。每个种子包含14个条件、4个出发时刻和61个起点。主方法与门槛在确认阶段保持不变；开发阶段额外检查2%与10%的门槛，未据此选择新的主方法。原始干净条件单独作为已知情形的对照。

对照包括快照规划、30 s趋势、半斜率趋势、三模型均值规划、两种门控，以及掌握完整未来的有限时域最小FED参照。各在线方法使用相同观测与动作约束，但门控需要更多规划计算；完整未来方法仅用于衡量空间。评价指标为到达率、FED、行走时间、立即折返以及相对快照的配对改善和伤害计数。配对FED相等容差为\(10^{-10}\)。本文不将起点、时刻或同源合成场当作独立物理火灾重复，不进行轨迹级独立样本显著性推断。

## 3 合成确认结果

'''
    results = f'''正式矩阵共完成{run['episodes']:,}条轨迹与{run['executed_moves']:,}个动作，其中合成确认组每种主方法包含{confirm['snapshot_hold']['n']:,}个起点—时刻—条件组合。所有方法在本图模型内到达出口。两项执行方式重复核查单独计数，不增加效果评估样本。

表1汇总合成确认组的结果。相对降低按配对FED差额之和除以对应基线FED总和计算，各组合等权，负值表示恶化。

| 方法 | 平均FED | 相对快照降低 | 更低／相同／更高 | 平均行走时间/s |
|---|---:|---:|---:|---:|
{table}

一致性门控相对直接集成执行的平均FED降低为{100*gate['relative_reduction_vs_ensemble']:+.3f}%，相对半斜率趋势为{100*gate['relative_reduction_vs_half']:+.3f}%。其相对快照的有害组合数为{gate['higher_vs_snapshot']}，切换门槛版本为{switch['higher_vs_snapshot']}，半斜率版本为{half['higher_vs_snapshot']}。平均剂量、局部伤害和改善机会因此需要同时解释，不能以其中单一指标概括方法优越性。

在一致性门控自身访问到的{gate['opportunities']}个不同首动作机会中，{gate['accepted_harmful']}个被接受的建议在“一步候选后恢复快照”的反事实下有害，{gate['rejected_helpful']}个被拒绝的建议在相同语义下有益。这些机会可能来自同一回合且相互重叠，不作为独立样本，也不将其差额相加替代完整回合收益。

![按条件呈现的相对FED变化]({(OUT/'confirmation_arm_effects.png').as_posix()})

## 4 讨论与研究范围

{interpretation} 切换门槛版本相对快照的汇总平均降低仅为{100*switch['relative_reduction_vs_snapshot']:.4f}%。两种门控在全部六个脉冲条件中均恶化，包括干净观测条件。将脉冲类总体权重由42.86%提高至约{100*switch_break['overall_transient_break_even_weight']:.2f}%时，切换门槛版本的平均优势也消失。这个权重分析属于事后缓存敏感性，未改变主结果权重或确认方法。

上述证据支持将本轮方法定位为预测止损与失败分析，保留快照作为主要在线基线，暂停扩大训练或叠加LLM。其判断限于事先指定的压力分布，不能外推未见建筑或真实人群中的控制效果。

三种预测共享同一批历史观测，仅改变时间窗口或斜率增益，因此三模型赞同不能当作三份独立证据。即使统一了候选计划与比较计划，评价完整固定计划、仅修改一个动作以及持续采用预测控制仍是不同的执行语义。本文另在预先抽取的84个确认条件中对这些语义进行缓存审计，完整记录保留于实验报告，未据此回调门槛。

本节观测是假定各区FER可获得的抽象输入，乘性噪声不模拟检出限、相关误差或传感器漂移。所有合成场共享一个原始火灾和一张地图，增加种子主要扩展数学扰动与测量噪声的覆盖，不能补充物理火灾独立性。恒定速度、单代理及出口吸收终止也限制现实解释；FED变化不等同于伤亡概率变化。

本节提供的是普通预测与建议门控的基线及失败证据。其结果不包含LLM预测、真人行为建模或RL重新训练。后续扩展需要先明确新的可验证信息和独立数据，避免通过增加模型复杂度补偿物理或行为资料缺失。

## 5 数据与计算核查

19项实现检查通过；另以原生时间断点上的独立梯形积分重算全部轨迹，最大整回合FED差额为{audit['maximum_FED_replay_error']:.3g}。动作合法性、首次出口终止、门控阈值、反事实标签和汇总数据均另行核查。两进程执行调整只改变场景调度，冻结的方法、参数和随机种子保持一致。

实测数值来源为[完整压力测试报告]({REPORT.as_posix()})与[汇总数据]({(OUT/'stage_summary.csv').as_posix()})。本节依据实际程序输出撰写；对机制的解释与后续独立验证需求属于讨论。原作者物理输入、跨建筑火灾和真实人群动态资料仍需另行补充。

资料来源：Fang等，Development of a tenability-based path planning model for building fire evacuations using reinforcement learning，Journal of Building Engineering 118 (2026), 115132，[DOI](https://doi.org/10.1016/j.jobe.2025.115132)；[固定版本公开仓库](https://github.com/hqfang3/SafeStep_v1/tree/573f613420e21e9a6731bc56fa36ea0b3f3a05f6)。原2026-10-17初稿保持原样，本节不追溯替换其方法口径或已冻结数值。
'''
    assert not PAPER.exists()
    PAPER.write_text(methods_text + results, encoding="utf-8")


def main():
    assert not OUT.exists() and not REPORT.exists() and not PAPER.exists()
    master = verify_manifest(RUN / "completion_manifest.json")
    verify_manifest(AUDIT / "completion_manifest.json")
    verify_manifest(SEMANTICS / "completion_manifest.json")
    verify_manifest(EXECUTION_CHECK / "completion_manifest.json")
    verify_manifest(SUMMARY_CHECK / "completion_manifest.json")
    audit_plan = json.loads((AUDIT / "audit_plan.json").read_text("utf-8"))
    assert sha(ROOT / "scripts/audit_safestep_robust_gate.py") == audit_plan["script_sha256"]
    semantics_plan = json.loads((SEMANTICS / "analysis_plan.json").read_text("utf-8"))
    assert sha(ROOT / "scripts/audit_safestep_gate_plan_semantics.py") == semantics_plan["script_sha256"]
    run = json.loads((RUN / "run_results.json").read_text("utf-8"))
    audit = json.loads((AUDIT / "audit_results.json").read_text("utf-8"))
    frozen = json.loads((RUN / "frozen_plan.json").read_text("utf-8"))
    cfg = frozen["configuration"]
    amendment = json.loads((RUN / "execution_amendment.json").read_text("utf-8"))
    assert sha(ROOT / "scripts/run_safestep_robust_gate_parallel.py") == amendment["source_script_sha256"]
    assert run["episodes"] == audit["episodes"] == cfg["execution"]["expected_episodes"]
    rows = load_rows(RUN / "all_world_departure_summary.csv")
    OUT.mkdir()
    save(OUT / "analysis_plan.json", dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        stage="FIXED_DESCRIPTIVE_AGGREGATION_AFTER_COMPLETED_FROZEN_MATRIX",
        primary_partition="synthetic_confirmation", pooling="equal world/departure/start matrix, ratio of dose sums",
        hypothesis_changes=False, new_policy_rollouts=0, statistics="no independent-fire significance tests"))
    summary = grouped(rows, ("stage", "method"))
    arms = grouped(rows, ("stage", "family", "sensor_id", "method"))
    departures = grouped(rows, ("stage", "departure_fire_s", "method"))
    worlds = grouped(rows, ("stage", "world_id", "method"))
    families = grouped(rows, ("stage", "family", "method"))
    family_index = {(r["family"], r["method"]): r for r in families if r["stage"] == "synthetic_confirmation"}
    base_means = {family: family_index[family, "snapshot_hold"]["mean_FED"]
                  for family in ("source", "accelerating_local", "transient_local")}
    mixtures, breakpoints = [], []
    source_weight = 1 / 7
    for method in ("consensus_gate", "switch_margin_gate"):
        delta = {family: family_index[family, method]["total_paired_difference_vs_snapshot"]
                  / family_index[family, method]["n"] for family in base_means}
        for p in (0., .25, .5, .75, 1.):
            weights = {"source": source_weight, "accelerating_local": (1-source_weight)*(1-p),
                       "transient_local": (1-source_weight)*p}
            absolute = sum(weights[family] * delta[family] for family in weights)
            baseline = sum(weights[family] * base_means[family] for family in weights)
            relative = -absolute / baseline
            mixtures.append(dict(method=method, source_weight=source_weight,
                conditional_transient_weight=p, overall_transient_weight=(1-source_weight)*p,
                mean_absolute_difference_FED=absolute, relative_reduction_vs_snapshot=relative,
                status="POST_HOC_CACHED_WEIGHTING_NOT_NEW_CONFIRMATION"))
            if p == .5:
                primary = next(r for r in summary if r["stage"] == "synthetic_confirmation" and r["method"] == method)
                assert abs(relative - primary["relative_reduction_vs_snapshot"]) < 1e-12
        denominator = (1-source_weight) * (delta["transient_local"] - delta["accelerating_local"])
        p_star = -(source_weight*delta["source"] + (1-source_weight)*delta["accelerating_local"]) / denominator
        breakpoints.append(dict(method=method, conditional_transient_break_even_weight=p_star,
            overall_transient_break_even_weight=(1-source_weight)*p_star, fixed_source_weight=source_weight,
            interpretation="Exploratory reweighting of cached family means; not a real fire prevalence estimate"))
    write_csv(OUT / "stage_summary.csv", summary)
    write_csv(OUT / "arm_summary.csv", arms)
    write_csv(OUT / "departure_summary.csv", departures)
    write_csv(OUT / "world_summary.csv", worlds)
    write_csv(OUT / "family_summary.csv", families)
    write_csv(OUT / "posthoc_mixture_weight_sensitivity.csv", mixtures)
    save(OUT / "mixture_break_even_weights.json", breakpoints)
    save(OUT / "analysis_results.json", dict(run=run, audit=audit,
        stage_summary=summary, primary_stage="synthetic_confirmation",
        execution_semantics=json.loads((SEMANTICS / "analysis_results.json").read_text("utf-8")),
        posthoc_mixture_breakpoints=breakpoints))
    figure(summary, arms)
    make_report(run, audit, cfg, summary, arms, departures, mixtures, breakpoints)
    make_paper_section(run, audit, summary, arms, breakpoints)
    links = re.findall(r"\]\((D:/[^)]+)\)", REPORT.read_text("utf-8"))
    assert all(Path(p).is_file() for p in links)
    for rel, digest in frozen["input_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in frozen["protected_previous_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    files = [p for p in OUT.iterdir() if p.is_file()] + [REPORT, PAPER, Path(__file__)]
    save(OUT / "completion_manifest.json", dict(status="COMPLETE", links_verified=len(links),
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    # Small review bundle: complete trajectories remain in the project, explicitly excluded here.
    archive = ROOT / "artifacts/SafeStep预测门控压力测试_汇总材料_2026-10-18.zip"
    assert not archive.exists()
    bundle_files = [REPORT, PAPER, RUN / "frozen_plan.json", RUN / "execution_amendment.json",
                    RUN / "run_results.json", AUDIT / "audit_results.json",
                    SEMANTICS / "analysis_results.json", SEMANTICS / "score_bins_and_execution_semantics.csv",
                    EXECUTION_CHECK / "verification.json", RUN / "all_world_departure_summary.csv",
                    SUMMARY_CHECK / "results.json"]
    bundle_files += [p for p in OUT.iterdir() if p.is_file()]
    readme = "汇总审阅包：先读reports下的完整报告。包含报告、汇总数据表、图和核查摘要。完整逐轨迹数据在原项目outputs/safestep_robust_gate_20261018/worlds中；不在本包内。Markdown转换为包内相对链接，转换后的字节哈希另存BUNDLE_MANIFEST.json；JSON保留原项目路径和来源哈希。仅一个原始火灾上的数学压力测试，不是独立物理验证，也不是独立可运行仿真软件。"
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in bundle_files}
    for path, content in list(payload.items()):
        if path.endswith(".md"):
            def replace(match):
                target = Path(match.group(1)).relative_to(ROOT).as_posix()
                assert target in payload, target
                return "](" + posixpath.relpath(target, posixpath.dirname(path)) + ")"
            payload[path] = re.sub(r"\]\((D:/[^)]+)\)", replace, content.decode("utf-8")).encode("utf-8")
    payload["README.txt"] = readme.encode("utf-8")
    bundle_manifest = dict(status="RESULTS_REVIEW_BUNDLE_NOT_STANDALONE_SIMULATOR",
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in bundle_files},
        archive_payload_sha256={path: hashlib.sha256(content).hexdigest() for path, content in payload.items()},
        transformation="Markdown absolute project links converted to relative included targets; original files unchanged")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for path, content in payload.items():
            z.writestr(path, content)
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(bundle_manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path, digest in bundle_manifest["archive_payload_sha256"].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == digest
    print(json.dumps(dict(status="COMPLETE", report=str(REPORT), archive=str(archive),
        confirmation=[r for r in summary if r["stage"] == "synthetic_confirmation"]), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
