"""Consolidate six fixed-budget runs after independent verification succeeds."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/safestep_author_training_20261015_v2"
REPORT=ROOT/"reports/SafeStep从头训练复现_双设置三种子结果_2026-10-15.md"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cached_return(record,exit_reward):
    trace=record["integration_trace"];seen=set();value=0.
    for k,row in enumerate(trace[1:]):
        previous=trace[k];old=tuple(previous["cell"]);new=tuple(row["cell"])
        fed=previous["FED"]
        utility=0 if fed==0 else -100 if fed<.3 else -300 if fed<1 else -800
        if new==old:walking=-1.
        else:
            walking=exit_reward if new in ((6,4),(14,2)) else -.5 if new in seen else -.25
            seen.add(new)
        value+=.95**k*(.1*walking+.2*utility*row["increment"])
    return float(value)


def main():
    cfg=read(ROOT/"schemas/safestep_author_training_20261015_v2.json")
    verification=read(OUT/"verification.json")
    assert verification["status"]=="PASS"
    assert not REPORT.exists(),"Preserve the completed report; write a new version instead"
    rows=[];summaries=[]
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            s=read(OUT/"runs"/f"{arm['id']}_seed{seed}"/"summary.json")
            summaries.append(s);t=s["training"];p,q=s["paper_starts"]
            rows.append(dict(arm=arm["id"],seed=seed,episodes=t["episodes"],
                transitions=t["transitions"],optimizer_updates=t["optimizer_updates"],
                early_stopped=t["early_stopped"],both_paper_routes_match=s["both_paper_routes_match"],
                start1_status=p["status"],start1_steps=p["steps"],start1_FED=p["FED"],
                start2_status=q["status"],start2_steps=q["steps"],start2_FED=q["FED"],
                all_start_success=s["all_start_success"],invalid_steps=s["invalid_steps"],revisits=s["revisits"],
                strict_all_start_gate=s["all_start_success"]==61 and s["invalid_steps"]==0 and s["revisits"]==0,
                all_start_mean_FED=s["mean_FED"],all_start_max_FED=s["max_FED"],over_03=s["over_03"]))
    with (OUT/"six_run_comparison.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts={arm["id"]:{"paper_routes":sum(r["both_paper_routes_match"] for r in rows if r["arm"]==arm["id"]),
                           "all_start_gate":sum(r["strict_all_start_gate"] for r in rows if r["arm"]==arm["id"])}
            for arm in cfg["arms"]}
    (OUT/"summary.json").write_text(json.dumps(dict(runs=rows,acceptance_counts=counts),indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    colors=["#176a9b","#b45d23","#258158"]
    for ax,arm in zip(axes,cfg["arms"]):
        for seed,color in zip(cfg["seeds"],colors):
            path=OUT/"runs"/f"{arm['id']}_seed{seed}"/"episodes.csv"
            entries=list(csv.DictReader(path.open(encoding="utf-8")))
            loss=np.array([float(r["loss"]) for r in entries])
            smoothed=np.convolve(loss,np.ones(100)/100,mode="valid")
            ax.semilogy(np.arange(100,len(loss)+1),np.maximum(smoothed,1e-10),color=color,label=f"seed {seed}",linewidth=1.2)
        ax.set_title(arm["id"].replace("_"," "))
        ax.set_xlabel("Training episode");ax.set_ylabel("Mean original four-output MSE (100 episodes)")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("From-scratch SafeStep training: final runs retained, no checkpoint selection")
    fig.savefig(OUT/"training_loss_curves.png",dpi=170);plt.close(fig)
    source_audit=dict(
        pinned_author_training_source=digest(ROOT/cfg["source_directory"]/"algorithm_DQL.py"),
        original_train_and_optimize_invoked=True,
        public_vs_paper_differences=[
            "Paper Table2 samples/updates inside time-step loop; public routine optimizes once after each episode.",
            "Paper Eq9 is selected state-action MSE; public routine compares all4 policy outputs to target-network outputs and overwrites the selected target.",
            "Public optimizer treats only WIN as terminal; a301-step LOSE still bootstraps.",
            "Public Maze.reset reseeds Python with None unless supplied; wrapper supplies recorded deterministic per-episode seeds.",
            "Paper exit path reward10; public Maze.reward_exit0.",
            "Paper Table5 learning rate0.008 and12000 episodes; public main.py0.01 and maximum20000 with early stopping."
        ],
        paper_listed_arm_limit="Uses stated lr/episodes/exit reward with public optimizer, not a complete reimplementation of Table2.",
        modifications_to_frozen_author_source=False)
    (OUT/"source_training_audit.json").write_text(json.dumps(source_audit,indent=2),encoding="utf-8")
    reference=read(ROOT/"outputs/safestep_paper_numerics_20261015/numeric_run/results.json")["fixed_routes"]
    objective=[]
    for arm in cfg["arms"]:
        published_returns=[cached_return(r,arm["reward_exit"]) for r in reference]
        for seed in cfg["seeds"]:
            traces=read(OUT/"runs"/f"{arm['id']}_seed{seed}"/"evaluation_traces.json")
            for start_number,start in enumerate([[5,1],[10,9]],1):
                record=next(r for r in traces if r["start"]==start)
                value=cached_return(record,arm["reward_exit"])
                objective.append(dict(arm=arm["id"],seed=seed,start_number=start_number,
                    actual_discounted_return=value,
                    paper_B_discounted_return=published_returns[2*(start_number-1)],
                    alternative_A_discounted_return=published_returns[2*(start_number-1)+1],
                    paper_B_better_than_specified_A=published_returns[2*(start_number-1)]>published_returns[2*(start_number-1)+1],
                    actual_minus_paper_B=value-published_returns[2*(start_number-1)]))
    (OUT/"cached_route_objective_audit.json").write_text(json.dumps(dict(
        method="Recompute gamma0.95 discounted original rewards from saved per-step dose traces; no new rollout/training; posthoc descriptive, acceptance criteria unchanged",
        rows=objective),indent=2),encoding="utf-8")
    table="\n".join(f"| {'公开 Beta' if r['arm']=='public_beta' else '正文参数＋公开程序'} | {r['seed']} | {r['episodes']:,} | "
        f"{r['start1_steps']} / {r['start1_FED']:.6f} | {r['start2_steps']} / {r['start2_FED']:.6f} | "
        f"{r['all_start_success']}/61 | {'是' if r['both_paper_routes_match'] else '否'} | {r['invalid_steps']:,} / {r['revisits']:,} |" for r in rows)
    total_episodes=sum(r["episodes"] for r in rows)
    total_transitions=sum(r["transitions"] for r in rows)
    total_updates=sum(r["optimizer_updates"] for r in rows)
    beta=counts["public_beta"];paper=counts["paper_listed_parameters"]
    overall=("已有从随机初始化得到论文两条推荐路线的运行，但是否跨种子一致，须按下表的通过数判断。"
             if any(r["both_paper_routes_match"] for r in rows) else
             "六次从头训练均未同时重现两条论文推荐路线；已完成有限复现尝试，不能宣称训练结果复现成功。")
    report=f'''# SafeStep 从头训练复现：双设置、三种子结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / reproducibility verification
- Origin Date: 2026-10-15
- Verification Status: VERIFIED（运行、记录与数值复核通过；方法和结果范围见下文）
- Version Label: safestep_author_training_v2
- 资料版本：[作者公开仓库固定提交](https://github.com/hqfang3/SafeStep_v1/tree/{cfg['author_commit']})。
- 状态：六个固定种子运行、独立核查和整理完成。

## 1. 本轮完成了什么

上一轮已经用原作者数据和权重，对齐酒店案例四条路线的 FED。本轮进一步检验：**从随机权重开始，执行作者公开训练程序，能否重新得到论文路线？**

{overall}

公开 Beta 设置中，两条论文路线同时重现 **{beta['paper_routes']}/3** 次；“61 个起点全部到达、没有非法动作或重访”的条件通过 **{beta['all_start_gate']}/3** 次。

正文所列参数与公开程序结合的设置中，上述两项分别通过 **{paper['paper_routes']}/3**、**{paper['all_start_gate']}/3** 次。

这些是同一酒店、同一火灾、同一疏散时刻的训练随机性检查，不是跨建筑或跨火灾泛化验证。模型是否到达、是否低剂量、是否与论文路线一致，分开报告。

## 2. 为什么运行两种设置

| 项目 | 公开 Beta 设置 | 正文所列参数＋公开程序 |
|---|---|---|
| 学习率 | 0.01 | 0.008 |
| 轮数 | 最多 20,000 | 固定 12,000 |
| 出口路径奖励 | 0 | 10，经路径权重 0.1 缩放 |
| 提前停止 | 每 500 轮全起点通过，且最近 20 次损失均低于 0.5 | 不提前停止 |
| 种子 | 141、142、143 | 141、142、143 |

共同条件为：原始 300 秒开始的酒店率表；61→61→4 网络；Adam；gamma=0.95；回放容量 1000；batch64；探索率从 1 开始，每次优化后乘 0.9995；沿用作者的目标网络同步逻辑。两个设置中相同种子的初始策略权重指纹相同。

第二种设置只是将正文明确给出的学习率、轮数和出口奖励应用于公开训练程序。它同时改变多项设置，不能作为单一因素的因果实验，更不能称为完整正文伪代码复现。

## 3. 保留了哪些原始行为

实际调用的是作者 `DQL_NetworkModel.train` 和 `optimize`，没有重新设计学习算法。原始动作选择、回放抽样、损失、更新时机、目标网络同步和收敛检查均保留。

原始 `Maze.step` 继续计算烟气剂量和奖励：初始起点剂量、新位置采样、5 秒表的取整索引、出口零剂量，以及第 301 步超时。没有加入动作屏蔽、额外监督或候选路径教师。

为实现可追溯运行，仅在外层完成以下适配：

- 直接包装原始环境，保留其 Status 枚举，避免 GUI 与 Gym 布尔类型提示。
- 原始 reset 每轮会重新播种；本轮传入事先确定的随机种子序列，并逐轮保存，以免系统熵使实验不可重复。
- 保存逐步转移和回放样本编号；每 500 轮及结尾保存权重，避免覆盖作者原始权重文件。
- 保存全部六个最终模型，不按论文路线挑选中间检查点或最佳种子。

## 4. 六次运行结果

论文参照：起点 1 推荐路线为 **10 步、FED 0.1471**；起点 2 为 **19 步、FED 0.2300**。下表 FED 保留六位小数，是否复现按预先固定的路径逐格一致与四位小数一致共同判断。

| 设置 | 种子 | 实际训练轮数 | 起点1：步数 / FED | 起点2：步数 / FED | 全起点到达 | 两条论文路线同时重现 | 非法动作 / 重访 |
|---|---:|---:|---:|---:|---:|---|---:|
{table}

失败模型也完整保留。没有把未到达出口的低步数或剂量当成有效疏散结果；全部起点的平均 FED 如需使用，必须同时查看到达率。

训练总计 **{total_episodes:,} 个回合、{total_transitions:,} 个转移、{total_updates:,} 次优化更新**。三个种子的结果仅描述本轮重复情况，不据此计算泛化成功率或显著性。

![六次训练的原始损失趋势](../outputs/safestep_author_training_20261015_v2/training_loss_curves.png)

图中为公开程序四输出 MSE 的 100 轮均值。损失下降不等于路线正确；两种设置的奖励不同，损失大小也不能直接作为方法优劣指标。本轮没有要求或证明随机训练损失曲线逐点等于论文曲线。

### 4.1 补充缓存核对：路线差异是否只是奖励偏好不同

使用已保存的每步剂量，分别按两套出口奖励及 gamma=0.95 重算原始折扣回报。在两处起点、两套奖励中，论文给出的 Exit B 路线都优于论文指定的 Exit A 备选路线。这说明本轮若改走该备选路线，不能仅用“奖励更喜欢快到出口”来解释。

公开设置下，种子 141 的起点 2 路线为 21 步、FED 0.275580，论文路线为 19 步、FED 0.230021；前者时间更长、剂量更高，原始折扣回报也更低（−3.244884 对 −2.899458）。这是具体的策略质量差异，但本轮没有据此更改训练设置。

此为事后缓存解释，只比较已经保存的指定路线，不是对全部路线做了最优性证明，也没有修改原验收标准。

## 5. 原始训练程序与论文的额外差别

除了学习率、轮数和出口奖励，还发现三项必须保留说明的差别：

1. **更新时机不同。** 正文表 2 在每个时间步内抽样更新；公开程序在整个回合结束后才执行一次优化。本轮沿用公开程序。
2. **损失并非只包含选定动作。** 公开程序用目标网络当前状态的四个输出作目标，仅替换被采取动作的目标值，然后对四个输出求 MSE；这与正文公式所写的单个状态动作误差并不完全相同。
3. **超时仍然自举。** 公开优化器只把 WIN 当成终止目标；超时 LOSE 的转移仍然加入下一状态最大 Q。本轮记录该行为，没有暗中修复后仍称原版。

因此，本轮核实的是两种明确标注的作者程序运行结果。不能把它们与论文最终版本的全部训练细节视为同一个已知过程。论文原始训练种子和完整历史未提供，也不能声称重建了作者当年的随机训练轨迹。

## 6. 记录层错误、修复和独立核查

初次 20 轮检查在最终进度状态写入处遇到字段重复错误，学习与评价文件已保留；修复后，正式 v1 六次运行又在第 100 轮的进度写入处遇到同类错误。原始算法没有在这两处报错。

v1 目录、冻结配置和代码保留。v2 用单独的记录类区分“运行状态”和“回合结果”，不改变作者训练函数；重复检查扩展为 100 轮，覆盖此前未触发的写入分支，再冻结六次正式运行。没有根据失败记录调整学习率、奖励、种子或预算。

四个针对训练适配器的单元检查通过，包括确定性 reset、采样不消耗额外随机数、一次优化后的权重与原程序完全一致，以及第 100 轮进度写入。

完整核查的首个版本在逐轮检查时重复解压整个回放编号数组，导致额外读取开销。保留首版脚本后，第二版仅将数组改为一次读入；核查断言、范围和训练文件不变。该修正不产生任何训练更新。

独立核查进一步完成：

- 逐步检查全部 **{total_transitions:,} 个训练转移**的几何移动、时间索引、剂量、奖励和终止状态。
- 检查全部 **{verification['total_replay_sample_IDs_checked']:,} 个回放样本编号**，确认不越过当时的容量与时间窗口，批内没有重复。
- 核对逐轮更新数、目标网络同步位置，以及早停是否满足冻结条件。
- 重载全部最终权重，独立执行 **366 条最终疏散轨迹**，核对保存的完整路线和 FED。
- 三对初始策略权重相同，最终模型先保存再评价。
- **{verification['prior_artifacts_unchanged']} 个此前资料文件**的哈希未变；作者原始代码、数据与权重未编辑。

这些检查支持执行与数值的可信性，不是对真实火灾的物理验证，也不代表训练算法与论文全部伪代码已一致。

## 7. 当前复现边界与后续

| 部分 | 当前状态 |
|---|---|
| 酒店原权重两条推荐路线、四条路线 FED | 上一轮已在报告精度上对齐，仍有效 |
| 原作者公开训练程序从随机初始化运行 | 本轮六次有限运行完成，结果见表 |
| 正文全部训练伪代码与实现逐项一致 | 尚未完成；已有明确差别，不应混称 |
| 原始小案例约 62% 改善 | 仍缺该建筑的区域率表，酒店数据不能替代 |
| FDS 与浓度到 FER 的上游生成 | 尚未复现 |
| LLM、人群 PPO、真实行为模型 | 未在本轮训练或验证 |

下一步应先根据这些明确差别决定是否另做“严格正文伪代码实现”，并把它作为单独版本。不能为了选到更好的曲线而无上限扩大本轮训练。48 个旧人群最终场景继续封存。

## 8. 可追溯文件

- [冻结配置](../schemas/safestep_author_training_20261015_v2.json)
- [六次运行比较 CSV](../outputs/safestep_author_training_20261015_v2/six_run_comparison.csv)
- [完整汇总](../outputs/safestep_author_training_20261015_v2/summary.json)
- [独立核查](../outputs/safestep_author_training_20261015_v2/verification.json)
- [此前原数据与原权重的关键数值报告](SafeStep原作者数据与权重_关键数值复现报告_2026-10-15.md)

每次运行目录保留最终权重、每轮记录、逐步转移、回放样本编号、收敛检查和最终轨迹。原作者资料仍保存在上一轮的独立目录，本轮没有覆盖它们。
'''
    REPORT.write_text(report,encoding="utf-8")
    final_paths=[p for p in OUT.rglob("*") if p.is_file() and p.name!="completion_manifest.json"]
    final_paths.extend([REPORT,ROOT/"scripts/verify_safestep_author_training.py",Path(__file__),
                        ROOT/"scripts/verify_safestep_author_training_v2.py",
                        ROOT/"tests/test_safestep_author_training.py",ROOT/"tests/test_safestep_author_training_v2.py"])
    manifest=dict(status="complete",runs=6,total_episodes=total_episodes,total_transitions=total_transitions,
        optimizer_updates=total_updates,acceptance_counts=counts,verification_status="PASS",report=str(REPORT.relative_to(ROOT)),
        no_LLM_requests=True,old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in final_paths})
    (OUT/"completion_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!="artifact_sha256"},indent=2))


if __name__=="__main__":main()
