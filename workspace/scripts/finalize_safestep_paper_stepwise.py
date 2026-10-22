"""Write the bounded Table2 reconstruction report after full verification."""
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"schemas/safestep_paper_stepwise_20261015.json"
OUT=ROOT/"outputs/safestep_paper_stepwise_20261015"
REPORT=ROOT/"reports/SafeStep正文训练流程复现_三种子结果与边界_2026-10-15.md"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not REPORT.exists(),"Preserve frozen reports"
    cfg=read(CFG);verification=read(OUT/"verification.json")
    assert verification["status"]=="PASS"
    rows=[]
    for label,parent,pattern in [
        ("原公开程序＋正文参数",ROOT/"outputs/safestep_author_training_20261015_v2/runs","paper_listed_parameters_seed{}"),
        ("正文逐步更新流程",OUT/"runs","seed{}")]:
        for seed in cfg["seeds"]:
            summary=read(parent/pattern.format(seed)/"summary.json")
            training=summary["training"];a,b=summary["paper_starts"]
            rows.append(dict(method=label,seed=seed,episodes=training["episodes"],transitions=training["transitions"],
                updates=training["optimizer_updates"],start1_status=a["status"],start1_steps=a["steps"],start1_FED=a["FED"],
                start2_status=b["status"],start2_steps=b["steps"],start2_FED=b["FED"],
                both_paper_routes_match=summary["both_paper_routes_match"],all_start_success=summary["all_start_success"],
                illegal_steps=summary["invalid_steps"],revisits=summary["revisits"],
                all_start_mean_FED=summary["mean_FED"],all_start_max_FED=summary["max_FED"],
                starts_FED_at_least03=summary["over_03"]))
    with (OUT/"paired_comparison.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    old=rows[:3];new=rows[3:]
    matches=sum(r["both_paper_routes_match"] for r in new)
    all_start=sum(r["all_start_success"]==61 and r["illegal_steps"]==0 and r["revisits"]==0 for r in new)
    new_updates=sum(r["updates"] for r in new);old_updates=sum(r["updates"] for r in old)
    comparison=dict(rows=rows,new_paper_route_matches=matches,new_all_start_gate=all_start,
        old_paper_route_matches=sum(r["both_paper_routes_match"] for r in old),
        new_total_updates=new_updates,old_total_updates=old_updates,optimizer_budget_ratio=new_updates/old_updates,
        no_equal_compute_causal_claim=True)
    (OUT/"summary.json").write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    for seed,color in zip(cfg["seeds"],["#176a9b","#b45d23","#258158"]):
        entries=list(csv.DictReader((OUT/"runs"/f"seed{seed}"/"episodes.csv").open(encoding="utf-8")))
        values=np.asarray([float(row["mean_update_loss"]) for row in entries])
        smoothed=np.convolve(values,np.ones(100)/100,mode="valid")
        ax.semilogy(np.arange(100,len(values)+1),np.maximum(smoothed,1e-10),color=color,label=f"Seed {seed}",linewidth=1.1)
    ax.set_title("SafeStep paper-flow reconstruction: selected-action TD loss")
    ax.set_xlabel("Episode");ax.set_ylabel("Selected-action MSE (100-episode average)")
    ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/"stepwise_training_loss.png",dpi=170);plt.close(fig)
    table="\n".join(f"| {r['method']} | {r['seed']} | {r['updates']:,} | {r['start1_status']}，{r['start1_steps']} / {r['start1_FED']:.6f} | "
        f"{r['start2_status']}，{r['start2_steps']} / {r['start2_FED']:.6f} | {r['all_start_success']}/61 | {'是' if r['both_paper_routes_match'] else '否'} |" for r in rows)
    safety="\n".join(f"| {r['seed']} | {r['illegal_steps']:,} | {r['revisits']:,} | {r['all_start_mean_FED']:.6f} | {r['all_start_max_FED']:.6f} | {r['starts_FED_at_least03']} |" for r in new)
    headline=(f"本轮三个种子中，**{matches}/3** 同时对齐论文两条推荐路线，**{all_start}/3** 满足全部61个起点到达、零非法动作和重访。")
    interpretation=("本轮三个种子都重现了两条指定论文路线。但这一结果限于一个酒店火灾场景，不能换算为跨建筑成功率。"
        if matches==3 else "本轮仍存在种子差异，不能称为稳定复现论文路线。应保留不一致模型，不能通过选择最佳检查点掩盖差异。")
    report=f'''# SafeStep 正文训练流程复现：三种子结果与边界

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / reproducibility verification
- Origin Date: 2026-10-15
- Verification Status: VERIFIED（下述训练执行与数值核查）
- Version Label: safestep_paper_stepwise_v1
- 论文：Fang et al., Journal of Building Engineering 118 (2026), 115132；[DOI](https://doi.org/10.1016/j.jobe.2025.115132)。
- 原始环境与数据：[作者固定提交](https://github.com/hqfang3/SafeStep_v1/tree/{cfg['author_commit']})。
- 状态：三个固定种子训练、独立复算与整理完成。没有正在扩展的训练。

## 1. 本轮回答什么

上一轮使用公开训练程序，发现它与正文表2及公式9不同。本轮单独重建正文明确给出的训练流程，检验从头训练能否更一致地重现酒店两条指定路线。

{headline}

{interpretation}

原公开程序配正文参数时，两条路线同时重现2/3。新旧实际优化次数分别为 **{new_updates:,}** 与 **{old_updates:,}**，比值 **{new_updates/old_updates:.2f}**。因此这是训练流程的复现比较，**不是相同计算预算下的算法改进证据**。

## 2. 对照正文修改了什么

| 项目 | 上轮公开程序配正文参数 | 本轮正文流程重建 | 来源或解释 |
|---|---|---|---|
| 优化时机 | 一整个回合结束后一次 | 回放就绪后，每个控制步一次 | 正文表2，PDF第7页 |
| 损失 | 四输出 MSE，被采取动作的目标被替换 | 只对采样动作 Q(s,a) 求 MSE，目标断开梯度 | 公式9，PDF第6页 |
| 终止目标 | 只有WIN停止自举 | WIN与回合超时均令后续价值为0 | 表2的回合终止分支；超时长度沿用原代码 |
| 目标网络同步 | 在回合末判断累计步数是否超过50 | 每50个全局控制步同步，在该步优化后进行 | 表2“Every C steps” |

三种种子均固定为141、142、143；每种12000回合；学习率0.008；gamma0.95；batch64；回放容量1000；出口路径奖励10，仍由路径权重0.1缩放。

没有新增动作屏蔽、观测信息、教师、模仿损失、奖励项或候选路径选择器。最终模型保存后才评分，不提前停止，不挑选最佳种子或检查点。

## 3. 正文没有唯一确定的部分

这轮称为“正文流程重建”，不能称为作者完整训练历史的唯一精确还原。未明确的细节逐项记录如下。

- 网络为原公开版本的61→61→4，隐藏层ReLU，优化器Adam。
- 探索率每回合内保持不变，回合后乘0.9995；这与上轮每回合一次更新的衰减节奏一致。没有把它改为每次新优化都衰减，否则探索会在很少回合内耗尽。正文未给出完整衰减日程。
- `len(replay)>64`后才抽样优化，沿用原代码的缓冲区就绪条件。
- 保留原环境第301步回合超时，并把这个回合结束按表2的终止分支处理。位置输入仍不包含剩余时长、累积FED和访问历史，因此仍有状态信息不完整的限制。
- 保留原始积分：初始起点剂量、新位置采样、5秒率表取整、出口零剂量，以及原始奖励的先后次序。这些代码级细节未在正文全部写明，但上一轮已用它们重算对齐四条酒店路径的FED。
- 相同种子的初始策略权重和12000个重置种子序列均与上轮配对一致；探索和回放抽样的消耗则会随新更新频率而变化。

这些是透明说明的实施选择，不能把它们当成已经得到作者确认的未公开设置。

## 4. 结果：相同种子的六行对照

两个目标为起点1的10步、FED0.1471，以及起点2的19步、FED0.2300。路线需逐格相同，FED需在报告的四位小数上相同，才计为复现。

| 设置 | 种子 | 实际优化次数 | 起点1：状态，步数 / FED | 起点2：状态，步数 / FED | 全起点到达 | 两条论文路线同时对齐 |
|---|---:|---:|---:|---:|---:|---|
{table}

WIN代表到达出口，LOSE代表回合上限内未到达。若存在LOSE，长时间累积剂量或平均值必须结合失败率理解，不能用成功回合均值掩盖失败。

本轮全部起点的附加诊断如下。它们共享同一建筑与火灾时序，不是独立真实火灾样本。

| 种子 | 非法动作 | 重访次数 | 全起点平均FED，含失败 | 最大FED | FED≥0.3起点数 |
|---|---:|---:|---:|---:|---:|
{safety}

即便全部到达，也不自动意味着所有起点低于0.3参考阈值，更不意味着全局最优。没有以本轮结果选择或改造新测试场景，48个旧人群最终场景仍未读取。

## 5. 更新量和训练曲线

本轮共36000回合，**{verification['total_transitions']:,} 个控制步、{new_updates:,} 次优化**。每个种子前64个控制步用于就绪缓冲区，之后每个控制步更新一次。

![正文流程的选定动作损失](../outputs/safestep_paper_stepwise_20261015/stepwise_training_loss.png)

曲线是每回合更新损失均值再作100回合平均。它与上一轮四输出MSE的定义、优化次数不同，不能直接用纵轴高低判断方法优劣；路线、到达和剂量结果才是本轮验收依据。

## 6. 独立核查证据

三个公式单元检查通过，分别覆盖终止分支、仅选定动作接收预测误差梯度，以及非终止Bellman目标。

训练前，同一种子两次100回合运行的逐步转移、回放编号、预测、目标与最终权重完全一致。短训练上另行复核了18281个控制步、18217次优化以及9次完整Adam更新。

完整训练后独立核查：

- 全部 **{verification['total_transitions']:,} 个转移**的移动、时间索引、FED、奖励和终止。
- 全部 **{verification['total_replay_IDs_checked']:,} 个回放编号**的合法时间窗口、容量与批内唯一性。
- 全部已保存选定动作预测与目标构成的MSE。
- **{verification['total_terminal_TD_labels_checked']:,} 个终止TD标签**均严格等于对应即时奖励，自举项为0。
- 按训练前指定位置保存的 **{verification['exactly_replayed_Adam_updates']} 次完整优化上下文**，重载策略、目标及Adam状态后，标签、损失和更新后权重逐项完全一致。
- 重载三个最终模型，独立重算 **183条最终完整轨迹**及其FED。
- 与上一轮三个对应种子的初始权重、重置种子序列一致。
- 上一轮冻结的 **{verification['old_artifacts_unchanged']} 个文件**哈希未变，作者原始代码与数据未编辑。

优化日志分块存储并一次读取核查，未在逐条检查中重复解压整个数组。核查通过说明执行与数值可信，不代表物理火灾模型获得了外部验证。

## 7. 现在可以与不可以说什么

可以报告：在披露了未指定细节的条件下，正文逐步更新、选定动作损失和终止处理已经实施，并完成三个固定种子的有限运行。酒店关键路线的复现数量按表如实报告。

不能报告：改动中哪一项单独造成结果差异、相同计算量下优于公开程序、跨建筑或人群泛化、真实救生效果，或摘要中小建筑约62%改善已经复现。

该小建筑的率表仍未取得；本轮也没有重新运行FDS或从CO/HCN/CO2浓度生成率。酒店现成率表不能替代这些缺口。

## 8. 后续决定

保留当前版本、三个最终模型及全部结果。先整理原权重复算、公开程序从头训练和本轮正文流程三条证据，明确不一致发生在哪里；不按结果临时追加种子或调参。要继续论文小案例，应优先获得对应建筑的原始率表。

## 9. 可追溯文件

- [冻结方案](../schemas/safestep_paper_stepwise_20261015.json)
- [逐种子比较CSV](../outputs/safestep_paper_stepwise_20261015/paired_comparison.csv)
- [汇总与验收](../outputs/safestep_paper_stepwise_20261015/summary.json)
- [独立核查](../outputs/safestep_paper_stepwise_20261015/verification.json)
- [上一轮从头训练报告](SafeStep从头训练复现_双设置三种子结果_2026-10-15.md)

原作者资料、既有实验和旧报告均独立保留。本轮为0次LLM请求，没有读取旧48个封存人群场景。
'''
    REPORT.write_text(report,encoding="utf-8")
    links=[p for p in re.findall(r'\]\(([^)]+)\)',report) if not p.startswith(('http://','https://'))]
    assert all((REPORT.parent/p).resolve().exists() for p in links)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json']
    paths.extend([REPORT,CFG,ROOT/"src/fire_evac/safestep_paper_stepwise.py",ROOT/"scripts/run_safestep_paper_stepwise.py",
        ROOT/"scripts/verify_safestep_paper_stepwise.py",Path(__file__),ROOT/"tests/test_safestep_paper_stepwise.py"])
    manifest=dict(status="complete",runs=3,episodes=36000,transitions=verification['total_transitions'],updates=new_updates,
        paper_routes_matched=matches,all_start_gate_passed=all_start,independent_verification="PASS",
        equal_compute_comparison=False,report=str(REPORT.relative_to(ROOT)),old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths})
    (OUT/"completion_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!='artifact_sha256'},indent=2))


if __name__=="__main__":main()
