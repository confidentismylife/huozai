"""Summarize all six fixed-budget runs after independent verification."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_frequency_ablation_20261015"
REPORT = ROOT / "reports/SafeStep更新频率单因素实验_六次运行结果_2026-10-15.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label):
    return f"[{label}]({Path(path).as_posix()})"


def main():
    assert not REPORT.exists()
    assert not (OUT / "completion_manifest.json").exists()
    cfg = read(ROOT / "schemas/safestep_frequency_ablation_20261015.json")
    verified = read(OUT / "verification.json")
    assert verified["status"] == "PASS" and len(verified["runs"]) == 6
    results, flat = {}, []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            name = f"{arm}_seed{seed}"
            summary = read(OUT / "runs" / name / "summary.json")
            training = summary["training"]
            audit = next(r for r in verified["runs"] if r["run"] == name)
            results[arm, seed] = (summary, audit)
            row = dict(arm=arm, seed=seed, control_steps=training["control_steps"],
                completed_episodes=training["completed_episodes"], updates=training["optimizer_updates"],
                epsilon=training["epsilon"], training_wall_seconds=training["wall_seconds"],
                arrivals=summary["success"], blocked_starts=audit["trajectory_categories"]["blocked_self_loop"],
                cycle_starts=audit["trajectory_categories"]["multi_cell_cycle"],
                invalid_steps=summary["invalid_steps"], revisits=summary["revisits"],
                both_paper_routes_match=summary["both_paper_routes_match"])
            for i, p in enumerate(summary["paper_starts"], 1):
                row.update({f"start{i}_status": p["status"], f"start{i}_steps": p["steps"],
                            f"start{i}_FED": p["FED"], f"start{i}_paper_path_match": p["path_match"]})
            flat.append(row)
    with (OUT / "frequency_comparison.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    pairs = []
    for seed in cfg["seeds"]:
        a, b = [results[arm, seed][0] for arm in cfg["arms"]]
        delta = b["success"] - a["success"]
        pairs.append(dict(seed=seed, episode_end_arrivals=a["success"], every_step_arrivals=b["success"],
            arrival_count_delta=delta, arrival_percentage_point_delta=delta / 61 * 100,
            optimizer_count_ratio=b["training"]["optimizer_updates"] / a["training"]["optimizer_updates"]))
    with (OUT / "paired_differences.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    deltas = [p["arrival_count_delta"] for p in pairs]
    if all(d > 0 for d in deltas):
        interpretation = "三个成对种子中，按步更新的到达起点数均更高，但两组都还没有学到可靠策略。当前早期有限预算下，单独增加更新频率没有重现上一轮相对较差的表现方向；这个结果不能排除高频更新在更长训练后出现退化。因此不能把上一轮崩坏简单归为‘每步更新’，也不能据此认定根因已定位；其他改动、相互作用与训练阶段仍需区分。"
    elif all(d < 0 for d in deltas):
        interpretation = "三个成对种子中，按步更新的到达起点数均更低。这为‘在当前学习率、回放与同步规则下，单独提高更新频率会损害这个有限预算内的策略表现’提供了局部证据；仍不能认定它是上一轮失败的唯一原因，也不能断定继续训练后的方向。"
    elif all(d == 0 for d in deltas):
        interpretation = "三个成对种子的到达起点数差均为零。这个指标在当前有限预算下没有区分两种更新方式；这不等于证明两个训练过程等价，也不能定位上一轮失败原因。"
    else:
        interpretation = "成对种子的差值方向不完全一致，或包含持平。当前有限预算没有给出一致的更新频率优劣方向，不能将上一轮失败归因于更新频率这一项。"
    table = []
    paper_table = []
    labels = {"episode_end": "每回合一次", "every_step": "每控制步一次"}
    for row in flat:
        table.append(f"| {labels[row['arm']]} | {row['seed']} | {row['arrivals']}/61 | {row['blocked_starts']} | {row['cycle_starts']} | {row['completed_episodes']:,} | {row['updates']:,} | {row['epsilon']:.4f} |")
        s1 = f"{row['start1_status']}，{row['start1_steps']} 步，{row['start1_FED']:.6f}"
        s2 = f"{row['start2_status']}，{row['start2_steps']} 步，{row['start2_FED']:.6f}"
        paper_table.append(f"| {labels[row['arm']]} | {row['seed']} | {s1} | {s2} | {'是' if row['both_paper_routes_match'] else '否'} |")
    pair_table = [f"| {r['seed']} | {r['arrival_count_delta']:+d} | {r['arrival_percentage_point_delta']:+.2f} | {r['optimizer_count_ratio']:.1f} 倍 |" for r in pairs]
    updates = verified["total_updates"]
    report = f"""## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：run + validate。
- Origin Date：2026-10-15。
- Verification Status：六次固定控制步预算运行完成，独立核查通过。
- Version Label：safestep_frequency_ablation_v1。
- 范围：同一作者酒店案例的机制诊断；不是论文 12,000 回合完整训练、真实火灾验证或新建筑泛化测试。

# SafeStep 更新频率单因素实验

本轮完成两种更新方式、三个种子的六次训练，每次恰好 **200,000 个环境控制步**。六次均完成固定预算，没有超时、追加训练或挑选最佳检查点。

{interpretation}

六个预算最终模型均未满足全部61起点到达、无非法动作和重访的要求，也没有任何一个同时匹配两条论文推荐路线。本轮没有得到可靠可用的新策略，产出是一次有边界的机制比较。

## 1. 问题与冻结设计

上一轮正文流程重建同时改变更新频率、损失、终止目标和目标网络同步，并在一小时上限中断。不能从那一轮直接确定哪项造成了中途策略失败。

本轮以作者公开程序的训练机制为参照，在共同实现中仅改变优化调度：

| 项目 | 每回合一次组 | 每控制步一次组 |
|---|---|---|
| 优化触发 | 回放超过64条后，仅在环境回合结束时更新 | 回放超过64条后，每次转移后更新 |
| 损失 | 原公开程序四动作输出 MSE，未选动作有目标网络锚定 | 相同 |
| 超时样本 | 保留原程序的下一状态自举 | 相同 |
| 探索率规则 | 在符合预热条件的已完成回合末乘0.9995 | 相同 |
| 目标同步规则 | 仅在已完成回合末，累计步数超过50才同步 | 相同 |
| 模型与输入 | 61→61→4；只输入位置 | 相同 |
| 其他参数 | 学习率0.008、折扣0.95、批量64、回放1000、出口奖励参数10 | 相同 |
| 预算 | 每种子200,000环境步 | 相同 |

种子为141、142、143；三对初始网络权重完全一致。六次运行前冻结配置、代码及作者资料哈希；每次事先设置900秒上限，结果出来后不延长、不重试、不改变超参数。

**控制的是环境交互预算，不是优化次数或耗时。** 更新频率的变化会改变轨迹、已完成回合数和回放内容。探索规则和同步规则虽然相同，实际探索率、实际同步时刻也可能随轨迹变化。因此本轮检验这个调度选择的整体有限预算效果，没有单独隔离其每一种后续作用。

## 2. 共同适配与原版边界

为让有限实验可执行，两组共同采用批量计算的原版四输出损失。运行前，使用作者原优化函数连续八次更新作参照，比较损失、策略梯度与 Adam 参数；按相对容差2e-5、绝对容差2e-6通过。批量矩阵计算可能产生浮点差异，不能声称逐位复刻旧训练轨迹。目标网络锚定值在共同实现中停止求梯度；目标网络不由策略优化器更新，对策略梯度的数值一致性已在上述检查范围内核实。

两组共同分离探索动作与回放抽样的随机数流，避免多抽回放样本顺带改变探索动作随机数。作者 reset 仍收到事先确定的逐回合种子，但它内部的全局随机数重置不干扰本轮的局部生成器。这是共同实验适配，不是宣称与旧公开程序运行历史完全相同。

预算若在回合中耗尽，最后一条转移保留 PLAYING 标记，记录为 BUDGET_CUT；不伪造超时、不额外执行回合末优化／探索衰减／同步。最终权重按恰好200,000步保存，再作评价。未用50,000或100,000步检查点挑成绩。

## 3. 六次运行结果

“原地循环”指最终反复选择不能移动的方向；“多格循环”指确定性贪心策略在多个格子间反复行走。两类均以未到达出口为前提，按起点计数，三列之和每行均为61。

| 更新方式 | 种子 | 到达出口 | 原地循环 | 多格循环 | 完成训练回合 | 优化更新数 | 结束探索率 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

按步更新组每次均为199,936次更新，即200,000减去64步预热。按回合组的次数取决于预算内实际完成的回合，详细记录保留未完成的最后回合。

| 成对种子 | 到达起点数差：按步减按回合 | 到达比例差：百分点 | 按步／按回合更新次数比 |
|---|---:|---:|---:|
{chr(10).join(pair_table)}

这些是三个种子在同一酒店地图全部61起点上的描述。没有把61个起点当成61栋独立建筑，也没有计算显著性或宣称泛化提升。

## 4. 两个论文起点

论文推荐路线参照为：起点1，10步、FED四位小数0.1471；起点2，19步、FED四位小数0.2300。路径是否同时匹配按逐格比较决定。

| 更新方式 | 种子 | 起点1：状态／步数／FED | 起点2：状态／步数／FED | 两条推荐路径同时匹配 |
|---|---:|---|---|---|
{chr(10).join(paper_table)}

LOSE轨迹按原环境一直积分到301步，不能当成有效疏散路线，更不能仅按FED大小比较优劣。即使本轮某模型得到论文路径，也只说明当前固定预算模型的路线吻合，不意味着完成论文全部训练与案例。

## 5. 独立核查

- 核对全部 **{verified['total_control_steps']:,} 个训练转移**的几何位置、危险率采样、FED、奖励、回合终止与预算截断。
- 检查 **{updates:,} 次优化**及 **{verified['total_replay_IDs_checked']:,} 个回放样本编号**，逐次重建局部回放随机数生成器，所有编号匹配。
- 分别重建两组的更新时刻、探索衰减和回合末目标同步，确保没有混入上一轮的逐50步同步或超时零自举。
- 从完整上下文精确重放 **{verified['exactly_replayed_Adam_contexts']} 次事先指定的 Adam 更新**；这验证本轮共同实现，不代表与作者标量实现全程逐位一致。
- 从保存的预算最终权重独立执行 **366 条路线**，核查路径、剂量、到达数、非法动作、循环和论文路径匹配。
- 三对初始权重和分歧前前65条转移一致；每种设置两次5,000步预运行的全部转移、优化日志和最终权重一致。
- 上一轮完成清单中的 **{verified['earlier_artifact_hashes_unchanged']} 个登记文件**哈希未变；原作者资料哈希也通过本轮冻结检查。

预运行独立审计首版在读取结构化数组的奖励字段时遇到 NumPy 内存步长与张量转换不兼容。保留首版审计器后，仅将该字段转换成连续float32数组；相同训练文件通过修正后的审计。这个错误发生在审计读取层，没有导致训练重跑或参数修改。

## 6. 科学解释与下一步边界

{interpretation}

本轮预算使两组处于有限学习阶段，不能替代此前12,000回合的完成运行，更不能与上轮8,000至9,500回合中途权重作等训练阶段比较。尤其不能把当前组间差异直接解释成对上一轮全部失败的充分解释。

下一轮如果继续，应先根据本轮差异选择一个仍有区分力的机制问题并重新冻结计划。若要严格比较同一探索率轨迹下的纯更新频率，需要另设按全局控制步固定的探索与同步安排；那会是新的实验设定。若要主张正文训练能复现论文终点，则必须另行安排完整训练预算，不能拿本轮代替。

从代码机制看，更值得单独核对的另一项是损失定义：原程序对四个输出平均，而单动作损失只对选中输出平均；在同一批样本和同一目标值下，选中动作误差项的系数会相差四倍，同时还移除了未选动作的锚定。这是由公式可见的差别，不是本轮已经验证的失败原因。后续可在固定更新频率下单独检验这一项，避免直接把学习率、终止目标和观测一起修改。

本轮没有启动上述追加运行。当前证据应先与此前原始权重、公开程序从头训练和正文流程中断结果分开呈现。

## 7. 常见误读检查

完成11/11项检查：按种子分别呈现以避免合并掩盖差异；不从酒店模型外推个体真人行为；全部61起点和全部六个种子结果保留以限制选择／幸存者偏差；没有按成功路线作条件筛选或回归控制；分母及回合／控制步／更新数明确；没有挑最差种子重试再宣称改善；没有寻找显著结果或挑最佳检查点；配置形成时间与共同实现适配公开；更新次数与失败之间不作反向或唯一因果解释。仅报告描述性成对差，不进行显著性检验。

## 8. 资料与保持封存的范围

作者来源：[SafeStep_v1](https://github.com/hqfang3/SafeStep_v1)，固定提交`{cfg['author_commit']}`。

- {link(ROOT / 'schemas/safestep_frequency_ablation_20261015.json', '冻结实验设计')}
- {link(OUT / 'frequency_comparison.csv', '六次运行结果CSV')}
- {link(OUT / 'paired_differences.csv', '三个种子的成对差值')}
- {link(OUT / 'verification.json', '独立核查汇总')}
- {link(OUT / 'campaign_status.json', '运行状态与耗时')}
- {link(ROOT / 'reports/SafeStep正文训练流程_超时与中途策略审计_2026-10-15.md', '上一轮中断与缓存诊断报告')}
- {link(ROOT / 'reports/SafeStep原作者数据与权重_关键数值复现报告_2026-10-15.md', '此前酒店关键数值复现')}

没有LLM请求、收费API或PPO训练；48个旧人群最终场景继续封存。小建筑约62%改善、原始危险率缺口、FDS与浓度到危险率上游过程仍未复现。此前成功对齐的酒店原权重四条路线剂量结果保留有效。
"""
    REPORT.write_text(report, encoding="utf-8")
    result = dict(status="SIX_FIXED_CONTROL_BUDGET_RUNS_VERIFIED", created_at=datetime.now(timezone.utc).isoformat(),
        comparison_scope=cfg["comparison_scope"], pairs=pairs, interpretation=interpretation,
        total_control_steps=verified["total_control_steps"], total_updates=updates,
        paper12000_episode_training_completed=False, LLM_requests=0,
        old_final48_crowd_scenarios_accessed=False, report=str(REPORT.relative_to(ROOT)))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    local_links = [p for p in re.findall(r"\]\(([^)]+)\)", report) if not p.startswith("https://")]
    for p in local_links:
        assert Path(p).is_absolute() and Path(p).is_file(), p
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += [REPORT, Path(__file__).resolve(), ROOT / "schemas/safestep_frequency_ablation_20261015.json",
              ROOT / "src/fire_evac/safestep_frequency_ablation.py", ROOT / "scripts/run_safestep_frequency_ablation.py",
              ROOT / "scripts/verify_safestep_frequency_ablation.py", ROOT / "scripts/verify_safestep_frequency_ablation_v1.py",
              ROOT / "tests/test_safestep_frequency_ablation.py"]
    manifest = dict(status="SIX_FIXED_CONTROL_BUDGET_RUNS_AND_AUDIT_COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        completed_runs=6, control_steps_per_run=200000, final_model_type="budget_final_not_paper12000_episode_final",
        post_score_training_extensions=0, old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))})
    (OUT / "completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for p, expected in manifest["artifact_sha256"].items():
        assert digest(ROOT / p) == expected, p
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Artifact hashes verified:", len(manifest["artifact_sha256"]))


if __name__ == "__main__":
    main()
