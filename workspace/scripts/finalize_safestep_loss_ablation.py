"""Complete the nine-model loss comparison after independent verification."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_evac.safestep_author_numerics import AuthorDQN


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def describe_direction(deltas, contrast):
    if all(d > 0 for d in deltas):
        return f"{contrast}：三个种子的到达起点数差均为正。这个方向只适用于本轮固定预算与酒店案例，不能据此认定跨种子总体效应或最终收敛效果。"
    if all(d < 0 for d in deltas):
        return f"{contrast}：三个种子的到达起点数差均为负。在本轮条件下没有显示到达数改善，不能断定其他训练阶段也如此。"
    if all(d == 0 for d in deltas):
        return f"{contrast}：三个种子的到达起点数差均为零。该指标未区分两组；不代表权重、路径或训练过程等价。"
    return f"{contrast}：三个种子的到达数差方向不一致或包含持平，本轮没有一致的改善方向。"


def main():
    torch.set_num_threads(1)
    cfg = read(ROOT / "schemas/safestep_loss_ablation_20261015.json")
    out = ROOT / cfg["output_directory"]
    report_path = ROOT / "reports/SafeStep损失函数拆分实验_九模型对照_2026-10-15.md"
    assert not report_path.exists() and not (out / "completion_manifest.json").exists()
    verification = read(out / "verification.json")
    assert verification["status"] == "PASS" and len(verification["runs"]) == 9
    probe = read(out / "common_context_adam_probe_v2.json")
    assert probe["status"] == "COMMON_CONTEXT_ONE_STEP_PROBE_COMPLETE"
    assert probe["version"] == "2-independent-optimizer-moment-copies"
    assert probe["script_sha256"] == digest(ROOT / "scripts/probe_safestep_loss_adam_contexts.py")
    for path, expected in probe["source_sha256"].items():
        assert digest(ROOT / path) == expected

    def directory(arm, seed):
        if arm == "full_anchor":
            return ROOT / cfg["baseline_directory"] / "runs" / f"every_step_seed{seed}"
        return out / "runs" / f"{arm}_seed{seed}"

    results, rows = {}, []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            d = directory(arm, seed)
            s = read(d / "summary.json")
            t = s["training"]
            v = next(v for v in verification["runs"] if v["run"] == d.name)
            results[arm, seed] = (s, v)
            row = dict(loss=arm, seed=seed, reused_baseline=arm == "full_anchor", control_steps=t["control_steps"],
                updates=t["optimizer_updates"], completed_episodes=t["completed_episodes"], epsilon=t["epsilon"],
                arrivals=s["success"], blocked_starts=v["trajectory_categories"]["blocked_self_loop"],
                cycle_starts=v["trajectory_categories"]["multi_cell_cycle"], invalid_steps=s["invalid_steps"],
                revisits=s["revisits"], both_paper_routes_match=s["both_paper_routes_match"],
                all_start_arrival_without_invalid_or_revisit=s["success"] == 61 and s["invalid_steps"] == 0 and s["revisits"] == 0)
            for i, p in enumerate(s["paper_starts"], 1):
                row.update({f"start{i}_status": p["status"], f"start{i}_steps": p["steps"],
                            f"start{i}_FED": p["FED"], f"start{i}_paper_path_match": p["path_match"]})
            rows.append(row)
    write_csv(out / "nine_model_comparison.csv", rows)
    contrasts = []
    for seed in cfg["seeds"]:
        a, b, c = [results[arm, seed][0]["success"] for arm in cfg["arms"]]
        contrasts.append(dict(seed=seed, full_anchor_arrivals=a, selected_arrivals=b, selected_quarter_arrivals=c,
            remove_anchor_delta_C_minus_A=c - a, remove_anchor_percentage_points=(c - a) / 61 * 100,
            scale_objective_delta_B_minus_C=b - c, scale_objective_percentage_points=(b - c) / 61 * 100))
    write_csv(out / "paired_contrasts.csv", contrasts)

    # Compare all three fixed seed pairs, without selecting trajectories or checkpoints.
    invariance = []
    for seed in cfg["seeds"]:
        dirs = [directory(arm, seed) for arm in ("selected", "selected_quarter")]
        weights = [torch.load(d / "budget_final_model.pth", weights_only=True, map_location="cpu") for d in dirs]
        assert all(torch.isfinite(t).all() for w in weights for t in w.values())
        diffs = [weights[0][k] - weights[1][k] for k in weights[0]]
        models = [AuthorDQN(61), AuthorDQN(61)]
        for model, weight in zip(models, weights):
            model.load_state_dict(weight)
            model.eval()
        with torch.no_grad():
            identity = torch.eye(61)
            qs = [torch.stack([model(identity[s]) for s in range(61)]) for model in models]
        records = []
        for d in dirs:
            arrays = []
            for p in sorted((d / "transitions").glob("*.npz")):
                with np.load(p) as z:
                    arrays.append(z["records"])
            records.append(np.concatenate(arrays))
        same = records[0] == records[1]
        mismatch = np.flatnonzero(~same)
        invariance.append(dict(seed=seed, compared_transitions=len(same), equal_transition_rows=int(same.sum()),
            first_different_transition_ID=int(mismatch[0]) if len(mismatch) else None,
            exactly_equal_final_weights=all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0]),
            maximum_absolute_parameter_difference=max(float(d.abs().max()) for d in diffs),
            parameter_difference_L2=float(torch.sqrt(sum((d ** 2).sum() for d in diffs))),
            maximum_absolute_Q_difference=float((qs[0] - qs[1]).abs().max()),
            identical_final_greedy_actions=int((qs[0].argmax(1) == qs[1].argmax(1)).sum())))
    (out / "selected_scale_invariance.json").write_text(json.dumps(dict(
        purpose="All seed pairs, same fixed final budget: does global loss scaling yield exactly identical closed-loop histories?",
        caveat="Differences do not identify optimizer epsilon versus finite precision versus subsequent feedback as separate causes",
        comparisons=invariance), indent=2), encoding="utf-8")

    anchor_text = describe_direction([r["remove_anchor_delta_C_minus_A"] for r in contrasts], "去掉其他动作的锚定（C减A）")
    scale_text = describe_direction([r["scale_objective_delta_B_minus_C"] for r in contrasts], "将单动作损失整体乘四（B减C）")
    passes = sum(r["all_start_arrival_without_invalid_or_revisit"] for r in rows)
    paper_passes = sum(r["both_paper_routes_match"] for r in rows)
    decision = ("本轮没有候选模型通过全部到达检查，不把任何一组视为已经修复的复现模型。"
                if passes == 0 else "到达检查通过的模型仍须与论文路径、剂量和完整训练要求分别核对，不能合并宣称整篇复现成功。")
    names = dict(full_anchor="A 四输出锚定（复用）", selected="B 单动作", selected_quarter="C 单动作÷4")
    table = [f"| {names[r['loss']]} | {r['seed']} | {r['arrivals']}/61 | {r['blocked_starts']} | {r['cycle_starts']} | {r['completed_episodes']:,} | {r['epsilon']:.4f} |" for r in rows]
    paired_table = [f"| {r['seed']} | {r['remove_anchor_delta_C_minus_A']:+d} | {r['remove_anchor_percentage_points']:+.2f} | {r['scale_objective_delta_B_minus_C']:+d} | {r['scale_objective_percentage_points']:+.2f} |" for r in contrasts]
    scale_table = [f"| {r['seed']} | {r['equal_transition_rows']:,}/{r['compared_transitions']:,} | {r['first_different_transition_ID']} | {r['maximum_absolute_parameter_difference']:.6g} | {r['identical_final_greedy_actions']}/61 |" for r in invariance]
    paper_table = [f"| {names[r['loss']]} | {r['seed']} | {r['start1_status']} / {r['start1_steps']} / {r['start1_FED']:.6f} | {r['start2_status']} / {r['start2_steps']} / {r['start2_FED']:.6f} | {'是' if r['both_paper_routes_match'] else '否'} |" for r in rows]

    def link(path, label):
        return f"[{label}]({path.as_posix()})"

    report = f"""## Material Passport

- Origin Skill：academic-research-suite / experiment-agent。
- Origin Mode：run + validate。
- Origin Date：2026-10-15。
- Verification Status：六次新增固定预算训练完成；连同三个复用基线，九个模型独立核查通过。
- Version Label：safestep_loss_components_v1。
- 范围：同一作者酒店案例、每步更新、200,000控制步预算的损失机制比较；不是论文12,000回合训练的完整复现。

# SafeStep 损失函数拆分实验

本轮**新训练六次，复用上一轮已冻结的三个每步更新基线**，合计比较九个模型。每个模型均为200,000环境控制步、199,936次优化更新。没有依据成绩延长训练或挑选检查点。

{anchor_text}

{scale_text}

九个模型中，{passes}个满足61个起点全部到达且无非法动作、无重访的检查，{paper_passes}个同时匹配两条论文推荐路径。到达与路径检查不能自动代表真实火灾安全性或跨建筑泛化。

{decision}

## 1. 为什么设置三组

上一轮发现“每步更新”在早期固定预算下没有比每回合更新更差，仍不足以解释更早正文流程中断时的策略问题。本轮继续缩小问题，固定每步更新，只拆分损失。

令D为批内选中动作的平均TD平方误差，U为批内其他三个动作相对目标网络输出的平方误差之和的平均：

| 组别 | 损失 | 要检验的变化 |
|---|---|---|
| A 四输出锚定 | (D+U)/4 | 与作者公开优化目标相同的共同实现参照 |
| B 单动作 | D | 同时去掉U，并将D的系数变为A的四倍 |
| C 单动作÷4 | D/4 | 保持D的系数与A相同，只去掉U |

主要比较为C对A的“移除锚定”，以及B对C的“整体缩放”。B与A的直接差异同时含两项，不能当成唯一机制解释。

学习率固定0.008，优化器为Adam。**损失／梯度整体乘四不等于参数更新乘四，也不等于学习率乘四。** Adam的一阶矩与二阶矩归一化会抵消恒定正比例缩放的一部分作用；其epsilon、浮点误差及后续轨迹反馈可使两组不能完全一致。本轮做了梯度组成与实际Adam更新检查，而不是仅凭标量损失大小判断。

## 2. 控制条件与复用边界

三个种子为141、142、143。全部使用相同位置输入、61→61→4网络、危险率数据、奖励、批量64、回放1000、折扣0.95及拆分随机数流。超时仍按公开程序自举；目标同步仍只在回合末按超过50步的规则触发；探索率仍在符合预热条件的完成回合末乘0.9995。

不同损失会改变轨迹，因此实际完成回合数、探索率和同步时刻可能随之变化。这里控制的是规则和环境步／优化次数预算，不是强制各组具有相同训练数据或探索率轨迹。结果是损失选择在这个闭环设置下的总效果，不能单独分离每一种中间作用。

A组复用上一轮三个模型，到达数4、10、31事先已知；不是本轮新跑三次，也不是盲测。三个种子全部复用，没有按成绩选择。新训练前登记基线目录与文件哈希；一个5,000步原损失预运行精确重现旧预运行的全部转移、优化日志与最终权重，确认共同执行过程未变。

本轮不修改被冻结的旧训练源文件，在独立进程中临时替换损失计算入口，并强制沿用每步更新调度，退出后恢复入口。六个新增运行在启动前冻结配置、代码、复用来源；每次900秒上限，无自动重试和追加预算。未完成回合的最后一步保留PLAYING标签，预算结束不伪造环境终止。

## 3. 全部九个结果

| 损失 | 种子 | 到达出口 | 原地循环起点 | 多格循环起点 | 完成训练回合 | 结束探索率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

每行前三类起点合计61。循环按独立重建的确定性贪心轨迹核查。失败模型与失败路线全部保留。

| 种子 | 移除锚定：C−A，到达数差 | 对应百分点 | 整体乘四：B−C，到达数差 | 对应百分点 |
|---|---:|---:|---:|---:|
{chr(10).join(paired_table)}

这是三个种子在同一个酒店的描述性成对差，不是九个独立建筑实验，也未做显著性推断。不同目标的原始损失大小不能直接比较，特别不能把C损失更小称为“拟合更好”。

## 4. 整体缩放是否产生完全相同的过程

对B与C的全部三个种子逐条比较200,000个转移，并比较预算终点权重和所有61格的贪心动作：

| 种子 | 完全相同的转移行 | 首个不同转移编号（从0起） | 最终参数最大绝对差 | 相同贪心动作格子 |
|---|---:|---:|---:|---:|
{chr(10).join(scale_table)}

转移行的比较包括回合编号、控制步、状态、动作、奖励、终止和FED；首个差异为None表示全部一致。这个比较能检验严格相同，不能分别量化Adam epsilon、浮点误差与后续反馈各自贡献。即使最后到达数相同，也不代表权重与全部轨迹相同。

另做了一项明确标为事后探索的缓存诊断：使用全部30个事先保存的A组更新上下文，让三种损失分别从相同网络、样本、目标与A组Adam历史矩出发，各模拟一步。30次A更新均精确重现，B的梯度为C的四倍，但B／C实际参数变化L2范数的比值最小为{probe['parameter_step_ratio_min']:.3f}、中位数为{probe['parameter_step_ratio_median']:.3f}、最大为{probe['parameter_step_ratio_max']:.3f}，并非固定四倍。

v2这90个临时单步分支不是六次正式训练之外的新策略训练，也不代表B、C从头训练时各自的Adam历史。缓存探针首版没有隔离各分支的优化器历史张量，跨分支比值无效，其90个分支结果已保留为v1_invalid，不用于结论。修正版本为每个分支深拷贝历史矩，并检查分支初始矩一致、原始矩未变化；本报告仅使用v2。正式训练每个运行有独立进程和优化器，独立核查每个上下文只重放一次，不涉及该探针首版问题。

## 5. 论文两个起点

论文推荐路径的参照是起点1：10步、FED0.1471；起点2：19步、FED0.2300。以下FED保留六位；是否匹配按逐格路径核对，匹配时再核对论文四位小数剂量。

| 损失 | 种子 | 起点1：状态／步数／FED | 起点2：状态／步数／FED | 两路径同时匹配 |
|---|---:|---|---|---|
{chr(10).join(paper_table)}

LOSE是原环境跑到301步的失败路线，其高剂量来自继续积分，不能当成有效疏散路径或真人仍能行动的证据。

## 6. 执行与独立核查

- 三项公式检查通过：完整目标与共同标签一致；C保留A选中动作误差的梯度系数、去掉未选动作锚定；B的选中梯度为C四倍，但示例中Adam首次参数更新近似相同。
- 每个新损失各做两次5,000步预运行，转移、优化日志和最终权重逐位一致；A预运行与旧基线逐位一致。
- 独立核查九个模型共 **{verification['total_control_steps']:,} 个转移**，其中本轮新训练 **{verification['new_control_steps']:,} 个**，其余为复用基线重新核查。
- 核查 **{verification['total_updates']:,} 次优化**、**{verification['total_replay_IDs_checked']:,} 个回放样本编号**，逐次重建回放随机数与窗口。
- 按各自损失独立重放 **{verification['exactly_replayed_Adam_contexts']} 次预先指定的Adam更新**，核对完整上下文、选中预测、标签、标量损失及更新后权重。
- 独立执行 **549条预算最终模型轨迹**，重算路径、FED、循环、非法动作与论文路径匹配。
- 三组配对初始权重及前65条转移相同；旧完成清单的 **{verification['earlier_artifact_hashes_unchanged']} 个文件**哈希未变，三个复用目录的登记文件也再次核对。

这些检查验证计算与执行，不能代替对原论文未知训练细节、物理上游过程或真实行为的验证。

## 7. 当前结论及边界

{anchor_text}

{scale_text}

预算终点表现不能回答更长训练是否退化、是否最终收敛，也不能直接与更早8,000至9,500回合中途模型作相同阶段比较。即使某一损失在这里改善，也只是在公开程序的超时自举、回合末同步等条件下的结果；更早正文流程同时采用了别的终止和同步规则，仍可能存在交互作用。

{decision} 这也不推翻此前作者原权重的数值吻合或已完成的公开程序训练结果，不能据此宣告原论文不可复现。

下一步应依据成对方向决定是否有一个值得确认的机制，而不是同时修改学习率、观测、奖励与终止。若差异不稳定，应先承认目前不能定位唯一原因；若要认定原论文完整终点复现，仍需另行冻结完整训练预算。本轮结束后未自动追加训练。

统计误读检查覆盖11/11类：按种子分列；不外推真人或建筑群体；全部种子和失败保留；不对成功轨迹作条件筛选；列明61起点分母与复用／新跑数量；不挑极端结果重试；不筛显著性或最佳检查点；公开既知基线和配置冻结时间；不把相关机制、梯度缩放与最终性能作唯一因果或反向因果解释。未做显著性检验。

## 8. 可追溯文件

- {link(ROOT / 'schemas/safestep_loss_ablation_20261015.json', '冻结设计与三种损失')}
- {link(out / 'baseline_reuse_before_training.json', '新训练前登记的基线复用来源')}
- {link(out / 'nine_model_comparison.csv', '全部九个模型结果')}
- {link(out / 'paired_contrasts.csv', '移除锚定与整体缩放的成对差值')}
- {link(out / 'selected_scale_invariance.json', '整体缩放的轨迹与权重比较')}
- {link(out / 'common_context_adam_probe_v2.json', '同状态单步Adam探针（v2有效结果）')}
- {link(out / 'verification.json', '独立核查汇总')}
- {link(ROOT / 'reports/SafeStep更新频率单因素实验_六次运行结果_2026-10-15.md', '上一轮更新频率实验')}

作者来源为[SafeStep_v1](https://github.com/hqfang3/SafeStep_v1)，固定提交`{cfg['author_commit']}`。没有LLM请求、收费API或PPO训练，48个旧人群最终场景继续封存。酒店原始权重关键数值复现仍有效；小建筑约62%改善、缺失危险率、FDS与浓度到危险率的上游仍未复现。
"""
    report_path.write_text(report, encoding="utf-8")
    summary = dict(status="SIX_NEW_RUNS_AND_THREE_REUSED_BASELINES_VERIFIED", created_at=datetime.now(timezone.utc).isoformat(),
        contrasts=contrasts, anchor_interpretation=anchor_text, scale_interpretation=scale_text,
        all_start_checks_passed=passes, both_paper_paths_matched=paper_passes,
        new_control_steps=verification["new_control_steps"], total_control_steps_compared=verification["total_control_steps"],
        paper12000_episode_training_completed=False, LLM_requests=0, old_final48_crowd_scenarios_accessed=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in re.findall(r"\]\(([^)]+)\)", report):
        if not path.startswith("https://"):
            assert Path(path).is_absolute() and Path(path).is_file(), path
    files = [p for p in out.rglob("*") if p.is_file()]
    files += [report_path, Path(__file__).resolve(), ROOT / "schemas/safestep_loss_ablation_20261015.json",
              ROOT / "src/fire_evac/safestep_loss_ablation.py", ROOT / "tests/test_safestep_loss_ablation.py",
              ROOT / "scripts/run_safestep_loss_ablation.py", ROOT / "scripts/verify_safestep_loss_ablation.py",
              ROOT / "scripts/probe_safestep_loss_adam_contexts.py", ROOT / "scripts/probe_safestep_loss_adam_contexts_v1.py"]
    manifest = dict(status="LOSS_ABLATION_AND_AUDIT_COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        new_training_runs=6, reused_baseline_runs=3, control_steps_per_model=200000,
        final_model_type="budget_final_not_paper12000_episode_final", post_score_extensions=0,
        old_final48_crowd_scenarios_accessed=False,
        artifact_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))})
    (out / "completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path, expected in manifest["artifact_sha256"].items():
        assert digest(ROOT / path) == expected, path
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("interpretation")}, indent=2))
    print("Verified artifact hashes:", len(manifest["artifact_sha256"]))


if __name__ == "__main__":
    main()
