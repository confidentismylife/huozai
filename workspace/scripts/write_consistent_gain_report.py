"""Report cached semantic repair and its limitations, without new simulation."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/consistent_gain_crossfit_development_v1"
REPORT = ROOT / "reports/统一提名与覆盖收益交叉拟合诊断报告_2026-10-03.md"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def link(label, path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def main():
    assert not REPORT.exists()
    s, v, m, frozen = (read(path) for path in ("summary.json", "verification.json", "development_manifest.json", "frozen_models.json"))
    assert v["all_checks_passed"]
    lookup = {(r["block"], r["arm"]): r for r in s["summary_rows"]}
    def results(block, arms):
        return table(["方案", "平均缓存规划收益，百分点", "相对缓存教师主遗憾，百分点", "601起点", "有益/有害/持平覆盖"],
            [(a, f"{lookup[block,a]['cached_primary_gain_pp_family_equal']:+.4f}", f"{lookup[block,a]['cached_primary_regret_pp_family_equal']:.4f}",
              f"{lookup[block,a]['selected_601_states']}/{lookup[block,a]['states']}",
              "/".join(str(lookup[block,a][k]) for k in ("beneficial_override_states", "harmful_override_states", "tied_override_states"))) for a in arms])
    old_arms = ("pure_ppo_601", "single_class_ungated", "single_gain_ungated", "single_class_guarded", "single_gain_consistent",
                "set_class_ungated", "set_gain_ungated", "set_class_guarded", "set_gain_consistent")
    oof_arms = ("pure_ppo_601", "ridge_gain_ungated", "ridge_gain_consistent", "cached_teacher_primary")
    available = table(["SGDEV诊断方案", "已有实际执行缓存的起点", "缺失起点"],
        [(a, f"{lookup['viewed_sgdev_initial',a]['actual_execution_cache_available_states']}/48",
          48-lookup['viewed_sgdev_initial',a]['actual_execution_cache_available_states'])
         for a in ("single_gain_ungated", "set_gain_ungated", "single_gain_consistent", "set_gain_consistent", "ridge_gain_ungated", "ridge_gain_consistent")])
    folds = table(["拟合", "拟合家庭", "隔离家庭", "拟合状态", "隔离OOF状态"],
        [(r["name"], len(r["fit_family_ids"]), len(r["heldout_family_ids"]) if r["name"] != "final" else "12个外部校准家庭（未拟合）",
          r["fit_states"], r["heldout_states"]) for r in frozen["models"]])
    source_links = "、".join(link(label, path) for label, path in (
        ("执行前方案", ROOT / "reports/统一提名与覆盖收益交叉拟合开发方案_2026-10-03.md"),
        ("冻结开发清单", OUT / "development_manifest.json"), ("模型及余量", OUT / "frozen_models.json"),
        ("结果摘要", OUT / "summary.json"), ("完整逐起点诊断", OUT / "diagnostic_records.csv"),
        ("家庭隔离OOF预测", OUT / "crossfit_predictions.npz"), ("独立复核", OUT / "verification.json"),
        ("统一决策实现", ROOT / "scripts/consistent_gain_policy.py"), ("执行入口", ROOT / "scripts/run_consistent_gain_crossfit.py")))
    report = f"""# 统一提名与覆盖收益：代码修正、交叉拟合与缓存诊断

2026年10月3日。已完成本轮；这是已有缓存上的开发诊断，没有新增火灾模拟、神经网络或PPO训练、LLM调用。新拟合只有5个轻量线性收益模型。旧资料和资格阈值保留。

## 1. 本轮解决了什么，还有什么没解决

提名与覆盖已统一使用同一个分数：每个候选的相对601预测收益先扣误差余量，取最大分数，再用该分数判断是否严格超过+0.10个百分点。601的收益与最终分数固定为0，精确并列优先601。主路径不再先按独立分类概率提名，再用另一个数决定覆盖。

独立复核检查了576条统一决策记录：所有非601执行都满足同一分数>0.10个百分点。旧分类无门控在检查点与SGDEV资料中共有14条“预测收益为负但仍提名非601”的记录；新收益排名消除了这一语义冲突。此修正只写入新模块和新输出，旧控制器与缓存没有覆盖。

性能问题仍未解决。家庭隔离预测中的线性原型，统一有门控30次覆盖包含8次有益、4次有害、18次持平；有害发生在3家庭。在已查看SGDEV中，新原型允许6次覆盖，按缓存规划主结果全部持平。旧神经网络收益头改成统一排名后依然全部回退601。不能把逻辑一致当成准确预测，更不能称已经提升实际疏散安全。

## 2. 通俗解释新的决策

旧流程有两套意见：分类头说“选A”，收益头却可能说“A比601差”。门控再拦截A时，可能直接回到601，忽略另一个有较好扣余量分数的B。

新流程只用一把尺。例如A预测改善2个百分点，但扣除2.5个百分点经验误差后为−0.5；B预测改善1个百分点，扣除0.3后为+0.7。新流程会按+0.7选择B，并用同一个+0.7与+0.10的判据比较。数字仅为示意，不是本轮观测结果。

代码执行规则为：

```text
score[p] = predicted_gain_vs601[p] - penalty[p]
score[601] = 0
nominee = argmax(score)  # 同分优先601
execute = nominee if score[nominee] > 0.10个百分点 else 601
```

零分不是人为给601加优势，而是相对601收益的定义。601不应该因为自身预测误差余量被赋予负的“相对自身收益”。低于阈值的候选仍会回退；不会为增加覆盖数量降低原判据。

## 3. 两部分诊断怎样区分

第一部分复用上轮10个冻结神经网络的收益预测，每监督目标5个。保留原k、样本标准差和0.10个百分点尺度下限；只将分类提名改为收益排名。无门控取平均主收益最大计划；统一门控按每计划`μ−k×max(σ,0.001)`先排名、再用同一数覆盖。这样可以比较“相同预测下改变提名”的后果，不涉及重新训练。

第二部分用线性原型做家庭交叉拟合。采用49维观测，建议特征28:33清零；目标是原规划primary(601)−primary(plan)，17个主收益，601恰为0。不保留独立分类头，不拟合其他五个次级字段。岭惩罚固定0.01，float64计算；截距不惩罚，输入标准化只取拟合家庭，家庭等权且同家庭各状态等权。没有搜索超参数或检查点。

84训练家庭、592状态按6冲突类型各14家庭，用固定种子20262101随机排列后轮转分四折。同家庭人口和时刻一起隔离。

{folds}

4折产生592个完全不在对应模型拟合中的OOF预测，主诊断取168个初始状态。第5模型用全部84家庭拟合，用于外部旧校准和已查看SGDEV诊断。交叉拟合隔离的是家庭，没有使先前已经查看过的数据变成新盲数据。

## 4. 余量怎么得到，为什么仍不能保证安全

固定线性提名规则后，OOF在每家庭两个初始人口状态中取被提名计划的最大乐观误差。84家庭分数的80%经验higher分位数得到1.7557个百分点。最终模型另在6个余量家庭计算相同误差，经验分位数为0，按方案下限计为0.10个百分点。

原型使用两者及下限的最大值：δ=1.7557个百分点。所有非601预测先扣同一δ，601不扣；覆盖仍需扣后分数严格超过0.10个百分点。旧两个神经网络门控k=8.5025/10.7085完全保留。

六个余量家庭中只有1家庭产生非601提名，其最大乐观误差为3.6248个百分点；其余5家庭提名601，误差为0。80% higher分位数没有取唯一最大值，因此得到0。这说明这六家庭对“真正提名非601时是否可靠”的信息很少，不能把0分位数解释为模型误差很小或覆盖80%安全。OOF较大余量提供了额外保守项，但仍没有安全保证。

OOF预测确实排除了当前家庭的拟合数据；然而共同δ包含这些OOF误差，所以**有门控OOF结果是复用校准资料的描述，并非完整隔离的交叉验证性能估计**。无门控OOF提名没有使用本家庭标签调整；本轮不以这些结果宣称确认性优势。

## 5. 家庭隔离预测的结果

以下“收益”均为缓存规划主收益，正数代表比601的最坏相容世界平均规划失能低；“遗憾”是距缓存候选库主最优的差。它们不等于独立实际执行的疏散收益。

{results('train_oof_initial', oof_arms)}

无门控覆盖70/168起点：15有益、12有害、43持平。加入固定余量后覆盖30/168：8有益、4有害、18持平。允许覆盖减少，缓存有害覆盖也减少，但有益覆盖也少了；不是无损安全过滤。统一门控的平均缓存规划收益只有+0.0835个百分点，不应拿它与上一轮实际失能改善混作同一个指标。

有害覆盖家庭为PREX0014、PREX0044、TSDEV0002。它们的状态均未进入对应OOF模型拟合，说明这不是只靠移除训练内预测偏差就能解决的问题。这里没有估计现实伤害概率、统计显著性或正式安全合格率。

## 6. 相同旧预测下改变提名是否更好

这是已查看SGDEV的48初始状态、24家庭的缓存规划比较：

{results('viewed_sgdev_initial', old_arms)}

单标签旧分类提名平均缓存收益+0.2980，换为主收益排名后+0.2025；集合旧分类为+0.3356，换为收益排名后+0.1157。收益排名虽然排除了预测为负的提名，但没有增加这里的缓存规划收益。两组统一有门控全部回到601，平均缓存收益0。

因此，直接删掉分类头或改成收益argmax不能视为已完成性能优化。现有收益预测在候选间的排序和误差仍不够准确；分类头可能在某些情形提供有用选择，而两头又会产生冲突。本轮能验证一致决策逻辑，尚未解决怎样学到可靠的统一收益值。

在6个旧检查点家庭中，线性无门控仅2/12起点覆盖、两者有益，平均缓存收益+0.5787；有门控12/12回退。旧数据量小且已查看，不能据此选模型或改余量。

## 7. 原型在SGDEV与实际结果缺失

{results('viewed_sgdev_initial', ('pure_ppo_601', 'ridge_gain_ungated', 'ridge_gain_consistent', 'cached_teacher_primary'))}

原型无门控在14个起点覆盖：2有益、0有害、12持平，平均缓存收益+0.1042。原型统一有门控覆盖6个，全部缓存主结果持平；平均收益0。它没有因为统一逻辑而释放出有益机会。

SGDEV存了所有17计划的规划结果，实际执行只覆盖原实验曾提名过的部分计划。新选择的实际结果并不总有缓存：

{available}

本轮只统计覆盖率，没有把缺失结果填0，没有只对有缓存的子集报告平均实际改善，也没有暗中补跑火灾模拟。即使某组缓存全部可用，这一轮也不把已查看同一银行解释为新验证；完整实际收益需另行冻结小预算新场景方案。

“cached_teacher_primary”只按主规划指标和601优先同分规则取最优，用于计算主遗憾；它不同于上一轮使用六字段词典序的robust3教师策略，主指标并列时可能选不同计划。不能用本表+0.9983规划收益替换上一轮教师−0.4601实际失能差值。

## 8. 代码与资料核查

新增决策模块10项测试通过。独立复核重新加载5个线性模型，检查其正规方程残差而没有再次拟合；标准化、家庭权重、四折分配、592个OOF预测、余量与1392条逐起点策略记录全部可重算。576条一致决策都满足同分数排名/覆盖判据；原十个神经网络也重新加载核对。

{v['source_inputs_unchanged']}个冻结输入哈希未改。新增模拟0、神经网络训练0、PPO训练0、LLM调用0。旧报告、模型、原SRC、缓存和阶段历史保全；当前状态另存上一阶段精确字节后更新。本轮拟合的是5个线性原型，不能把“没有PPO训练”误写为“完全没有拟合”。

## 9. 下一项值得做什么

应先补足“值得覆盖”的预测与校准证据，而非扩大PPO训练。目前六余量家庭仅1个提供非601提名样本；OOF中仍有4次有害覆盖，新规则在SGDEV未保留有益覆盖。下一方案可以在训练家庭OOF上区分基线、值得覆盖和预测失效，校准时报告真正被提名的覆盖家庭数，提前确定补样与预算，不能看到这里的δ较大就临时降低它。

若后续模型在隔离诊断中产生值得验证的覆盖，再固定统一规则，用全新小型开发银行补齐实际执行，并同时报告失能、FED、拥堵和覆盖机会；不以缓存规划成本代替实际结果。当前不启动正式第三轮，不改变95% / 10% / 5% / 95%资格门槛，也没有建立LLM独立贡献。

本轮的明确进展是把决策语义修正并验证，而不是已经找到了稳定提升方法。负结果保留，后续工作由可观测收益识别和充分校准样本决定。

可复现文件：{source_links}。

只读复核命令，不会新增模拟或拟合：

```powershell
& 'D:/miniconda3/python.exe' scripts/verify_consistent_gain_crossfit.py
```
"""
    REPORT.write_text(report, encoding="utf-8")
    print(str(REPORT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
