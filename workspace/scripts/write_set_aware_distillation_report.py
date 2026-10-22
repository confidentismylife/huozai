"""Write a Chinese evidence-linked report from completed pilot artifacts."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/set_aware_distillation_development_v1"
REPORT = ROOT / "reports/集合监督与覆盖不确定性开发实验报告_2026-10-03.md"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def number(x, digits=4):
    return f"{x:+.{digits}f}"


def link(label, path):
    return f"[{label}]({Path(path).resolve().as_posix()})"


def main():
    assert not REPORT.exists(), "preserve completed report"
    s, v, q, f, m = (read(n) for n in ("summary.json", "verification.json", "descriptive_quantification.json", "frozen_students.json", "development_manifest.json"))
    assert v["all_checks_passed"]
    effects = {r["treatment"]: r for r in s["all_paired_effects"]}
    arm_results = {r["arm"]: r for r in s["arm_summary"]}
    main_arms = ("single_ungated", "single_guarded", "set_ungated", "set_guarded", "pure_ppo_601", "teacher_robust3", "legacy_large_rank_ensemble")
    main_table = table(["方案", "最终失能率", "相对601差值，百分点", "探索性95%区间", "改善/变差/持平家庭"],
        [(a, f"{arm_results[a]['final_incapacitation']*100:.4f}%", number(effects[a]["delta_incap_pp"]),
          f"[{number(effects[a]['exploratory_ci_low_pp'])}, {number(effects[a]['exploratory_ci_high_pp'])}]",
          f"{effects[a]['improved_families']}/{effects[a]['worsened_families']}/{effects[a]['tied_families']}") for a in main_arms])
    all_table = table(["全部执行组", "相对601失能差值，百分点", "探索性95%区间", "改善/变差/持平"],
        [(a, number(effects[a]["delta_incap_pp"]), f"[{number(effects[a]['exploratory_ci_low_pp'])}, {number(effects[a]['exploratory_ci_high_pp'])}]",
          f"{effects[a]['improved_families']}/{effects[a]['worsened_families']}/{effects[a]['tied_families']}") for a in m["arms"]])
    factor_table = table(["主失能对比", "差值，百分点", "探索性95%区间"],
        [(r["treatment"] + " − " + r["control"], number(r["delta"]), f"[{number(r['exploratory_ci_low'])}, {number(r['exploratory_ci_high'])}]")
         for r in s["factorial_effects"] if r["metric"] == "incap_pp"] +
        [("门控效果的集合−唯一交互", number(s["incap_interaction"]["delta_pp"]),
          f"[{number(s['incap_interaction']['exploratory_ci_low_pp'])}, {number(s['incap_interaction']['exploratory_ci_high_pp'])}]")])
    gate_table = table(["监督目标", "k", "非601提名", "允许覆盖", "拦截", "被拦有益/有害/持平起点"],
        [(g["objective"], f"{g['k']:.4f}", f"{g['nonbaseline_nomination_states']}/48", f"{g['allowed_override_states']}/48",
          g["blocked_nomination_states"], "/".join(str(v["gate_actual_outcome_descriptions"][g["objective"]]["blocked"][k])
             for k in ("actual_beneficial_states", "actual_harmful_states", "actual_tied_states"))) for g in s["gate_summary"]])
    secondary_table = table(["组", "平均FED", "拥堵累计量", "平均步数", "实际平均放行", "保持阻塞率"],
        [(a, f"{arm_results[a]['final_fed']:.6f}", f"{arm_results[a]['additional_congestion']:.4f}", f"{arm_results[a]['steps']:.4f}",
          f"{arm_results[a]['mean_release']:.4f}", f"{arm_results[a]['hold_block_rate']*100:.2f}%") for a in main_arms])
    type_labels = {"clear": "无额外冲突", "compound_conflict": "复合冲突", "forecast_error": "预测错误",
                   "safe_exit_congested": "安全出口拥堵", "safe_exit_long_block": "安全出口长期阻塞", "safe_exit_short_block": "安全出口短暂阻塞"}
    type_table = table(["冲突类型（每类4家庭）", "唯一无门控", "集合无门控", "robust3教师", "集合有门控"],
        [(type_labels[k], *[number(v["descriptive_effects_by_conflict_type"][k][a]["delta_vs601_pp"])
          for a in ("single_ungated", "set_ungated", "teacher_robust3", "set_guarded")]) for k in sorted(type_labels)])
    training_table = table(["模型", "所选轮次/实际训练轮次", "检查点主遗憾，百分点"],
        [(r["arm"], f"{r['selected_epoch']}/{r['actual_epochs']}", f"{r['best_checkpoint_primary_regret_pp']:.4f}") for r in f["models"]])
    retention_table = table(["组", "收益点估计/教师收益", "学生收益−0.5×教师收益，百分点", "探索性95%区间"],
        [(r["arm"], f"{r['descriptive_point_retention_percent']:.2f}%", number(r["benefit_minus_half_teacher_pp"]),
          f"[{number(r['exploratory_ci_low_pp'])}, {number(r['exploratory_ci_high_pp'])}]") for r in q["teacher_retention"]])
    links = "、".join(link(label, path) for label, path in (
        ("训练前开发方案", ROOT / "reports/集合监督与覆盖不确定性小规模开发方案_2026-10-03.md"),
        ("开发冻结清单", OUT / "development_manifest.json"), ("模型与余量冻结记录", OUT / "frozen_students.json"),
        ("全部结果摘要", OUT / "summary.json"), ("独立复核", OUT / "verification.json"),
        ("补充量化", OUT / "descriptive_quantification.json"), ("逐家庭差值", OUT / "family_paired_deltas.csv"),
        ("门控诊断", OUT / "gate_diagnostics.csv"), ("完整逐回合结果", OUT / "evaluation_episode_results.csv")))
    report = f"""# 集合监督与覆盖不确定性：开发实验结果与研究判断

2026年10月3日。本报告记录已经完成的训练、评估与只读复核；范围是新开发实验。原正式资格阈值95% / 10% / 5% / 95%未改，第三轮正式PPO训练与最终银行未启动。

## 1. 先讲结论

本轮没有建立“集合监督＋经验覆盖门控”优于强固定PPO601的证据。两个有门控方案在48个初始起点都执行601，因此主比较差值为0，逐回合结果完全相同。这个零区间反映了本次行为相同；不能把它解释为已经证明两个不同控制器总体等效，也不能称为门控安全认证。

集合无门控平均失能下降0.1302个百分点，探索性区间为[−0.6772，+0.3386]，3家庭改善、3家庭变差；唯一标签无门控下降0.2315个百分点，区间[−0.6944，0]，全部改善集中在1家庭。集合相对唯一的差值+0.1013个百分点，区间跨零，未建立集合目标的额外优势，也不足以断言其普遍更差。

离线robust3教师在本轮仍有机会：比601下降0.4601个百分点，探索性区间[−0.9115，−0.1301]。教师使用隐藏人口初始化和仿真未来，不能把它的成绩当成可部署学生或LLM的成绩。当前瓶颈更接近“怎样从实际观测判断值得覆盖”，暂不支持扩大同一流程的PPO训练。

## 2. 为什么做这一轮

前一轮发现：同一个初始观测，可能因隐藏人口初始化、随机规划未来而产生不同教师标签；其中不少不同标签只是主指标并列。15个选定冲突家庭的交叉诊断说明两种随机来源和交互都存在；这是诊断，不是控制收益试验。

本轮改两件事。第一，允许主指标足够接近的多个计划都成为监督答案，减少被迫复制唯一并列标签。第二，覆盖601之前扣除模型分歧和旧校准集的经验误差余量，让模型在证据薄弱时退回601。两件事采用2×2对照，便于区分监督损失和门控的影响。

通俗地说，离线教师的反事实模拟给出一组候选疏散方案；学生只看49维观测，先提名一个方案；门控再判断预测改善够不够抵消经验误差。实际执行的是完整闭环候选策略，每一步仍按实时观测控制方向和放行量。本轮只在初始时刻做一次元计划提名，没有实现全过程反复更换元计划。

## 3. 训练、校准、评估怎样隔离

训练标签完全复用原673个缓存状态。84训练家庭592状态用于梯度更新；12个旧校准家庭81状态按每类家庭ID固定拆成6个检查点家庭和6个余量家庭。余量家庭不进入拟合，也不参与选检查点。原校准资料此前已被查看，拆分后的两组仍是开发资料，不能称盲校准。

两个监督目标分别使用801—805五个初始化，共10次新监督训练。相同种子的两目标初始网络哈希相同。网络49→64→64→119，为17计划各输出六个相对601收益和一个分类logit。输入28:33的建议特征清零；所有输入和输出尺度只由训练家庭计算。使用AdamW，学习率0.003，weight decay 0.001，全批，最多300轮，每10轮检查，耐心80轮；选检查点用6家庭初始状态的家庭等权教师主遗憾，并列保留最早。

两组的损失均含主收益MSE、0.1倍其他五字段MSE及分类项。唯一目标复制六字段词典序最优计划；集合目标把主失能距最优≤0.10个百分点的计划都视为可接受，分类损失为`−log(可接受计划softmax质量之和)`。六收益头均减去自身601预测，601收益恰为0。可接受集合只约束主规划失能，不能保证FED、拥堵和效率也近优。

{training_table}

592训练状态中227个（38.34%，按家庭等权为39.63%）允许全部17计划，170个（28.72%）只有一个可接受计划。全17计划可接受时，集合分类项在数学上为0，不提供计划之间的区分；辅助收益回归仍参与更新。这能解释一种信息稀释机制，但本试验没有因果隔离它，不能把结果全归因于这个原因。浮点复核显示float32与原float64标签选择、集合成员均无变化。

每目标五个模型的平均分类建议分数提名；精确并列时按平均六收益及候选索引打破。十个单模型也全部评估，避免以新结果挑最好种子。旧大排序集成、五个固定PPO、robust3教师及实时规则一并报告，共22组。两新目标间网络、输入和训练预算一致；旧学生架构与检查点流程不同，固定PPO的演员使用原34维接口，它们是冻结参考而非新的匹配架构消融。

覆盖余量只用6个余量家庭初始状态。在各家庭的两人口及全部17计划中取最大乐观残差，按80%经验分位数higher方式确定k，下限1。用`L=μ−k×max(σ,0.10个百分点)`；非601提名只有L>0.10个百分点才获准覆盖。σ是五初始化样本标准差，建议softmax、L及k都不是安全概率或有覆盖保证的置信界。六个分数取higher的80%分位数实际取第二大值；没有校准总体80%安全率。

模型、余量和代码冻结后，用种子20262001生成的24个SGDEV家庭进行模拟，六冲突类型各4个。场景参数与旧训练、校准、PDDEV/PRAU/PTDEV、两轮正式训练和最终场景银行精确去重；人口、规划、执行和传感器随机命名空间也与原1137缓存状态去重。

每家庭2人口起点、每起点3独立检验未来。48起点只有24个不同初始观测向量，同家庭两人口观测相同；人口初始化在模拟中不同。这不是48个独立家庭。统计单位始终是24家庭，先平均人口和未来，再对六类型内家庭重采样2000次，固定种子20262002。所有区间均为固定控制器下、未作多重性调整的探索性区间，不是最终确认。

## 4. 新场景里的实际结果

以下差值定义为“处理组失能率−601失能率”，负数代表改善；单位是百分点，不能误写为相对百分比。

{main_table}

601的失能率是3.2205%；集合无门控为3.0903%，唯一无门控为2.9890%。按每1000个模拟人员标准化，点估计分别对应少1.30和2.31个失能事件，教师为少4.60；这不是现实火灾中的救援人数。集合的区间仍容许改善或伤害，唯一的区间上端为0且依赖一个家庭，均不足以建立稳定优势。

{factor_table}

所有有门控控制均等于601，所以“集合有门控−唯一有门控”的零值来自相同执行；交互也不代表已经发现一个新的安全机制。门控相对无门控平均损失了现有点估计收益，没有保留任何覆盖机会。

### 4.1 收益集中在哪里

{type_table}

上表为各类型平均相对601差值。每类型仅4家庭，属于描述。唯一方案全部主收益来自安全出口拥堵家庭SGDEV0004：该家庭平均改善5.5556个百分点，除以24后形成总体0.2315个百分点。集合方案同时包含该收益和其他收益、伤害：SGDEV0012/0013改善0.0694/2.5694个百分点；SGDEV0019/0020/0024变差1.3194/0.6250/3.1250个百分点。不能用总体均值掩盖这些伤害。

### 4.2 保留了多少教师机会

本表是事后补充描述；没有用它重新训练或调整资格线。直接构造每家庭“学生收益−0.5×教师收益”并重采样，避免仅凭比值点估计宣称达到50%。

{retention_table}

唯一无门控的比值点估计约50.31%，但对应差值区间跨零，未证明至少保留一半。集合无门控为28.30%；两个门控方案为0%。教师收益本身也是本轮小型开发证据，比例不能成为现实安全或正式资格指标。

## 5. 为什么门控全部退回601

{gate_table}

唯一目标k=8.5025，集合k=10.7085。非601提名的L范围分别是[−17.5301，−7.3009]和[−37.1351，−1.4893]个百分点，全都低于固定的+0.10百分点覆盖门槛。这说明当前预测和经验误差余量组合没有产生任何合格覆盖。

唯一方案拦下12个提名：其中2起点执行结果有益、0有害、10持平。集合方案拦下26个：5有益、6有害、15持平。起点有益/有害按各自3检验未来的平均失能相对601判断；同家庭起点相关，不是独立安全事件。这说明门控既挡住了伤害，也挡住了机会，不能称已经学会准确区分。

还有一个模型语义问题：分类提名与收益回归是两套输出。唯一/集合非601提名的平均预测收益最低分别为−0.8391/−5.0766个百分点，仍然可能被分类头提名。这与“只在值得时覆盖”的需求不一致。全17候选最大残差的余量也可能由很少被提名的候选主导。本轮不能判断哪一原因占多少，不能仅靠降低k解决。

按固定方案，即使门控完全退回也不修改k、容差或门槛。本次没有看结果后放松门控，没有重新挑模型。

## 6. 主失能之外的指标与初始化变化

{secondary_table}

所有组最终剩余率为0，但已经失能的人仍计入失能率，因此不能把“无剩余”当“全部安全撤离”。FED是当前简化模拟中的累计危险暴露输出，不能直接换算现实中毒风险。

集合无门控FED比601高0.008652；它相对唯一高0.008972，探索性区间[−0.002168，+0.020506]。集合无门控拥堵累计量15.0802，601为13.3995，平均步数27.8472 vs 27.5694。没有建立主收益与次级指标一起改善。

集合五个单初始化的相对601效果范围为−0.3414至+0.0463个百分点；唯一为−0.2315至0。集合804在检查点集主遗憾最好（0.2315），新评估却出现+0.0463的伤害点估计。说明选检查点的经验成绩不能保证新场景成绩。所有单模型均见文末表，没有挑803重新命名为主结果。

五个PPO参考602—605在本轮整体比601更差，但在个别家庭仍可能是合适候选；601是此前已固定的强参考，没有使用本轮结果重新挑控制。不能用击败弱参考来替代击败601。

## 7. 实际预算、核查与文件保全

训练10个监督模型；新增PPO训练0、新LLM调用0、新训练标签模拟0。新增4896个规划回合、933个去重实际执行回合，共5829个完整回合，低于7344上限。22组产生3168条逻辑执行记录，其中同计划/未来复用已有结果，没有冒充3168次独立模拟。新模拟采集阶段耗时144.21秒；这不含训练、复核和写报告时间，不能当全部研究成本。

记录25275步实际动作，其中1839步保持阻塞、275步中性方向但正放行。中性正放行实际联合动作−1不属于策略请求的九动作之一；复核保留它，没有错误记成“等待”。本次实际执行轨迹、逐回合首请求/首实际动作、平均放行、等待请求和保持阻塞率全部重算一致。

独立核查重新加载十个实际检查点、重算输入/收益标准化、初始网络哈希配对、检查点选择、余量、22组选择、所有同计划复用、人口守恒、全部主/因素Bootstrap与原冻结来源。{v['source_inputs_unchanged']}个输入哈希未改，8项新增目标测试已通过；没有重跑无关的全项目测试。

独立复核第一次在最终JSON序列化时遇到NumPy整数不能写入的问题，所有计算断言已运行至写出阶段；保存失败版本后仅修复复核器标量序列化，再次只读运行通过。训练器、分析器、协议、模型和模拟缓存没有为此修改，没有重复训练或模拟。

本报告及新输出独立保存，旧报告、模型、缓存与失败记录保留。结束时另存本轮前状态精确字节，核对历轮完成清单和历史完整报告，再更新当前开发状态。最终的保全证明见同目录completion_manifest.json。

## 8. 继续研究的投入判断

本轮的作用是排除“集合标签本身就能稳定解决问题”和“加入保守门控即可保留收益”这两个过强推断。当前设置的教师仍有可利用的机会，但现有学生只学到不稳定、集中于少数家庭的覆盖；门控全回退。这个负结果值得保留，继续重复同一训练并不能回答新问题。

下一轮建议先处理提名与覆盖的一致性：定义相对601的“是否值得覆盖”和“覆盖哪个计划”，在训练家庭内部做交叉拟合，使置信/余量学习看到未参与拟合的预测误差；再用独立校准家庭拟合固定的提名条件余量。全计划最大残差仍可作为保守诊断比较，不能看到本轮全阻塞就任意改成更宽松规则。后续每项改动应先保存预算、判断标准和隔离规则，再生成新开发数据。

先回答“能否拒绝明显有害覆盖，同时允许一部分真正有益覆盖”，再考虑PPO微调。若仍不能从观测稳定识别，应转向补充可观测人口/队列信息或研究建议失效机制，而非无期限扩算。当前初始元计划评估没有解决动态重规划问题，后续方法也需要单独验证。

本轮没有验证LLM独立贡献，没有行为模型/观测噪声/地图迁移，没有真实火场物理或现实人群验证。教师新开发优势不等于总体优势；24家庭并非功效论证支持的确认样本。原0.20个百分点是探索投入目标，集合点估计未达到，但区间容许达到，不能据此认定路线已经无效。正式第三轮继续暂缓，本轮结果不用于正式冻结通过。

## 9. 全部执行组与复现入口

{all_table}

资料：{links}。

代码：{link('训练及新评估入口', ROOT / 'scripts/run_set_aware_distillation.py')}、{link('独立复核入口', ROOT / 'scripts/verify_set_aware_distillation.py')}、{link('事后描述量化', ROOT / 'scripts/describe_set_aware_distillation.py')}。

只读重算命令（没有新训练或新模拟）：

```powershell
& 'D:/miniconda3/python.exe' scripts/verify_set_aware_distillation.py
& 'D:/miniconda3/python.exe' scripts/describe_set_aware_distillation.py
```

完整试验已运行完，不需再次执行训练入口。冻结代码、训练曲线、模型、48个NPZ原始缓存和独立核查均在本轮输出目录。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(str(REPORT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
