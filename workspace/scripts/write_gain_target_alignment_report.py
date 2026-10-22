"""Save a source-grounded Chinese diagnostic report and exportable figure."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/gain_target_alignment_diagnostic_development_v1"
REPORT=ROOT/"reports/收益目标对齐与十二未来复核诊断报告_2026-10-03.md"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(p,label):return f"[{label}]({Path(p).resolve().as_posix()})"
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    assert not REPORT.exists() and not (OUT/"report_assets_manifest.json").exists()
    s=read(OUT/"summary.json");v=read(OUT/"verification.json");m=read(OUT/"development_manifest.json")
    assert v["all_checks_passed"]
    rows=s["targeted_families"];n=len(rows);y=np.arange(n)
    fig,ax=plt.subplots(figsize=(10.2,5.6))
    actual=np.array([r["actual_gain_pp"] for r in rows]);lo=np.array([r["future_only_ci_low_pp"] for r in rows]);hi=np.array([r["future_only_ci_high_pp"] for r in rows])
    ax.errorbar(actual,y-.18,xerr=np.array([actual-lo,hi-actual]),fmt="o",color="#0072B2",capsize=3,label="Actual source, 12 futures (future-only 95% interval)")
    ax.scatter([r["predicted_gain_pp"] for r in rows],y+.18,marker="x",s=52,color="#D55E00",label="Frozen prediction")
    ax.scatter([r["robust_gain_pp"] for r in rows],y+.04,marker="s",s=35,facecolors="none",edgecolors="#009E73",label="Difference of worst-world losses (R)")
    ax.scatter([r["minimum_world_gain_pp"] for r in rows],y-.04,marker="^",s=35,color="#CC79A7",label="Minimum paired world gain (M)")
    ax.axvline(0,color="#555555",lw=.8);ax.axvline(.1,color="#999999",lw=.8,ls=":")
    ax.set_yticks(y,[r["family"] for r in rows]);ax.invert_yaxis();ax.set_xlabel("601 incapacitation minus proposal incapacitation (percentage points)")
    ax.set_title("Targeted replication on eight viewed development families")
    ax.grid(axis="x",alpha=.2);ax.legend(fontsize=8,loc="lower right")
    fig.tight_layout();fig.savefig(OUT/"targeted_gain_comparison.png",dpi=200);fig.savefig(OUT/"targeted_gain_comparison.svg");fig.savefig(OUT/"targeted_gain_comparison.pdf");plt.close(fig)
    groups={(g["block"],g["scope"]):g for g in s["cached_groups"]}
    cal=groups["calibration","eligible_proposals"];test=groups["test","eligible_proposals"]
    rep=table(OUT/"replication_state_results.csv")
    meaning_conflicts=[r for r in rep if float(r["robust_gain_pp"])>.1 and float(r["minimum_world_gain_pp"])< -1e-10]
    assert len(meaning_conflicts)==2 and {r["family"] for r in meaning_conflicts}=={"WCALP0020"}
    text="""# 收益目标对齐与十二未来复核诊断报告

2026年10月3日。本轮已完成；这是已查看开发场景的诊断和定向复核，不是最终验证。数值来自冻结缓存和本轮新增完整模拟。

## 结论先讲清楚

现在有更多证据支持：问题不只是门控太严格，也不只是上轮只抽了3个随机未来。当前线性收益预测在有收益的长时间封闭场景上低估，在两个高分持平对照上高估。新增12未来后，四个收益较大的长封闭家庭仍有益，原有害家庭仍有害；两个高分持平对照仍没有失能改善。单独降低收益门槛不能修复“持平高分、有益低分”的排序；联合分类器能否可靠筛除伤害，还需要新的独立证据。

另一个需要明确的语义是：分别取两策略的最坏世界损失再相减，并不等于每个世界都改善。本轮发现一个短封闭家庭在真实世界略有益，但另一个世界略有害。它可以满足“最大失能损失有所下降”，却不满足“所有支持世界都不伤害”。这是两种不同的优化目标，不能互换解释，也不能仅凭这个诊断将既有算法称为实现错误。

本轮没有重训、没有改阈值、没有启动正式第三轮。支持的是继续做一次受限的目标与拟合验证；尚不支持扩展PPO训练、宣布稳定安全收益或宣称LLM带来独立增益。

## 1. 为什么需要这一步

上一轮补了30个提案校准家庭和24个自然生成检验家庭。条件余量由1.7557升到3.3954个百分点；分类器与收益门控交集为0，实际主差值为0。两个长封闭自然检验家庭的有益建议均被挡住。分类器只是额外拒绝条件，无法弥补收益预测偏低；原有害案例的有益分类分约0.645，未过0.80阈值，也不能把分类器argmax解释为安全接受。

本轮要区分三个解释：模型未学好它自己的规划目标、规划目标与真实源结果不同、少量随机未来让标签不稳。区分后再决定应改标签、改拟合，还是增加标签抽样精度。

## 2. 固定了什么，实际花了多少

原17个候选计划及冻结PPO601—605、GainRidge、分类器、两个余量和所有阈值保留。真实源只用于结果归因。完整复核复用原人口起点，但随机未来全部新取；不增加新场景、不重新提名。

先固定保存方案和源码哈希，随后运行分析。全量分解108起点，其中78起点有非601提案；自然检验区块的30个601提案没有从记录删除，也没有放入提案误差平均值。8个定向家庭选自上一轮结果：全部6个非持平家庭，再加校准和自然检验区块各1个最高预测分的持平提案家庭。该选择使用过结果，因此不能代表自然场景出现率。

每家庭2人口×2源世界×2策略×12未来，共768完整回合；原真实场景另跑4回合，逐指标核对源世界复用结果。实际772，上限800。没有任何新增拟合、PPO训练或LLM调用。新未来共192个种子，世界与策略内共享，人口间分离；传感器种子也与旧记录和新未来互不重叠。共享种子不能保证策略分支中每个个体都收到同一个随机扰动。

独立只读验证核对108个分解、16个复核起点、768个复核回合、4个源场景审计回合、9895个真实源执行步；877步被指示保持规则阻塞，中性指示正放行为0。1513个冻结输入哈希保持一致。数学测试8项通过，覆盖世界聚合语义、恒等式、单位、支持集和配对抽样。没有用验证器重跑模拟。

## 3. 用一个等式拆开误差

P是模型预测收益；R是两策略各自最坏世界平均损失之差；T是已知真实火源世界的规划3未来收益；A是独立实际3未来收益。全部收益为“601失能率−提案失能率”，正数有益，单位为百分点。

**P−A＝(P−R)+(R−T)+(T−A)。**

例如P低于R，表示模型连其训练目标都预测偏低；R与T不同，表示火源世界聚合改变了收益数值；T与A不同，表示两组随机未来结果不同。该恒等式仅是算术分解，不能把一项绝对误差占比解释为因果贡献。正负差额可能相互抵消，各项绝对均值不需要加成总绝对误差。

| 原区块，仅非601提案 | 起点/家庭 | P平均 | R平均 | T平均 | A平均 | 对R的MAE | 世界聚合差额绝对均值 | 随机未来差额绝对均值 | 对A的MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for label,g in (("条件富集校准",cal),("原自然检验，现已查看开发数据",test)):
        text+=f"| {label} | {g['states']}/{g['families']} | {g['predicted_gain_pp']:.4f} | {g['robust_gain_pp']:.4f} | {g['true_planning_gain_pp']:.4f} | {g['actual_gain_pp']:.4f} | {g['mean_abs_prediction_to_robust_pp']:.4f} | {g['mean_abs_robust_to_true_pp']:.4f} | {g['mean_abs_true_to_actual_pp']:.4f} | {g['mean_abs_total_error_pp']:.4f} |\n"
    text+="""
家庭等权，家庭内人口等权。校准区块人为富集提案，不和自然区块合并推算总体效果。在这两组已查看提案中，模型对R的误差绝对均值大于世界聚合差额和未来样本差额。这支持先检查预测质量，不能据此排除模型误设、人口变化或未来数不足。

上轮108状态中，未观察到R>0.10而某世界内收益为负的状态；12未来定向复核后出现2状态。有限抽样增加后能发现新的语义冲突，所以“上轮没看见”不等于不存在。

## 4. 十二未来复核：机会与高估都还在

下表每项为两个人口平均；R与M使用本轮同一批12未来。M是各世界配对收益的最小值。区间只重抽12个未来ID，固定家庭、两个原人口和策略，不是跨场景置信区间，也没有多重比较调整。

| 家庭 | 固定提案 | P | 原实际3未来 | 新实际12未来 | 未来抽样95%区间 | 新R | 新M | 结果 |
|---|---|---:|---:|---:|---|---:|---:|---|
"""
    cn={"beneficial":"有益","harmful":"有害","neutral_or_small":"持平/微小"}
    for r in rows:
        text+=f"| {r['family']} | {r['proposal']} | {r['predicted_gain_pp']:.4f} | {r['original_actual_gain_pp']:.4f} | {r['actual_gain_pp']:.4f} | [{r['future_only_ci_low_pp']:.4f}, {r['future_only_ci_high_pp']:.4f}] | {r['robust_gain_pp']:.4f} | {r['minimum_world_gain_pp']:.4f} | {cn[r['replicated_class']]} |\n"
    text+="\n"+f"![固定提案预测与定向复核收益]({(OUT/'targeted_gain_comparison.png').resolve().as_posix()})\n"
    text+="""
四个较大改善的长封闭家庭是WCALP0087、WCALP0129、WTEST0003、WTEST0021。它们两个人口标签一致，前6和后6未来也均为有益。新R分别约7.2569、2.2917、1.5972、2.3611个百分点，明显高于P的0.1710、0.9655、0.3628、0.7531：低估不仅出现在真实世界收益上，也出现在模型原本要拟合的R目标上。

WCALP0068和WTEST0017是预先选定的高分持平对照：P分别6.2804、3.5264，新实际收益均0，新R也均0。这里“持平”仅指最终失能指标，不能推广为FED、拥堵等所有指标等效。

WCALP0165的wait6_then_live提案仍有害：新实际平均−0.6250，未来抽样区间[−0.9028,−0.3646]。模型却预测+1.6993。这继续提供明确的负例，但原/新门控本来均未接受它，不能称为门控已执行的安全事故。

短封闭WCALP0020改善较小：新实际+0.2257，区间下端约0.0868，跨越0.10的有意义收益界；前6未来+0.4167，后6仅+0.0347，2个人口分半标签都不一致。它提示小收益仍受抽样精度影响，应与四个较大收益家庭分开解释。

总体16起点标签变化为：10个原有益仍有益、2个原有害仍有害、4个原持平仍持平。标签按点估计判定，不能把“全部一致”当作真实期望标签已知；尤其短封闭案例区间和分半结果并不稳定。

## 5. “最坏世界更好”与“每个世界不变坏”

令L601(w)、L提案(w)为世界w中的12未来平均失能损失：

R＝max_w L601(w)−max_w L提案(w)；M＝min_w[L601(w)−L提案(w)]。

R衡量最大损失的改善，两次max可以选不同世界；M要求同一世界内作对比，然后取最差对比。二者都能服务于明确的研究目标，但用R训练分类器后，不能将其“有益”标签解释为所有可能世界均不伤害。

具体案例WCALP0020的两人口：

| 起点 | 新R | 实际源收益 | 世界0收益 | 世界1收益 | 新M | 601/提案最坏世界 |
|---|---:|---:|---:|---:|---:|---|
"""
    for r in meaning_conflicts:
        text+=f"| {r['state_id']} | {float(r['robust_gain_pp']):.4f} | {float(r['actual_gain_pp']):.4f} | {float(r['world0_gain_pp']):.4f} | {float(r['world1_gain_pp']):.4f} | {float(r['minimum_world_gain_pp']):.4f} | {r['baseline_worst_source']}/{r['proposal_worst_source']} |\n"
    text+="""
这两个状态R超过0.10，但M为负。当前真实源收益有利，另一个支持世界有轻微负收益。该证据来自2起点且每世界12未来，不能直接当成已知期望伤害；它已经足够要求下一版明确标签语义并保留世界内配对反例。

## 6. 门控为什么依然没有收益

没有任何阈值调节，固定门控在这16定向起点上：

| 条件 | 接受起点 | 新实际标签 |
|---|---:|---|
"""
    labels={"prior_guard_accepts":"原1.7557余量门控","conditional_guard_accepts":"新3.3954余量门控","filtered_guard_accepts":"新余量＋价值分类器","worth_condition_passes":"仅分类器条件，诊断用"}
    for r in s["fixed_gate_diagnostics"]:
        counts="、".join(f"{cn[k]}{n}" for k,n in r["replicated_classes"].items()) or "无接受"
        text+=f"| {labels[r['gate']]} | {r['accepted']} | {counts} |\n"
    text+="""
原门控只捕捉小收益短封闭案例和两个持平对照；新门控只允许两个持平对照。分类器条件单独允许WCALP0087和WCALP0129的4个有益起点，但它们的低收益预测未过余量门槛；两者取交集依然为0。此处接受数是定向复核状态诊断，不能取代上一轮24自然家庭的主效应估计。

这个证据解释了“继续加一个过滤器”为何没有立刻奏效：过滤器能拒绝部分高估，但无法提升有益提案的低预测值。简单扩大PPO训练也不直接解决这个预测与标签问题。

## 7. 对论文方向与下一步的决定

当前继续研究的价值在于明确了可检验的失败模式：有意义机会并非全是3未来偶然波动；预测却同时存在机会低估和持平高估。值得再做一次小预算、严格隔离的目标拟合验证，暂缓大规模训练。不能从本轮得出稳定总体改善或发布安全门控。

下一轮应先把预测问题独立出来，固定候选与提名机制，比较原R标签与世界内配对收益标签，并直接保存独立实际未来的收益作外部检验。预先区分“降低最大损失”和“所有支持世界均无负收益”，不要把两种语义都塞进一个模糊的有益标签。训练、校准与检验使用全新家庭且互不交叉；本轮8家庭只作为已查看反例，不用于后续最终检验。

拟合方面优先检查一个小模型能否区分长封闭收益、清晰持平和等待伤害，配合零预测参考、家庭等权MAE、有益召回和实际有害接受记录；不先搜索更多模型或降低当前门槛。未来标签需要给出抽样波动，不能仅用单次正负号。当真实源不可由当前观测区分时，模型应预测基于可观测信息的条件收益，不能输入真实源或用事后最有利世界替代可部署预测。

本轮较大收益的建议均为其他冻结纯PPO种子（602或603），仍然主要说明不同冻结策略之间存在可选择的机会。它未证明LLM建议比规则或多种子PPO选择更好。恢复LLM+RL论文主张前，还需要相同信息、候选和预算下的LLM消融、跨人群/观测噪声检验以及正式独立场景评估；本轮不承担这些检验。

95% / 10% / 5% / 95%资格线保持不变，正式第三轮训练与最终银行未启动，原源码、模型、协议、报告及失败文件保留。

## 8. 可复核文件

"""
    for path,label in ((m and ROOT/"reports/收益目标对齐与十二未来复核开发方案_2026-10-03.md","运行前固定方案"),
        (OUT/"development_manifest.json","场景、选择理由、种子和冻结输入"),(OUT/"cached_decomposition.csv","108起点误差分解"),
        (OUT/"replication_state_results.csv","16起点12未来复核"),(OUT/"replication_family_results.csv","8家庭汇总与区间"),
        (OUT/"summary.json","结构化结果"),(OUT/"verification.json","独立验证"),
        (OUT/"targeted_gain_comparison.pdf","可导出图PDF"),(ROOT/"scripts/run_gain_target_alignment.py","执行脚本"),
        (ROOT/"scripts/verify_gain_target_alignment.py","只读验证脚本"),(ROOT/"tests/test_gain_target_diagnostic.py","数学语义测试")):
        text+="- "+link(path,label)+"\n"
    REPORT.write_text(text,encoding="utf-8")
    assets={"report_sha256":sha(REPORT),"writer_sha256":sha(__file__),"summary_sha256":sha(OUT/"summary.json"),
        "plots_sha256":{p.name:sha(p) for p in OUT.glob("targeted_gain_comparison.*")},
        "figure_data_source":"summary.targeted_families","interval_scope":"paired_future_IDs_only_two_fixed_populations",
        "report_figures_not_generalization_evidence":True}
    (OUT/"report_assets_manifest.json").write_text(json.dumps(assets,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"saved_report":str(REPORT),"characters":len(text),"plots":list(assets["plots_sha256"])},ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
