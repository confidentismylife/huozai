"""Write the finite-library coverage audit, preserving oracle limitations."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/candidate_library_coverage_audit_development_v1"
REPORT=ROOT/"reports/完整候选计划覆盖审计报告_2026-10-03.md"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def link(p,label):return f"[{label}]({Path(p).resolve().as_posix()})"


def main():
    assert not REPORT.exists() and not (OUT/"report_assets_manifest.json").exists()
    m,s,v=(read(OUT/p) for p in ("development_manifest.json","summary.json","verification.json"));assert v["all_checks_passed"]
    states=table(OUT/"state_opportunity_audit.csv");plans=table(OUT/"full_library_plan_evidence.csv");arms=table(OUT/"arm_state_results.csv")
    lookup={(r["state_id"],int(r["plan_index"])):r for r in plans};armlookup={(r["state_id"],r["arm"]):r for r in arms}
    missed=sorted((r for r in states if r["missed_by_old_proposal"]=="True"),key=lambda r:r["state_id"])
    family_ids=sorted({r["family"] for r in missed});same_obs={}
    for family in family_ids:
        with np.load(OUT/"states"/(family+"_00_planning.npz"),allow_pickle=False) as z:x=z["observation"].copy()
        with np.load(OUT/"states"/(family+"_01_planning.npz"),allow_pickle=False) as z:y=z["observation"].copy()
        same_obs[family]=bool(np.array_equal(x,y))
    chart_rows=[lookup[r["state_id"],int(r["empirical_best_index"])] for r in missed]
    fig,ax=plt.subplots(figsize=(10.2,4.1));yy=np.arange(len(chart_rows))
    ax.scatter([float(r["old_prediction_gain_pp"]) for r in chart_rows],yy+.11,color="#D55E00",marker="x",s=55,label="Frozen prediction for that candidate")
    ax.scatter([float(r["actual_gain_pp"]) for r in chart_rows],yy-.11,color="#0072B2",s=40,label="Actual 6-future mean (hindsight-selected candidate)")
    ax.axvline(0,color="#555555",lw=.8);ax.set_yticks(yy,[r["state_id"]+": "+r["plan"] for r in chart_rows]);ax.invert_yaxis()
    ax.set_xlabel("601 incapacitation minus candidate incapacitation (percentage points)");ax.set_title("Missed opportunities in a viewed development subset",fontsize=12)
    ax.grid(axis="x",alpha=.2);ax.legend(fontsize=8,loc="lower left");fig.tight_layout()
    for suffix in ("png","svg","pdf"):fig.savefig(OUT/("missed_candidate_prediction_comparison."+suffix),dpi=200)
    plt.close(fig)
    cn={"beneficial":"有益","harmful":"有害","neutral_or_small":"持平/微小"};primary=s["primary_paired_effect"];op=s["opportunity_summary"]
    effects={(r["treatment"],r["control"]):r for r in s["paired_effects"]}
    text=f"""# 完整候选计划覆盖审计报告

2026年10月3日。固定12个已查看GTEST开发家庭、24个原人口起点、全部17个既有计划；新增{ s['total_unique_full_rollouts'] }次完整模拟。没有新增训练或LLM调用，资格线和现行门槛保持不变。

## 结论与研究决定

在这个固定开发子集中，旧提案确实漏掉有限库里的经验收益机会：{op['missed_by_old_states']}起点、{op['missed_by_old_families']}家庭有收益>0.10个百分点的候选，但旧提案都选择601。全部机会来自其他冻结PPO种子；12个手工计划在本批实际6未来中没有出现有意义失能改善。

这不是只有事后挑最佳的现象。完整规划库先选计划、再用独立实际未来检验，R_point抓住3个遗漏起点、M_point抓住2个，隐藏真源Oracle抓住4个。R/M不读真实源标签，但能调用模拟器和固定人口状态，因此仍是具有模拟特权的离线教师；当前轻量学生并未学会这种选择。

主比较M_point−旧提案为{primary['delta_incap_pp']:+.6f}个百分点，探索性区间[{primary['exploratory_ci_low_pp']:.6f}, {primary['exploratory_ci_high_pp']:.6f}]；2家庭更好、0更差、10持平，区间上端为0。它支持继续检查提名覆盖，不能宣布稳定总体疗效。保留3.395401个百分点余量后，R_margin、M_margin及旧完整门控均没有主指标改善。没有将无余量规划参照替换成现行门控。

下一步优先修复“只给旧提名方案产生训练标签”的数据接口：至少固定覆盖全部5个冻结PPO候选，而不是让旧收益预测先排除所有负分方案。先检查小候选集合能否保留这些机会，再讨论新的排名训练；暂不扩大PPO训练。LLM的独立贡献仍未建立，当前正例主要是多种子策略选择。

## 1. 为什么上一轮没有正例不等于候选库没有收益

上一轮只评价601和旧GainRidge提案，新检验28个非601提案中26持平、2有害，其余起点执行601。那只能检验提案器已给出的方案，无法验证被它打低分的其他计划。本轮把既有有限库全部评价后，在GTEST0004和GTEST0007发现其他PPO策略有益；两个家庭旧提案均为601，所以按“非601提案才进入收益训练”规则，它们不会提供候选替代正例。

本轮使用每类型按ID取前2家庭的固定规则，未按风险、失能结果或候选收益选样；并保留原两个人口。GTEST已经查看，本报告只能是开发诊断，不是新自然发生率估计或最终检验。12家庭中的2家庭不能直接换算成部署收益机会概率。

## 2. 实际怎么审计

每起点两个相容火源世界，各17计划×6全新规划未来，得204回合；原真实场景另17×6个全新实际未来，得102回合。共4896规划＋2448实际＝7344，上限7344。没有扩充家庭、未来或模型；人口种子有意复用，新规划/实际/传感器流与旧记录隔离。

每起点完成规划后，保存规划哈希与R/M/Oracle/旧提案的冻结选择，再开始实际模拟。完整库全部计划共享同一实际未来组，比较组只引用这些结局，不分别重跑。共享未来种子并不保证不同策略路径中每个个体接收一一对应的随机扰动。

R(p)＝601最大世界平均失能损失−计划p最大世界平均失能损失；M(p)＝每世界内601−p收益的最小值。最高评分严格>0.10个百分点才覆盖，同分优先601。R/M_margin另外扣除原3.395401余量；原完整门控仍保留旧分类器条件。新全库参照未擅自套用只在旧提案上训练的分类器。

真实源Oracle使用真源世界的规划6未来选择，再在独立实际未来检验；其真源信息明确标为特权。事后最佳使用实际6未来选计划，有选择乐观偏差，只展示有限样本机会，不给它控制效力区间，也不称全球最优或理想安全上界。

## 3. 漏掉的是哪些计划

以下“实际最佳”是事后有限样本选择，只用于定位遗漏；全部收益单位为百分点，正数有益。旧预测为原GainRidge对这个候选的预测值，不是新教师评分。

| 起点 | 类型 | 旧提案 | 事后有益候选 | 旧预测该候选 | 新实际6未来 | 前3实际未来 | 后3实际未来 |
|---|---|---|---|---:|---:|---:|---:|
"""
    for r in missed:
        q=lookup[r["state_id"],int(r["empirical_best_index"])]
        text+=f"| {r['state_id']} | {r['conflict_type']} | {r['old_plan']} | {q['plan']} | {float(q['old_prediction_gain_pp']):.4f} | {float(q['actual_gain_pp']):.4f} | {float(q['actual_first3_gain_pp']):.4f} | {float(q['actual_last3_gain_pp']):.4f} |\n"
    text+="\n"+f"![遗漏候选的旧预测与实际收益]({(OUT/'missed_candidate_prediction_comparison.png').resolve().as_posix()})\n"
    text+="""
拥堵GTEST0004的602旧预测约−1.3751，但两个原人口实际收益约+1.7361、+0.4861；R和M规划也都认为它有益。该例提供最直接的提名失配证据：在两种世界评分下都有候选收益，旧收益预测仍把它排除。

GTEST0007的603旧预测约−8.2473，两个原人口实际收益约+1.9444、+1.8750。clear表示没有附加出口冲突，并不表示没有火灾或零危险。前后半实际未来均有益，两个原人口也均有益；但6未来的分半一致仍不是独立新家庭复现或已知真实期望。

GTEST0007还反映源世界目标的限制：在实际源有益的计划，另一个支持源可能有害。因此不能把它的所有遗漏都简单归因于拟合错误。逐状态规划如下：

| 起点 | R_point选什么/实际收益 | M_point选什么/实际收益 | 该事后候选的规划R | 该事后候选的规划M |
|---|---|---|---:|---:|
"""
    for r in missed:
        state=r["state_id"];rr=armlookup[state,"R_point"];mm=armlookup[state,"M_point"];q=lookup[state,int(r["empirical_best_index"])]
        text+=f"| {state} | {rr['plan']} / {float(rr['actual_gain_pp']):+.4f} | {mm['plan']} / {float(mm['actual_gain_pp']):+.4f} | {float(q['R_gain_pp']):+.4f} | {float(q['M_gain_pp']):+.4f} |\n"
    text+="""
GTEST0007_00的603规划R约+0.2778、M约−0.8333；另一人口R约−1.2500、M约−3.7500。R取不同世界的最大损失后作差，M直接惩罚有害世界，拒绝它有明确目标原因。隐藏源Oracle才在两人口均捕捉实际有益计划，不能把这种信息特权给未识别火源的学生。

"""
    if same_obs.get("GTEST0007"):
        text+="经直接数组比对，GTEST0007两个人口的初始49维观测完全相同，而R教师选择不同。教师模拟固定人口和新规划流所得到的状态差异，并未都包含在学生输入中。修复候选覆盖不等于能逐状态复刻教师选择，后续仍需估计可观测条件收益及其不确定性。\n"
    text+="\n## 4. 规划先选、实际后验的效果\n\n相对601的失能差值：负数较好。家庭内两人口平均后，12家庭等权；六类型内家庭Bootstrap2000次，未多重调整。小样本区间触及0，不宣称稳定收益。\n\n| 组 | 覆盖起点 | 实际有益/有害起点 | 失能差值 | 探索性95%区间 | 改善/变差/持平家庭 |\n|---|---:|---|---:|---|---|\n"
    summaries={r["arm"]:r for r in s["arm_summary"]}
    for arm in ("old_proposal","frozen_full","R_point","M_point","R_margin","M_margin","oracle_source_plan"):
        q=summaries[arm];e=effects[arm,"pure_ppo_601"];c=q["actual_classes"]
        text+=f"| {arm} | {q['overrides']} | {c.get('beneficial',0)}/{c.get('harmful',0)} | {e['delta_incap_pp']:+.6f} | [{e['exploratory_ci_low_pp']:.6f}, {e['exploratory_ci_high_pp']:.6f}] | {e['improved_families']}/{e['worsened_families']}/{e['tied_families']} |\n"
    text+="""
R_point选了5个非601计划，其中3有益、2主指标持平；M_point选2个，均有益。R_margin选1个非601计划但主指标持平，M_margin与旧完整门控均不覆盖。保留现行余量后没有拾回主收益，说明提名修复之外还存在预测误差余量和接受目标的限制；本轮没有为制造收益降低门槛。

Oracle与事后最优在这批有限实际样本上恰好选到相同有益计划，不能据此认为Oracle达到真实全球上界。前者先在规划数据选择，后者在实际数据选，是两种不同性质的记录。

## 5. 严格M正收益的结构性限制

24起点中有18个在至少一个支持世界的601规划失能均值为0。所有计划失能损失非负，所以这些起点任何计划M≤0，M_point必然退回601。M≥0表示每世界不恶化的经验条件，M>0.10表示每世界都有有意义改善，后者更严格。

本批18个零基线世界起点中，实际全库收益机会数为0，所以这项结构限制**没有直接解释本次4个遗漏**。遗漏发生在另6个起点中；GTEST0007的M拒绝来自另一个世界的负收益，不能靠把正收益条件改成简单非负就自动解决。不能将这个数学区别包装成当前改门槛后必有收益的证据。

规划6未来中零失能不证明真实期望损失等于0；这里的不可改善结论针对本次有限样本标签。后续应明确研究究竟要求“所有世界严格获益”还是“每世界不恶化且某些情况下获益”，并先存新协议，不能在本次结果上偷换目标。

## 6. 审计可信度、执行统计和限制

独立只读验证重算408个计划×起点结局记录、216个比较组×起点记录、24个计划选择及选择前后时间哈希；验证先冻结规划选择再生成实际结果，检查真实源Oracle与事后最佳标记，并重算全部12家庭配对区间。没有验证器复拟合或重新模拟。

"""
    text+=f"8项数学测试通过；{v['source_inputs_unchanged']}个冻结输入哈希不变。真实执行轨迹共{v['actual_trace_steps_verified']}步，指示保持阻塞{v['hold_block_steps_verified']}步、中性指示正放行{v['neutral_positive_release_steps_verified']}步。中性正放行按实际joint action −1记录，并非把请求方向当成已经执行；全库包括等待转放行计划，本轮如实保留这些执行反馈。\n"
    text+="""
限制分别是：已查看子集、两个原人口、6未来的有限精度、仅17计划库、R/M的模拟访问及固定隐藏人口特权、Oracle隐藏源、事后候选选择乐观偏差、未作多重调整，以及未覆盖地图/行为/噪声变化。没有新增LLM或PPO训练，也没有证明LLM相对规则/多种子策略选择的独立贡献。正式第三轮和最终银行仍未启动。

## 7. 下一步的具体修改方向

先固定一个常驻小候选集合，至少包含601—605全部冻结PPO；用现有完整库证据检查它是否保留已有机会。当前两场景事后最佳和R/M有益选择都属于这个集合，12个手工计划没有补出实际正例，因此先做小候选对照有明确成本理由，但不能据此永久删掉手工库或认为其他场景无手工收益。

新的训练数据接口需要给候选集合内每个计划产生相对601标签，保留旧提名为601的场景；不要让一个已知有偏差的旧预测先过滤掉全部拥堵样本，再要求收益模型学会拥堵收益。全候选标签与固定小集合对照先审计、后另存训练/校准/检验划分，下一轮仍须新家庭独立验证。

同时区分两条失败链：GTEST0004是可支持世界参照也能拾回的提名失配；GTEST0007包含世界间收益冲突，需要源信息/条件收益问题，不能用单纯增加训练量承诺解决。现行余量仍挡住所有主收益，所以候选覆盖修复也不等于最终门控有效。下一轮先做接口与低成本小候选验证，暂缓更长PPO训练。

## 8. 可复核材料

"""
    for p,label in ((ROOT/"reports/完整候选计划覆盖审计固定方案_2026-10-03.md","运行前固定方案"),(OUT/"development_manifest.json","固定子集、种子与输入哈希"),
        (OUT/"full_library_plan_evidence.csv","全部408计划起点证据"),(OUT/"state_opportunity_audit.csv","24起点机会与遗漏"),
        (OUT/"family_opportunity_audit.csv","12家庭机会汇总"),(OUT/"arm_state_results.csv","216组起点执行结果"),(OUT/"paired_effects.csv","先选后验的配对比较"),
        (OUT/"summary.json","结构化结果"),(OUT/"verification.json","独立验证"),(OUT/"missed_candidate_prediction_comparison.pdf","可导出图PDF"),
        (ROOT/"scripts/run_candidate_library_coverage.py","执行脚本"),(ROOT/"scripts/verify_candidate_library_coverage.py","只读验证脚本"),
        (ROOT/"tests/test_candidate_library_scores.py","数学语义测试")):
        text+="- "+link(p,label)+"\n"
    REPORT.write_text(text,encoding="utf-8")
    assets={"report_sha256":sha(REPORT),"writer_sha256":sha(__file__),"summary_sha256":sha(OUT/"summary.json"),
        "plots_sha256":{p.name:sha(p) for p in OUT.glob("missed_candidate_prediction_comparison.*")},
        "figure_candidates_selected_after_actual_results_explicitly_marked":True,"identical_initial_observations_in_missed_families":same_obs,
        "zero_world_limitation_not_attributed_to_this_round_missed_opportunities":True,"hindsight_not_used_for_efficacy_interval":True}
    (OUT/"report_assets_manifest.json").write_text(json.dumps(assets,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(REPORT),"characters":len(text),"missed_states":len(missed),"identical_initial_observations":same_obs,"primary":primary},ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
