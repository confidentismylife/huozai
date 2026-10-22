"""Report the fixed four-cell development comparison, including negative evidence."""
from collections import Counter
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/proposal_gain_factorial_development_v1"
REPORT=ROOT/"reports/收益标签与轻量模型四组对照实验报告_2026-10-03.md"
NAMES=("R_linear","R_kernel","M_linear","M_kernel","frozen","zero")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(p,label):return f"[{label}]({Path(p).resolve().as_posix()})"
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def main():
    assert not REPORT.exists() and not (OUT/"report_assets_manifest.json").exists()
    s=read(OUT/"summary.json");m=read(OUT/"development_manifest.json");v=read(OUT/"verification.json");g=read(OUT/"frozen_gates.json")
    assert v["all_checks_passed"]
    quality={(r["block"],r["model"],r["scope"]):r for r in s["prediction_quality"]}
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.4))
    for ax,block,title in zip(axes,("calibration","test"),("Calibration: 22 proposals / 11 families","Test: 28 proposals / 14 families; zero benefits")):
        values=[quality[block,n,"all_proposals"]["actual_mae_pp"] for n in NAMES]
        ax.bar(np.arange(6),values,color=["#0072B2","#56B4E9","#009E73","#8AC6A9","#D55E00","#777777"])
        ax.set_xticks(np.arange(6),NAMES,rotation=30,ha="right");ax.set_ylabel("Family-weighted actual-gain MAE (pp)");ax.set_title(title,fontsize=10)
        for i,val in enumerate(values):ax.text(i,val+.015*max(values),f"{val:.3f}",ha="center",fontsize=8)
        ax.set_ylim(0,max(values)*1.16);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Frozen nominees: four new fits vs old prediction and zero reference",fontsize=12);fig.tight_layout()
    for suffix in ("png","svg","pdf"):fig.savefig(OUT/("proposal_gain_mae_comparison."+suffix),dpi=200)
    plt.close(fig)
    with np.load(OUT/"training_dataset.npz",allow_pickle=False) as z:families=z["families"];targets=z["targets"]
    cn={"beneficial":"有益","harmful":"有害","neutral_or_small":"持平/微小"}
    def clas(g):return "harmful" if g< -1e-12 else "beneficial" if g>.001 else "neutral_or_small"
    training_family_classes={t:{c:len(set(families[np.array([clas(v)==c for v in targets[:,i]])])) for c in cn} for i,t in enumerate(("R","M"))}
    pred=table(OUT/"prediction_evidence.csv")
    evidence=[r for r in pred if r["model"]=="zero"]
    cal=[r for r in evidence if r["block"]=="calibration" and r["eligible"]=="True"]
    test=[r for r in evidence if r["block"]=="test" and r["eligible"]=="True"]
    main=s["primary_paired_effect"];proposal_effect=next(r for r in s["test_paired_effects"] if r["treatment"]=="point_proposal" and r["control"]=="pure_ppo_601")
    text=f"""# 收益标签与轻量模型四组对照实验报告

2026年10月3日。新家庭、训练/校准/检验分离的同生成器开发实验已完成；不是正式第三轮最终评估。以下单位“百分点”表示失能比例差×100。

## 结论与决定

本轮完成4次小模型拟合、{s['total_unique_full_rollouts']}次完整模拟，结果仍未建立有效的覆盖门控。主比较M_kernel完整门控相对冻结旧完整门控的实际失能差值{main['delta_incap_pp']:.4f}个百分点，探索性区间[{main['exploratory_ci_low_pp']:.4f}, {main['exploratory_ci_high_pp']:.4f}]；36家庭全部持平。两门控均在所有检验起点执行601，零差值来自执行策略一致，不能称为安全认证、等效检验通过或新方法有效。

新模型对独立实际收益的误差小于旧模型，但四组均没有超过零预测参考。更关键的是：新检验中只有26个持平提案和2个有害提案，没有有益提案。它能够提供高估/有害拒绝的有限证据，却不能回答“能否找回有益覆盖”。本轮不把有益召回写成0；其分母为0，指标记为不可估计。

因此暂缓扩大PPO、换更大网络或反复拟合。下一项更有辨识力的低预算工作是审计固定提案器的候选覆盖：在固定小场景子集上比较更完整的计划库，查清它是否漏掉有益替代计划；如果候选库本身几乎无收益，则需改研究任务和场景辨识度。当前没有搜全候选，不能说那些没提案的场景不存在好方案。

## 1. 本轮改变了什么

上一轮12未来复核确认：有益建议被低估，高分持平建议被高估；同时“最大世界损失之差R”不能解释为“每世界均无负收益”。本轮将两个标签与两个拟合方法交叉，隔离标签语义和拟合形式的影响：

| 标签/拟合 | 线性岭回归 | 固定高斯核岭回归 |
|---|---|---|
| R：两策略各自最大世界损失之差 | R_linear | R_kernel |
| M：各世界内配对收益的最小值 | M_linear | M_kernel |

同一批训练数据、相同66维输入、相同家庭权重和0.01岭系数；不搜索带宽、随机种子或超参数。核宽度仅由训练的非恒定特征维数决定。4次闭式拟合是确定性的，不是4个PPO种子，也未增加深层网络训练。

所有模型只评价“601或冻结旧提案”二选一，不重新搜索17候选，不更换旧PPO601—605。输入为49个可观测量（28:33清零）和17计划one-hot；真实源、冲突类型与未来模拟结果不进入预测。真实源用于生成实际反事实结局。原价值分类器保留68维旧元特征与原预测，其权重与0.80/0.10条件不改。

R衡量最坏损失是否下降；M是同世界比较的保守有限样本代理。二者解决不同问题，不用对实际A的MAE简单宣布某个标签普遍更正确；尤其M较保守，可能在真实源有益而其他世界不利时主动拒绝。M仍不是所有真实未来都不伤害的证明。

## 2. 样本隔离、预算与真实训练支持

固定96训练家庭、24校准、36检验；每家庭2人口，各世界6规划未来，校准和检验另用6个独立实际未来，训练不采集实际未来。三个银行完整参数与所有旧银行去重，家庭ID也不重用。银行按生成顺序使用，没有按结果富集，没有因检验没收益而补场景。原人口和规划/实际、传感器种子隔离。

训练规划采集→四模型冻结→校准采集→余量冻结→检验启动与采集，时间标记及模型哈希可复核。检验初始观测未提前生成；新校准实际结局不参与拟合或余量。

| 区块 | 生成家庭 | 总起点 | 非601提案家庭 | 可评价提案起点 |
|---|---:|---:|---:|---:|
"""
    for block,label in (("train","训练"),("calibration","校准"),("test","检验")):
        er=[r for r in evidence if r["block"]==block and r["eligible"]=="True"]
        text+=f"| {label} | {s['family_counts'][block]} | {2*s['family_counts'][block]} | {s['eligible_family_counts'][block]} | {len(er)} |\n"
    text+="\n实际模型只拟合68起点/34家庭，不能把96生成家庭全部算成拟合样本。相同提案输入的两人口也不是68个独立家庭。\n\n"
    text+="| 训练标签 | 有益起点/含有益家庭 | 持平起点/含持平家庭 | 有害起点/含有害家庭 |\n|---|---:|---:|---:|\n"
    for target in ("R","M"):
        counts=s["training_support"]["training_target_class_counts"][target]
        text+=f"| {target} | {counts.get('beneficial',0)}/{training_family_classes[target]['beneficial']} | {counts.get('neutral_or_small',0)}/{training_family_classes[target]['neutral_or_small']} | {counts.get('harmful',0)}/{training_family_classes[target]['harmful']} |\n"
    text+=f"""
某家庭不同人口可以含不同标签，因此按标签统计的家庭数可能重叠。M的正例稀少，解释其收益预测与召回时要保留这个限制。训练只见过6个非601计划；未见过的计划仍报告预测误差，但四组执行时共同退回601，没有从检验学习它们的收益。

预算上限9000，设计最坏8928；实际{s['total_unique_full_rollouts']}＝{s['budget']['planning_rollouts']}规划＋{s['budget']['evaluation_rollouts']}实际执行。相同计划共享回合，四组模型共享全部结局，不分别重复模拟。没有新增LLM调用或PPO训练。

## 3. 预测质量：与零预测比较才有意义

MAE为平均绝对预测误差；先对每家庭人口平均，再对家庭等权。只在非601提案上计算，避免大量601自比较零值掩盖错误。下表均为百分点：

| 模型 | 校准对R MAE | 校准对M MAE | 校准对独立实际A MAE | 检验对R MAE | 检验对M MAE | 检验对A MAE |
|---|---:|---:|---:|---:|---:|---:|
"""
    for n in NAMES:
        c=quality["calibration",n,"all_proposals"];t=quality["test",n,"all_proposals"]
        vals=[c[k+"_mae_pp"] for k in ("R","M","actual")]+[t[k+"_mae_pp"] for k in ("R","M","actual")]
        text+=f"| {n} | "+" | ".join(f"{x:.4f}" for x in vals)+" |\n"
    text+="\n"+f"![收益预测MAE对照]({(OUT/'proposal_gain_mae_comparison.png').resolve().as_posix()})\n"
    mk=quality["test","M_kernel","all_proposals"];old=quality["test","frozen","all_proposals"];zero=quality["test","zero","all_proposals"]
    text+=f"""
M_kernel检验实际MAE从旧模型的{old['actual_mae_pp']:.4f}下降到{mk['actual_mae_pp']:.4f}，但零预测仅{zero['actual_mae_pp']:.4f}。所有新模型在校准和检验的实际MAE仍高于零参考。新模型减小旧模型的严重高估是有用诊断，不能独立当成“学会识别覆盖价值”。本检验绝大多数收益为0，接近0的预测就容易获得较小MAE；有益召回和实际控制收益仍需有正例的独立检验。

四组共享相同新拟合预算，属于公平的目标×模型比较；旧冻结模型来自更大且包含中间时刻的旧训练数据，作为参考而非相同预算因子。下面的差值使用相同家庭提案误差配对，负数代表前者MAE较小；点差是探索性比较，没有据此挑新获胜模型：

| 检验评价目标 | 比较 | MAE差值 |
|---|---|---:|
"""
    for r in s["factor_mae_differences"]:
        text+=f"| {r['target_evaluation']} | {r['treatment']} − {r['control']} | {r['delta_mae_pp']:+.4f} |\n"
    text+="\n## 4. 新结局里的正例、负例与语义矛盾\n\n| 区块 | 实际有益 | 实际有害 | 实际持平/微小 | 含实际有益家庭 |\n|---|---:|---:|---:|---:|\n"
    for label,er in (("校准",cal),("检验",test)):
        counts=Counter(r["actual_class"] for r in er)
        text+=f"| {label} | {counts['beneficial']} | {counts['harmful']} | {counts['neutral_or_small']} | {len({r['family'] for r in er if r['actual_class']=='beneficial'})} |\n"
    text+="""
“没有有益检验提案”只针对冻结旧提案器输出的方案，不能解释为全部17计划均没有好方案。校准有正例和负例，但校准数据已用于余量，不能把在校准找出的规则另称独立检验。6实际未来产生的是经验标签，不能把持平视为真实期望收益严格为0。

校准的原始收益分（>0.10）对4个实际有益起点：R_linear找回2、R_kernel找回4、两个M模型均未找回；这只是余量扣除前的诊断。M目标更保守且训练正例少，不能直接以0召回断定M语义无效。检验无有益分母，各模型召回均不可估计。

"""
    text+=f"检验提案中R/M标签不同{s['test_R_M_class_disagreements']}起点；R/独立实际不同{s['test_R_actual_class_disagreements']}，M/独立实际不同{s['test_M_actual_class_disagreements']}。本次M标签与实际类别一致不构成可靠概率或总体安全结论：只有14提案家庭、且没有实际有益提案。逐世界收益与规划前3/后3未来结果已保存在明细。\n"
    text+="\n## 5. 校准余量与执行结果\n\n4个余量均保持3.395401个百分点；没有使用独立实际结局调门槛。完整门控需要收益门控通过、计划在训练中出现，以及原价值分类器有益分≥0.80且有害分≤0.10。只收益门控作为诊断另外记录。\n\n| 执行组 | 校准覆盖起点与实际类别 | 检验覆盖起点与实际类别 |\n|---|---|---|\n"
    gates={(r["block"],r["arm"]):r for r in s["gate_summary"]}
    def describe(row):return str(row["overrides"])+"；"+("、".join(cn[c]+str(n) for c,n in row["actual_classes"].items()) or "无覆盖")
    for arm in ("point_proposal","frozen_gain","frozen_full")+tuple(n+suffix for n in NAMES[:4] for suffix in ("_gain","_full")):
        text+=f"| {arm} | {describe(gates['calibration',arm])} | {describe(gates['test',arm])} |\n"
    baseline=next(r for r in s["test_arm_summary"] if r["arm"]=="pure_ppo_601")
    text+=f"""
检验所有门控（包括只收益门控）都没有覆盖，因此这次不是分类器又把大量已通过收益门控的机会挡掉。已有提案在检验没有实际正例，新预测也未达到保留的余量门槛。不能为了让接受数非零而临时降低门槛。

36家庭的601失能率为{baseline['final_incapacitation']*100:.6f}%；所有门控与其逐起点执行一致。固定未门控提案相对601失能增加{proposal_effect['delta_incap_pp']:+.6f}个百分点，探索性区间[{proposal_effect['exploratory_ci_low_pp']:.6f}, {proposal_effect['exploratory_ci_high_pp']:.6f}]，{proposal_effect['worsened_families']}家庭变差、{proposal_effect['improved_families']}改善、{proposal_effect['tied_families']}持平。样本小且差值区间触及0，不扩展为稳定危害估计。

FED、剩余、拥堵和步数也保存；所有门控由于选择相同601而逐指标相同，点提案在这些指标上可能不同。“主指标持平”本身不能证明不同方案在所有指标等效。

## 6. 验证与文件保全

独立只读验证重新计算312起点、1872个预测记录、1440个执行组记录；核验训练标准化、四个闭式正规方程、两个标签、旧提案和分类器分数、校准残差与不降低余量、执行选择及36家庭Bootstrap。没有重新拟合或模拟；四模型正规方程残差最大约{max(v['fit_normal_equation_residuals'].values()):.2g}。

数学测试11项通过。独立验证核对{v['source_inputs_unchanged']}个冻结输入、{v['actual_trace_steps_verified']}实际执行步；{s['budget']['hold_block_steps']}步指示保持阻塞、中性正放行{s['budget']['neutral_positive_release_steps']}。旧源码、模型、报告、协议及历史失败文件保持原哈希，仅当前进度索引在封装时更新并保存旧字节。

原95% / 10% / 5% / 95%资格线不变；正式第三轮训练和最终银行未启动。本实验属于同生成器、小型开发验证，不含行为/地图/观测噪声迁移、PPO多种子训练或真实火场验证。较大收益的旧诊断主要来自选择其他冻结PPO种子；本轮未建立LLM独立控制贡献。

## 7. 下一步应做什么，以及停止条件

先冻结本轮结果，停止继续追加同类拟合。最有价值的下一步是一个小预算候选覆盖审计：预先固定少量场景、完整候选计划与独立实际未来，比较“旧提案”与“可观测世界下完整候选选出的参照”。旧GTEST已查看，若用它做审计只能标成开发诊断，不能再称最终检验。

该审计的决定点很具体：完整候选若能找到旧提案漏掉的收益，优先解决提名覆盖与收益预测的接口；若连完整候选参照都很少改善，则当前场景/目标不足以检验覆盖收益，调整可辨识任务比扩大训练更合理。不得把事后真实源最优计划包装为可部署建议。

后续若需要有益/有害挑战集，须预先声明如何按可观测参数构建，另保留自然场景集；不能在测试后补正例再报自然总体效果。LLM角色也需在相同信息、候选和预算下另作消融，不能将多PPO种子选择的收益归因给LLM。

## 8. 可复核材料

"""
    for p,label in ((ROOT/"reports/收益标签与轻量模型四组对照开发方案_2026-10-03.md","运行前固定方案"),(OUT/"development_manifest.json","新银行、种子与输入哈希"),
        (OUT/"frozen_models.json","模型冻结与训练支持"),(OUT/"frozen_gates.json","校准余量冻结"),(OUT/"prediction_evidence.csv","逐提案预测、世界标签与实际结果"),
        (OUT/"execution_arm_results.csv","逐起点执行组结果"),(OUT/"test_paired_effects.csv","36家庭配对主次比较"),(OUT/"summary.json","结构化结果"),
        (OUT/"verification.json","独立验证"),(OUT/"proposal_gain_mae_comparison.pdf","图表PDF"),(ROOT/"scripts/run_proposal_gain_factorial.py","运行脚本"),
        (ROOT/"scripts/verify_proposal_gain_factorial.py","只读验证脚本"),(ROOT/"tests/test_proposal_gain_models.py","数学测试")):
        text+="- "+link(p,label)+"\n"
    REPORT.write_text(text,encoding="utf-8")
    assets={"report_sha256":sha(REPORT),"writer_sha256":sha(__file__),"summary_sha256":sha(OUT/"summary.json"),
        "plots_sha256":{p.name:sha(p) for p in OUT.glob("proposal_gain_mae_comparison.*")},"figure_data_source":"summary.prediction_quality.actual_mae_pp",
        "no_actual_beneficial_test_proposals_explicitly_reported":True,"zero_reference_better_than_all_new_models_explicitly_reported":True,
        "no_winner_selected_from_test":True,"training_class_family_counts":training_family_classes}
    (OUT/"report_assets_manifest.json").write_text(json.dumps(assets,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(REPORT),"characters":len(text),"primary":main,"zero_vs_M_kernel_actual_mae":[zero['actual_mae_pp'],mk['actual_mae_pp']],"test_proposal_classes":s['test_proposal_actual_classes']},ensure_ascii=False))


if __name__=="__main__":sys.stdout.reconfigure(encoding="utf-8");main()
