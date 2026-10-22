"""Write all prespecified results, including mixed positive and negative signals."""
from pathlib import Path
import csv
import json
import sys
from run_five_ppo_fresh_student import ROOT, OUT, read, write, sha

REPORT=ROOT / "reports/五PPO全新家庭轻量选择实验报告_2026-10-04.md"
def table(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def link(p,label):return f"[{label}]({Path(p).resolve().as_posix()})"


def main():
    assert not REPORT.exists() and not (OUT / "report_assets_manifest.json").exists()
    s,m,v,d=(read(OUT / p) for p in ("summary.json","development_manifest.json","verification.json","miss_diagnostic.json"));assert v["all_checks_passed"]
    effects={(r["treatment"],r["control"]):r for r in s["all_paired_effects"]};arms={r["arm"]:r for r in s["arm_summary"] if r["block"]=="test"}
    primary=s["primary_paired_effect"];testop=next(r for r in s["opportunity_summary"] if r["block"]=="test")
    text=f"""# 五PPO全新家庭轻量选择实验报告

2026年10月4日。固定24训练、12校准、24留出家庭，全部与历史场景参数/ID及随机流隔离。完成7200规划＋2160实际＝9360次完整模拟、4个多输出轻量模型拟合、2个训练均值参照估计；没有PPO更新或新LLM调用。所有报告数值来自本轮冻结后独立留出评价。

## 结论

修复全候选标签覆盖后，新留出仍有{testop['families_with_library_benefit']}家庭、{testop['beneficial_candidate_rows']}个候选存在实际收益机会，但尚未建立学生控制优势。4个学习模型和2个均值参照的余量版本均24/24回退601。固定主比较M_kernel_margin−mean_M_margin为{primary['delta_incap_pp']:+.6f}个百分点，区间[0,0]：两组实际执行相同，不代表证明了部署安全或有效性。

不能只看主比较0就忽略无余量信号：R线性参照失能差值约−0.300926个百分点，2家庭改善、1变差，探索区间[−1.001157,+0.101273]包含0；R核参照约+0.041956，1改善、2变差；M核参照覆盖13家庭但失能全部持平。正向点估计和有害家庭均如实保留，不能把无余量方案替换成已验证门控。

9个实际有益候选中，6个M规划标签未达到0.10个百分点的正收益条件，另3个M标签有益却全部被M核模型预测为无益。这分别是源世界目标与实际收益的差异、以及学生预测泛化失败；单纯降低余量不能修复这些原始提名问题。

下一步先审计初始可观测信息和规划收益反转，尤其F5TE0018与相近训练家庭F5TR0006；本轮留出现在也已查看，只能作后续开发诊断。继续暂停扩大PPO和追加学生训练，不用当前留出正例调门槛再宣布成功。LLM独立贡献仍未建立。

## 1. 本轮究竟学什么

选择器只在初始49维观测上选一次冻结PPO601—605中的一个；选中策略随后以原9动作闭环完成疏散。它是初始策略库选择实验，不是运行中逐步切换策略，也不是LLM预测错误的在线纠偏实验。

输入沿用28:33屏蔽，不含真实火源、潜在人口属性或实际结局。4个候选分别有输出头，601预测严格0，使观测能改变候选差序；没有沿用“共享观测线性项加候选常数”只能产生固定差序的加法模型。

R/M各拟合线性多输出岭回归、核岭回归，固定正则0.01；核宽度仅由训练特征有效维数决定。另用24训练家庭的各候选平均R/M作两个常数分数参照。本轮两组平均分数的非601分量均为负，因此均值参照的无余量版也选择601。零预测只用于预测误差比较，无需拟合。

模型、预处理和均值参照在任何校准/留出拟合前冻结；校准仅选择余量，留出不选择超参、候选或模型。训练24家庭只有96个非601候选：R有意义正标签9个、M正标签4个，正标签仍稀少，负标签幅度可到−41.3194/−44.9306个百分点。

## 2. 新家庭和预算如何隔离

F5TR24、F5CA12、F5TE24，六冲突类型在各块平衡。完整场景参数签名和ID均与旧记录及跨块无重复；人口、规划未来、实际未来、传感器流另开命名空间。没有因为收益少或结果不佳重抽家庭、重换种子或追加模拟。

每家庭两个人口，每人口两源世界×五候选×六规划未来＝120规划回合；60家庭共7200。校准和留出另每家庭两人口×五候选×六实际未来＝60实际回合；36家庭共2160。21个比较组引用同一五候选实际结局，未按组重复模拟。

同一家庭两个初始观测经数组比对相同。标签先在每源世界汇总两人口×六未来的损失，再计算R/M。这是新的“汇总人口后”经验估计对象，不能等同于过去逐人口R/M取平均；非线性最大/最小运算的顺序会改变目标。它也不是精确真实条件期望，只有两人口、六未来，仍存在潜在状态特权和模拟噪声。

共600条R/M候选目标记录，对应300个家庭×候选配对；模型实际使用24个训练家庭行，每行含五候选目标。未来、候选或同家庭人口不作为独立家庭来计算区间。

## 3. 全部21组留出结果

失能差值相对601，负数更好，单位百分点。区间是六类型内家庭Bootstrap2000次，未多重调整。本轮属于同生成器小规模开发，不能宣布最终泛化、真实火灾效力或资格通过。

有益家庭需收益>0.10个百分点；有害为失能增加。区间的改善/变差计数按实际非零差值，可能包含低于0.10个百分点的小改善，不能与有益分类混同。

| 组 | 非601家庭数 | 实际有益/有害家庭 | 相对601失能差值 | 探索性95%区间 |
|---|---:|---|---:|---|
"""
    for name in sorted(arms):
        ar=arms[name];ef=effects[name,"pure_ppo_601"];cl=ar["actual_family_classes"]
        text+=f"| {name} | {ar['override_families']} | {cl.get('beneficial',0)}/{cl.get('harmful',0)} | {ef['delta_incap_pp']:+.6f} | [{ef['exploratory_ci_low_pp']:.6f}, {ef['exploratory_ci_high_pp']:.6f}] |\n"
    text+="""
teacher组使用模拟器、固定人口和源世界规划，具有信息特权；其效果不能归到学生或LLM。M教师的改善全部来自一个家庭，区间上端为0；固定604在四个收益家庭都可能有益，但另四个家庭变差，整体区间很宽。因此候选库存在机会不等于固定换一个种子就稳定更好。

M_kernel_point的24家庭失能全部持平，但FED、拥堵和步数不全部相同；它有13个非601选择，不能说完全没有执行变化。全部次要结局和48人口的有害/有益分类已保存，不能用失能持平声称全面等效。

## 4. 预测比简单参照好吗

下面只比较留出的96个非601候选，避免601恒零稀释误差。预测对应自身R/M目标，实际收益MAE另外针对真实场景独立未来，二者不是同一估计对象。

| 预测器 | R标签MAE | M标签MAE | 实际收益MAE | 预测有益候选数 | 实际9个有益候选中预测有益的个数 |
|---|---:|---:|---:|---:|---:|
"""
    for q in s["prediction_quality"]:
        if q["block"]=="test":text+=f"| {q['model']} | {q['R_MAE_pp']:.6f} | {q['M_MAE_pp']:.6f} | {q['actual_MAE_pp']:.6f} | {q['predicted_beneficial_candidates']} | {q['beneficial_candidates_predicted_positive']} |\n"
    text+="""
单位均为百分点。全部4个学习模型的实际收益MAE仍高于零预测的1.507161。R核模型仅预测9个实际有益候选中的1个有益，M核模型为0个；本轮存在9个有益候选，所以0/9召回是可估计的0，区别于上一轮没有有益提案时召回不可估计。均值/零参照没有正预测，其正预测精度不可估计。

预测M目标和预测真实源实际收益必须分开：在每支持源都需要获益的目标下，真实源有益的候选可以有负M。实际收益误差不能单独用于替换这个目标或重新训练，必须先明确研究要求。

## 5. 为什么带余量全部回退

校准12家庭，每家庭取四个非601候选的最大乐观残差，再取95% higher经验分位数。本例该分位数就是12家庭最大值。余量只增不减，最低仍是原3.395401个百分点。

| 预测器 | 冻结余量，百分点 |
|---|---:|
"""
    for name,margin in s["margins_pp"].items():text+=f"| {name} | {margin:.6f} |\n"
    text+="""
余量约7—12个百分点，未降低来制造收益。新规则取所有候选的最大残差及更高分位数，比过去仅旧提名的规则更保守；不能将余量增加全解释为模型性能相对上一轮退化。12校准家庭与经验95%分位数也不构成95%部署安全覆盖保证。

新增余量参照没有套用只训练过旧提名的Worth分类器，也没有替换旧完整门控。原95%/10%/5%/95%资格线未变，资格未通过；正式第三轮PPO与最终银行未启动。对于M核模型，去掉余量也未抓住收益，因此没有“只要放宽余量就能有效”的证据。

## 6. 九个正例的遗漏链

以下由实际结局事后定位，只是描述性诊断，不能反过来参与模型/门槛选择。所有单位为百分点。

| 家庭 | 候选 | 实际收益 | 规划R | 规划M | R核预测 | M核预测 |
|---|---|---:|---:|---:|---:|---:|
"""
    positive=table(OUT / "actual_beneficial_candidate_audit.csv")
    for r in positive:text+=f"| {r['family']} | {r['plan']} | {float(r['actual_gain_pp']):+.4f} | {float(r['planning_R_pp']):+.4f} | {float(r['planning_M_pp']):+.4f} | {float(r['predicted_R_kernel_pp']):+.4f} | {float(r['predicted_M_kernel_pp']):+.4f} |\n"
    text+="\n"+f"![实际收益、M规划标签与学生预测]({(OUT / 'fresh_candidate_miss_diagnostic.png').resolve().as_posix()})\n"
    text+="""
9个实际有益候选里6个M标签未达正收益条件、5个R标签未达条件，存在源世界目标与实际源收益的差异。留出19家庭至少一支持世界的601样本损失为0，严格M正收益不可能；但4个实际收益家庭均不在这19个家庭里，不能把当前遗漏直接归因于这项零基线结构。

另3个M标签有益候选全在F5TE0018，M核预测均为负。其中603实际收益约16.1458，M标签约14.1319，学生却预测−18.6072。这是明确的教师标签与学生预测不一致，不能用风险目标不同掩盖。

训练M的全局正标签最大约5.2083个百分点，F5TE0018的三个候选M值全部超过各自训练最大值；训练604从未出现M正标签。这里表明这批训练标签没有覆盖该正收益强度，并不证明输入分布整体越界，也不能推断再训练多少步就能解决。

## 7. 相近训练家庭的提示与下一步

在M核模型训练标准化距离下，F5TE0018最近训练家庭是F5TR0006，核相似度约0.754887。603的训练近邻M标签约−16.3194，而当前留出M标签约+14.1319；604近邻约−0.5903，而留出约+13.4375。距离、相似度和每候选训练范围全部保存。

这只是所选距离下的描述性近邻，不能证明两场景完全不可区分、不能证明收益反转由某一个变量导致，也不能将核相似度当可辨识性检验。需要进一步核查可观测参数、隐藏危险时序与源世界结局是否在相似输入下发生反转，区分有限未来噪声、缺失信息和模型平滑造成的遗漏。

下一步优先用已有缓存做上述信息与标签可辨识性审计，暂缓追加模型训练和PPO预算。若决定改收益目标或加入较晚烟气观测，应先另存协议，并给简单规则/纯PPO同样信息与决策时刻；不能在当前已查看留出上修改方案再称最终检验。LLM贡献需要单独同信息、同预算对照，不能由多PPO种子选择的收益代替。

## 8. 执行和完整性

8项针对多输出、人口汇总、家庭隔离、余量与严格阈值的测试通过。独立验证核对60全新家庭、2124个冻结输入、4模型正规方程、训练预处理、2均值参照、6余量及全部选择先于实际结局的链，重算600条目标记录、756家庭组记录和1512人口组记录、全部区间与预测MAE。验证没有新拟合或重跑模拟。

实际轨迹55940步；指示保持阻塞3529步、中性指示正放行308步。中性正放行的实际joint action仍记录−1，未把请求动作当实际执行。五冻结PPO参数在运行前后保持不变。

启动时有一次顺序错误：准备命令尚在运行时提前调用执行，因manifest未生成退出；该次未开始模拟/拟合。记录保存，等待准备完成后才成功执行完整固定预算，没有丢弃已生成失败实验或偷偷重采样。

## 9. 可复核材料

"""
    paths=[(ROOT / "reports/五PPO全新家庭轻量选择实验固定方案_2026-10-04.md","运行前固定方案"),
        (OUT / "development_manifest.json","家庭、种子、输入哈希与预算"),(OUT / "training_dataset.npz","仅训练家庭的数据"),
        (OUT / "frozen_models.json","模型与均值参照冻结记录"),(OUT / "frozen_gates.json","校准余量冻结记录"),
        (OUT / "collection_complete.json","完整预算和缓存哈希"),(OUT / "pooled_candidate_labels.csv","600条人口汇总目标记录"),
        (OUT / "arm_family_results.csv","756个家庭组结果"),(OUT / "arm_population_results.csv","1512个人口组结果"),
        (OUT / "family_library_opportunities.csv","36家庭的候选机会"),(OUT / "paired_effects.csv","预设比较与探索区间"),
        (OUT / "prediction_quality.csv","预测与简单参照比较"),(OUT / "actual_beneficial_candidate_audit.csv","9正例遗漏及近邻诊断"),
        (OUT / "miss_diagnostic.json","描述性诊断摘要"),(OUT / "fresh_candidate_miss_diagnostic.pdf","图PDF"),
        (OUT / "summary.json","结构化结果"),(OUT / "verification.json","独立验证"),
        (ROOT / "scripts/five_ppo_student.py","人口汇总与多输出模型"),(ROOT / "scripts/run_five_ppo_fresh_student.py","固定运行流水线"),
        (ROOT / "scripts/verify_five_ppo_fresh_student.py","独立验证脚本"),(ROOT / "tests/test_five_ppo_student.py","8项测试"),
        (ROOT / "artifacts/five_ppo_fresh_startup_record_20261004.json","启动顺序错误记录")]
    for p,label in paths:assert p.is_file();text+="- "+link(p,label)+"\n"
    REPORT.write_text(text,encoding="utf-8")
    write(OUT / "report_assets_manifest.json",{"report_sha256":sha(REPORT),"writer_sha256":sha(__file__),"summary_sha256":sha(OUT / "summary.json"),
        "miss_diagnostic_sha256":sha(OUT / "miss_diagnostic.json"),"links_including_image":len(paths)+1,
        "all21_test_arms_reported":True,"all7_prediction_references_reported":True,
        "positive_R_linear_signal_and_harmful_cases_explicit":True,"pooling_estimand_and_initial_only_selection_explicit":True,
        "figure_visually_inspected":True,"essential_figure_data_clipped":False,"startup_failure_preserved":True})
    print(json.dumps({"report_written":str(REPORT),"characters":len(text),"links":len(paths)+1},ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
