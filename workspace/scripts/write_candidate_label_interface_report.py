"""Chinese report of the cache-only five-candidate interface audit."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ppo_candidate_label_interface_development_v1"
REPORT = ROOT / "reports/五PPO候选集合与全候选标签接口审计报告_2026-10-03.md"
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def link(p, label): return f"[{label}]({Path(p).resolve().as_posix()})"


def main():
    assert not REPORT.exists() and not (OUT / "report_assets_manifest.json").exists()
    m, s, v = (read(OUT / p) for p in ("development_manifest.json", "summary.json", "verification.json"))
    assert v["all_checks_passed"]
    c = s["label_interface_counts"]; input_audit = s["input_label_audit"]
    arms = {r["arm"]: r for r in s["arm_summary"]}
    labels = {(r["state_id"], int(r["plan_index"])): r for r in table(OUT / "planning_label_rows.csv")}
    actual = table(OUT / "candidate_actual_diagnostic.csv")
    text = f"""# 五PPO候选集合与全候选标签接口审计报告

2026年10月3日。固定复用上一轮12个已查看开发家庭、24个人口起点，检查601—605五个冻结PPO及全候选标签接口。新增模拟0、新增拟合0、PPO更新0、LLM调用0。没有重跑上一轮7344次模拟。

## 结论

五个冻结PPO在本批缓存中保留了完整17库的全部4个实际收益起点，24起点的有限样本最大失能收益也相同。它支持将五候选作为下一轮低成本对照集合；不证明其他场景手工计划无用，不证明五候选全局最优。

新接口已生成{c['all_candidate_rows']}条规划标签，即24起点×5候选。旧提案为601的{c['old601_states_retained']}起点及其{c['old601_rows_retained']}候选行全部保留；601本身的{c['baseline_rows_retained']}条零收益基准也保留。旧接口在同一子集仅允许{c['old_nominee_only_rows_in_this_subset']}条非601提案行，漏掉全部{c['actual_beneficial_candidate_rows']}条实际有益候选。本轮补全了开发诊断数据出口，尚未训练新的选择器。

五候选R无余量参照仍有3个有益起点、M参照仍有2个；五候选保留现行余量后均回退601，失能收益仍为0。候选覆盖修复和最终门控有效是不同的完成条件。初始观测也存在标签冲突，因此不能把下一步简单变成更多训练步数。

## 1. 为什么需要修改标签出口

历史四组收益模型的采集函数只模拟601和旧提案，拟合函数又通过plan!=12筛掉旧提案为601的起点。这不是统计样本自然没有替代正例，而是旧提案器先决定哪个候选有资格获得标签。

本轮将“旧提案分数”从候选数据覆盖条件中移除。固定五候选对每个起点完整输出R和M目标；旧提案仍作为独立诊断元数据，不能决定保留哪些标签。历史脚本和模型保持原样，新的规划标签生成接口及资格检查可供下一轮新数据流水线接入；本轮没有修改历史拟合结果。

| 同一开发子集的接口统计 | 旧非601提名行 | 全五候选行 |
|---|---:|---:|
| 行数 | {c['old_nominee_only_rows_in_this_subset']} | {c['all_candidate_rows']} |
| 规划R有益标签 | {c['old_only_R_beneficial_rows']} | {c['all_R_beneficial_rows']} |
| 规划M有益标签 | {c['old_only_M_beneficial_rows']} | {c['all_M_beneficial_rows']} |
| 实际有益候选行，仅用于诊断 | {c['actual_beneficial_rows_kept_by_old_nominee_only']} | {c['actual_beneficial_candidate_rows']} |

120行不等于120个独立场景：仍只有12个家庭、24个原人口、每起点5个候选的重复结构。601零收益标签提供锚点，不是新的收益正例。

## 2. 五候选是否保留机会

候选种子固定为601—605，不按实际结局重新挑种子。模型参数不变；直接引用上一轮对应计划的规划和实际缓存。以下仍是已查看数据的跟进诊断，不能重新称为独立复现。

| 组 | 非601选择起点 | 实际有益/有害起点 | 相对601失能差值，百分点 |
|---|---:|---|---:|
"""
    for name in ("full17_R_point", "five_R_point", "full17_M_point", "five_M_point", "full17_R_margin", "five_R_margin", "full17_M_margin", "five_M_margin"):
        row = arms[name]; classes = row["actual_classes"]
        text += f"| {name} | {row['overrides']} | {classes.get('beneficial',0)}/{classes.get('harmful',0)} | {100*row['mean_delta_final_incapacitation_vs601']:+.6f} |\n"
    text += """
失能差值负数更好。这里只重算缓存均值，没有新增有效样本或新的独立置信区间。上一轮探索区间和局限仍适用；不能因五候选结果相同就把证据强度翻倍。

实际有益候选及新规划标签如下。实际收益由上一轮实际6未来得到，候选机会有有限样本选择偏差；R/M仍是模拟特权教师的规划标签。

| 起点 | 候选 | 规划R，百分点 | 规划M，百分点 | 实际收益，百分点 |
|---|---|---:|---:|---:|
"""
    for row in sorted((r for r in actual if r["actual_class"] == "beneficial"), key=lambda r: r["state_id"]):
        q = labels[row["state_id"], int(row["plan_index"])]
        text += f"| {row['state_id']} | {row['plan']} | {float(q['target_R_pp']):+.4f} | {float(q['target_M_pp']):+.4f} | {float(row['actual_gain_pp']):+.4f} |\n"
    text += """
拥堵家庭0004的两个602候选在R和M标签下均有益；无附加出口冲突家庭0007的603实际有益，但M标签为负，另一个人口的R也为负。增加候选标签并不消除支持源世界间的收益冲突，不能把全部漏选都归因于训练不充分。

## 3. 失能相同不表示所有结果相同

五候选替换了完整库的部分手工选择。R_point在长期封闭家庭GTEST0003的两个起点，由wait3_then_live改选602；R_margin在其中一个起点改选601。M两组选择均未改变。

下表为五候选减完整17库，在24起点等权平均的变化。FED为模拟累计暴露量，拥堵为现有模拟统计，步数为模拟决策步；这里没有将它们擅自换算成真实伤害、人数或秒。

| 规划/门控 | 改变起点数 | 失能变化，百分点 | FED变化 | 拥堵变化 | 步数变化 |
|---|---:|---:|---:|---:|---:|
"""
    for row in s["five_vs_full17"]:
        text += f"| {row['objective_gate']} | {row['changed_states']} | {100*row['mean_five_minus_full17_final_incapacitation']:+.6f} | {row['mean_five_minus_full17_final_fed']:+.6f} | {row['mean_five_minus_full17_additional_congestion']:+.6f} | {row['mean_five_minus_full17_steps']:+.6f} |\n"
    text += """
本批中五候选的R参照降低了FED和拥堵，但步数略增；失能、疏散率和剩余率相同。这些是次要指标的缓存描述，不作事后多指标显著性结论，也不宣布全面优于完整库。

## 4. 可以节省多少计算

按同样两世界×6未来，完整17库每起点需要204次规划回合，五候选需要60次；实际评价分别102和30次。计划数量带来的设计回合减少为70.588235%。这是固定设计的算术比较，不是本轮实测运行时间加速，也不包括数据生成、模型训练或其他开销。

本轮只读缓存，实际新增模拟为0。五候选集合只作为低成本研究对照，17库及其手工计划、缓存和代码全部保留。

## 5. 新标签接口包含什么

all_candidate_planning_labels.npz共有120行：features为120×54；前49列来自可观测输入，沿用28:33屏蔽，后5列为候选one-hot。target_R和target_M为失能损失比例，例如0.01代表1个百分点，报告表格才乘100。

R是601最大世界平均损失减候选最大世界平均损失；M是各支持世界内601−候选收益的最小值。逐世界逐未来配对收益保存为120×2×6，供离线核查，不当成模型传感器输入。目标保留原6未来的有限精度及模拟潜在人口特权。

features不含真实火源、实际结局、潜在人口属性、旧收益预测或旧提案。家庭ID、人口起点ID、候选ID、支持世界和种子是分组/追溯元数据；不能将元数据随意拼到模型特征中。实际结局另存candidate_actual_diagnostic.csv，不写进规划标签NPZ。

12家庭每家庭总样本权重1/12，每家庭10行，各行权重1/120。未来划分必须将同家庭的全部人口和候选放在同一组，不能随机按120行拆训练和检验。

规划标签包先生成并保存哈希，再进入实际缓存诊断阶段；独立验证只用规划数组重算全部标签。但原缓存早已查看，因此这个顺序只是实现层面的标签来源隔离，不能将本轮变成盲法实验。

## 6. 相同输入为什么会有不同标签

120行对应60个唯一输入向量，每个输入出现两次：同家庭两个人口初始观测相同，同候选one-hot也相同，模拟中固定的潜在人口及未来结局却可能不同。

| 目标 | 相同输入但标签不同的组数 | 本批有限标签的最小MAE，百分点 | 最小RMSE，百分点 |
|---|---:|---:|---:|
"""
    for target in ("R", "M"):
        q = input_audit["finite_empirical_fit_lower_bounds"][target]
        text += f"| {target} | {q['conflicting_feature_groups']} / {input_audit['unique_feature_vectors']} | {q['finite_label_minimum_MAE_pp']:.6f} | {q['finite_label_minimum_RMSE_pp']:.6f} |\n"
    text += """
这些下限来自同输入组的中位数/均值误差算术，没有拟合模型。任意只用该54维输入的确定性函数，都不能同时精确输出同输入下不同的逐起点标签。

下限只针对本批固定标签与权重，不能推断真实期望误差下限、新场景泛化误差或需要多少训练数据，也不能区分潜在人口影响和6未来模拟噪声。实际部署可以研究条件期望与不确定性，训练目标不必是逐人口的无噪声教师标签。

## 7. 本轮没有把开发缓存当成新训练集

全部GTEST数据仍标记viewed_development_diagnostic_not_training。提供require_training_partition资格检查，当前数据会被拒绝用于新的训练；未来只有明确标记为fresh_family_disjoint_training且家庭在许可训练集合、与留出集合无交集时才允许通过。直接绕过此检查自行拟合不会获得新的独立证据。

本轮生成的是完整、可复核的诊断标签出口，没有训练选择器、校准新门槛或部署五候选门控。原95%/10%/5%/95%资格线、覆盖严格0.10个百分点条件和3.395401个百分点余量保持不变。正式第三轮PPO和最终银行未启动。

## 8. 验证、修复记录和下一步

独立验证重算120条规划标签、120条实际诊断行、192条组×起点结果、24个库覆盖记录、全部改变选择及六项结局；核对规划标签包与实际诊断的时间/哈希、候选范围、权重和同输入冲突下限。2088个冻结输入未改变。

针对新接口的10项测试最终通过。首次测试为5失败5通过：一处NumPy混合索引交换了世界和候选轴，一处测试误把浮点减法结果当作精确阈值。修正索引和测试用例后保留严格门槛；失败代码、测试和完整日志均保存。源快照还发现同名失败版本覆盖快照名称的问题，保留原快照并补充frozen_runtime下正确版本，运行源哈希和分析结果不受影响。前序冻结文件未修改。

下一步应先另存全新家庭的训练/校准/留出方案，使用本轮全候选接口，估计可观测条件收益并保留601场景；再做小规模轻量排名学习对照。必须同时检验候选覆盖和是否值得覆盖，并与简单多种子策略选择比较，避免把PPO种子差异当作LLM贡献。不能直接用当前4个正例调门槛，再在同一GTEST上宣布成功。更长PPO训练继续暂缓。

## 9. 可复核材料

"""
    artifacts = [
        (ROOT / "reports/五PPO候选集合与全候选标签接口审计方案_2026-10-03.md", "运行范围与固定方案"),
        (OUT / "development_manifest.json", "冻结输入与数据范围"),
        (OUT / "all_candidate_planning_labels.npz", "全候选规划标签NPZ"),
        (OUT / "planning_label_rows.csv", "120行规划标签"),
        (OUT / "label_bundle_manifest.json", "诊断标签包资格及哈希"),
        (OUT / "candidate_actual_diagnostic.csv", "实际结局独立诊断表"),
        (OUT / "state_library_coverage.csv", "24起点候选覆盖"),
        (OUT / "arm_state_results.csv", "192组起点结果"),
        (OUT / "changed_library_choices.csv", "改变的选择与六项结果"),
        (OUT / "identical_input_label_conflicts.csv", "相同输入标签冲突"),
        (OUT / "summary.json", "结构化结果"), (OUT / "verification.json", "独立验证"),
        (ROOT / "scripts/candidate_label_interface.py", "全候选接口与拟合资格检查"),
        (ROOT / "scripts/run_candidate_label_interface_audit.py", "缓存审计脚本"),
        (ROOT / "scripts/verify_candidate_label_interface_audit.py", "独立验证脚本"),
        (ROOT / "tests/test_candidate_label_interface.py", "10项接口测试"),
        (ROOT / "artifacts/candidate_label_interface_test_repair_20261003/repair_note.json", "首次测试修复记录"),
        (OUT / "snapshot_name_collision_note.json", "快照名称修复记录")]
    for path, label in artifacts:
        assert path.is_file(); text += "- " + link(path, label) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    write_value = {"report_sha256": sha(REPORT), "writer_sha256": sha(__file__), "summary_sha256": sha(OUT / "summary.json"),
        "source_inputs_unchanged": len(m["source_input_sha256"]), "local_links": len(artifacts),
        "all_eight_arms_and_four_secondary_comparisons_reported": True,
        "cache_only_not_new_independent_replication_explicit": True,
        "viewed_labels_fit_rejection_and_secondary_tradeoffs_explicit": True,
        "failed_tests_and_snapshot_repairs_preserved_and_reported": True}
    (OUT / "report_assets_manifest.json").write_text(json.dumps(write_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_written": str(REPORT), "characters": len(text), "links": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
