"""Audit newly retrieved primary materials and record bounded gap conclusions.

Reads downloaded text/video metadata and historical source manifests. Does not
execute downloaded code, launch FDS, train, or contact authors.
"""
from datetime import datetime, timezone
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_gap_material_search_20261016"
MATERIALS = OUT / "materials"
REPORT = ROOT / "reports/SafeStep缺失资料补齐检索_发现与可用性判断_2026-10-16.md"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    assert not p.exists(), p
    p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_save(p, values):
    assert not p.exists(), p
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def link(p, label):
    return f"[{label}]({p.as_posix()})"


def main():
    commercial = (MATERIALS / "example_commercial.fds").read_text("utf-8")
    guide = (MATERIALS / "FDS_User_Guide_6.9.1_text.txt").read_text("utf-8")
    data = read(OUT / "repo_tree.json")
    commits = read(OUT / "repo_commits.json")
    branches = read(OUT / "repo_branches.json")
    assert commits[0]["sha"] == "573f613420e21e9a6731bc56fa36ea0b3f3a05f6"
    assert len(commits) == 6 and len(branches) == 1
    assert read(OUT / "repo_tags.json") == read(OUT / "repo_releases.json") == read(OUT / "repo_issues.json") == []
    # Verify the original Git blob identity, not merely a local checksum.
    official_tree = read(OUT / "fdgen_official_tree.json")["tree"]
    git_blob_matches = []
    for name in ("README.md", "LICENSE.md", "example/example_commercial.fds", "example/example_Single_Story_Gas_1.fds"):
        p = MATERIALS / Path(name).name
        content = p.read_bytes()
        sha = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
        expected = next(x["sha"] for x in official_tree if x["path"] == name)
        assert sha == expected, name
        git_blob_matches.append(name)
    # Parse only simple namelist observations; this is not an FDS syntax/run validator.
    blocks = re.findall(r"&([A-Z][A-Z0-9_]*)\b([^/]*?)/", commercial, re.I | re.S)
    counts = {name: sum(k.upper() == name for k, _ in blocks) for name in ("MAIN", "MESH", "REAC", "DEVC", "SLCF", "RAMP", "OBST")}
    assert counts["MESH"] == 6 and counts["REAC"] == 1 and counts["RAMP"] == 0
    assert "CO_YIELD=0.042" in commercial and "SOOT_YIELD=0.198" in commercial
    assert not re.search(r"HCN|HYDROGEN CYANIDE", commercial, re.I)
    assert not re.search(r"CARBON DIOXIDE", commercial, re.I)
    assert not re.search(r"SETPOINT\s*=\s*150(?:\.0)?\b", commercial, re.I)
    assert "22.10.18" in guide and "Fractional Effective Dose" in guide and "HCN_YIELD" in guide
    comparisons = [
        dict(item="geometry", paper="Hotel floor:four corridors,24 rooms,longest corridor45.7m", template="Named TOP/RIGHT/BOTTOM/LEFT/ROOM24 meshes; six meshes; TOP x0..44m", verdict="RELATED_GEOMETRY_CANDIDATE_NOT_EXACT_IDENTITY"),
        dict(item="reaction", paper="Polyurethane; CO yield0.042; soot yield0.198", template="POLYURETHANE_GM27;CO_YIELD0.042;SOOT_YIELD0.198", verdict="SPECIFIED_YIELDS_MATCH"),
        dict(item="fire_location", paper="Centre ofRoom24", template="Room24 burner position randomized by IFSL/MRND/GFSL", verdict="DIFFERENT_SETUP"),
        dict(item="heat_release", paper="t-squared,peak5000kW at326s", template="Randomized incipient/growth/peak/decay;peak1000..8000kW;template SURF references generated ramp", verdict="DIFFERENT_NOT_READY_TO_RUN_AS_PAPER_CASE"),
        dict(item="door_control", paper="Fire room door opens at100s", template="Three opening stages with sampled ranges100..150,200..250,250..300s;placeholder timers500s", verdict="DIFFERENT_SETUP"),
        dict(item="window_control", paper="Temperature breakage150C", template="No explicit SETPOINT150 and a permanent OPEN exterior vent in downloaded template", verdict="PAPER_CONTROL_NOT_FOUND_IN_TEMPLATE"),
        dict(item="gas_observations", paper="CO/CO2/HCN at58 region centres,z2.0m", template="243 DEVC entries are thermocouples/TIME/heat-flux;7 slice entries include CO but no explicit HCN or CO2 token", verdict="INSUFFICIENT_FOR_PAPER_FER_TABLE"),
        dict(item="output_time", paper="Hotel rates up to900s at5s intervals", template="T_END900;no exact58-region gas device layout or5s gas CSV output defined", verdict="DURATION_MATCH_OUTPUT_MAPPING_INCOMPLETE"),
        dict(item="direct_execution", paper="A fixed FDS case", template="Contains FD-Gen directives including MAIN/IFSL/MRND/GHRC;RAMP_Q references a generated ramp", verdict="NEEDS_GENERATION_OR_EXPLICIT_DERIVATION_NOT_UNCHANGED_FDS_INPUT"),
    ]
    csv_save(OUT / "paper_vs_fdgen_comparison.csv", comparisons)
    inventory_specs = [
        ("mmc1.mp4", "https://ars.els-cdn.com/content/image/1-s2.0-S2352710225033698-mmc1.mp4", "PAPER_ORIGINAL_SUPPLEMENT", "Case1/Case2 route and hazard-category visualization;no continuousFER"),
        ("mmc2.mp4", "https://ars.els-cdn.com/content/image/1-s2.0-S2352710225033698-mmc2.mp4", "PAPER_ORIGINAL_SUPPLEMENT", "Hotel starting position1 route visualization"),
        ("mmc3.mp4", "https://ars.els-cdn.com/content/image/1-s2.0-S2352710225033698-mmc3.mp4", "PAPER_ORIGINAL_SUPPLEMENT", "Hotel starting position2 route visualization"),
        ("example_commercial.fds", "https://github.com/usnistgov/FD-Gen/blob/main/example/example_commercial.fds", "AUTHOR_RELATED_OFFICIAL_TEMPLATE", "Partial hotel geometry and reaction scaffold;not originalSafeStep case"),
        ("example_Single_Story_Gas_1.fds", "https://github.com/usnistgov/FD-Gen/blob/main/example/example_Single_Story_Gas_1.fds", "AUTHOR_RELATED_OFFICIAL_TEMPLATE", "Generic residential example;not original24-cell corridor"),
        ("NIST.TN.2332.pdf", "https://doi.org/10.6028/NIST.TN.2332", "OFFICIAL_TECHNICAL_DOCUMENTATION", "FD-Gen parameters,geometry lineage and generation workflow"),
        ("FDS_User_Guide_6.9.1.pdf", "https://github.com/firemodels/fds/releases/download/FDS-6.9.1/FDS_User_Guide.pdf", "OFFICIAL_VERSIONED_GUIDE", "Gas output/units/combustion/FED reference;paperFDS version not confirmed"),
        ("thermal_risks_nist.bin", "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=957841", "RELATED_PRIMARY_PAPER", "Commercial-building temperature study;not original toxic-gas data"),
    ]
    inventory = [dict(file=name, source_url=url, evidence_class=level, usable_for=use,
                      local_file=(MATERIALS/name).relative_to(ROOT).as_posix(), bytes=(MATERIALS/name).stat().st_size,
                      sha256=digest(MATERIALS/name)) for name, url, level, use in inventory_specs]
    csv_save(OUT / "material_inventory.csv", inventory)
    gaps = [
        dict(gap="small_building_FER_and62percent", status="STILL_MISSING", gained="Three original video attachments;small-case topology and routes independently observable", still_needed="Original24-state time series or matchingCO/HCN/CO2 data and conversion;originalsmall-buildingFDS input"),
        dict(gap="exact_final_paper_training_flow", status="NO_NEW_PUBLIC_CLARIFICATION_FOUND", gained="Currentauthor repository branches,history,releases,issues checked;samepinnedcommit", still_needed="Finalpapercode/version correspondence,architecture/seed/schedules/terminal details"),
        dict(gap="hotel_FDS_upstream", status="PARTIAL_SCAFFOLD_FOUND", gained="OfficialFD-Gen commercialgeometry/reaction scaffold and guide", still_needed="Fixedpaperfire,door/windowcontrols,58-regiongasdevices,conversioncode and numericalmatch"),
        dict(gap="concentration_to_FER", status="FORMULAS_AVAILABLE_PARAMETERS_AND_ORIGINAL_INPUTS_MISSING", gained="PaperEq3 plusofficialFDSunit/FEDdocumentation", still_needed="Gasinputs,breathingV/incapacitationD,unit/timebasis,HCN handling,gridmapping and originalconversionimplementation"),
    ]
    csv_save(OUT / "gap_status.csv", gaps)
    known = dict(source="Originalpaper/localmanuscript andoriginalSupplement1;geometrypreviouslyrebuiltfrompaperfigures",
        not_original_FER_data=True, corridor_cells=20, total_traversable_cells_including_rooms_and_exits=24,
        cell_size_m=2, walking_speed_m_per_s=.3,
        topology=dict(corridor="R1..R20insequence", exit_A="aboveR1", exit_B="aboveR20", room_A="aboveR18", room_B="belowR14"),
        cases=[dict(case=1, evacuation_start_s=220, recommended_steps=15, paper_FED=.0441, DA_steps=8, paper_DA_FED=.1185),
               dict(case=2, evacuation_start_s=565, recommended_steps=15, paper_left_FED=.0695, paper_right_FED=.5853)],
        source_targets_are_scoring_only=True, continuous_rates_recoverable_from_colour_alone=False)
    save(OUT / "small_case_available_constraints.json", known)
    report = f"""## Material Passport

- Origin Skill：academic-research-suite / deep-research；pdf技能用于技术资料检查。
- Origin Mode：针对SafeStep复现缺口的定向公开资料检索与材料核对。
- Origin Date：2026-10-16。
- Verification Status：八项下载材料已保存、SHA256登记；四个GitHub源码文件的Git blob SHA1一致；FDS文本审计与视频抽帧检查完成。未运行FDS或新训练。
- Version Label：safestep_gap_material_search_v1。

# SafeStep缺失资料补齐检索：发现与可用性判断

**找到了有价值的材料，但最关键的小建筑逐格危险率仍没有补齐。** 新增成果是三个原始补充视频、一份与酒店案例高度相关的官方FDS生成模板、FD-Gen技术说明及版本化FDS用户手册。它们分别帮助核对路线、补充上游建模框架和澄清单位/接口，不能直接替代原论文的数值输入。

## 1. 这次具体检查了什么

检索论文DOI和出版商页面、NIST论文与新闻页面、作者GitHub三个公开仓库、SafeStep的分支/标签/发行版/全部Issue/提交历史、作者主页仓库，以及相关NIST官方FD-Gen项目与技术资料。另进行了限定域名的NIST数据、Zenodo和GitHub检索。未发现并不意味着所有位置都不存在；本报告只描述本次检查范围。

论文公开索引列出三个补充视频，并注明数据可按请求提供。[论文页面](https://www.sciencedirect.com/science/article/abs/pii/S2352710225033698)。本次出版商全文直接读取返回403，但三条公开补充文件地址均成功返回视频。没有把访问失败记成“附件不存在”。

SafeStep当前主分支最新提交仍为`{commits[0]['sha']}`，与原归档相同；当前API返回一个分支、六次提交、零标签、零发行版和零Issue。作者公开仓库列表为FD-Gen_beta、个人主页和SafeStep_v1。未检索到可用的公开SafeStep_v2实现或新增的原始危险率文件。[作者仓库](https://github.com/hqfang3/SafeStep_v1)

## 2. 三个原始补充视频：直接有用，但不能倒推出精确剂量

| 文件 | 内容 | 本地材料 |
|---|---|---|
| Supplement1 | 两个长走廊案例的路线与危险等级变化 | {link(MATERIALS/'mmc1.mp4','mmc1.mp4')} |
| Supplement2 | 酒店起点1 | {link(MATERIALS/'mmc2.mp4','mmc2.mp4')} |
| Supplement3 | 酒店起点2 | {link(MATERIALS/'mmc3.mp4','mmc3.mp4')} |

Supplement1的开头可见Case1与Case2两幅图，均有RoomB起点和左右出口；抽查终帧可见向左到ExitA的轨迹。它提供论文图外的动态视觉核对。我们此前已从正文图重建小地图，视频属于补强核对，不能说是首次获得拓扑。导出的帧是检查用原始视频帧，保留在materials目录。

这些视频显示的是危险等级配色和路径，不是逐格连续气体浓度或FER表。一个颜色对应一段数值范围，多种连续危险率都可能产生同样颜色。因此不能靠调颜色或插值反推四位小数的原始FED，更不能用论文终点目标反向拟合后称为独立复现。

视频播放时长与模拟火灾物理时间不同，不用视频帧率代替疏散时间步。其余轨迹长度、起始时刻及论文FED目标已整理成{link(OUT/'small_case_available_constraints.json','小案例可用约束')}，明确标注这些不是原始危险率数据。

## 3. 最值得继续利用的线索：NIST官方商业建筑模板

来源为同一作者参与维护的[usnistgov/FD-Gen](https://github.com/usnistgov/FD-Gen)，下载文件为[example_commercial.fds](https://github.com/usnistgov/FD-Gen/blob/main/example/example_commercial.fds)。其几何命名包含TOP、RIGHT、BOTTOM、LEFT和ROOM24，与SafeStep酒店的走廊/起火房间高度相关；CO产率0.042和烟炱产率0.198也与论文表4对应。

这个关联是候选同源建模框架的证据，尚不能证明该文件就是论文实际工况。逐项核对发现：

| 项目 | 论文酒店工况 | 下载模板 | 判断 |
|---|---|---|---|
| 燃料与产率 | 聚氨酯；CO0.042、烟炱0.198 | POLYURETHANE_GM27，产率相同 | 指定产率一致 |
| 火源位置 | Room24中心 | Room24范围内随机火源 | 需要固定与确认 |
| 热释放率 | 326秒达到5MW的t²火 | 随机初期、增长、峰值和衰减；峰值1–8MW | 不是同一条曲线 |
| 火房门 | 100秒开启 | 三阶段随机开启；含占位计时器 | 不一致 |
| 窗户 | 150℃破裂 | 未找到该显式阈值；存在永久OPEN边界 | 边界需核对 |
| 气体观测 | 58区域、2米高度、CO/CO₂/HCN | 243个DEVC条目；未显式配置对应58区域三气体设备 | 未补齐 |
| HCN/CO₂ | 论文公式需要 | 模板全文无HCN/HYDROGEN CYANIDE及CARBON DIOXIDE字符串 | 不能直接生成所需表 |

模板包含FD-Gen专用随机化命令，且热释放率引用待生成的RAMP；本次扫描没有找到RAMP定义块。它是生成器输入，不能不经处理就当成原论文FDS案例执行。上述是文本核对，未做FDS语法验证或仿真。

完整比较见{link(OUT/'paper_vs_fdgen_comparison.csv','九项差异表')}。本地原样文件为{link(MATERIALS/'example_commercial.fds','商业建筑模板')}；另下载的{link(MATERIALS/'example_Single_Story_Gas_1.fds','住宅模板')}用于了解建模接口，不对应论文的24格长走廊。

## 4. 技术说明和手册可以补什么

{link(MATERIALS/'NIST.TN.2332.pdf','NIST TN2332')}介绍FD-Gen的输入结构、火源与HRR/开门时间的随机化及商业建筑例子，来源为[官方技术说明](https://doi.org/10.6028/NIST.TN.2332)。它补充如何构建和生成FDS案例的依据，不提供SafeStep小建筑的原始FER。

已保存{link(MATERIALS/'FDS_User_Guide_6.9.1.pdf','FDS6.9.1用户手册')}。选择这个版本是因为FD-Gen说明其测试过FDS6.9.1，不代表已经确认SafeStep论文使用的确切FDS版本。当前FDS官方主页的现行版本另有更新；为复现不能自动把最新软件行为当成原实验行为。[FDS官方文档入口](https://pages.nist.gov/fds-smv/manuals.html)

手册关于气体观测与FED的部分给出了有用核对点：VOLUME FRACTION、ppm与百分数的区别；部分FED表达式的时间单位为分钟；HCN生成与输出需要明确配置。SafeStep式3还涉及呼吸量V、致失能剂量参数D，以及CO₂对呼吸的放大作用。这些参数和气体原始序列目前没有一套经来源确认的完整转换实现。

此外，FDS6.9.1内置FED的总项、HCN函数形式与SafeStep式3并非逐项同一。后续应从相同气体序列按论文式3独立计算并逐项核对，不能直接导出FDS内置FED列替代作者表；也不能只因公式写法不同就认定作者转换有错。

相关的{link(MATERIALS/'thermal_risks_nist.bin','商业建筑热风险论文')}提供同一研究方向的温度监测背景，是建筑框架的辅助线索；其温度数据不能替代CO/HCN/CO₂序列。

## 5. 三个主要缺口现在补到哪里

| 复现缺口 | 本轮进展 | 是否已经补齐 |
|---|---|---|
| 小建筑四个剂量与约62%改善 | 有原始补充视频，可再次核对几何与路线 | **否，连续FER/浓度缺失** |
| 正文训练完整流程 | 核实公开仓库仍为原固定版本，无新增公开澄清 | **否** |
| 酒店FDS与转换上游 | 找到强相关官方生成模板、技术说明和手册 | **部分，有框架但不是原工况** |

需要更正先前容易产生的误解：浓度到FER的**数学公式已在论文中公开**。真正的缺口是原始浓度输入、参数和单位落实、网格/测点映射，以及产生作者危险率表的实现；不是说完全没有公式。

## 6. 下一步的合理边界

现在值得做的是以下载模板为起点，逐项补出“酒店FDS重建候选”的静态配置差异清单和58区域气体测点设计。只有固定工况、确认转换与观测后，才有理由评估是否运行一次小规模FDS验证。派生文件应另立版本，标注为重建候选，保留原始下载文件；是否恢复作者数值仍以独立比较决定。

小建筑方面继续保留为原始资料需求：若拿不到相应FER或原始气体序列，只能做注明假设的重建，不能认定0.0441等数值和约62%已经复现。增加DQN训练量解决不了这个输入缺口。

最小资料需求已整理为{link(OUT/'minimal_missing_materials.md','精确缺失资料清单')}。本轮未联系作者、未提交Issue，未运行FDS、未重训、未调用付费LLM/API；48个旧人群最终场景未读取。

## 7. 来源与下载记录

{link(OUT/'material_inventory.csv','八项材料清单')}包含网址、本地路径、字节数、用途和SHA256。GitHub API与下载日志保存在同目录；四个下载源码文件还核对了Git blob SHA1。部分数据站点与DOI元数据端点返回404，已记录为本次获取失败，不能据此断言对应数据在所有渠道都不存在。

新增材料只写入`outputs/safestep_gap_material_search_20261016`。旧复现报告和结果没有改写；本报告作为截至2026-10-16的资料补充。
"""
    missing = """# SafeStep最小缺失资料清单

这是材料需求记录，本轮没有向作者或其他人员发送。

1. **小建筑Case1/Case2危险率**：每个状态与绝对火灾时间对应的FER表、单位、输出间隔、状态坐标与编号；最好包括0—900秒序列。两个案例起始时刻为220秒和565秒。需要确认出口剂量与起点初始剂量的约定。
2. **等价原始气体输入**：若不提供FER，需CO、HCN、CO₂的原始时间序列、监测高度/位置、浓度单位、V和D取值及转换代码。文中已公开公式，缺的是可执行的完整输入与实现对应关系。
3. **小建筑FDS案例**：原始.fds、FDS版本、网格/房间尺寸、火源面积与位置、3MW/253秒火曲线、材料与CO/烟炱/HCN设置、通风与开门/破窗、气体输出设备。
4. **酒店原始FDS案例**：用于Room24、5MW/326秒、100秒开门、150℃破窗的固定工况，58个2米高气体测点与作者表区域编号的对应关系；确认与FD-Gen商业模板是否同源以及修改内容。
5. **最终论文训练版本**：与公开Beta的版本关系、网络结构、随机种子、探索率规则、优化频率、四输出/选中动作损失、超时自举、同步时机与采样/初始剂量约定。论文描述存在歧义的部分不能仅靠一次跑分反推。

优先级：小建筑FER或等价气体序列最高；酒店固定FDS及转换实现其次；训练细节同时列出，但暂不因此扩大训练。
"""
    save(OUT / "minimal_missing_materials.md", missing)
    save(REPORT, report)
    save(OUT / "README.md", f"""# SafeStep缺口资料检索入口

- {link(REPORT,'检索报告与可补齐判断')}
- {link(OUT/'material_inventory.csv','下载材料清单与来源')}
- {link(OUT/'paper_vs_fdgen_comparison.csv','酒店论文工况与FD-Gen模板差异')}
- {link(OUT/'gap_status.csv','当前缺口状态')}
- {link(OUT/'minimal_missing_materials.md','最小缺失资料清单')}
- {link(MATERIALS/'mmc1.mp4','小建筑原始补充视频')}
- {link(MATERIALS/'example_commercial.fds','官方商业建筑模板')}

原始材料保存在materials目录。下载代码未执行，模板不是已验证的原论文工况，视频不是连续危险率表。
""")
    preserved = read(ROOT / "outputs/safestep_reproduction_package_20261016/source_manifest.json")["input_sha256"]
    for name, expected in preserved.items():
        assert digest(ROOT / name) == expected, name
    verification = dict(status="PASS_MATERIAL_AUDIT_NOT_NUMERICAL_REPRODUCTION", inventory_count=len(inventory),
        git_blob_identity_matches=git_blob_matches, commercial_namelist_counts=counts,
        unchanged_previous_selected_inputs=len(preserved), source_commit_unchanged=commits[0]["sha"],
        original_small_case_continuous_FER_found=False, original_small_case_FDS_found=False,
        related_hotel_FDS_template_found=True, supplemental_videos_downloaded=3,
        FDS_runs=0, training_updates=0, LLM_requests=0, external_messages_sent=0,
        old_final48_crowd_scenarios_accessed=False)
    save(OUT / "verification.json", verification)
    artifacts = [p for p in OUT.rglob("*") if p.is_file()] + [REPORT, Path(__file__)]
    save(OUT / "completion_manifest.json", dict(status="TARGETED_MATERIAL_SEARCH_COMPLETE",
        created_at=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts}))
    print(json.dumps(verification, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
