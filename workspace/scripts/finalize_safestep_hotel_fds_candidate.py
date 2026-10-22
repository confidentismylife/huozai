"""Write the Chinese audit report and freeze current reconstruction artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import build_safestep_hotel_fds_candidate as builder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/safestep_hotel_fds_candidate_20261016"
REPORT = ROOT / "reports/SafeStep酒店FDS重建候选_区域映射与浓度接口_2026-10-16.md"


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    assert not path.exists(), path
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def link(path, label):
    return f"[{label}]({path.as_posix()})"


def main():
    assert not (OUT / "completion_manifest.json").exists(), "Completed artifacts are frozen"
    summary = load(OUT / "build_summary.json")
    checks = load(OUT / "verification.json")
    spec = load(OUT / "frozen_specification.json")
    source = load(OUT / "source_manifest.json")
    for name, sha in {**source["protected_previous_sha256"], **source["input_sha256"]}.items():
        assert digest(ROOT/name) == sha, name
    mapping = builder.get_mapping(ROOT/spec["source_mapping"])
    source_blocks = builder.blocks((ROOT/spec["source_template"]).read_text("utf-8"))
    rows = builder.propose_locations(mapping, source_blocks)
    draft, devices, _, _, _ = builder.derive_input(source_blocks, rows, spec)
    assert draft == (OUT/"safestep_hotel_candidate.fds.in").read_text("utf-8")
    assert devices == (OUT/"gas_devices_58x3.fds.fragment").read_text("utf-8")
    modules = ["build_safestep_hotel_fds_candidate.py", "safestep_gas_to_fer.py",
               "verify_safestep_hotel_fds_candidate.py", "finalize_safestep_hotel_fds_candidate.py"]
    for name in modules:
        compile((ROOT/"scripts"/name).read_text("utf-8"), name, "exec")
    save(OUT/"finalization_audit.json", {
        "draft_exactly_matches_final_builder_regeneration": True,
        "gas_fragment_exactly_matches_final_builder_regeneration": True,
        "protected_previous_files_unchanged": len(source["protected_previous_sha256"]),
        "python_modules_compile": modules,
        "diagram_v2_visually_inspected": True,
        "figure_revision": "Only bottom corridor label placement changed; original image retained",
        "post_verification_draft_change": "NIST source/date attribution comments added; namelist records unchanged",
        "FDS_runs": 0,
    })
    report = f"""# SafeStep酒店FDS重建候选：区域映射、火灾输入与浓度接口

日期：2026-10-16。对象：Fang等人的SafeStep酒店案例。性质：**原始材料约束下的重建草案，不是作者原始FDS算例，也没有新增物理仿真结果。**

这一步已经把“找到一个相关模板”推进到了“可以逐项审查的火灾输入草案”。完成了58个候选物理区域与61个非出口RL网格的对应、174路气体测量配置、论文明确火灾条件的写入，以及浓度到FER的显式单位接口。尚缺的物理参数保留为空，没有拿既有FED成绩反向调参。

## 1. 为什么先做这一项

此前已经用作者公开的危险率表和权重复现酒店四条路线的FED，四位小数与论文一致。这说明公开下游数据和规划程序可以重放；它不能证明我们已经重建“火灾产生烟气→气体浓度→危险率”的上游过程。

新找到的NIST FD-Gen商业建筑模板与酒店布局相关，具有相同的CO和烟炱产率，还包含58个走廊温度测点。这让上游重建有了具体起点。但它使用随机火源、随机热释放曲线和分段开门，不是可直接拿来运行的论文原始输入。

本轮只推进酒店上游重建。小建筑的两种案例仍缺原始连续危险率和物理输入；论文约62%的降幅**没有因此被复现**。LLM/PPO扩展也没有恢复训练。

## 2. 材料来源与可追溯性

| 材料 | 来源与地位 | 本轮用途 |
|---|---|---|
| 论文正文、Table 4、Eq. 3 | [论文DOI](https://doi.org/10.1016/j.jobe.2025.115132)；已有本地作者稿 | 火灾条件与数学表达式 |
| example_commercial.fds | [NIST FD-Gen固定提交中的模板](https://github.com/usnistgov/FD-Gen/blob/428a4153239efa9c4c4b6b20670990c52eaf926a/example/example_commercial.fds) | 几何、网格与温度测点的候选来源 |
| coordinates2areanumber.py | [SafeStep公开仓库](https://github.com/hqfang3/SafeStep_v1/tree/573f613420e21e9a6731bc56fa36ea0b3f3a05f6)；已有本地固定版本 | 作者RL网格和区域标签 |
| FDS 6.9.1用户指南 | [FDS官方版本手册](https://github.com/firemodels/fds/releases/download/FDS-6.9.1/FDS_User_Guide.pdf)；已有本地PDF与文本 | 控制逻辑、气体输出、单位和时间平均行为 |

商业模板SHA256：`{spec['source_template_sha256']}`。原文件的Git blob身份也已核对。NIST来源、修改日期与完整许可随草案保存。

本轮前后核对了**{checks['protected_previous_files_unchanged']}个先前选定文件**的哈希，全部保持不变。此数量只代表这份清单的覆盖范围，不代表全盘所有历史文件都经过审计。

## 3. 已写入草案的条件

| 项目 | 当前草案 | 证据性质 |
|---|---|---|
| 环境 | 20℃、相对湿度40% | 论文明确；湿度以百分数写入 |
| 火灾位置 | Room24；候选燃烧器位于模板室内测点附近 | 房间编号来自论文；精确中心未核实 |
| CO/烟炱产率 | 0.042 / 0.198 kg/kg | 论文与模板一致 |
| 起火增长 | t²增长，326秒达到5,000 kW | 论文明确 |
| 火灾房门 | 初始关闭，100秒整门移除 | 论文开门时间明确；替换了模板三个重叠门体 |
| 窗口触发阈值 | 150℃ | 论文明确；温度的具体定义未知 |
| 气体采样 | 58处，高2.0 m，CO/CO₂/HCN各一路 | 气体、数量和高度来自论文；XY对应仍为推断 |
| 输出时间 | 0—900秒、间隔5秒 | 对齐公开危险率表的接口要求 |
| 疏散计时 | 火灾时间300秒开始疏散 | 论文明确；接口同时保存两种时钟 |

热释放率采用1秒间隔的离散RAMP。增长段整数秒节点与t²函数一致；线性连接造成的最大理论热释放率误差约{checks['ramp_linear_interpolation_max_HRR_error_kW']:.5f} kW，增长段积分误差约{checks['ramp_growth_integral_error_kJ']:.3f} kJ。这只是规定曲线的离散误差，不是FDS实际燃烧热释放率的验证。

## 4. 四项会改变烟气结果的近似

1. **燃烧器尺寸与精确位置。** 采用模板生成器的2×1×0.2 m尺寸，候选边界为x=3.4—5.4、y=23.7—24.7、z=0—0.2 m。它位于模板Room24温度测点附近，不能证明等于房间几何中心或作者原燃烧器。使用规定HRR的燃烧面，也不能称为已经复现聚氨酯床垫热解。
2. **326秒之后的曲线。** 当前草案保持5 MW到900秒。论文没有提供足够的峰后曲线信息，这是明确的重建假设，可能改变后续剂量。
3. **窗户模型。** 原模板窗口是一个永久HOLE。草案把它改为受控HOLE：初始未开洞，窗边气温超过150℃后开洞。这样能够表达“先封闭、后打开”，但初始屏障仍继承INERT墙体，既没有玻璃受热传导，也没有验证玻璃自身温度。论文所说的窗户温度不能直接等同于这个气温代理量。
4. **浓度输出。** 当前指定湿基体积分数、瞬时点值和5秒输出。FDS默认设备输出会在输出间隔内做时间平均；草案显式关闭时间平均。作者实际采用哪一种并未提供，两种处理经过非线性毒性公式后可能得到不同FER。

窗外的OPEN边界在房间外侧计算域边缘，不等于窗洞。草案保留它，只修改墙上的窗口控制，避免把环境边界误当成需要封闭的窗户。

## 5. 58个物理区域如何接到61个RL格

| 区域 | 候选物理位置规律 | 数量 |
|---|---|---:|
| R1—R21 | 顶部走廊，y=27.1 m，x=2—42 m | 21 |
| R22—R36 | 底部走廊，y=12.1 m，x=12—40 m | 15 |
| R37—R43 | 左侧上段，x=10.2 m，y=13.1—25.1 m | 7 |
| R44—R50 | 右侧走廊，x=33.2 m，y=13.1—25.1 m | 7 |
| R51—R55 | 左侧下段，x=10.2 m，y=11.1—3.1 m | 5 |
| R56—R58 | 电梯前室，x=22.6 m，y=14.7/17.1/19.7 m | 3 |

每个候选XY都能找到模板中对应的原温度测点。但“该物理点就是作者R编号的中心”仍是拓扑推断，表中每一行都带有`INFERRED_NOT_AUTHOR_CONFIRMED`标记。

作者映射里的R56、R57、R58各对应两个RL格，因此是58个物理区域、61个非出口格，另有两个出口。不能误把61个格当成61个独立气体位置。

![候选物理测点与作者RL网格]({(OUT/'region_mapping_candidate_v2.png').as_posix()})

图中左侧只显示计算网格范围与采样位置，没有画全部墙体；左右两图的比例不同。实测点坐标与RL坐标也不能简单用一个全局等比例变换对应。

静态检查结果：6个网格共{int(summary['mesh_cells']):,}个计算单元，58个候选点均在网格范围内，且在原始连续坐标下未命中障碍物。58个点至少有一个坐标恰在计算网格面上；这不是已发现的错误，但后续必须检查FDS实际取值单元。

继承几何中有{summary['no_positive_volume_mesh_overlap_obstructions']}个障碍物与当前网格没有正体积交叠，其中可能包含域外建筑对象、边界对象或零厚度对象；没有把它们自动删掉。当前检查不包含完整的网格吸附、墙洞相互作用、网格连通性或求解器初始化验证，不能据此宣称几何完全有效。

## 6. 浓度转FER接口已经实现什么

实现文件：{link(ROOT/'scripts/safestep_gas_to_fer.py', '显式单位转换脚本')}。

接口按作者稿Eq. 3的表达式分别计算CO与HCN贡献，再相加。当前提出的计算约定是CO和HCN使用ppm，CO₂使用百分数；它们从FDS的`mol/mol`体积分数分别乘以10⁶和100。**这些是待核实的接口约定，尚不能称为作者实现已经采用的单位。**

公式中的CO₂倍增因子为`exp(CO2_percent/5)`；CO项包含`V/D`，HCN项不包含这个比值。V是每分钟呼吸空气体积，D是导致失能的COHb百分数。脚本要求调用者明确提供V、D、公式速率的时间基准以及参数来源。

当公式的速率单位被明确设为“每分钟”时，脚本除以60后输出`FER_CO_per_s`、`FER_HCN_per_s`和`FER_total_per_s`；如果被明确设为“每秒”，则不做这个除法。程序不会自动猜测哪个才是作者设置。原参数模板中的V、D和时间基准全部为null，直接使用会报错。

接口同时保留`fire_time_s`与`evacuation_elapsed_s=fire_time_s-300`，因此0秒火灾数据的疏散相对时间为−300秒，900秒对应600秒。预疏散数据不会被截成0，避免把火灾时间和疏散时间混用。

程序会拒绝缺失HCN、负浓度、NaN、把质量分数当作体积分数、重复时间、缺失单位或不完整的0—900秒时间网格。不会把未提供的HCN填成零，不会静默插值，也没有把FDS内置FED直接替代作者公式。

## 7. 本轮验证实际支持什么

本轮{checks['check_count']}项脚本断言通过，其中大量是文件哈希、ID引用和逐设备对应检查；**它们不是{checks['check_count']}个独立实验。**

浓度接口的端到端测试构造了58区域×181时间点，共{checks['synthetic_export_rows']:,}行输出。输入是明确标记的常量合成浓度，V=30、D=30也只是任意算术测试数值。检验了单位等价、分钟与秒的60倍关系、CO₂放大因子、已知恒定速率的积分，以及缺失输入的拒绝行为。这证明代码和数据格式接通，不能作为真实火灾或人体反应的验证数据。

| 项目 | 本轮结果 |
|---|---|
| 物理区域及作者格映射 | 58区域、61非出口格，候选映射已列出 |
| 新气体设备 | 174路，ID、坐标、气体名称完整 |
| 静态坐标检查 | 58/58通过；不等同于求解器验证 |
| 单位与数据接口 | 合成测试通过，缺失参数明确报错 |
| 先前选定文件保护 | {checks['protected_previous_files_unchanged']}个哈希保持不变 |
| FDS解析或初始化 | 0次 |
| 真实火灾仿真 | 0次 |
| 新训练、新LLM调用 | 均为0 |
| 原48个最终人群场景 | 未读取 |

## 8. 是否可以直接跑论文火灾

**目前不可以。** 本机能够找到`C:/Program Files/firemodels/FDS6/bin/fds.exe`，但其版本尚未核实，安装有程序也不等于输入和物理参数已经齐备。

草案使用`.fds.in`后缀，HCN产率写成`{{{{REQUIRED_HCN_YIELD_KG_PER_KG}}}}`占位符，刻意保留为不能直接执行的输入。FDS手册所列HCN默认产率为0不能代替论文缺失值；若无说明直接使用默认值，会把一个关键气体贡献悄悄去掉。

需要区分两类缺口：

- **运行缺口：** 需要显式设定有依据的HCN产率，核对求解器版本，然后进行初始化、设备单元定位与短时稳定性检查。
- **精确复现缺口：** 还需要原始几何与区域坐标、原燃烧器与峰后曲线、窗户温度定义、气体与FER转换参数、原始浓度数据及数值设置。即使我们选一组透明假设把算例跑通，也只能得到重建模拟，不能改称原论文精确复现。

## 9. 后续推进边界

下一项可执行工作是**单独建立一个带参数来源的探索性版本**，先核对本机FDS版本，用极短算例验证输入能被解析、门窗控制方向和测点所在单元。它应与当前草案分目录保存，任何用于排错的临时产率或简化设置都不能流入“论文复现结果”。

900秒完整烟气计算应放在输入检查之后。此前缺失的V、D等可先保留为转换层的参数，不妨碍独立验证火灾求解器，但没有这些信息就不能对齐作者FER/FED。若无法获得原始参数，研究目标应明确改为带假设和敏感性分析的独立重建，不再无限增加训练次数来试图补偿物理输入缺失。

## 10. 交付文件

- {link(OUT/'README.md', '候选包入口与文件说明')}
- {link(OUT/'safestep_hotel_candidate.fds.in', '酒店FDS输入草案（含必填占位符）')}
- {link(OUT/'region_mapping_candidate.csv', '58区域与RL格对应表')}
- {link(OUT/'parameter_register.csv', '已知、假设与未知参数登记表')}
- {link(OUT/'gas_devices_58x3.fds.fragment', '174路气体设备片段')}
- {link(OUT/'fer_parameters_REQUIRED.json', '缺失参数模板')}
- {link(OUT/'readiness.json', '运行与精确复现的缺口清单')}
- {link(OUT/'verification.json', '静态与合成接口验证明细')}

本轮完成的是一份可追溯、可审查的上游重建草案；已有酒店下游数值复现的结论保持原样，没有新增加“已复现原始烟气物理过程”的结论。
"""
    readme = f"""# SafeStep酒店FDS候选包

本目录由NIST FD-Gen商业建筑模板派生，修改日期2026-10-16。来源为National Institute of Standards and Technology，完整许可见FD-Gen_LICENSE.md。原下载文件保持不变。

**状态：静态草案与合成接口测试完成；尚未运行FDS，不是作者原始算例。**

优先阅读{link(REPORT, '中文报告')}与{link(OUT/'readiness.json', '缺口清单')}。

| 文件 | 用途 |
|---|---|
| safestep_hotel_candidate.fds.in | 含必填HCN产率占位符的输入草案 |
| parameter_register.csv / frozen_specification.json | 区分论文条件、公开表格要求、候选假设与未知参数 |
| region_mapping_candidate.csv | 58个候选物理点与61个非出口RL格 |
| region_point_geometry_screen.csv | 连续坐标下的初步几何检查 |
| region_mapping_candidate_v2.png / .svg | 已人工查看的区域图；旧图也保留 |
| gas_devices_58x3.fds.fragment | CO/CO₂/HCN共174路测量配置 |
| specified_HRR_ramp.csv | 规定的热释放率曲线，不是FDS实测输出 |
| fer_parameters_REQUIRED.json | V、D、时间基准尚待提供，不可直接用于真实转换 |
| template_derivation_log.json | 删除生成器指令、替换火源与门窗控制的记录 |
| synthetic_interface_checks/ | 全部为合成算术测试；不得用于论文火灾效果评价 |
| source_manifest.json / completion_manifest.json | 原输入与新交付文件哈希 |

转换脚本可在将来获得完整浓度和有依据参数后使用。以下是命令格式，示例路径不会自动存在，必须指定新输出目录：

```powershell
& 'D:/miniconda3/python.exe' scripts/safestep_gas_to_fer.py --input '<完整FDS设备CSV路径>' --parameters '<已补齐且注明来源的参数JSON路径>' --output-directory '<新转换输出目录>'
```

不能只把.fds.in改成.fds就当成已经完成物理重建，也不能把合成测试中的V=30、D=30复制成论文参数。已有冻结候选和验证不覆盖重写；后续物理排错与敏感性版本需另存。
"""
    save(REPORT, report)
    save(OUT/"README.md", readme)
    checked_links = []
    for document in (REPORT, OUT/"README.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text("utf-8")):
            if not target.startswith("https://"):
                assert Path(target).is_file(), target
                checked_links.append(target)
    files = sorted(p for p in OUT.rglob("*") if p.is_file()) + [REPORT]
    files += [ROOT/"scripts"/name for name in modules] + [builder.CONFIG]
    save(OUT/"completion_manifest.json", {
        "status": "STATIC_RECONSTRUCTION_CANDIDATE_COMPLETE_PHYSICAL_INPUTS_STILL_MISSING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        "local_links_verified": len(checked_links),
        "protected_previous_files_unchanged": len(source["protected_previous_sha256"]),
        "physical_FDS_runs": 0, "training_updates": 0, "LLM_requests": 0,
        "external_messages_sent": 0, "old_final48_crowd_scenarios_accessed": False,
    })
    print(json.dumps({"report": REPORT.as_posix(), "artifact_count": len(files),
                      "local_links_verified": len(checked_links), "physical_FDS_runs": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()
