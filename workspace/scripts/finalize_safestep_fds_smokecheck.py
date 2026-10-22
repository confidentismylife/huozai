"""Freeze the solver diagnostic package and write a scope-limited Chinese report."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/safestep_hotel_fds_smokecheck_20261016"
REPORT = ROOT/"reports/SafeStep酒店FDS短时运行与门窗控制检查_2026-10-16.md"


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
    assert not (OUT/"completion_manifest.json").exists(), "Preserve a completed package"
    summary = load(OUT/"run_summary.json")
    verification = load(OUT/"verification.json")
    manifest = load(OUT/"source_manifest.json")
    clocks = load(OUT/"native_timestamp_audit.json")
    for name, sha in manifest["protected_previous_sha256"].items():
        assert digest(ROOT/name) == sha, name
    with (OUT/"run_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        runs = list(csv.DictReader(f))
    total_s = sum(float(r["elapsed_s"]) for r in runs)
    run_table = "\n".join(f"| {r['case']} | {r['classified_status']} | {float(r['reached_fire_time_s']):g} | {float(r['elapsed_s']):.2f} |" for r in runs)
    warnings = sorted({line.strip() for case in (OUT/"runs").iterdir() for line in (case/"stderr.log").read_text("utf-8", errors="replace").splitlines() if line.strip().startswith("WARNING:")})
    # Independent last audit of both output and stderr, not just return code.
    for row in runs:
        case = OUT/"runs"/row["case"]
        stderr = (case/"stderr.log").read_text("utf-8", errors="replace")
        output = case/(case.name+".out")
        combined = stderr+(output.read_text("utf-8", errors="replace") if output.exists() else "")
        if row["classified_status"] != "FDS_ERROR":
            assert not re.search(r"^\s*ERROR(?:\(|:)", combined, re.M), case.name
            assert "STOP: Set-up only" in combined or "STOP: FDS completed successfully" in combined, case.name
    save(OUT/"finalization_audit.json", {"protected_previous_files_unchanged": len(manifest["protected_previous_sha256"]),
         "both_stderr_and_out_checked": True, "success_requires_FDS_completion_marker": True,
         "all_processes_finished": all((OUT/"runs"/r["case"]/"run_result.json").is_file() for r in runs),
         "initial_v1_status_in_run_result_is_process_status_only": "V1 returned0 despite anFDS input error; corrected interpretation retained inrun_comparison.csv",
         "all_input_and_output_revisions_preserved": True, "global_environment_modified": False})
    report = f"""# SafeStep酒店FDS短时运行、气体输出与门窗控制检查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-16
- Verification Status: VERIFIED，仅限下述本地软件诊断
- Version Label: safestep_hotel_fds_smokecheck_20261016
- Evidence scope: 求解器原始输入、日志、CSV、SMV；不包含论文原始浓度或新疏散效果实验

## 本轮实际完成了什么

**已经真正启动FDS并完成酒店短时计算。** 共执行6次输入运行：首次初始化被旧设备命名格式拦住；修正后的初始化通过；随后完成2次酒店布局的2秒运行和2次微型温度控制的1秒运行。所有输入版本与失败日志均保留，没有启动900秒完整火灾，没有计算FER/FED。

酒店58个区域的174路气体输出全部存在，单位为`mol/mol`。门体与窗洞的控制方向已经在求解器输出中得到核实。对FDS初始化后生成的障碍物网格进行检查，58个测点相邻的候选单元均未落入体积障碍物。

## 1. 运行环境与参数边界

| 项目 | 本轮使用 |
|---|---|
| FDS版本 | FDS-6.11.1-0-gff928db-release |
| MPI库 | Intel MPI 2021.16 for Windows |
| 并行设置 | 1个MPI进程，2个OpenMP线程 |
| 酒店计算网格 | 继承候选草案的6个网格、93,981个单元 |
| 单次运行硬超时 | 240秒；本轮未触发 |
| HCN产率 | **0.01 kg/kg，仅为任意指定的软件诊断参数** |
| 人体参数V、D与公式时间基准 | 仍为空，没有参与计算 |
| 原48个最终人群场景 | 未读取 |

0.01不是从论文找回的HCN产率，也不是经过实验校准的材料参数。使用非零值是为了让含氮反应和HCN输出通道能够参与软件检查；本轮输出不得用于论文毒气剂量、路线效果或材料燃烧结论。

本轮启动器仅为子进程补入FDS安装目录内`bin/mpi`的DLL搜索路径，未修改全局PATH或系统配置。

## 2. 两个实际运行问题已经处理

**启动依赖问题。** 直接调用程序得到Windows错误`0xC0000135`，对应缺少DLL依赖。将安装包自带MPI路径加入当前子进程环境后，`-v`版本查询和求解器均可启动。

**设备名称兼容性。** 第一次初始化报`ERROR(899)`：旧模板中的设备名称含逗号，例如`T1_(2,27)`。新的诊断版本将232个此类温度设备改为`LEGACY_TC_001`等名称，保留完整原名对应表；174个新气体设备ID本来就不含逗号，没有被改名。这个修正没有改变物理几何。

另发现：这次FDS初始化报错后，进程退出码仍为0。第一版`run_result.json`中的进程状态不能当成求解成功证据；独立分析已经将该次运行判为`FDS_ERROR`。后续检查同时要求无FDS错误、存在成功结束标志以及预期输出完整。

## 3. 完整运行记录

| 算例 | 核实后的状态 | 到达的模拟时间（秒） | 实际耗时（秒） |
|---|---|---:|---:|
{run_table}

六次输入运行耗时合计约{total_s:.1f}秒。该时间不能用于线性推算900秒火灾的成本：后续火势、流速和稳定性要求会改变时间步长。

- `hotel_init_v1`：保留首次错误，不覆盖。
- `hotel_init_v2`：零时长初始化，解析完整酒店布局和设备配置。
- `hotel_short`：保留100秒开门和150℃窗边气温触发，只计算起火后0—2秒。
- `hotel_control_timers`：专门验证执行方向，诊断中将门改为0.5秒打开，窗洞改为0.75秒的时间触发；它不验证窗户温度模型。
- 两个`micro_temperature`：单独搭建的小网格、不含燃烧。分别以20℃和160℃环境检验150℃气温阈值。

## 4. 门和窗是否真的变化

不只检查设备逻辑，还读取了SMV中的`HIDE_OBST`几何事件。

| 检查 | 观察到的设备逻辑 | 观察到的几何结果 |
|---|---|---|
| 酒店原时序短算例 | 前2秒门、窗均未触发 | 没有提前移除门或窗口填充体 |
| 酒店加速开门 | 0.519543秒，门设备由T变F | 0.692724秒，左右相邻两个网格中的门体均移除 |
| 酒店加速开窗 | 0.865905秒，窗口设备由F变T | 1.039086秒，Room24网格中的窗口填充体移除 |
| 20℃微型算例 | 未达到150℃阈值 | 窗口保持封闭 |
| 160℃微型算例 | 初始气温160℃，0秒达到触发条件 | 0.01秒，窗口填充体移除 |
| 两个微型算例的门 | 均在0.5秒切换状态 | 均在0.51秒移除门体 |

结果支持“门初始存在，触发后移除；窗洞初始被填充，触发后打开”的程序执行方向正确。**它不支持玻璃传热、温差破裂或真实破窗时间已经被验证。** 酒店草案仍采用窗边气温代理，初始屏障继承INERT墙体。

逻辑触发与几何移除并非完全同时。酒店本次步长约0.173秒，几何移除比设备逻辑晚约一个时间步。因此不能把输入中写了100秒，直接表述成“恰在100.000秒完成几何变化”；完整计算时仍需核查时间步误差。

## 5. 58个测点位置与174路数据

初始化阶段输出包含全部174个气体设备及对应物种：CO、CO₂、HCN各58路。读取后的CSV包含13个原始时间点，整理为58×13＝**754行诊断浓度记录**，保留原始时间戳与体积分数单位。

检查没有发现缺失气体列、重复时间、NaN、负浓度、超过1的体积分数或气体体积分数之和超过1。读取代码复用了上一轮已经冻结的原生CSV读取接口，没有强行通过不满足条件的规则时间表转换。

使用FDS实际生成SMV中的障碍物网格边界检查采样位置。因为58个测点都至少有一个坐标在网格面上，本轮保守地检查其相邻的全部候选单元，而非猜测求解器具体选了哪一侧。**58/58位置的所有相邻候选单元均未命中初始化后的体积障碍物。**

这个检查比原始连续坐标检查更进一步，但没有找回作者原始区域坐标，也没有证明这些点一定代表正确的区域中心；零厚度壁面及跨网格流动连通性仍需另外核查。

## 6. 新发现：输出周期不等于精确时间网格

虽然诊断设置`DT_DEVC=0.1`秒，实际输出时间前几项为：

```text
0.00000000, 0.17318098, 0.34636197, 0.51954295, ... , 2.00000000
```

实际相邻输出间隔约为{clocks['actual_min_interval_s']:.6f}—{clocks['actual_max_interval_s']:.6f}秒，明显不是精确的0.1秒整数网格。原始FDS按实际求解步输出，不能只凭输出频率设置，就假定数据已经是规则时间表。

上一轮转换脚本对0—900秒、每5秒一行的严格校验是一个有用的拒绝条件，但它不能直接保证能接收FDS原生输出。后续需要单独定义：按原始时间积分，还是用明确的方法生成5秒表；若插值，应说明在浓度层还是FER层执行。毒性公式非线性，两种做法不能随意互换。本轮没有对时间或浓度做插值，也没有修改冻结的旧转换脚本。

## 7. 这轮不能说明烟气效果

酒店短时算例只覆盖起火后的前2秒。实际记录的最高HRR约为{summary['actual_HRR_peak_kW_in2s_diagnostic']:.6f} kW，远未达到5,000 kW火灾阶段。58个走廊测点的CO、HCN最大值均为0；CO₂包含环境背景值。这可以证明对应气体列被正常写出，**不能证明非零毒气到达走廊后的浓度预测准确**。

本轮没有计算暴露剂量、比较路线、重新训练RL，也没有检验300秒开始疏散时的环境。已有酒店四条路线FED复现结论保持原样，小建筑约62%的降幅仍未完成复现。

## 8. 尚未消除的警告与后续边界

求解器仍报告两条燃料相关警告：建议用SPEC定义燃料组成；`REAC_FUEL`不在预定义物种表，需要核对未定义物性的处理。没有为了清除警告而擅自填入热物性数据。这些警告不妨碍本轮启动和短时软件检查，但在解释物理火灾结果前必须处理或明确其影响。

后续应先完成**有来源的物理参数表和原始时间处理规则**，再规划300秒或900秒计算。参数表至少要交代HCN产率、燃料物性、燃烧器、峰后曲线和窗户代理；无法找回原始值的条目，应以独立重建假设和敏感性分析呈现。V、D与公式时间基准仍是FER转换的独立缺口。

本轮已经解决“能否解析、能否短时运行、门窗是否按正确方向动作、气体数据能否写出”这组问题，下一轮无需再重复相同的启动测试，也不应把任意HCN参数的诊断算例直接扩展成论文效果实验。

## 9. 可追溯交付

- {link(OUT/'README.md', '本轮文件入口')}
- {link(OUT/'run_comparison.csv', '全部运行与失败记录')}
- {link(OUT/'control_event_timing.csv', '逻辑与几何事件时间表')}
- {link(OUT/'initialized_region_cell_screen.json', '初始化后的测点邻接单元检查')}
- {link(OUT/'native_diagnostic_gas_long.csv', '754行原始时间诊断浓度（不可用于论文效果结论）')}
- {link(OUT/'native_timestamp_audit.json', '时间戳问题明细')}
- {link(OUT/'readiness_after_diagnostics.json', '已解决与未解决项')}
- {link(OUT/'verification.json', '全部核查断言与范围')}

本轮共有{verification['check_count']}项程序断言通过，其中多数为哈希、单元和逐行数值校验，不是{verification['check_count']}个独立实验。已核对{summary['protected_previous_files_unchanged']}个先前选定文件的哈希，全部保持不变；旧草案、原论文数据、已有模型及报告没有被本轮修复覆盖。
"""
    readme = f"""# SafeStep酒店FDS运行诊断包

日期：2026-10-16。**只用于软件诊断，不是论文物理结果复现。**

{link(REPORT, '中文报告')}记录了成功运行、首次失败及所有参数限制。

| 文件/目录 | 内容 |
|---|---|
| frozen_diagnostic_plan.json | 运行前固定的范围、临时参数与超时 |
| version_query.json | FDS6.11.1与进程环境修正 |
| runs/ | 6次独立运行输入、日志、原始CSV与SMV；初次错误保留 |
| runs/hotel_init_v2/derivation.json | 232个旧设备名称的对应表 |
| run_comparison.csv | 根据FDS日志核实的状态；不能只看退出码 |
| control_event_timing.csv | 设备逻辑切换与实际几何移除的时间差 |
| initialized_region_cell_screen.json | 58处测点相邻单元的初始化后检查 |
| native_diagnostic_gas_long.csv | 754行诊断数据，保留原始时间戳；不是原论文浓度 |
| native_timestamp_audit.json | 输出周期与实际时间网格的区别 |
| readiness_after_diagnostics.json | 下轮物理参数与时间处理缺口 |
| source_manifest.json / completion_manifest.json | 历史文件保护与新输出哈希 |

酒店诊断用HCN产率0.01只是任意软件参数；不允许从它生成论文FED或宣称复现了真实毒气。两个微型温度算例也不代表酒店玻璃物理。NIST FD-Gen来源及完整许可见FD-Gen_LICENSE.md，派生输入标注了修改。

启动、修复、运行和分析脚本均随完成清单记录。运行器拒绝覆盖已启动的同名算例。后续参数、时间处理与长时运行版本必须另存，不能覆盖本轮日志。
"""
    save(REPORT, report)
    save(OUT/"README.md", readme)
    links = []
    for document in (REPORT, OUT/"README.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text("utf-8")):
            assert Path(target).is_file(), target
            links.append(target)
    scripts = ["prepare_safestep_hotel_fds_smokecheck.py", "repair_safestep_fds_init_v2.py", "run_safestep_fds_diagnostic.py",
               "prepare_safestep_fds_short_cases.py", "analyze_safestep_fds_smokecheck.py", "finalize_safestep_fds_smokecheck.py"]
    paths = sorted(p for p in OUT.rglob("*") if p.is_file()) + [REPORT]
    paths += [ROOT/"scripts"/name for name in scripts] + [ROOT/"schemas/safestep_hotel_fds_smokecheck_20261016.json"]
    for name in scripts:
        compile((ROOT/"scripts"/name).read_text("utf-8"), name, "exec")
    save(OUT/"completion_manifest.json", {
        "status": "BOUNDED_FDS_DIAGNOSTIC_COMPLETE_NOT_PAPER_FIRE_REPRODUCTION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "FDS_input_runs": 6, "dynamic_successful_runs": 4, "original_fire_effect_experiments": 0,
        "full900s_runs": 0, "FER_FED_values_computed": 0,
        "training_updates": 0, "LLM_requests": 0, "external_messages_sent": 0,
        "old_final48_crowd_scenarios_accessed": False,
        "protected_previous_files_unchanged": len(manifest["protected_previous_sha256"]),
        "links_verified": len(links), "all_processes_finished": True,
    })
    print(json.dumps({"report": REPORT.as_posix(), "artifact_count": len(paths), "links": len(links),
                      "dynamic_runs": 4, "full_fire_runs": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()
