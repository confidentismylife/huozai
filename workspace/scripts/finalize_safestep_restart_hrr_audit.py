"""Document the supported restart mechanism and freeze this bounded audit."""
from datetime import datetime, timezone
import json
import re

from prepare_safestep_restart_hrr_audit import ROOT, OUT, OLD, sha, save

REPORT = ROOT / "reports/SafeStep续算放热异常原因与修正路线_2026-10-17.md"


def link(path, text):
    return f"[{text}]({path.as_posix()})"


def main():
    assert not (OUT / "completion_manifest.json").exists()
    assert not REPORT.exists()
    results = json.loads((OUT / "audit_results.json").read_text("utf-8"))
    runs = json.loads((OUT / "solver_run_summary.json").read_text("utf-8"))
    evidence = json.loads((OUT / "source_code_evidence.json").read_text("utf-8"))
    sources = json.loads((OUT / "source_manifest.json").read_text("utf-8"))
    additional = json.loads((OUT / "additional_source_manifest.json").read_text("utf-8"))
    for source in sources["sources"] + additional:
        assert sha(ROOT / source["path"]) == source["sha256"]
    old = json.loads((OLD / "completion_manifest.json").read_text("utf-8"))
    older = json.loads((OLD / "source_manifest.json").read_text("utf-8"))["protected_previous_sha256"]
    previous = {**older, **old["artifact_sha256"]}
    bad = [name for name, expected in previous.items() if sha(ROOT / name) != expected]
    assert not bad, bad
    assert runs["new_solver_processes"] == runs["passed"] == 6
    checkpoint_copies = 0
    for case in (OUT / "runs").iterdir():
        derivation = json.loads((case / "derivation.json").read_text("utf-8"))
        for name, expected in derivation["copied_parent_files_sha256"].items():
            if name.endswith(".restart"):
                assert sha(case / name) == expected
                checkpoint_copies += 1
    for case in (OUT / "microfixture").iterdir():
        if (case / "derivation.json").exists():
            derivation = json.loads((case / "derivation.json").read_text("utf-8"))
            for name, expected in derivation["copied_files_sha256"].items():
                if name.endswith(".restart"):
                    assert sha(case / name) == expected
                    checkpoint_copies += 1
    hotel = results["hotel"]; micro = results["microfixture"]
    periodic = micro["comparisons"]["micro_periodic_resume"]
    stop = micro["comparisons"]["micro_stop_resume"]
    save(OUT / "final_verification.json", {
        "checked_at": datetime.now(timezone.utc).isoformat(), "new_solver_runs_passed": 6,
        "last_stage_frozen_files_unchanged": len(old["artifact_sha256"]),
        "earlier_selected_files_unchanged": len(older), "unique_previous_files_verified": len(previous),
        "previous_hash_mismatches": bad, "pinned_source_files_verified": len(sources["sources"])+len(additional),
        "unmodified_restart_copies_verified": checkpoint_copies,
        "hotel_cadence_controls_all_419_channels_exact_equal": hotel["coarse_dense_all_device_values_exact_match"],
        "micro_periodic_and_stop_Q_state_contrast_verified": True,
        "no_FDS_recompile_or_binary_checkpoint_edit": True,
        "training_updates": 0, "LLM_requests": 0, "final48_scenarios_accessed": False,
        "300s_or900s_runs": 0, "route_FED_computed": False})
    report = f"""# SafeStep续算放热异常：原因、对照证据与修正路线

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + implementation audit
- Origin Date: 2026-10-17
- Verification Status: VERIFIED，限本机版本源码、原生断点解析和六次诊断运行
- Version Label: safestep_restart_hrr_audit_20261017
- Claim boundary: 诊断探索性火灾模型的数值续算问题，不是原论文物理火灾或疏散效果的完整复现

**本轮定位到了具体原因：本机FDS 6.11.1在检测到停止文件后，会清零放热场并提前退出燃烧反应计算；等待辐射状态完成的几步仍会推进时间。恢复后出现了额外的燃烧瞬态。原来的HRR先低后高，不能只解释成CSV显示异常。**

修正路线是改用正常计算中定期保存的断点，重新建立干净基准。本轮没有改FDS程序、没有编辑断点字节，也没有对异常曲线做平滑。

## 1. 为什么要查这一项

上一轮酒店探索计算在115.49119秒通过原生停止文件保存断点，随后续算到120秒。虽然程序正常结束、旧设备记录保留，续算附近的平均HRR却从约621 kW降到433 kW，再升到798 kW。当时原因没有确认，保留了异常记录。

这次检查表明：**退出码正常、原生续算成功、旧CSV保留，都不足以证明断点前后的物理数值状态没有受到扰动。** 上一轮关于“成功计算到120秒”的执行记录仍成立，但不能把它升级成这120秒火灾轨迹已经通过物理连续性验证。

## 2. 源码和断点里找到了什么

核对的是与本机可执行文件一致的官方提交：`{sources['commit']}`，版本`FDS-6.11.1-0-gff928db-release`，不是直接参照当前开发分支。

主循环先检测停止文件并设置用户停止状态；退出前可能等待辐射计算完成。这意味着出现停止状态之后，计算仍可能推进若干步。[官方主循环与停止处理]({evidence['stop_flag_set_before_combustion']['official_url']})、[等待条件]({evidence['wait_for_radiation_completion']['official_url']})

燃烧模块在每次调用时先清零Q和CHI_R，随后在停止状态不为零时提前返回。因此上述等待过程可能不再执行正常化学反应。[官方燃烧模块]({evidence['skip_chemistry_when_stop_flag_present']['official_url']})

我们按该提交的二进制记录结构只读解析了六个原始酒店断点文件。它们的保存时刻一致，均为{hotel['checkpoint_s']:.11f}秒；六个网格的Q放热场全部为零。所有原文件及本轮复制的断点字节均保留。

同时，断点保留了尚未写入CSV的累计放热量。最后一次HRR输出在{hotel['last_HRR_dump_s']:.11f}秒，保存的部分区间能量约{hotel['saved_partial_HRR_energy_kJ']:.6f} kJ，对应约{hotel['saved_partial_mean_HRR_kW']:.3f} kW的平均值。这一低值已经进入断点，并非恢复时才凭空出现在CSV里。

HRR输出由累计量除以距上次输出的时长得到，是区间平均量。[官方输出实现]({evidence['native_HRR_time_average']['official_url']}) 因此第一条433 kW记录混合了断点前后的状态，不能当作恢复后的瞬时火势。

## 3. 实际运行了哪六次计算

| 运行 | 场景及对照 | 结束方式 | 目的 |
|---|---|---|---|
| 酒店重复续算 | 原六网格、同一115.49119秒断点，HRR间隔0.5秒 | 117.6秒正常结束 | 检查原异常是否能复现 |
| 酒店密集输出续算 | 同一断点，HRR请求间隔0.025秒 | 117.6秒正常结束 | 查看被0.5秒平均值掩盖的变化 |
| 小模型连续计算 | 2304个网格单元、丙烷燃烧器、固定0.02秒步长 | 5.5秒正常结束，3秒附近定期保存断点 | 建立连续参考 |
| 小模型定期断点续算 | 从上述正常保存的3秒断点恢复 | 5.5秒正常结束 | 检查不使用停止文件的续算 |
| 小模型停止文件计算 | 同一模型，运行中加入空停止文件 | 3秒保存断点后停止 | 直接复现停止状态对燃烧的影响 |
| 小模型停止后续算 | 从停止文件生成的断点恢复 | 5.5秒正常结束 | 观察恢复后的放热与温度变化 |

六次均通过各自预定的结束标志、退出码和错误检查，没有超时。求解器进程累计运行约{runs['total_solver_wall_s']:.2f}秒；它不是本轮检索、编程、分析的总耗时。

### 酒店模型：异常可重复，输出频率不会把它消除

0.5秒重复续算与上一轮在共同时间点上的五个HRR值完全一致。密集输出中，下一条短区间平均值达到1079.042 kW；利用断点保存的累计量扣除首条混合区间的旧能量，可反推出恢复第一步约{hotel['first_resumed_step_HRR_inferred_kW']:.1f} kW。后者受CSV时间和数值打印精度限制，是重建值，不是新增的瞬时测量。

两种输出频率下，{hotel['device_rows']}个时间点、{hotel['device_channels_excluding_time']}个设备通道的数值完全一致；重建的HRR区间能量只相差{abs(hotel['coarse_dense_energy_difference_kJ']):.8f} kJ。说明改变输出频率没有消除数值轨迹中的瞬态，粗输出只是把它平均得更平缓。

### 小模型：定期断点与停止文件生成的断点表现不同

下表的HRR偏差只比较3秒之后的共同原生时间点。温度偏差比较整个轨迹中可直接匹配的探针记录，包括停止前后的扰动。

| 指标 | 正常定期断点续算 | 停止文件生成断点后续算 |
|---|---:|---:|
| 恢复后HRR最大绝对偏差，相对连续参考 | {periodic['post_restart_max_abs_HRR_difference_kW']:.5f} kW | {stop['post_restart_max_abs_HRR_difference_kW']:.5f} kW |
| 恢复后HRR最大相对偏差 | {100*periodic['post_restart_max_relative_HRR_difference']:.4f}% | {100*stop['post_restart_max_relative_HRR_difference']:.2f}% |
| 恢复后最高区间平均HRR | {periodic['post_restart_peak_HRR_kW']:.3f} kW | {stop['post_restart_peak_HRR_kW']:.3f} kW |
| 探针温度最大绝对偏差 | {periodic['max_matching_probe_temperature_difference_C']:.3f}℃ | {stop['max_matching_probe_temperature_difference_C']:.3f}℃ |
| 0—5.5秒重建放热能量相对差 | {100*periodic['relative_window_energy_difference']:.6f}% | {100*stop['relative_window_energy_difference']:.4f}% |

停止文件路径中，2.86—3.00秒的八条HRR输出为零，随后恢复第一条约227.43 kW；连续参考在相应时刻约128 kW。正常定期断点中的Q场包含非零燃烧放热，而停止断点中的Q场全为零。温度、燃料和CO₂探针也存在差异，支持这是一项数值状态扰动。

停止后恢复所释放的额外热量与燃料在反应暂停期间累积、恢复后继续反应的解释一致；本轮没有另做完整燃料质量与化学能守恒审计，不能把该解释改写成已逐项闭合的能量证明。

**总放热量接近，仍可能掩盖先停烧、后集中释放的过程错误。** 火灾疏散依赖烟气和温度到达的时间，不能只检查整段能量是否相近。

![续算机制对照]({(OUT/'restart_mechanism_controls.png').as_posix()})

图中HRR点是输出区间平均值，连线用于观察趋势。A为酒店短时续算；B、C为独立小模型，时间轴不代表同一场火灾。

## 4. 时间与能量怎样统计

本轮按源码定义，把HRR输出均值乘以对应原生时间区间，重建输出窗口能量；没有继续把这些平均值当作瞬时点做梯形积分。第一段酒店续算还扣除了断点中保存的部分区间累计量。

CSV打印时间具有有限精度，重建能量只是对输出统计的一致性检查，不能替代独立能量守恒验证。正常定期续算也不是逐字节相同：小模型续算相较连续参考少了一条4.1599998秒的HRR输出，形成一个较长的原生平均区间。我们没有补点或静默插值，直接匹配的点数及完整差异都保存在JSON中。

## 5. 对已有结果的处理

1. 原120秒文件全部保留，作为探索执行与故障诊断记录。停止附近及其后的数值状态不继续用作已验证的火灾物理证据。
2. 原115.49119秒停止断点不再作为300秒或900秒计算的起点。删掉两条异常CSV记录也不能修复已受到扰动的状态。
3. 本轮结果支持在下一次酒店计算中采用定期断点；小模型精度不能直接推广成酒店精度。
4. 已有公开危险率与权重的下游SafeStep路线/FED重放属于另一条证据链，本轮没有重新计算或更改它。上游自建火灾仍未达到原论文物理复现的证据等级。

## 6. 下一步的具体执行顺序

先从零重新运行同一探索性酒店模型到120秒，采用正常终止和定期断点，例如每20秒保存；给足运行预算，避免为了接近墙钟上限而创建停止文件。若触及硬上限，记录中断，从已确认写完整的定期断点另建目录恢复，不使用停止状态生成的断点。

随后从正常保存的一个中间断点短时恢复，与同一连续酒店轨迹比较HRR、CO、CO₂、HCN、温度和门事件。只有确认酒店层面的续算偏差可接受，才延长到300秒。之后仍需处理原始几何、HCN迁移、窗口气温代理等物理假设，不能把续算修正等同于参数已经验证。

**本轮没有启动新的连续120秒酒店计算，也没有进入300秒、900秒或路线FED计算。** 没有PPO训练、LLM/API调用、对外邮件或读取48个封存最终场景。

## 7. 文件与核验

- {link(OUT/'audit_results.json', '完整数值分析及限制')}
- {link(OUT/'source_code_evidence.json', '固定版本源码位置')}
- {link(OUT/'solver_run_summary.json', '六次运行记录')}
- {link(OUT/'figure_data.json', '图表原始绘图数据')}
- {link(OUT/'final_verification.json', '本轮最终核验')}
- {link(OLD/'completion_manifest.json', '上一轮冻结清单，未修改')}

核验了上一轮{len(old['artifact_sha256'])}个冻结文件，以及更早选定的{len(older)}个文件，均未改变。官方源码8个文件与来源哈希一致，14个用于续算的断点副本没有修改字节。GitHub未认证API最初返回403限流，随后通过官方发布页定位完整提交并下载固定源码；没有因此更换源码版本或改动求解器。
"""
    REPORT.write_text(report, "utf-8")
    readme = f"""# FDS restart HRR audit, 2026-10-17

Source-pinned, six-run diagnosis of native stop-file chemistry transients.

The previous stop checkpoint is preserved but must not seed longer physical runs.
Passive periodic saves were close to a continuous reference in one small fixture;
clean hotel restart equivalence remains to be measured.

Report: {link(REPORT, REPORT.name)}

This audit did not change the FDS executable, any checkpoint bytes, or earlier artifacts.
No LLM calls, training, sealed final scenarios, or longer fire runs were used.
"""
    (OUT / "README.md").write_text(readme, "utf-8")
    links = re.findall(r"\]\((D:/[^)]+)\)", report)
    for target in links:
        assert __import__("pathlib").Path(target).exists(), target
    scripts = [ROOT / "scripts" / name for name in (
        "prepare_safestep_restart_hrr_audit.py", "run_safestep_restart_hrr_controls.py",
        "run_safestep_restart_microfixture.py", "analyze_safestep_restart_hrr_audit.py",
        "finalize_safestep_restart_hrr_audit.py")]
    artifacts = sorted(p for p in OUT.rglob("*") if p.is_file()) + [REPORT] + scripts
    save(OUT / "completion_manifest.json", {
        "status": "COMPLETE_BOUNDED_RESTART_MECHANISM_AUDIT", "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        "previous_files_verified_unchanged": len(previous), "new_solver_runs_passed": 6,
        "local_report_links_verified": len(links), "clean_hotel120_rerun_started": False,
        "300s_or900s_runs": 0, "route_FED_computed": False,
        "training_updates": 0, "LLM_requests": 0, "final48_scenarios_accessed": False})
    print(json.dumps({"status": "COMPLETE", "new_artifacts_frozen": len(artifacts),
        "previous_files_verified": len(previous), "report": REPORT.as_posix()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
