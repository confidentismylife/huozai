"""Freeze clean hotel execution evidence and report the predeclared screens."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

from prepare_safestep_clean_hotel120 import ROOT, OUT, PREFIX, sha, save
from analyze_safestep_fds_smokecheck import parse_smv

REPORT = ROOT / "reports/SafeStep正常120秒酒店基准与定期断点续算核验_2026-10-17.md"


def link(path, label):
    return f"[{label}]({path.as_posix()})"


def main():
    assert not (OUT / "completion_manifest.json").exists() and not REPORT.exists()
    plan = json.loads((OUT / "frozen_plan.json").read_text("utf-8"))
    results = json.loads((OUT / "analysis_results.json").read_text("utf-8"))
    captured = json.loads((OUT / "captured_checkpoint.json").read_text("utf-8"))
    preflight = json.loads((OUT / "preflight.json").read_text("utf-8"))
    bad = [name for name,expected in plan["protected_previous_sha256"].items() if sha(ROOT/name) != expected]
    assert not bad, bad
    parent = OUT/"runs"/PREFIX
    resumed = OUT/"runs/hotel_clean_checkpoint_resume110"
    assert sha(parent/f"{PREFIX}.fds") == preflight["input_sha256"]
    for name, expected in captured["copied_sha256"].items():
        assert sha(ROOT/captured["path"]/name) == expected and sha(resumed/name) == expected
    assert (parent/f"{PREFIX}_git.txt").read_text("utf-8").strip() == plan["FDS_version"]
    assert not list(OUT.rglob("*.stop"))
    math_checks = json.loads((OUT/"energy_comparison_math_verification.json").read_text("utf-8"))
    assert math_checks["status"] == "PASS" and len(math_checks["checks"]) == 4
    complete = results["all_screens_passed"]
    status_text = "全部通过" if complete else "存在未通过项"
    next_text = ("本轮预先规定的数值一致性筛查全部通过，可以准备300秒探索计算；仍需保留所有物理假设标签，并继续使用正常定期断点。"
                 if complete else "保留正常连续120秒轨迹；先在封闭双腔最小反例中修正并验证压力分区合并状态的恢复，再返回酒店核验。当前不进入300秒计算。")
    followup_rows=[]; followup_runs=0; followup_wall=0.0
    for subdir, metric_key in [("dynamic_door_diagnostic","channels"),("door_boundary_diagnostic","metrics"),("sealed_zone_diagnostic","metrics")]:
        diagnostic=json.loads((OUT/subdir/"diagnostic_results.json").read_text("utf-8"))
        followup_runs+=diagnostic["solver_runs"];followup_wall+=diagnostic["total_solver_wall_s"]
        for contrast,value in diagnostic["results"].items():
            metrics=value[metric_key]
            case=OUT/subdir/(contrast+"_continuous")
            _,events=parse_smv(case/(contrast+".smv"))
            native_hrr=pd.read_csv(case/(contrast+"_hrr.csv"),skiprows=1)
            for suffix in ("_continuous","_resumed"):
                status=json.loads((OUT/subdir/(contrast+suffix)/"run_result.json").read_text("utf-8"))
                assert status["passed"] and not status["timed_out"]
            followup_rows.append({"contrast":contrast,"directory":subdir,
                "checkpoint_s":value.get("checkpoint_time_s",value.get("checkpoint_s")),
                "door_geometry_times_s":[e["geometry_event_time_s"] for e in events if e["ID"]=="DOOR"],
                "pressure_zone_count":len([c for c in native_hrr if c.startswith("ZONE_")]),
                "max_temperature_difference_C":max(metrics[c]["max_abs_difference"] for c in ("LEFT_T","RIGHT_T")),
                "max_CO2_difference_ppm":max(metrics[c]["max_abs_difference"] for c in ("LEFT_CO2","RIGHT_CO2"))*1e6,
                "max_interface_U_difference_m_s":metrics["INTERFACE_U"]["max_abs_difference"],
                "pressure_endpoints":value.get("pressure_zones",{})})
    assert followup_runs==16
    continuous_hrr=pd.read_csv(parent/f"{PREFIX}_hrr.csv",skiprows=1)
    resumed_hrr=pd.read_csv(resumed/f"{PREFIX}_hrr.csv",skiprows=1)
    pressures={c:{"continuous_interpolated_110s_Pa":float(np.interp(110,continuous_hrr.Time,continuous_hrr[c])),
        "resumed_native_110s_Pa":float(resumed_hrr.iloc[-1][c])} for c in ("ZONE_1","ZONE_12")}
    divg=OUT/"pressure_zone_source/divg.f90"
    source_manifest=json.loads((divg.parent/"source_manifest.json").read_text("utf-8"))
    assert sha(divg)==source_manifest["sha256"]
    prior_source=ROOT/"outputs/safestep_restart_hrr_audit_20261017/materials"
    dump=(prior_source/"dump.f90").read_text("utf-8")
    restart_block=dump[dump.index("SUBROUTINE DUMP_RESTART("):dump.index("END SUBROUTINE READ_RESTART")]
    assert "CONNECTED_ZONES" not in restart_block
    assert "IF (N_ZONE>0 .AND. OBST_CREATED_OR_REMOVED) CALL MERGE_PRESSURE_ZONES" in divg.read_text("utf-8")
    save(OUT/"followup_diagnosis_summary.json",{"status":"PRESSURE_ZONE_MERGE_RESTART_FAILURE_SUPPORTED_BY_MINIMAL_CONTRAST",
        "followup_solver_runs":followup_runs,"followup_solver_wall_s":followup_wall,"rows":followup_rows,
        "hotel_110s_pressure_Pa":pressures,"open_fixtures_had_no_pressure_zones":True,
        "source_evidence":{"CONNECTED_ZONES_not_in_restart_read_write_block":True,
            "pressure_merge_rebuild_guarded_by_geometry_change_flag":True,"pinned_divg_sha256":sha(divg)},
        "interpretation":"Dynamic pressure-zone coupling restoration is a supported mechanism hypothesis; original hotel screening remains failed",
        "solver_fix_implemented":False,"patched_solver_validation_done":False,"original_frozen_gates_changed":False})
    diagnostics_table="\n".join(f"| {r['contrast']} | {r['pressure_zone_count']} | {r['max_temperature_difference_C']:.6f}℃ | {r['max_CO2_difference_ppm']:.6f} ppm |" for r in followup_rows)
    pressure_table="\n".join(f"| {zone} | {r['continuous_interpolated_110s_Pa']:.3f} | {r['resumed_native_110s_Pa']:.3f} |" for zone,r in pressures.items())
    diagnostic_section=f"""## 4. 失败后已进一步缩小到压力分区连接的恢复

酒店主对照失败后，追加了16次秒级小模型计算，组成8组各自的连续/续算配对；累计求解器墙钟耗时约{followup_wall:.2f}秒。这些是有具体失效触发的机理诊断，没有放宽原酒店的冻结标准，也没有重复酒店长计算。

| 小模型对照 | 压力区个数 | 最大探针温度差 | 最大探针CO₂差 |
|---|---:|---:|---:|
{diagnostics_table}

最初六种开放小模型覆盖普通动态门、静态连通、单/双网格、火源侧薄门，以及断点恰逢移除门的时刻。它们的温度偏差均小于0.062℃，没有复现酒店现象。不过它们都没有密闭压力分区，不能检验压力区合并的恢复。

最后两组加入了封闭条件。动态开门形成两个相连压力区的模型，续算后出现明显的压力分离和约4.62℃温度差；静态连通、只有一个压力区的封闭模型，最大温差约0.0071℃。因此，“动态门本身”“两个网格本身”“薄门本身”不足以解释酒店失败；动态压力区连接的恢复是更具体的重点。

酒店缓存中，110秒时两个压力区的表压如下，连续参考在该时刻采用原生输出间插值，续算使用110秒终点原生值：

| 压力区 | 连续参考（Pa） | 正常断点续算（Pa） |
|---|---:|---:|
{pressure_table}

源码核对发现：连接矩阵初始化为对角项，压力区合并受几何变化标志控制；本机固定版本的断点读写段没有保存该连接矩阵。[固定版本的合并入口](https://github.com/firemodels/fds/blob/ff928dbf68a2623ced3a007be75391c6dc6c9f90/Source/divg.f90#L88)、[连接矩阵初始化](https://github.com/firemodels/fds/blob/ff928dbf68a2623ced3a007be75391c6dc6c9f90/Source/main.f90#L1440)

结合封闭双腔反例，证据支持“压力分区合并/耦合状态未在恢复后正确重建”这一具体机理。**本轮没有修改或重新编译求解器，尚未通过修复前后对照证明该机理已修好，也不排除酒店还有其他恢复差异。** 普通小模型通过不能覆盖酒店模型中已观察到的失败。

"""
    table = "\n".join(f"| {s['name']} | {s['metric']:.8g} | {s['limit']:.8g} | {'通过' if s['passed'] else '未通过'} |" for s in results["engineering_screens"])
    gas_table = "\n".join(f"| {g} | {r['max_abs_difference_ppm']:.6f} ppm | {100*r['peak_scaled_max_abs_difference']:.5f}% | {r['worst_channel']}，{r['worst_time_s']:.2f}秒 |" for g,r in results["gases"].items())
    subgroup_table = "\n".join(f"| {g} | {100*r['subgroups']['corridor58']['peak_scaled_max_abs_difference']:.5f}% | {100*r['subgroups']['room24']['peak_scaled_max_abs_difference']:.5f}% |" for g,r in results["gases"].items())
    p = results["continuous_native_peak_summary"]
    hrr = results["HRR"]
    logical_door = [e for e in results["controls"]["logical_continuous"] if e["ID"] == "SS_DOOR_OPEN"]
    windows = [e for e in results["controls"]["logical_continuous"] if e["ID"] == "SS_WINDOW_BREAK"]
    window_text = "120秒内没有触发150℃窗口气温代理" if not windows else f"窗口气温代理事件：{windows}"
    transition = results["earlier_stop_pilot_comparison"]["transition_common_rows"]
    transition_table = "\n".join(f"| {r['Time']:.5f} | {r['HRR_clean']:.5f} | {r['HRR_old']:.5f} |" for r in transition)
    if not transition_table:
        transition_table = "| 无相同原生输出时刻 | — | — |"
    save(OUT/"final_verification.json", {
        "verified_at": datetime.now(timezone.utc).isoformat(), "status": "EXECUTION_AND_ANALYSIS_VERIFIED",
        "predeclared_engineering_screen_passed": complete,
        "protected_previous_files_verified": len(plan["protected_previous_sha256"]), "mismatches": bad,
        "new_continuous_input_hash_preserved": True, "six_captured_and_resumed_restart_hashes_unchanged": True,
        "solver_version_verified_from_native_git_file": plan["FDS_version"],
        "normal_termination120s": True, "normal_restart_to110s": True, "stop_files_created": 0,
        "energy_math_checks_passed": 4, "300s_or900s_runs": 0,
        "hotel_solver_runs":2,"followup_fixture_solver_runs":followup_runs,"total_solver_runs":2+followup_runs,
        "pinned_pressure_zone_source_verified":True,"solver_fix_implemented":False,
        "route_FED_computed": False, "training_updates": 0, "LLM_requests": 0,
        "sealed_final48_scenarios_accessed": False})
    report = f"""# SafeStep正常120秒酒店基准与定期断点续算核验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + numerical comparison
- Origin Date: 2026-10-17
- Verification Status: VERIFIED，限本轮实际执行、数据处理和数值筛查；物理参数真实性未升级
- Version Label: safestep_clean_hotel120_20261017
- Claim boundary: 探索性酒店火灾计算与一个正常断点的短时一致性核验，不是原论文完整物理复现

**本轮已经从零正常计算到120秒，没有创建停止文件；复制了约100秒正常计算中的定期断点，再续算到110秒。预先冻结的数值一致性筛查：{status_text}。**

{next_text}

## 1. 本轮解决什么

上一轮查明：通过停止文件生成的断点可能受到燃烧反应暂停的影响。虽然程序可以恢复，但恢复后的HRR尖峰不能作为真实火灾变化。本轮实际执行了改进后的酒店计算，没有继续使用115.49119秒的旧停止断点。

| 项目 | 本轮设置 |
|---|---|
| 求解器 | {plan['FDS_version']}，与前轮同一可执行文件 |
| 网格 | 同一六网格酒店候选，93,981个网格单元 |
| 火源、材料、几何、门和窗代理 | 与前轮候选保持一致，没有通过调参消除异常 |
| 常规设备与HRR请求输出 | 0.5秒；保留实际原生时间戳 |
| 正常定期断点 | 每20秒请求保存，不创建停止文件 |
| 连续计算 | 0—120秒，正常结束，耗时{results['continuous_run']['elapsed_s']:.2f}墙钟秒 |
| 配对续算 | 从{captured['fire_time_s']:.8f}秒的正常断点恢复到110秒，耗时{results['paired_run']['elapsed_s']:.2f}墙钟秒 |
| 资源 | 1个MPI进程、2个OpenMP线程；主对照2次，失败后追加16次小模型运行 |

连续运行给足独立的墙钟预算，接近预算时不会创建停止文件。如果超过硬上限，程序会记录中断；已保存完整的定期断点才是后续恢复候选。这轮实际结束方式与超时情况以原生日志和运行JSON为准。

## 2. 如何确保复制到的是同一份完整断点

主计算继续运行时，程序在100秒附近检查六个网格的断点。只有六个原生保存时刻相同、文件大小与修改时间稳定、复制前后哈希一致，并且全部Fortran二进制记录首尾完整，才接受副本。

实际接受的时刻为{captured['fire_time_s']:.11f}秒。火源所在Room24的Q放热场包含非零值；它与上一轮停止断点中Q场全零的状态不同。120秒正常终止时的六个断点也通过时刻一致性检查，Room24放热场仍为非零。

短时续算沿用同一CHID、同一网格、同一设备数组和相同的断点字节，仅启用RESTART并缩短本次诊断终止时间。连续运行的输出目录没有被续算覆盖。

## 3. 结果比较与预先规定的标准

评价标准在本轮运行前写入冻结计划，没有根据最终成绩放宽。它们是本项目的探索性数值一致性筛查值，不是消防安全限值或材料校准标准。

| 检查项 | 实际值 | 冻结阈值/目标 | 结果 |
|---|---:|---:|---|
{table}

连续计算有{results['devices']['continuous_rows']}条设备时间记录，配对续算有{results['devices']['resumed_rows']}条。419个设备通道中，177个是58个区域及Room24的CO、CO₂、HCN，另有{results['devices']['temperature_channels']}个温度通道。断点之前保留的{results['devices']['exact_retained_prefix_rows']}条原生设备记录与连续参考完全相同。

配对比较使用{results['paired_grid']['start_s']:.2f}—{results['paired_grid']['end_s']:.2f}秒、间隔0.5秒的{results['paired_grid']['points']}个共同时间点。只在已有数据覆盖范围内插值，未外推、未平滑、未删除异常点。

### HRR

输出均值插值后的最大相对差为{100*hrr['max_interpolated_relative_error']:.5f}%，最大绝对差为{hrr['max_interpolated_abs_error_kW']:.5f} kW。它用于比较曲线形状，不能把平均输出当作瞬时燃烧强度。

重建放热能量从实际断点{captured['fire_time_s']:.8f}秒到{hrr['energy_end_s']:.8f}秒，连续参考为{hrr['reconstructed_energy_kJ']['continuous']:.6f} kJ，续算为{hrr['reconstructed_energy_kJ']['resumed']:.6f} kJ，相对差{100*hrr['relative_energy_error']:.6f}%。计算扣除了断点保存的{hrr['saved_pre_checkpoint_partial_energy_kJ']:.6f} kJ旧区间累计量。

终点是否为两条曲线共同的原生HRR输出时刻：{hrr['energy_end_is_shared_native_output_time']}。是否使用了部分平均窗口近似：{hrr['fractional_output_window_used']}。这项结果是输出统计一致性检查，受打印精度和区间均值假设限制，不是独立能量守恒证明。

### 气体与温度

| 气体 | 最大绝对差 | 按全体参考峰值及冻结下限归一后的最大差 | 最差通道与时刻 |
|---|---:|---:|---|
{gas_table}

CO₂在这个配对表中统一用ppm表达；另存的原生长表仍按既有约定用CO₂百分数。最大温度差为{results['temperature']['max_abs_difference_C']:.6f}℃，位于{results['temperature']['worst_channel']}，{results['temperature']['worst_time_s']:.2f}秒。

全体通道的参考峰值可能主要由火源房贡献。因此另列走廊和Room24各自的归一偏差，避免较大的房内浓度掩盖走廊局部变化；这些子组指标是描述性补充，没有加入或改变冻结的通过条件。

| 气体 | 走廊58区域最大归一偏差 | Room24最大归一偏差 |
|---|---:|---:|
{subgroup_table}

### 门与窗事件

门的逻辑事件记录：{logical_door}。{window_text}。逻辑事件与几何显示/隐藏事件的序列保留检查为{results['controls']['event_sequences_match']}。

本次约100秒断点是在门已经触发以后保存的，因而验证的是事件及其后状态的保留。它不能代替一次从开门之前恢复、重新跨越开门事件的对照。

![正常断点配对比较]({(OUT/'hotel_clean_checkpoint_comparison.png').as_posix()})

{diagnostic_section}
## 5. 与旧停止续算轨迹的关系

旧轨迹只用于展示停止附近的差异，没有作为本轮正常断点的主参考，更没有用它来重新拟合任何火源或产率。共同原生时刻的部分HRR如下。

| 原生火灾时间（秒） | 本轮正常结束HRR（kW） | 旧停止后续算HRR（kW） |
|---:|---:|---:|
{transition_table}

原停止前可匹配的HRR记录共{results['earlier_stop_pilot_comparison']['common_native_prefix_HRR_rows']}条，最大绝对差为{results['earlier_stop_pilot_comparison']['max_abs_common_prefix_HRR_difference_kW']:.8f} kW。这项核对仅限相同输出时刻，不把找不到相同时刻的点静默视为相同。

## 6. 正常120秒轨迹包含什么

下表是这套候选物理假设下的测点/区域峰值，仅用于描述早期传播。不能解释为原作者实测值、通道安全阈值或整座建筑危险程度。

| 指标 | Room24测点 | 58个区域所有记录的最大值 |
|---|---:|---:|
| CO | {p['room24']['CO']['max']:.3f} ppm | {p['corridor58']['CO']['max']:.3f} ppm |
| HCN | {p['room24']['HCN']['max']:.3f} ppm | {p['corridor58']['HCN']['max']:.3f} ppm |
| CO₂ | {p['room24']['CO2']['max']:.5f}% | {p['corridor58']['CO2']['max']:.5f}% |

Room24气温测点最高{p['room24']['temperature']['max']:.3f}℃；HRR输出区间均值最高{p['HRR_output_mean_peak_kW']:.3f} kW。气温测点不是玻璃温度，窗口仍使用未经原作者确认的气温代理。

![正常120秒轨迹概况]({(OUT/'hotel_clean120_overview.png').as_posix()})

## 7. 当前能说明什么，以及下一步

本轮建立了一条正常结束的120秒候选酒店火灾轨迹，并量化了一个正常定期断点在约10秒后续范围内的数值偏差。它比“程序能恢复”多了一项实际对照证据，但仍不等于整个火灾模型已经可靠，也不保证更长时段误差同样小。

{next_text}

原论文缺失的精确几何、原始HCN产率、燃料物性及窗口破裂实现仍未完整找回。当前HCN=0.007仍为跨材料迁移，其他假设沿用此前明确登记的状态。数值续算筛查不会把这些假设升级成原作者参数。

所有数据都早于300秒疏散开始。本轮没有计算路线FED、没有PPO训练或LLM调用，也没有读取48个封存最终场景；没有启动300秒或900秒长时计算。

## 8. 可追溯文件

- {link(OUT/'frozen_plan.json', '运行前冻结的计划与标准')}
- {link(OUT/'analysis_results.json', '完整结果、筛查值与限制')}
- {link(OUT/'captured_checkpoint.json', '正常断点复制与完整性证明')}
- {link(OUT/'normal_end120_checkpoint_metadata.json', '正常120秒终点断点核验')}
- {link(OUT/'clean120_native_gas_long.csv', '58区域原生时间气体长表')}
- {link(OUT/'paired_gas_comparison_grid.csv', '仅用于配对比较的共同时间点表')}
- {link(OUT/'figure_data.json', '图表数据')}
- {link(OUT/'final_verification.json', '最终核验')}
- {link(OUT/'followup_diagnosis_summary.json', '16次后续机理对照、压力缓存与源码证据')}

上一轮及更早的{len(plan['protected_previous_sha256'])}个选定文件再次核验，哈希均未改变。本轮原始输出和正常断点单独保存。
"""
    REPORT.write_text(report,"utf-8")
    for target in re.findall(r"\]\((D:/[^)]+)\)",report):
        assert Path(target).exists(),target
    (OUT/"README.md").write_text(f"# Clean 120s hotel run and passive checkpoint comparison\n\nStatus: {results['status']}\n\n{link(REPORT,'Chinese report')}\n\nExploratory physical assumptions remain unchanged; no route FED, training, LLM calls or sealed scenario access.\n", "utf-8")
    scripts=[ROOT/"scripts"/name for name in ("prepare_safestep_clean_hotel120.py","run_safestep_clean_hotel120.py",
         "analyze_safestep_clean_hotel120.py","finalize_safestep_clean_hotel120.py",
         "diagnose_safestep_dynamic_door_restart.py","diagnose_safestep_door_restart_boundary.py",
         "diagnose_safestep_sealed_zone_restart.py")]
    files=sorted(p for p in OUT.rglob("*") if p.is_file())+[REPORT]+scripts
    save(OUT/"completion_manifest.json", {"status":"COMPLETE_CLEAN120_AND_PAIRED_PASSIVE_RESTART",
        "created_at":datetime.now(timezone.utc).isoformat(),"engineering_screen_passed":complete,
        "artifact_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        "protected_previous_verified":len(plan["protected_previous_sha256"]), "solver_processes":2+followup_runs,
        "hotel_solver_processes":2,"post_failure_small_fixture_solver_processes":followup_runs,
        "normal_continuous_120s_runs":1,"short_passive_restart_runs":1,"300s_or900s_runs":0,
        "route_FED_computed":False,"training_updates":0,"LLM_requests":0,"sealed_final48_scenarios_accessed":False})
    print(json.dumps({"status":"COMPLETE","engineering_screen_passed":complete,"frozen_files":len(files),
        "report":REPORT.as_posix()},ensure_ascii=True))


if __name__ == "__main__":
    main()
