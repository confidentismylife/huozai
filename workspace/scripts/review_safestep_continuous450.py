"""Independent read-only review; preserve the completed run and first report."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import psutil

from prepare_safestep_continuous450 import ROOT, OUT, CASE, PREFIX, sha, save
from analyze_safestep_fds_smokecheck import parse_smv

REVIEW=ROOT/'outputs/safestep_continuous450_review_20261017'
ORIGINAL_REPORT=ROOT/'reports/SafeStep连续450秒候选火灾与四条路线FED_2026-10-17.md'
REPORT=ROOT/'reports/SafeStep连续450秒候选火灾与四条路线FED_核查版_2026-10-17.md'


def main():
    REVIEW.mkdir(exist_ok=False);assert not REPORT.exists()
    source_manifest=OUT/'completion_manifest.json';manifest=json.loads(source_manifest.read_text('utf-8'))
    mismatches=[p for p,h in manifest['artifact_sha256'].items() if sha(ROOT/p)!=h]
    assert not mismatches
    results=json.loads((OUT/'analysis_results.json').read_text('utf-8'))
    run=results['run'];assert run['passed'] and run['returncode']==0 and run['final_native_fire_s']==450
    table=results['candidate_routes'];traces=pd.read_csv(OUT/'candidate_route_residence_integrals.csv')
    for row in table:
        t=traces[traces.route==row['id']]
        assert len(t)==row['steps'] and abs(t.FED_increment.sum()-row['candidate_primary_FED'])<1e-12
        assert abs(t.duration_s.sum()-row['travel_time_s'])<1e-10
    ratios=[{'id':r['id'],
        'candidate_over_archived_same_primary':r['candidate_primary_FED']/r['archived_same_primary_FED'],
        'candidate_over_archived_same_legacy':r['candidate_legacy_compatibility_FED']/r['archived_legacy_paper_reproduction_FED'],
        'candidate_legacy_minus_primary_fraction':r['candidate_legacy_compatibility_FED']/r['candidate_primary_FED']-1}
        for r in table]
    text=(CASE/f'{PREFIX}.fds').read_text('utf-8')
    hole=re.search(r"&HOLE ID='SS_ROOM24_WINDOW', XB=([^/]+?)[,\s]+DEVC_ID=",text)
    assert hole
    expected=[float(x) for x in hole.group(1).split(',')]
    _,events=parse_smv(CASE/f'{PREFIX}.smv')
    window=[e for e in events if np.allclose(e['requested_clipped_XB'],expected,rtol=0,atol=1e-9)]
    assert len(window)==1 and window[0]['kind']=='HIDE_OBST'
    with (CASE/f'{PREFIX}_devc_ctrl_log.csv').open(encoding='utf-8-sig',newline='') as f:
        controls=[{k:v.strip() for k,v in row.items()} for row in csv.DictReader(f)]
    window_control=next(r for r in controls if r['ID']=='SS_WINDOW_BREAK')
    assert window_control['State']=='T' and float(window_control['Value'])>=150
    assert 0<=window[0]['geometry_event_time_s']-float(window_control['Time (s)'])<.1
    observed={'window_logical_event':window_control,'window_geometry_event':window[0],
       'identification':'Match the requested HOLE XB to SMV obstruction geometry, not only its generic ID',
       'original_automatic_ID_filter_omitted_window_entry':True,
       'solver_window_action_missing':False,'raw_data_changed':False}
    save(REVIEW/'verified_window_event.json',observed)
    start=json.loads((CASE/'process_start.json').read_text('utf-8'))
    original_solver_still_running=False
    if psutil.pid_exists(start['pid']):
        p=psutil.Process(start['pid']);original_solver_still_running=p.name().lower()=='fds_openmp.exe'
    assert not original_solver_still_running
    params=json.loads((OUT/'frozen_plan.json').read_text('utf-8'))
    assert sha(Path(params['FDS_executable']))==params['FDS_sha256']
    assert (CASE/f'{PREFIX}_git.txt').read_text('utf-8').strip()==params['FDS_version']
    gas=results['gas_summary'];ranks=results['route_rankings']
    ratio_rows='\n'.join(f"| {r['id']} | {r['candidate_over_archived_same_primary']:.3f} | {r['candidate_over_archived_same_legacy']:.3f} | {100*r['candidate_legacy_minus_primary_fraction']:+.2f}% |" for r in ratios)
    appendix=f'''

## 独立核对与结果解读

核对时间：2026-10-17。原始运行和第一版自动报告均保留，以下不涉及新模拟、参数调整或重新训练。

### 已确认的完成范围

FDS进程正常结束于450秒，实际用时 **{run['elapsed_wall_s']/60:.2f}分钟，约2小时13分钟**。原始输出包含 **{results['native_time_rows']}个原生时间节点**，最大间隔 **{results['native_max_gap_s']:.6f}秒**。四条路线逐段积分之和与汇总FED一致。

已复核本轮第一版清单内的 **{len(manifest['artifact_sha256'])}个冻结文件**，没有散列不一致。阶段结束时校验了 **{manifest['protected_previous_files_verified']}个既有文件**；安装的FDS版本和可执行文件散列也与计划一致。原求解器进程已退出。

### 排序保留，绝对量级没有复现

在这个候选模型的相同口径下，起点1的推荐路线相对替代路线FED低 **{100*ranks[0]['relative_reduction_vs_alternative']:.2f}%**，起点2低 **{100*ranks[1]['relative_reduction_vs_alternative']:.2f}%**。

这只是两组指定路线在一个假定火灾中的比较，不能表述为RL训练获得了约55%的收益，更不能当作论文另一场景的效应量复现。

更需要重视的是：主积分口径相同后，本轮FED仍为公开危险度数据下的约 **5.7–11.4倍**。这说明推荐路径排序一致，仍不足以验证上游火灾重建。改回旧兼容积分口径，也没有恢复原来的绝对量级。

| 路线 | 候选/公开数据：同主积分 | 候选/公开数据：同旧口径 | 本候选旧口径相对主积分差别 |
|---|---:|---:|---:|
{ratio_rows}

起点1替代路线的两种积分口径相差约27.52%，其余三条约相差5%。因此必须保留两种口径，不应把5秒取样结果当成原生时间积分结果。这四条路线的跨输入量级差距在两种口径下都存在，不能只用采样口径来解释。

### 烟气与通风边界

450秒内的走廊测点峰值：CO约 **{gas['CO']['corridor_max_ppm']:.1f} ppm**，HCN约 **{gas['HCN']['corridor_max_ppm']:.1f} ppm**，CO₂约 **{gas['CO2']['corridor_max_ppm']/10000:.3f}%**。这些是58个点中的最大值，不是每个人或整条路线的平均浓度。

本轮不能确定量级偏高分别由哪些因素造成。通风开口及出口对应关系、区域位置、HCN迁移产率、火源峰值后曲线和人体参数口径，都属于尚未排除的解释。单次运行不能分解它们的贡献，也不能据此倒推修改参数来贴近论文FED。

### 窗口动作核对：不是没有开窗

原生控制日志显示窗口温度代理在 **{float(window_control['Time (s)']):.6f}秒**触发，记录温度 **{float(window_control['Value']):.3f}°C**；SMV中对应几何移除在 **{window[0]['geometry_event_time_s']:.6f}秒**发生。

这个窗洞填充障碍在SMV中的ID是通用名称`Obstruction`，第一版自动结果按特定ID筛选时没有列出它。此次通过窗洞XB坐标匹配确认了实际动作。修正的是事件识别与报告说明，没有改变原始日志或重跑计算。这仍是气体温度触发的开孔代理，不是实际玻璃破裂验证。

### 下一步判断

本轮已完成“自生成烟气→明确单位的FER→四条固定路线FED”的候选计算链路。**暂不扩大到900秒，也不恢复强化学习训练。** 下一步应先核对出口与物理通风开口的对应、原作者气体转FER参数与时间口径，以及区域位置证据；已有差距不能靠继续训练解决。

第一版自动报告和数值清单均继续保留：[第一版报告]({ORIGINAL_REPORT.as_posix()})、[独立核查数据]({(REVIEW/'review_results.json').as_posix()})、[窗口事件核对]({(REVIEW/'verified_window_event.json').as_posix()})。
'''
    report=ORIGINAL_REPORT.read_text('utf-8')
    title='# SafeStep 连续450秒候选火灾与四条固定路线 FED'
    assert report.startswith(title)
    report=report.replace(title,title+'（核查版）',1)+appendix
    data={'reviewed_at':datetime.now(timezone.utc).isoformat(),
         'source_completion_manifest_sha256':sha(source_manifest),
         'source_frozen_artifacts_verified':len(manifest['artifact_sha256']),'mismatches':mismatches,
         'route_ratios':ratios,'ranking_comparisons':ranks,'verified_window_event':observed,
         'solver_still_running':original_solver_still_running,'extra_solver_runs':0,
         'source_run_or_formula_or_route_values_changed':False,'visual_overview_reviewed':True,
         'claim':'Complete continuous candidate calculation, not quantitative original-fire reproduction'}
    save(REVIEW/'review_results.json',data);REPORT.write_text(report,'utf-8')
    links=re.findall(r'\]\(([^)]+)\)',report)
    for target in links:assert Path(target).is_absolute() and Path(target).exists(),target
    assert all(sha(ROOT/p)==h for p,h in manifest['artifact_sha256'].items())
    files=[*REVIEW.glob('*.json'),REPORT,Path(__file__).resolve()]
    save(REVIEW/'completion_manifest.json',{'status':'READ_ONLY_REVIEW_COMPLETE_NO_NEW_SIMULATION',
         'source_manifest_sha256':sha(source_manifest),'source_frozen_files_verified':len(manifest['artifact_sha256']),
         'local_report_links_verified':len(links),'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print(json.dumps({'status':'READ_ONLY_REVIEW_COMPLETE','route_ratios':ratios,
         'window_geometry_time_s':window[0]['geometry_event_time_s'],'source_files_unchanged':True,
         'local_links_verified':len(links)}),flush=True)


if __name__=='__main__':main()
