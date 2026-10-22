"""Analyze one continuous candidate, preserving all author-original caveats."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from prepare_safestep_continuous450 import ROOT, OUT, CASE, PREFIX, AUTHOR, sha, save
from safestep_gas_to_fer import read_fds_gas_csv, gas_to_fer
from safestep_native_time import NativeRateCurve
from analyze_safestep_fds_smokecheck import parse_smv

REPORT=ROOT/'reports/SafeStep连续450秒候选火灾与四条路线FED_2026-10-17.md'


def link(path,label):return f'[{label}]({path.as_posix()})'


def author_curves():
    wb=openpyxl.load_workbook(AUTHOR/'FED rate - room24.xlsx',read_only=True,data_only=True)
    rows=list(wb['Sheet1'].values);wb.close();header=rows[0]
    values=np.array([r for r in rows[1:] if r[0] is not None],dtype=float)
    assert np.isfinite(values).all()
    return {str(name):NativeRateCurve(values[:,0],values[:,i],5.) for i,name in enumerate(header) if i>0 and str(name).startswith('R')}


def route_values(curves,routes):
    output=[];trace=[]
    for route in routes:
        total=0.
        for step,s in enumerate(route['segments']):
            c=curves[s['region']];start=s['start_fire_s'];end=s['end_fire_s']
            inc=c.integrate(start,end);total+=inc
            trace.append({'route':route['id'],'movement_interval':step,'region':s['region'],
                'start_fire_s':start,'end_fire_s':end,'duration_s':end-start,
                'FER_start_per_s':float(c.sample([start])[0]),'FER_end_per_s':float(c.sample([end])[0]),
                'FED_increment':inc,'FED_cumulative':total})
        assert abs(sum(s['end_fire_s']-s['start_fire_s'] for s in route['segments'])-route['travel_time_s'])<1e-9
        output.append({'id':route['id'],'steps':route['steps'],'travel_time_s':route['travel_time_s'],
           'end_fire_s':route['segments'][-1]['end_fire_s'],'FED':total})
    return output,trace


def finish_manifest(plan,summary):
    bad=[p for p,h in plan['protected_previous_sha256'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not bad,bad
    assert sha(CASE/f'{PREFIX}.fds')==plan['new_input_sha256']
    assert not list(OUT.rglob('*.stop'))
    # The supervisor still owns these two handles until analysis returns.
    # Solver raw stdout/stderr are already closed and remain in the manifest.
    live_names=['completion_manifest.json','latest_status.json','monitor_stdout.log','monitor_stderr.log']
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in live_names]
    files += [REPORT,ROOT/'scripts/prepare_safestep_continuous450.py',ROOT/'scripts/run_safestep_continuous450.py',
              ROOT/'scripts/read_safestep_continuous450_status.py',Path(__file__).resolve()]
    verification={'verified_at':datetime.now(timezone.utc).isoformat(),'status':summary['status'],
       'protected_previous_files_verified':len(plan['protected_previous_sha256']),'mismatches':bad,
       'native_input_unchanged':True,'stop_files_created':0,'restart_loaded':False,'solver_runs':1,
       'training_updates':0,'LLM_requests':0,'final48_accessed':False,'900s_runs':0,
       'original_author_fire_input_recovered':False,'candidate_FED_is_original_paper_validation':False}
    save(OUT/'final_verification.json',verification);files.append(OUT/'final_verification.json')
    save(OUT/'completion_manifest.json',{'status':summary['status'],'created_at':datetime.now(timezone.utc).isoformat(),
         'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},
         'protected_previous_files_verified':len(plan['protected_previous_sha256']),
         'supervisor_logs_excluded_because_writer_still_open':['monitor_stdout.log','monitor_stderr.log'],
         'solver_runs':1,'restart_loaded':False,'training_updates':0,'LLM_requests':0,'final48_accessed':False})
    save(OUT/'latest_status.json',{'status':summary['status'],'report':REPORT.relative_to(ROOT).as_posix(),
         'final_native_fire_s':summary.get('final_native_fire_s'),'finished_at':datetime.now(timezone.utc).isoformat()})


def main():
    assert not (OUT/'completion_manifest.json').exists()
    plan=json.loads((OUT/'frozen_plan.json').read_text('utf-8'))
    run=json.loads((CASE/'run_result.json').read_text('utf-8'))
    if not run['passed']:
        summary={'status':'INCOMPLETE_CONTINUOUS450_NO_ROUTE_FED','run':run,'route_FED_computed':False,
                 'final_native_fire_s':run['final_native_fire_s']}
        save(OUT/'analysis_results.json',summary)
        REPORT.write_text('# SafeStep 连续450秒候选火灾实验：未完成\n\n'
            f"本次运行未正常完成到450秒。已记录末次原生时间：{run['final_native_fire_s']}秒。\n\n"
            f"终止原因：`{run['termination_reason']}`；求解器错误：`{run['FDS_errors']}`。\n\n"
            '本轮没有计算完整四条路线FED，没有从断点恢复或自动重跑。原始输出保留。\n\n'
            +link(CASE/'run_result.json','运行记录')+'；'+link(OUT/'frozen_plan.json','冻结计划')+'。\n','utf-8')
        finish_manifest(plan,summary);print(json.dumps(summary),flush=True);return
    parameters=json.loads((OUT/'frozen_candidate_FER_parameters.json').read_text('utf-8'))
    library=json.loads((OUT/'frozen_routes.json').read_text('utf-8'));routes=library['routes']
    assert parameters['use_for_candidate_continuous450_route_FED'] and parameters['convention_status']=='reconstruction_assumption'
    times,samples=read_fds_gas_csv(CASE/f'{PREFIX}_devc.csv');times=np.array(times)
    assert times[0]==0 and abs(times[-1]-450)<1e-6
    all_data=pd.read_csv(CASE/f'{PREFIX}_devc.csv',skiprows=1)
    assert np.isfinite(all_data.to_numpy()).all() and np.array_equal(times,all_data.Time.to_numpy())
    curves={};long=[]
    for i in range(1,59):
        fer=[]
        for t,s in zip(times,samples):
            v=gas_to_fer({g:s[(i,g)] for g in ('CO','CO2','HCN')},{g:'mol/mol' for g in ('CO','CO2','HCN')},parameters)
            fer.append(v['FER_total_per_s']);long.append({'region':f'R{i}','fire_time_s':t,'evacuation_elapsed_s':t-300.,**v})
        curves[f'R{i}']=NativeRateCurve(times,fer,plan['analysis']['native_gap_max_s'])
    pd.DataFrame(long).to_csv(OUT/'native_gas_and_FER_long.csv',index=False,encoding='utf-8-sig')
    candidate,trace=route_values(curves,routes);archived,archive_trace=route_values(author_curves(),routes)
    pd.DataFrame(trace).to_csv(OUT/'candidate_route_residence_integrals.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(archive_trace).to_csv(OUT/'archived_rate_same_residence_integrals.csv',index=False,encoding='utf-8-sig')
    mapping={tuple(x['cell']):x['region'] for x in library['author_mapping']}
    compatibility=[]
    for r in routes:
        dt=library['grid_step_m']/library['walking_speed_m_s'];total=0.;ct=[]
        for k,cell in enumerate(r['path']):
            region=mapping[tuple(cell)];sample_time=300+5*int(dt*k/5)
            rate=float(curves[region].sample([sample_time])[0]) if region in curves else 0.
            inc=rate*dt;total+=inc;ct.append({'step':k,'region':region,'sample_fire_s':sample_time,'increment':inc})
        compatibility.append({'id':r['id'],'FED':total,'trace':ct})
    save(OUT/'author_compatibility_candidate_FED.json',{'note':'Separate legacy floor-sampled convention, not primary FED authority','results':compatibility})
    comparisons=[]
    for r,c,a,b in zip(routes,candidate,archived,compatibility):
        comparisons.append({**c,'candidate_primary_FED':c['FED'],'archived_same_primary_FED':a['FED'],
           'candidate_legacy_compatibility_FED':b['FED'],'archived_legacy_paper_reproduction_FED':r['archived_author_compatibility_FED']})
    pd.DataFrame(comparisons).drop(columns=['FED']).to_csv(OUT/'fixed_route_FED_comparison.csv',index=False,encoding='utf-8-sig')
    ranks=[]
    for start in (1,2):
        rec=next(r for r in comparisons if r['id']==f'hotel_position{start}_recommended')
        alt=next(r for r in comparisons if r['id']==f'hotel_position{start}_alternative')
        x=rec['candidate_primary_FED'];y=alt['candidate_primary_FED']
        ranks.append({'start':start,'recommended_FED':x,'alternative_FED':y,
           'recommended_strictly_lower':x<y,'absolute_difference_alternative_minus_recommended':y-x,
           'relative_reduction_vs_alternative':(y-x)/y if y>0 else None,
           'interpretation':'Fixed-route comparison within this assumed candidate, not learned policy benefit'})
    hrr=pd.read_csv(CASE/f'{PREFIX}_hrr.csv',skiprows=1)
    assert np.isfinite(hrr.to_numpy()).all() and np.all(np.diff(hrr.Time)>0)
    _,events=parse_smv(CASE/f'{PREFIX}.smv')
    zone_cols=[c for c in hrr.columns if c.startswith('ZONE_')]
    pressure_summary={c:{'min_gauge_Pa':float(hrr[c].min()),'max_gauge_Pa':float(hrr[c].max()),
          'final_gauge_Pa':float(hrr[c].iloc[-1])} for c in zone_cols}
    gas_summary={}
    for gas in ['CO','CO2','HCN']:
        cols=[f'SS_R{i:02d}_{gas}' for i in range(1,59)];arr=all_data[cols].to_numpy()*1e6
        arg=np.unravel_index(np.argmax(arr),arr.shape)
        gas_summary[gas]={'corridor_max_ppm':float(arr[arg]),'peak_time_s':float(times[arg[0]]),'peak_region':f'R{arg[1]+1}',
            'regions_exceeding_1ppm':int(np.sum(np.max(arr,axis=0)>1.)),
            'room24_max_ppm':float(all_data[f'ROOM24_{gas}'].max()*1e6)}
    prefix=json.loads((OUT/'continuous_prefix_check.json').read_text('utf-8'));assert prefix['passed']
    summary={'status':'COMPLETE_CONTINUOUS450_EXPLORATORY_CANDIDATE_NOT_ORIGINAL_PHYSICS',
        'completed_at':datetime.now(timezone.utc).isoformat(),'final_native_fire_s':float(times[-1]),'run':run,
        'native_time_rows':len(times),'native_max_gap_s':float(np.max(np.diff(times))),
        'gas_devices':174,'corridor_regions':58,'candidate_routes':comparisons,'route_rankings':ranks,
        'gas_summary':gas_summary,'pressure_zone_summary':pressure_summary,
        'ventilation_boundary_scope':'Only one explicit OPEN VENT, ROOM24 XMIN, in inherited input; RL exit A/B to physical open-boundary correspondence is unverified',
        'CO2_regions_above1ppm_is_not_smoke_arrival_evidence':'Initial ambient CO2 already exceeds 1ppm; use full baseline/time series for interpretation',
        'geometry_events':[e for e in events if e['ID'] in ['SS_DOOR24','SS_WINDOW24']],
        'prefix_check':prefix,'route_FED_computed':True,'model_trained':False,'original_fire_input_recovered':False,
        'claim_boundary':'Candidate results conditional on geometry, fuel, HCN, window, fire curve and physiology assumptions; no causal or original-material validation',
        'training_updates':0,'LLM_requests':0,'final48_accessed':False,'900s_runs':0}
    save(OUT/'analysis_results.json',summary)
    fig,axes=plt.subplots(3,1,figsize=(11,10),constrained_layout=True)
    axes[0].plot(hrr.Time,hrr.HRR,label='FDS output-window HRR')
    axes[0].plot(times,5000*np.minimum(times/326,1)**2,ls='--',label='Prescribed growth and plateau')
    axes[0].set(ylabel='HRR (kW)',title='Continuous candidate: prescribed and computed heat release');axes[0].legend()
    for gas,color in [('CO','#b03a2e'),('HCN','#1b6f5a')]:
        cols=[f'SS_R{i:02d}_{gas}' for i in range(1,59)]
        axes[1].plot(times,all_data[cols].max(axis=1)*1e6,label=f'{gas}, max of 58 regions',color=color)
    axes[1].set(ylabel='Wet concentration (ppm)',title='Corridor probes; no safety threshold inferred');axes[1].legend()
    idx=np.arange(4);axes[2].bar(idx-.18,[r['candidate_primary_FED'] for r in comparisons],.36,label='Candidate, native-rate integral')
    axes[2].bar(idx+.18,[r['archived_same_primary_FED'] for r in comparisons],.36,label='Archived rates, same integral')
    axes[2].set(xticks=idx,xticklabels=['Start1-B','Start1-A','Start2-B','Start2-A'],ylabel='Conditional FED',title='Fixed paths, common residence convention');axes[2].legend()
    for ax in axes[:2]:ax.axvline(300,color='gray',ls=':',label='Evacuation start');ax.set_xlabel('Fire time (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Exploratory assumed fire model; not original-paper physical validation')
    fig.savefig(OUT/'continuous450_overview.png',dpi=180);plt.close(fig)
    rows='\n'.join(f"| {r['id'].replace('hotel_position','起点').replace('_recommended',' 推荐').replace('_alternative',' 替代')} | {r['steps']} | {r['candidate_primary_FED']:.8g} | {r['archived_same_primary_FED']:.8g} | {r['archived_legacy_paper_reproduction_FED']:.8g} |" for r in comparisons)
    ranktext='\n'.join(f"- 起点{r['start']}：本候选模型下，推荐路线FED{'低于' if r['recommended_strictly_lower'] else '没有低于'}替代路线；替代减推荐的绝对差为 {r['absolute_difference_alternative_minus_recommended']:.8g}。" for r in ranks)
    report=f'''# SafeStep 连续450秒候选火灾与四条固定路线 FED

日期：2026-10-17。状态：连续运行正常完成，完成候选路线暴露计算；不是原作者完整物理复现。

## 本轮做了什么

使用原版 FDS `{plan['FDS_version']}`，从0秒启动同一个进程，正常运行到450秒。求解器用时 {run['elapsed_wall_s']/60:.2f} 分钟；没有读取断点，没有创建 `.stop` 文件，没有修改求解器或加入诊断标记。

本轮是用户授权的连续运行替代路线。此前酒店断点续算失败的结论保留，没有降低或重新解释那一轮的门槛。

与正常120秒父输入相比，只改变 CHID 和 T_END。几何、区域映射、燃烧器、产率、开门和窗户代理设置均保留。火源326秒后保持峰值，是继承的假设，不是本轮验证出的真实曲线。

## 运行与数据质量

- 测点输出 {len(times)} 个原生时间节点，覆盖0–450秒，最大相邻间隔 {summary['native_max_gap_s']:.6g} 秒；58区域、174个CO/CO₂/HCN通道。
- 原始CSV全部数值有限；气体使用湿体积分数，明确换算CO/HCN为ppm、CO₂为百分数。
- 0–119秒与旧正常连续运行的前缀检查通过；相同CSV测点值是否完全一致：{prefix['all_prefix_device_values_bitwise_equal_in_CSV']}。这只核验执行一致性，不是物理正确性验证。
- 各路线均在300秒出发，速度0.3m/s，网格步长2m；最长路线在426.6667秒到达出口，处于本轮观测覆盖范围内。

## 四条固定路线结果

主计算在每个原生气体节点计算FER，再对节点之间的分段线性FER精确积分。按出发单元在每个移动区间内驻留，进入出口时结束；不加入300秒之前的暴露。没有先降采样到5秒，也没有外推。

| 路线 | 步数 | 本轮候选FED：主积分 | 公开危险度：相同主积分 | 公开危险度：旧论文兼容口径 |
|---|---:|---:|---:|---:|
{rows}

{ranktext}

最后一列是之前复现论文数值所用的5秒向下取样、固定步长口径，属于不同离散化方法，不能与本轮主积分混为一谈。本轮另存了候选数据的相同旧口径结果，仅作可追溯比较。

两套危险度来自不同来源、几何和物理假设，表格不构成候选火灾的校准或独立验证。没有用论文FED拟合本轮参数，也没有根据结果重新选择路线。

FED是累计剂量指标，不是死亡概率。这里在到达网格出口时停止累计，是路线模型的边界约定，不证明该物理出口已经安全或与室外连通。

## 假设及局限

气体转FER沿用已记录的探索性参考：通气量25L/min、COHb阈值30%，公式分钟速率除以60转为每秒。原SafeStep的对应参数与上游实现尚未核实。

区域XY映射和几何仍有推断；HCN产率0.007来自跨材料文献迁移；燃料性质存在模型假设；窗户是气体温度触发的开孔代理；326秒之后采用恒定峰值。没有进行真人、材料或建筑验证。

另一个明确的边界条件缺口是：继承的输入中只有一条显式OPEN VENT，位于ROOM24的XMIN面。RL出口A/B没有独立具名的OPEN VENT，二者与物理通风边界的对应关系尚未核实。因此，数值正常结束不代表酒店通风模型正确；本轮保留原输入，没有事后加开口改善结果。各压力分区的原生压力范围单独保存在完整结果中，供后续审查通风假设。

这里只计算固定路线在假定环境下的气体暴露，没有重新训练DQN、PPO或LLM，也没有证明推荐策略有因果收益。48个保留最终场景仍未读取。

![连续450秒结果]({(OUT/'continuous450_overview.png').as_posix()})

## 可追溯材料

- {link(OUT/'frozen_plan.json','冻结计划')}。
- {link(OUT/'input_changes.diff','输入差异')}。
- {link(CASE/'run_result.json','运行记录')}。
- {link(OUT/'analysis_results.json','完整结果与事件记录')}。
- {link(OUT/'fixed_route_FED_comparison.csv','四路线比较表')}。
- {link(OUT/'candidate_route_residence_integrals.csv','逐移动区间积分')}。
- {link(OUT/'native_gas_and_FER_long.csv','原生气体与FER')}。
'''
    assert not REPORT.exists();REPORT.write_text(report,'utf-8')
    finish_manifest(plan,summary)
    print(json.dumps({'status':summary['status'],'candidate_routes':comparisons,'rankings':ranks}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save(OUT/'analysis_failure.json',{'status':'ANALYSIS_FAILED_RAW_OUTPUTS_PRESERVED','traceback':traceback.format_exc()})
        raise
