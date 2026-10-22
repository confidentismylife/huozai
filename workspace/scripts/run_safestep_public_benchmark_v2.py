"""Version 2: normalize the NumPy-derived counter before JSON serialization.

The completed first run is frozen; this entry point writes only fresh v2 outputs.
Policy, scoring, departures and evaluation budget are unchanged.
"""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from fire_evac.safestep_author_numerics import load_author_inputs,StoredAuthorPolicy
from fire_evac.safestep_public_benchmark import (
    Graph,RateField,nearest_exit_policy,snapshot_policy,future_min_fed_policy,
    forward_min_fed_reference,rollout,
)

CONFIG=ROOT/'schemas/safestep_public_benchmark_20261017_v2.json'
REPORT=ROOT/'reports/SafeStep公开危险率统一计分对照_2026-10-17_v2.md'
LABELS={
    'nearest_exit':'最近出口',
    'author_DQN_legal_mask':'作者DQN（共同合法动作约束）',
    'current_snapshot_replan':'当前危险率重规划',
    'full_future_minFED_reference':'完整未来最小FED参照',
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,data):
    assert not p.exists(),p
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_save(p,data):
    assert not p.exists() and data
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def legacy_score(path,inputs,departure,dt):
    times=inputs['rates']['Time'];base=int(np.flatnonzero(times==departure)[0])
    value=0.
    for k,cell in enumerate(path):
        label=inputs['mapping'][tuple(cell)]
        if label.startswith('EXIT'):continue
        index=base+int(dt*k/5)
        assert index<len(times),'No capped/extrapolated legacy data'
        value+=float(inputs['rates'][label][index])*dt
    return value


def group_summary(data):
    success=[r for r in data if r['reached']]
    fields=('FED','travel_time_s','distance_m','FED_excess_over_future_reference','revisits')
    result=dict(n=len(data),reached=len(success),arrival_rate=len(success)/len(data))
    for field in fields:
        values=[r[field] for r in success]
        result['mean_'+field]=float(np.mean(values)) if values else None
        result['median_'+field]=float(np.median(values)) if values else None
    result['successful_FED_at_most_0_3']=sum(r['FED']<=.3 for r in success)
    return result


def paired(data,a,b,tolerance):
    first={r['case_id']:r for r in data if r['method']==a}
    second={r['case_id']:r for r in data if r['method']==b}
    assert first.keys()==second.keys()
    both=[(first[k],second[k]) for k in first if first[k]['reached'] and second[k]['reached']]
    diffs=[x['FED']-y['FED'] for x,y in both]
    return dict(method_a=a,method_b=b,paired_reached=len(both),
        a_lower_FED=sum(d < -tolerance for d in diffs),ties=sum(abs(d)<=tolerance for d in diffs),
        a_higher_FED=sum(d > tolerance for d in diffs),
        mean_FED_a_minus_b=float(np.mean(diffs)) if diffs else None,
        a_Pareto_better=sum(x['FED']<=y['FED']+tolerance and x['steps']<=y['steps']
                           and (x['FED']<y['FED']-tolerance or x['steps']<y['steps']) for x,y in both),
        b_Pareto_better=sum(y['FED']<=x['FED']+tolerance and y['steps']<=x['steps']
                           and (y['FED']<x['FED']-tolerance or y['steps']<x['steps']) for x,y in both))


def make_figure(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles=[('#0072B2','o','-'),('#D55E00','s','-'),('#009E73','^','-'),('#666666','x','--')]
    with plt.rc_context({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        for (method,label),style in zip(LABELS.items(),styles):
            records=[r for r in summary if r['method']==method]
            times=[r['departure_fire_s'] for r in records]
            for ax,field in zip(axes,('mean_FED','mean_travel_time_s')):
                ax.plot(times,[r[field] for r in records],color=style[0],marker=style[1],
                        linestyle=style[2],label=label,linewidth=1.5,markersize=4)
        for ax in axes:
            ax.axvline(300,color='#AAAAAA',linestyle=':',linewidth=1)
            ax.set_xlabel('出发时的火灾绝对时间（秒）');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
        axes[0].set_ylabel('到达回合的平均FED');axes[1].set_ylabel('到达回合的平均行走时间（秒）')
        axes[0].set_title('同一公开危险率：61个起点的描述性均值')
        axes[1].set_title('暴露与行走时间需同时评价')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle('信息条件不同：灰色虚线使用完整未来；一个火灾、一个固定DQN权重',fontsize=11)
        fig.savefig(out/'departure_comparison.png',dpi=180)
        fig.savefig(out/'departure_comparison.svg');plt.close(fig)


def make_report(out,cfg,summary,pairwise,paper_starts,metadata):
    primary=cfg['primary_departure_fire_s']
    primary_rows=[r for r in summary if r['departure_fire_s']==primary]
    table='\n'.join(f"| {LABELS[r['method']]} | {r['reached']}/{r['n']} | {r['mean_FED']:.6f} | {r['mean_travel_time_s']:.2f} | {r['mean_FED_excess_over_future_reference']:.6f} |" for r in primary_rows)
    pair=next(p for p in pairwise if p['departure_fire_s']==primary and p['method_a']=='author_DQN_legal_mask' and p['method_b']=='nearest_exit')
    snapshot_pair=next(p for p in pairwise if p['departure_fire_s']==primary and p['method_a']=='current_snapshot_replan' and p['method_b']=='author_DQN_legal_mask')
    case_table='\n'.join(f"| {r['start_label']} | {LABELS[r['method']]} | {r['steps']} | {r['FED']:.6f} | {r['legacy_FED']:.6f} |" for r in paper_starts)
    transfer=[]
    for t in cfg['departure_fire_times_s']:
        v={r['method']:r for r in summary if r['departure_fire_s']==t}
        transfer.append(f"| {t} | "+' | '.join(f"{v[m]['mean_FED']:.6f}" for m in LABELS)+' |')
    content=f'''# SafeStep公开危险率统一计分对照

日期：2026-10-17。状态：完成。范围：同一酒店、同一公开火灾危险率、一个作者保存DQN权重；没有新训练、FDS模拟或LLM调用。

## 本轮问题与结果

本轮完成{metadata['case_count']}个起点—出发时刻组合、{metadata['rollouts']}条方法轨迹。它们是同一个火灾中的描述性组合，不是独立火灾重复，也不是新封存测试集。

在300秒出发、61个起点的主要结果中，作者DQN相对最近出口：{pair['a_lower_FED']}处暴露更低，{pair['ties']}处近似相同，{pair['a_higher_FED']}处更高。当前危险率重规划相对DQN：{snapshot_pair['a_lower_FED']}处更低，{snapshot_pair['ties']}处近似相同，{snapshot_pair['a_higher_FED']}处更高。比较容差FED={cfg['reporting']['paired_tolerance_FED']}。

| 方法 | 到达起点数 | 平均FED | 平均行走时间/秒 | 相对完整未来参照的平均FED差额 |
|---|---:|---:|---:|---:|
{table}

上述均值仅在对应方法成功到达的回合上计算。到达数和失败单列；配对比较只使用双方均到达的同一组合，避免把未到达轨迹的低部分剂量算成优点。

## 方法获得的信息并不相同

| 方法 | 推理时可用信息 | 地位 |
|---|---|---|
| 最近出口 | 地图和位置 | 距离基线 |
| 作者DQN | 位置；固定公开权重 | 场景策略重放，原训练预算和数据暴露未独立恢复 |
| 当前危险率重规划 | 地图、位置、当前所有区域精确FER | 每一步更新当前快照；规划时假设快照不变，不输入未来行 |
| 完整未来最小FED参照 | 整个评价时段的未来FER | 具备额外信息的有限时域最小暴露参照，不能当作公平在线竞争者 |

因此本轮是**共同环境、动作约束和计分下的方法对照**，不能写成相同观测、相同训练预算的公平训练实验。DQN不会因为本轮读取了新时刻危险率而自动获得感知烟气的能力。

## 冻结范围和计分

- 主要分析：已研究过的300秒出发，全部61个非出口起点。
- 出发时刻诊断：{cfg['departure_fire_times_s']}秒；其余八个时刻单独呈现。同一火灾条件下改变出发时间，不代表新火灾或新建筑泛化。
- 2m一步、0.3m/s，每步6.6667秒；共同最多45个合法动作，允许折返，没有等待动作。最长评价300秒，最后出发600秒时恰覆盖到公开表末尾900秒。
- 到任一网格楼梯出口立即终止；没有因FED超过某阈值提前终止，未声称模拟实际失能。
- 主指标：对公开5秒间隔FER进行分段线性插值，在每一步出发格驻留区间积分；初始区间只计一次，无出发前暴露，无出口停留暴露。率表覆盖5～900秒，没有插值出范围。
- 旧兼容指标：保留作者初始格项、后续格向下取样和固定时长算法，供追溯；它不参与本轮规划目标选择。

完整未来方法采用时间展开图上的反向动态规划，目标为45步内到达时的最小FED，暴露相同优先更少步数。它不是原DQN奖励函数的最优解；后者还含步数、重复访问及剂量分段权重。最小FED也不等于同时最小行走时间。

## 原论文两个起点：主积分和旧兼容口径

| 起点 | 方法 | 步数 | 主积分FED | 旧兼容FED |
|---|---|---:|---:|---:|
{case_table}

原作者路线仍按旧兼容口径匹配既有结果；主积分数值不同，不应当作“原论文四位小数复现失败”或悄悄替换原论文结果。

## 改变出发时间后的描述性结果

各格为相应方法在61个起点到达回合上的平均FED；到达计数和全部明细见CSV。

| 出发时刻/秒 | 最近出口 | 作者DQN | 当前快照重规划 | 完整未来参照 |
|---|---:|---:|---:|---:|
{chr(10).join(transfer)}

![出发时刻对照](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/departure_comparison.png)

图的连线仅连接预先固定的9个出发时刻的描述性均值；未平滑、未插补失败回合、没有置信区间。每个时刻共用61个起点，所有时刻共用同一个火灾。图片用于内部研究汇报，未声称符合某期刊版式要求。

## 可信度检查

- 冻结计划与输入哈希先于本轮方法评分保存。
- 四条已知路线的主积分与旧兼容积分均复核，误差在1e-10内。
- 300秒时作者DQN的61条路线与既有归档逐条一致：{metadata['stored_300_paths_match']}。合法动作约束改变原始Q最大动作的实际决策次数：{metadata['raw_DQN_action_mask_changes']}。
- 完整未来参照用独立的正向时间展开最短路核对{metadata['independent_reference_checks']}项；所有成功方法轨迹的FED都不低于对应参照，容差1e-10。
- 独立单元检查覆盖解析积分、起点与出口计分、有限时域、穷举小图、当前快照改道、失败处理。
- 既有冻结文件与输入在本轮前后复核通过；没有读取48个封存人群场景。

## 研究结论的边界

这轮能够量化在公开危险率背景下距离、固定策略与当前信息规划的差异，以及它们距离有限时域最小FED还有多少空间。它不能证明哪种方法适用于未见建筑、真实人群，或证明LLM有新增收益。

当前研究主线可固定为“公开数据下的路径规划数值复现与基线审计”。自建FDS的物理差距继续作为独立局限；无需用更多训练去补偿缺失的物理资料。后续是否值得引入LLM，需要先明确其新增信息和可检验的收益，不预设它一定优于这些基线。

## 可追溯文件

- [冻结方案](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/frozen_plan.json)
- [全部逐回合计分](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/episode_metrics.csv)
- [分时刻汇总](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/departure_summary.csv)
- [配对比较](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/paired_comparisons.csv)
- [全部路径与执行动作](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/rollouts.json)
- [验证与计数](D:/UserFolders/Documents/ChatGPT/LLM探索/{out.relative_to(ROOT).as_posix()}/verification.json)
'''
    assert not REPORT.exists();REPORT.write_text(content,encoding='utf-8')


def main():
    wall_start=time.perf_counter()
    cfg=json.loads(CONFIG.read_text('utf-8'));out=ROOT/cfg['output_directory']
    assert not out.exists() and not REPORT.exists()
    source=ROOT/cfg['source_directory']
    files=[CONFIG,Path(__file__),ROOT/'src/fire_evac/safestep_public_benchmark.py',
           ROOT/'tests/test_safestep_public_benchmark.py',ROOT/'src/fire_evac/safestep_author_numerics.py']
    files += [source/n for n in ('main.py','coordinates2areanumber.py','FED rate - room24.xlsx',
                                'fire hazard zones FED - room24.xlsx','maze_dql.pth')]
    files += [ROOT/'outputs/safestep_continuous450_20261017/frozen_routes.json',
              ROOT/'outputs/safestep_continuous450_20261017/analysis_results.json',
              ROOT/'outputs/safestep_paper_numerics_20261015/numeric_run/results.json']
    protected={}
    for name in ('safestep_continuous450_20261017','safestep_continuous450_review_20261017',
                 'safestep450_source_boundary_audit_20261017'):
        m=json.loads((ROOT/'outputs'/name/'completion_manifest.json').read_text('utf-8'))
        protected.update(m['artifact_sha256'])
    for rel,expected in protected.items():assert sha(ROOT/rel)==expected,rel
    input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    out.mkdir()
    save(out/'frozen_plan.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
         stage='FROZEN_BEFORE_NEW_POLICY_SCORING',configuration=cfg,input_sha256=input_hashes,
         protected_prior_sha256=protected))
    torch.set_num_threads(1)
    inputs=load_author_inputs(source,cfg['primary_departure_fire_s'])
    starts=[tuple(c) for c in inputs['empty']];assert len(starts)==61
    graph=Graph(starts+list(inputs['exits']),inputs['exits'])
    raw=inputs['rates'];times=np.asarray(raw['Time'])
    field=RateField(times,np.column_stack([raw[inputs['mapping'][c]] if c not in graph.exits
                                         else np.zeros(len(times)) for c in graph.cells]))
    dt=cfg['movement']['grid_step_m']/cfg['movement']['speed_m_per_s'];horizon=cfg['movement']['maximum_actions']
    assert max(cfg['departure_fire_times_s'])+horizon*dt<=times[-1]
    model=StoredAuthorPolicy(source,starts,int)
    dqn=np.full(len(graph.cells),-1,dtype=int);qdiagnostic=[]
    for i,c in enumerate(starts):
        q=model.q(c);raw_action=int(np.argmax(q));legal=[a for a,_ in graph.edges[i]]
        dqn[i]=min(legal,key=lambda a:(-float(q[a]),a))
        qdiagnostic.append(dict(cell_x=c[0],cell_y=c[1],raw_argmax=raw_action,
                                common_legal_argmax=int(dqn[i]),changed=raw_action!=dqn[i],
                                **{f'Q_{a}':float(q[a]) for a in range(4)}))
    nearest,distances=nearest_exit_policy(graph);assert np.isfinite(distances).all()
    # Preflight all four known routes under both conventions, without selecting routes.
    library=json.loads(files[-3].read_text('utf-8'))
    archived=json.loads(files[-2].read_text('utf-8'))
    old=json.loads(files[-1].read_text('utf-8'))
    expected_primary={r['id']:r['archived_same_primary_FED'] for r in archived['candidate_routes']}
    preflight=[]
    for route in library['routes']:
        total=sum(float(field.integrate(s['start_fire_s'],s['end_fire_s'])[graph.index[tuple(s['cell'])]])
                  for s in route['segments'])
        legacy=legacy_score(route['path'],inputs,300,dt)
        assert abs(total-expected_primary[route['id']])<1e-10
        assert abs(legacy-route['archived_author_compatibility_FED'])<1e-10
        preflight.append(dict(id=route['id'],primary_FED=total,legacy_FED=legacy))
    old_paths={tuple(r['start']):r['path'] for r in old['all_start_diagnostic']}
    episodes=[];rollouts=[];reference_checks=[];stored_matches=0;mask_changes=0
    methods=[m['id'] for m in cfg['methods']]
    for departure in cfg['departure_fire_times_s']:
        costs=field.interval_costs(departure,dt,horizon)
        future,values,_=future_min_fed_policy(graph,costs)
        snapshots=np.asarray([snapshot_policy(graph,field.sample(departure+k*dt)) for k in range(horizon)])
        policies=dict(nearest_exit=np.tile(nearest,(horizon,1)),author_DQN_legal_mask=np.tile(dqn,(horizon,1)),
                      current_snapshot_replan=snapshots,full_future_minFED_reference=future)
        for check_start in ((5,1),(10,9),(1,1),(5,13)):
            si=graph.index[check_start];forward,steps=forward_min_fed_reference(graph,costs,si)
            assert abs(forward-values[0,si])<1e-10
            reference_checks.append(dict(departure_fire_s=departure,start=check_start,
                                         forward_FED=forward,backward_FED=float(values[0,si]),forward_steps=steps))
        for start in starts:
            case_id=f't{departure:04d}_x{start[0]:02d}_y{start[1]:02d}';si=graph.index[start]
            for method in methods:
                run=rollout(graph,policies[method],costs,si)
                best=float(values[0,si]);assert np.isfinite(best)
                if run['reached']:assert run['FED']>=best-1e-10
                if method=='full_future_minFED_reference':
                    assert run['reached'] and abs(run['FED']-best)<1e-10
                if method=='author_DQN_legal_mask':
                    mask_changes+=sum(qdiagnostic[graph.index[tuple(s['source'])]]['changed'] for s in run['trace'])
                    if departure==300:
                        assert run['path']==[tuple(c) for c in old_paths[start]],start
                        stored_matches+=1
                legacy=legacy_score(run['path'],inputs,departure,dt)
                row=dict(case_id=case_id,departure_fire_s=departure,start_x=start[0],start_y=start[1],method=method,
                         reached=run['reached'],steps=run['steps'],travel_time_s=run['steps']*dt,
                         distance_m=run['steps']*cfg['movement']['grid_step_m'],FED=run['FED'],legacy_FED=legacy,
                         future_minimum_FED=best,FED_excess_over_future_reference=run['FED']-best if run['reached'] else None,
                         revisits=run['revisits'],exit_label=inputs['mapping'][tuple(run['exit'])] if run['exit'] else '',
                         evaluation_group='known_300s_primary' if departure==300 else 'departure_transfer_diagnostic')
                episodes.append(row)
                rollouts.append(dict(**row,path=run['path'],executed_steps=run['trace']))
        print(json.dumps(dict(completed_departure_fire_s=departure,rollouts_so_far=len(episodes))),flush=True)
    assert len(episodes)==len(starts)*len(cfg['departure_fire_times_s'])*len(methods)
    summary=[];pairwise=[]
    for departure in cfg['departure_fire_times_s']:
        at_time=[r for r in episodes if r['departure_fire_s']==departure]
        for method in methods:
            summary.append(dict(departure_fire_s=departure,method=method,
                                **group_summary([r for r in at_time if r['method']==method])))
        for a,b in [('author_DQN_legal_mask','nearest_exit'),('current_snapshot_replan','author_DQN_legal_mask'),
                    ('current_snapshot_replan','nearest_exit'),('author_DQN_legal_mask','full_future_minFED_reference')]:
            pairwise.append(dict(departure_fire_s=departure,**paired(at_time,a,b,cfg['reporting']['paired_tolerance_FED'])))
    metadata=dict(case_count=len(starts)*len(cfg['departure_fire_times_s']),rollouts=len(episodes),
                  independent_reference_checks=len(reference_checks),stored_300_paths_match=stored_matches,
                  raw_DQN_action_mask_changes=int(mask_changes),all_rollouts_reached=sum(r['reached'] for r in episodes),
                  successful_rollout_reference_bound_checks=sum(r['reached'] for r in episodes),
                  failed_rollouts=[r for r in episodes if not r['reached']],known_route_preflight=preflight,
                  reference_checks=reference_checks,solver_runs=0,training_updates=0,LLM_requests=0,
                  final48_accessed=False,independent_fire_count=1,stored_checkpoint_count=1)
    csv_save(out/'episode_metrics.csv',episodes);csv_save(out/'departure_summary.csv',summary)
    csv_save(out/'paired_comparisons.csv',pairwise);csv_save(out/'stored_policy_q_diagnostic.csv',qdiagnostic)
    save(out/'rollouts.json',rollouts)
    save(out/'information_conditions.json',cfg['methods'])
    paper_starts=[]
    for position,xy in (('起点1',(5,1)),('起点2',(10,9))):
        for r in episodes:
            if r['departure_fire_s']==300 and (r['start_x'],r['start_y'])==xy:
                paper_starts.append(dict(start_label=position,**r))
    csv_save(out/'paper_start_comparison.csv',paper_starts)
    make_figure(out,summary)
    metadata['elapsed_wall_s_before_report']=time.perf_counter()-wall_start
    save(out/'verification.json',metadata)
    make_report(out,cfg,summary,pairwise,paper_starts,metadata)
    for rel,h in input_hashes.items():assert sha(ROOT/rel)==h,rel
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    links=re.findall(r'\]\((D:/[^)]+)\)',REPORT.read_text('utf-8'))
    assert all(Path(p).is_file() for p in links)
    artifacts=[p for p in out.iterdir() if p.is_file()]+[REPORT,*files[:4]]
    save(out/'completion_manifest.json',dict(status='COMPLETE_PUBLIC_HAZARD_DESCRIPTIVE_BENCHMARK',
         protected_previous_files_verified=len(protected),input_files_unchanged=len(input_hashes),
         local_report_links_verified=len(links),
         artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts}))
    print(json.dumps(dict(status='COMPLETE',metadata={k:v for k,v in metadata.items() if k not in
          ('reference_checks','known_route_preflight','failed_rollouts')},
          primary=[r for r in summary if r['departure_fire_s']==300],
          primary_paired=[r for r in pairwise if r['departure_fire_s']==300]),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
