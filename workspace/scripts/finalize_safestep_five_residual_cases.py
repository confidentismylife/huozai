"""Cross-check fixed-exit bounds, finalize source-grounded case and prose outputs."""
from __future__ import annotations

import csv
import heapq
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
from analyze_safestep_five_residual_cases import ROOT,OUT,SOURCE,sha,save,csv_save
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_public_benchmark import Graph,RateField

REPORTS=[ROOT/'reports/SafeStep五个偏差案例分析与阶段结论_2026-10-17.md',
         ROOT/'reports/SafeStep公开危险率复现_方法结果与讨论初稿_2026-10-17.md']


def fixed_goal_forward(graph,costs,start,goal):
    horizon=len(costs);best={(0,start):0.};queue=[(0.,0,start)]
    while queue:
        value,k,i=heapq.heappop(queue)
        if value!=best[(k,i)]:continue
        if i==goal:return value
        if i in graph.exit_ids or k==horizon:continue
        for _,j in graph.edges[i]:
            new=value+float(costs[k,i]);key=(k+1,j)
            if new<best.get(key,float('inf')):
                best[key]=new;heapq.heappush(queue,(new,k+1,j))
    return float('inf')


def render_reviewed_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT/'segment_ledger.csv').open(encoding='utf-8-sig',newline='') as f:
        segments=list(csv.DictReader(f))
    cases=results['case_summaries'];x=np.arange(len(cases))
    with plt.rc_context({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
        exit_gaps=[r['exit_choice_gap'] for r in cases];local=[r['within_chosen_exit_gap'] for r in cases]
        axes[0].bar(x,exit_gaps,color='#D55E00',label='出口选择差额')
        axes[0].bar(x,local,bottom=exit_gaps,color='#0072B2',hatch='//',label='同出口内路线差额')
        axes[0].set_xticks(x,[r['region'] for r in cases])
        # Explicit headroom: preserve values, avoid the first draft touching frame.
        axes[0].set_ylim(0,max(r['absolute_gap'] for r in cases)*1.15)
        axes[0].set(xlabel='事后选出的全部5个偏差起点',ylabel='DQN相对最小FED参照的差额',
                    title='差额的目标函数分解，不代表训练原因')
        axes[0].legend(fontsize=8)
        ax=axes[1]
        for method,color,label in (('author_DQN_legal_mask','#D55E00','DQN既有路线'),
                                   ('full_future_minFED_reference','#0072B2','最小FED既有路线')):
            group=[r for r in segments if r['case_id']=='t0300_x08_y09' and r['method']==method]
            times=[300]+[float(r['end_fire_s']) for r in group]
            actual=[0]+[float(r['actual_FED_cumulative']) for r in group]
            frozen=np.cumsum([0]+[float(r['snapshot300_FED_increment']) for r in group])
            ax.plot(times,actual,color=color,label=label+'：时变计分')
            ax.plot(times,frozen,color=color,linestyle='--',label=label+'：冻结300秒快照')
            ax.plot(times[-1],actual[-1],marker='o',color=color,markersize=4)
        ax.set(xlabel='火灾绝对时间（秒）',ylabel='累计FED',title='R24：初始快照与时变计分的偏好反转')
        ax.set_ylim(bottom=0);ax.legend(fontsize=7,loc='upper left')
        for a in axes:a.grid(axis='y',alpha=.2)
        fig.suptitle('同一公开火灾的5个缓存案例：全体分析，无新增训练或策略回合',fontsize=11)
        for suffix in ('png','svg'):
            path=OUT/f'residual_case_analysis_v2.{suffix}';assert not path.exists()
            fig.savefig(path,dpi=180)
        plt.close(fig)


def main():
    assert not (OUT/'completion_manifest.json').exists()
    plan=json.loads((OUT/'analysis_plan.json').read_text('utf-8'))
    for key in ('input_sha256','protected_prior_sha256'):
        for rel,h in plan[key].items():assert sha(ROOT/rel)==h,rel
    results=json.loads((OUT/'results.json').read_text('utf-8'))
    frozen=json.loads((SOURCE/'frozen_plan.json').read_text('utf-8'))['configuration']
    inputs=load_author_inputs(ROOT/frozen['source_directory'],300)
    graph=Graph([tuple(c) for c in inputs['empty']]+list(inputs['exits']),inputs['exits'])
    field=RateField(inputs['rates']['Time'],np.column_stack([
        inputs['rates'][inputs['mapping'][c]] if c not in graph.exits else np.zeros(len(inputs['rates']['Time']))
        for c in graph.cells]))
    costs=field.interval_costs(300.,2/.3,45)
    cached=json.loads((SOURCE/'rollouts.json').read_text('utf-8'))
    checks=[];secondary=[]
    for case in results['case_summaries']:
        i=graph.index[(case['start_x'],case['start_y'])]
        for goal in graph.exit_ids:
            label=inputs['mapping'][graph.cells[goal]]
            forward=fixed_goal_forward(graph,costs,i,goal)
            backward=case['fixed_exit_A_min_FED' if label=='EXIT_A' else 'fixed_exit_B_min_FED']
            assert abs(forward-backward)<1e-10
            checks.append(dict(case_id=case['case_id'],exit=label,forward_FED=forward,backward_FED=backward))
        runs={r['method']:r for r in cached if r['case_id']==case['case_id']}
        d=runs['author_DQN_legal_mask'];f=runs['full_future_minFED_reference'];s=runs['current_snapshot_replan']
        gap=d['legacy_FED']-f['legacy_FED'];assert gap>1e-10
        suffix=case['shared_suffix_start_step']
        if suffix is not None:
            assert d['path'][suffix:]==f['path'][suffix:]
            assert all(abs(x['FED_increment']-y['FED_increment'])<1e-10
                for x,y in zip(d['executed_steps'][suffix:],f['executed_steps'][suffix:]))
        secondary.append(dict(case_id=case['case_id'],region=case['region'],DQN_legacy_FED=d['legacy_FED'],
            reference_path_legacy_FED=f['legacy_FED'],legacy_paired_gap=gap,
            snapshot_actual_FED=s['FED'],snapshot_actual_gap_to_reference=s['FED']-f['FED']))
    assert results['count_exit_choice']==2 and results['count_local_route']==3
    assert abs(results['exit_choice_share']+results['local_route_share']-1)<1e-10
    r24=next(r for r in results['case_summaries'] if r['region']=='R24')
    assert r24['snapshot_gap_DQN_minus_reference']<0 and r24['absolute_gap']>0
    csv_save(OUT/'secondary_legacy_and_snapshot_checks.csv',secondary)
    save(OUT/'independent_fixed_exit_checks.json',dict(checks=checks,count=len(checks),passed=True,
        additional_policy_rollouts=0,role='Independent offline value calculation, not a new simulation'))
    render_reviewed_figure(results)
    tokens={'84.01%','15.99%','0.189715','0.149545','0.305101','0.279993','0.286856','0.282254',
            '0.263760','0.259219','0.240307','0.237029'}
    link_count=0
    for report in REPORTS:
        content=report.read_text('utf-8')
        assert all(token in content for token in tokens),report
        links=re.findall(r'\]\((D:/[^)]+)\)',content)
        assert all(Path(p).is_file() for p in links)
        link_count+=len(links)
    for key in ('input_sha256','protected_prior_sha256'):
        for rel,h in plan[key].items():assert sha(ROOT/rel)==h,rel
    save(OUT/'final_verification.json',dict(status='CASE_ANALYSIS_AND_MANUSCRIPT_COMPLETE',
        all_five_included=True,independent_fixed_exit_checks=len(checks),all_checks_passed=True,
        legacy_paired_rankings_preserved=5,full_legacy_optimum_computed=False,
        quantitative_tokens_and_local_links_checked=True,local_links_verified=link_count,
        protected_prior_files_verified=len(plan['protected_prior_sha256']),
        original_analysis_inputs_unchanged=True,original_benchmark_unchanged=True,
        original_first_figure_kept=True,reviewed_figure_change='Added axis headroom only; values unchanged',
        additional_policy_rollouts=0,training_updates=0,FDS_runs=0,LLM_requests=0,final48_accessed=False))
    files=[p for p in OUT.iterdir() if p.is_file()]+REPORTS+[
        ROOT/'scripts/analyze_safestep_five_residual_cases.py',Path(__file__),
        Path('C:/Users/摩天轮大盗/.codex/skills/research-writing-skill/SKILL.md')]
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in files if p.is_relative_to(ROOT)}
    save(OUT/'completion_manifest.json',dict(status='COMPLETE_POSTHOC_CASE_EXPLANATION_AND_DRAFT',
        artifact_sha256=artifacts,protected_prior_files_verified=len(plan['protected_prior_sha256']),
        research_writing_skill_sha256=sha(files[-1]),
        independent_validation_claim=False,new_method_validation_claim=False))
    print(json.dumps(dict(status='COMPLETE',case_count=5,exit_choice_cases=2,local_route_cases=3,
        independent_checks=len(checks),local_links_verified=link_count,artifacts=len(artifacts)),ensure_ascii=True))


if __name__=='__main__':main()
