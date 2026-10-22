"""Explain every non-minimal 300 s DQN case using frozen public-benchmark cache.

No new policy rollout or training. Two offline constrained dynamic programs
decompose dose excess into exit choice and routing within the chosen exit.
Freezing FER at departure is an accounting diagnostic, not a physical replay.
"""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_public_benchmark import Graph,RateField,DELTAS

SOURCE=ROOT/'outputs/safestep_public_benchmark_20261017'
OUT=ROOT/'outputs/safestep_five_residual_cases_20261017'
METHODS=('nearest_exit','author_DQN_legal_mask','current_snapshot_replan','full_future_minFED_reference')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_save(path,rows):
    assert not path.exists() and rows
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixed_exit_values(graph,costs,goal_id):
    """Other exit remains an absorbing terminal, but is infeasible for this goal."""
    horizon,n=costs.shape
    values=np.full((horizon+1,n),np.inf);values[:,goal_id]=0.
    for k in range(horizon-1,-1,-1):
        for i in range(n):
            if i in graph.exit_ids:continue
            options=[costs[k,i]+values[k+1,j] for _,j in graph.edges[i]]
            if options:values[k,i]=min(options)
    return values


def main():
    assert not OUT.exists()
    protected={}
    frozen=json.loads((SOURCE/'frozen_plan.json').read_text('utf-8'))
    for key in ('protected_prior_sha256','input_sha256'):protected.update(frozen[key])
    for name in ('completion_manifest.json','review_manifest.json'):
        protected.update(json.loads((SOURCE/name).read_text('utf-8'))['artifact_sha256'])
        protected[(SOURCE/name).relative_to(ROOT).as_posix()]=sha(SOURCE/name)
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    input_files=[SOURCE/n for n in ('effect_summary.json','rollouts.json','episode_metrics.csv',
                                   'departure_summary.csv','paired_comparisons.csv','information_conditions.json')]
    input_files.append(Path(__file__))
    OUT.mkdir()
    save(OUT/'analysis_plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='POST_HOC_CACHE_EXPLANATION_NOT_CONFIRMATORY',
        prior_exposure='All five cached paths were inspected before this explanation plan; no preregistration claim',
        selection='Every author_DQN case at departure=300 with dose excess over future reference >1e-10',
        expected_cases=5,analyses=['actual cached route comparison','fixed-exit value decomposition',
            'departure-snapshot vs time-varying dose accounting','aligned source-cell interval differences'],
        model_or_route_tuning=False,extra_policy_rollouts=0,
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in input_files},
        protected_prior_sha256=protected))
    cfg=frozen['configuration'];dt=cfg['movement']['grid_step_m']/cfg['movement']['speed_m_per_s']
    inputs=load_author_inputs(ROOT/cfg['source_directory'],300)
    starts=[tuple(c) for c in inputs['empty']];graph=Graph(starts+list(inputs['exits']),inputs['exits'])
    mapping=inputs['mapping'];times=inputs['rates']['Time']
    field=RateField(times,np.column_stack([inputs['rates'][mapping[c]] if c not in graph.exits
                                         else np.zeros(len(times)) for c in graph.cells]))
    costs=field.interval_costs(300.,dt,45);snapshot=field.sample(300.)
    values={mapping[graph.cells[e]]:fixed_exit_values(graph,costs,e) for e in graph.exit_ids}
    cache=json.loads((SOURCE/'rollouts.json').read_text('utf-8'))
    selected=[r for r in cache if r['method']=='author_DQN_legal_mask' and r['departure_fire_s']==300
              and r['FED_excess_over_future_reference']>1e-10]
    assert len(selected)==5
    declared=json.loads((SOURCE/'effect_summary.json').read_text('utf-8'))
    assert {r['case_id'] for r in selected}=={r['case_id'] for r in declared['remaining_cases']}
    case_rows=[];segment_rows=[];aligned=[];details=[]
    for dqn in selected:
        case=dqn['case_id'];runs={r['method']:r for r in cache if r['case_id']==case}
        assert set(runs)==set(METHODS)
        reference=runs['full_future_minFED_reference'];cell=(dqn['start_x'],dqn['start_y']);i=graph.index[cell]
        constrained={e:float(v[0,i]) for e,v in values.items()}
        global_min=min(constrained.values())
        assert abs(global_min-dqn['future_minimum_FED'])<1e-10
        chosen_min=constrained[dqn['exit_label']]
        exit_gap=chosen_min-global_min;within_gap=dqn['FED']-chosen_min
        assert exit_gap>=-1e-10 and within_gap>=-1e-10
        delta=dqn['FED']-reference['FED']
        assert abs(delta-exit_gap-within_gap)<1e-10
        run_details=[]
        static={}
        for method in METHODS:
            r=runs[method];static[method]=0.;actual=0.
            for k,step in enumerate(r['executed_steps']):
                source=tuple(step['source']);j=graph.index[source]
                a=300.+k*dt;b=300.+(k+1)*dt
                increment=float(field.integrate(a,b)[j]);constant=float(snapshot[j])*dt
                assert abs(increment-step['FED_increment'])<1e-10
                actual+=increment;static[method]+=constant
                segment_rows.append(dict(case_id=case,method=method,step=k,region=mapping[source],
                    source_x=source[0],source_y=source[1],start_fire_s=a,end_fire_s=b,
                    actual_FED_increment=increment,snapshot300_FED_increment=constant,
                    temporal_change_increment=increment-constant,actual_FED_cumulative=actual))
            assert abs(actual-r['FED'])<1e-10
            run_details.append(dict(method=method,exit=r['exit_label'],steps=r['steps'],FED=r['FED'],
                frozen300_FED=static[method],path_cells=r['path'],path_regions=[mapping[tuple(c)] for c in r['path']]))
        diff_steps=[]
        for k in range(max(dqn['steps'],reference['steps'])):
            a=dqn['executed_steps'][k] if k<dqn['steps'] else None
            b=reference['executed_steps'][k] if k<reference['steps'] else None
            inc_a=a['FED_increment'] if a else 0.;inc_b=b['FED_increment'] if b else 0.
            same=a is not None and b is not None and a['source']==b['source']
            if not same:diff_steps.append(k)
            aligned.append(dict(case_id=case,step=k,start_fire_s=300+k*dt,end_fire_s=300+(k+1)*dt,
                DQN_region=mapping[tuple(a['source'])] if a else 'ALREADY_TERMINAL',
                reference_region=mapping[tuple(b['source'])] if b else 'ALREADY_TERMINAL',
                same_cell_same_time=same,DQN_increment=inc_a,reference_increment=inc_b,
                difference=inc_a-inc_b))
        suffix=None
        if len(dqn['path'])==len(reference['path']):
            suffix=next((k for k in range(len(dqn['path'])) if dqn['path'][k:]==reference['path'][k:]),None)
        snapshot_delta=static['author_DQN_legal_mask']-static['full_future_minFED_reference']
        temporal_residual=delta-snapshot_delta
        row=dict(case_id=case,region=mapping[cell],start_x=cell[0],start_y=cell[1],
            DQN_exit=dqn['exit_label'],reference_exit=reference['exit_label'],
            DQN_steps=dqn['steps'],reference_steps=reference['steps'],
            DQN_FED=dqn['FED'],reference_FED=reference['FED'],absolute_gap=delta,
            reduction_percent=100*delta/dqn['FED'],
            fixed_exit_A_min_FED=constrained['EXIT_A'],fixed_exit_B_min_FED=constrained['EXIT_B'],
            exit_choice_gap=exit_gap,within_chosen_exit_gap=within_gap,
            classification='EXIT_CHOICE' if exit_gap>1e-10 and within_gap<=1e-10 else
                'LOCAL_ROUTE_WITH_SAME_EXIT' if exit_gap<=1e-10 and within_gap>1e-10 else 'MIXED',
            DQN_snapshot300_FED=static['author_DQN_legal_mask'],
            reference_snapshot300_FED=static['full_future_minFED_reference'],
            snapshot_gap_DQN_minus_reference=snapshot_delta,
            temporal_accounting_residual=temporal_residual,
            snapshot_actual_preference_reversal=snapshot_delta*delta < -1e-12,
            shared_suffix_start_step=suffix,
            shared_suffix_start_fire_s=300+suffix*dt if suffix is not None else None,
            shared_suffix_start_region=mapping[tuple(dqn['path'][suffix])] if suffix is not None else None)
        case_rows.append(row)
        details.append(dict(**row,methods=run_details,differing_departure_steps=diff_steps))
    for row in case_rows:
        assert abs(sum(r['difference'] for r in aligned if r['case_id']==row['case_id'])-row['absolute_gap'])<1e-10
    total=sum(r['absolute_gap'] for r in case_rows)
    switches=sum(r['exit_choice_gap'] for r in case_rows)
    locals_=sum(r['within_chosen_exit_gap'] for r in case_rows)
    summary=dict(status='ALL_FIVE_POST_HOC_CASES_EXPLAINED_WITH_CACHE',case_count=5,
        total_gap_across_five=total,exit_choice_gap_sum=switches,local_route_gap_sum=locals_,
        exit_choice_share= switches/total,local_route_share=locals_/total,
        mean_gap_over_all61=total/61,
        count_exit_choice=sum(r['classification']=='EXIT_CHOICE' for r in case_rows),
        count_local_route=sum(r['classification']=='LOCAL_ROUTE_WITH_SAME_EXIT' for r in case_rows),
        case_summaries=case_rows,offline_constrained_value_computations=2,
        new_policy_rollouts=0,new_training_updates=0,new_FDS_runs=0,new_LLM_requests=0,
        final48_accessed=False,
        limitations=['Objective-specific mathematical decomposition, not a causal diagnosis of training',
                    'Frozen300 scoring holds rates constant for cached paths; not a physically simulated alternate fire',
                    'The three elevator cases share R56-R58 representation; they are not independent failure mechanisms'])
    save(OUT/'results.json',summary);save(OUT/'case_details.json',details)
    csv_save(OUT/'case_summary.csv',case_rows);csv_save(OUT/'segment_ledger.csv',segment_rows)
    csv_save(OUT/'aligned_difference_ledger.csv',aligned)
    make_figure(case_rows,segment_rows)
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    save(OUT/'cache_verification.json',dict(protected_prior_files_verified=len(protected),
        case_selection_all_five=True,actual_integrals_match_cache=True,
        fixed_exit_decomposition_matches_global_reference=True,aligned_difference_sums_match=True,
        source_files_changed=False))
    print(json.dumps(summary,ensure_ascii=True),flush=True)


def make_figure(cases,segments):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False}):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
        x=np.arange(len(cases));exit_gap=[r['exit_choice_gap'] for r in cases]
        local=[r['within_chosen_exit_gap'] for r in cases]
        axes[0].bar(x,exit_gap,color='#D55E00',label='出口选择差额')
        axes[0].bar(x,local,bottom=exit_gap,color='#0072B2',hatch='//',label='同出口内路线差额')
        axes[0].set_xticks(x,[r['region'] for r in cases]);axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel('DQN相对最小FED参照的差额');axes[0].set_xlabel('事后选出的全部5个偏差起点')
        axes[0].set_title('差额的目标函数分解，不代表训练原因');axes[0].legend(fontsize=8)
        ax=axes[1];case='t0300_x08_y09'
        for method,color,label in (('author_DQN_legal_mask','#D55E00','DQN既有路线'),
                                    ('full_future_minFED_reference','#0072B2','最小FED既有路线')):
            group=[r for r in segments if r['case_id']==case and r['method']==method]
            times=[300]+[r['end_fire_s'] for r in group]
            actual=[0]+[r['actual_FED_cumulative'] for r in group]
            frozen=np.cumsum([0]+[r['snapshot300_FED_increment'] for r in group])
            ax.plot(times,actual,color=color,linestyle='-',label=label+'：时变计分')
            ax.plot(times,frozen,color=color,linestyle='--',label=label+'：冻结300秒快照')
            ax.plot(times[-1],actual[-1],marker='o',color=color,markersize=4)
        ax.set(xlabel='火灾绝对时间（秒）',ylabel='累计FED',title='R24：初始快照与实际时变计分的偏好反转')
        ax.set_ylim(bottom=0);ax.legend(fontsize=7,loc='upper left')
        for ax in axes:ax.grid(axis='y',alpha=.2)
        fig.suptitle('同一公开火灾的5个缓存案例：全体分析，无新增训练或策略回合',fontsize=11)
        fig.savefig(OUT/'residual_case_analysis.png',dpi=180)
        fig.savefig(OUT/'residual_case_analysis.svg');plt.close(fig)


if __name__=='__main__':main()
