"""Verify immutable trajectory cache and recover JSON/report finalization.

Initial execution completed all trajectories but could not serialize a NumPy
integer counter. Original runner and its pre-score manifest stay unchanged.
No trajectory is replaced or selected during this recovery.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from fire_evac.safestep_author_numerics import load_author_inputs
from fire_evac.safestep_public_benchmark import Graph,RateField,forward_min_fed_reference,future_min_fed_policy
from run_safestep_public_benchmark import CONFIG,REPORT,sha,save,csv_save,legacy_score,make_report


def read_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def numeric_rows(p,strings):
    output=[]
    for r in read_rows(p):
        output.append({k:v if k in strings else None if v=='' else
                       (float(v) if any(c in v.lower() for c in '.e') else int(v)) for k,v in r.items()})
    return output


def main():
    clock=time.perf_counter()
    cfg=json.loads(CONFIG.read_text('utf-8'));out=ROOT/cfg['output_directory']
    assert (out/'rollouts.json').is_file() and not (out/'completion_manifest.json').exists()
    initial={p.relative_to(ROOT).as_posix():sha(p) for p in out.iterdir() if p.is_file()}
    plan=json.loads((out/'frozen_plan.json').read_text('utf-8'))
    for group in ('input_sha256','protected_prior_sha256'):
        for rel,h in plan[group].items():assert sha(ROOT/rel)==h,rel
    source=ROOT/cfg['source_directory'];inputs=load_author_inputs(source,300)
    starts=[tuple(c) for c in inputs['empty']];graph=Graph(starts+list(inputs['exits']),inputs['exits'])
    times=inputs['rates']['Time'];field=RateField(times,np.column_stack([
        inputs['rates'][inputs['mapping'][c]] if c not in graph.exits else np.zeros(len(times)) for c in graph.cells]))
    dt=cfg['movement']['grid_step_m']/cfg['movement']['speed_m_per_s'];horizon=cfg['movement']['maximum_actions']
    rollouts=json.loads((out/'rollouts.json').read_text('utf-8'))
    summaries=numeric_rows(out/'departure_summary.csv',{'method'})
    pairwise=numeric_rows(out/'paired_comparisons.csv',{'method_a','method_b'})
    paper=numeric_rows(out/'paper_start_comparison.csv',{'start_label','case_id','method','reached','exit_label','evaluation_group'})
    assert len(rollouts)==2196 and len({(r['case_id'],r['method']) for r in rollouts})==2196
    caches={};reference_checks=[]
    for t in cfg['departure_fire_times_s']:
        costs=field.interval_costs(t,dt,horizon)
        _,values,_=future_min_fed_policy(graph,costs)
        caches[t]=(costs,values)
        for start in ((5,1),(10,9),(1,1),(5,13)):
            i=graph.index[start];forward,steps=forward_min_fed_reference(graph,costs,i)
            assert abs(forward-values[0,i])<1e-10
            reference_checks.append(dict(departure_fire_s=t,start=start,forward_FED=forward,
                backward_FED=float(values[0,i]),forward_steps=int(steps)))
    old=json.loads((ROOT/'outputs/safestep_paper_numerics_20261015/numeric_run/results.json').read_text('utf-8'))
    old_paths={tuple(r['start']):r['path'] for r in old['all_start_diagnostic']}
    q=read_rows(out/'stored_policy_q_diagnostic.csv')
    q_by_cell={(int(r['cell_x']),int(r['cell_y'])):r for r in q}
    stored_matches=0;mask_changes=0;executed_steps=0
    for r in rollouts:
        path=r['path'];costs,values=caches[r['departure_fire_s']]
        assert len(path)==r['steps']+1 and len(r['executed_steps'])==r['steps']
        assert tuple(path[0])==(r['start_x'],r['start_y'])
        assert r['steps']<=horizon
        total=0.
        for k,s in enumerate(r['executed_steps']):
            i=graph.index[tuple(path[k])];j=graph.index[tuple(path[k+1])]
            assert graph.destination(i,s['action'])==j
            assert s['source']==path[k] and s['destination']==path[k+1]
            expected=float(costs[k,i]);assert abs(expected-s['FED_increment'])<1e-10
            total+=expected;assert abs(total-s['FED_cumulative'])<1e-10
            if r['method']=='author_DQN_legal_mask':mask_changes+=int(q_by_cell[tuple(path[k])]['changed']=='True')
        executed_steps+=r['steps'];assert abs(total-r['FED'])<1e-10
        assert r['reached']==(tuple(path[-1]) in graph.exits)
        assert abs(legacy_score(path,inputs,r['departure_fire_s'],dt)-r['legacy_FED'])<1e-10
        minimum=float(values[0,graph.index[tuple(path[0])]])
        assert abs(minimum-r['future_minimum_FED'])<1e-10
        if r['reached']:assert r['FED']>=minimum-1e-10
        if r['method']=='full_future_minFED_reference':assert r['reached'] and abs(r['FED']-minimum)<1e-10
        if r['departure_fire_s']==300 and r['method']=='author_DQN_legal_mask':
            assert path==old_paths[tuple(path[0])];stored_matches+=1
    library=json.loads((ROOT/'outputs/safestep_continuous450_20261017/frozen_routes.json').read_text('utf-8'))
    old_scores=json.loads((ROOT/'outputs/safestep_continuous450_20261017/analysis_results.json').read_text('utf-8'))
    expected={r['id']:r['archived_same_primary_FED'] for r in old_scores['candidate_routes']}
    preflight=[]
    for route in library['routes']:
        primary=sum(float(field.integrate(s['start_fire_s'],s['end_fire_s'])[graph.index[tuple(s['cell'])]]) for s in route['segments'])
        legacy=legacy_score(route['path'],inputs,300,dt)
        assert abs(primary-expected[route['id']])<1e-10
        assert abs(legacy-route['archived_author_compatibility_FED'])<1e-10
        preflight.append(dict(id=route['id'],primary_FED=primary,legacy_FED=legacy))
    metadata=dict(case_count=549,rollouts=len(rollouts),executed_steps=executed_steps,
                  independent_reference_checks=len(reference_checks),stored_300_paths_match=stored_matches,
                  raw_DQN_action_mask_changes=int(mask_changes),all_rollouts_reached=sum(r['reached'] for r in rollouts),
                  successful_rollout_reference_bound_checks=sum(r['reached'] for r in rollouts),
                  failed_rollouts=[{k:v for k,v in r.items() if k not in ('path','executed_steps')} for r in rollouts if not r['reached']],
                  known_route_preflight=preflight,reference_checks=reference_checks,
                  solver_runs=0,training_updates=0,LLM_requests=0,final48_accessed=False,
                  independent_fire_count=1,stored_checkpoint_count=1,
                  algorithm_unit_checks_passed=6,
                  observed_test_command="python -c import sys,pytest; sys.path.insert(0,'src'); pytest.main(['tests/test_safestep_public_benchmark.py','-q'])",
                  recovery='Reconstructed verification from unchanged cached trajectories; no trajectory rerun or replacement',
                  recovery_wall_s=time.perf_counter()-clock)
    save(out/'initial_postprocess_issue.json',dict(kind='JSON_NUMPY_INT64_SERIALIZATION',
        failed_stage='verification.json serialization after all 2196 trajectories, CSV tables and figures were written',
        source='Observed runner exception, manually recorded summary; not a raw stderr file',
        affected_counter='raw_DQN_action_mask_changes',raw_trajectories_changed=False,
        original_runner_kept_unchanged=True,recovery_script=Path(__file__).relative_to(ROOT).as_posix()))
    save(out/'verification.json',metadata)
    make_report(out,cfg,summaries,pairwise,paper,metadata)
    with REPORT.open('a',encoding='utf-8') as f:
        f.write('\n归档说明：初次运行完成全部轨迹后，汇总JSON遇到NumPy整数序列化错误。独立归档脚本从已有轨迹逐步复核并补齐报告，保留原冻结脚本、失败说明和全部原始输出，未替换轨迹。\n')
    for rel,h in initial.items():assert sha(ROOT/rel)==h,rel
    for group in ('input_sha256','protected_prior_sha256'):
        for rel,h in plan[group].items():assert sha(ROOT/rel)==h,rel
    links=re.findall(r'\]\((D:/[^)]+)\)',REPORT.read_text('utf-8'))
    assert all(Path(p).is_file() for p in links)
    artifacts=[p for p in out.iterdir() if p.is_file()]+[REPORT,Path(__file__),
        ROOT/'scripts/run_safestep_public_benchmark.py',CONFIG,ROOT/'src/fire_evac/safestep_public_benchmark.py',
        ROOT/'tests/test_safestep_public_benchmark.py']
    save(out/'completion_manifest.json',dict(status='COMPLETE_PUBLIC_HAZARD_DESCRIPTIVE_BENCHMARK',
         protected_previous_files_verified=len(plan['protected_prior_sha256']),
         original_cached_outputs_verified=len(initial),input_files_unchanged=len(plan['input_sha256']),
         local_report_links_verified=len(links),
         artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts}))
    print(json.dumps(dict(status='COMPLETE',case_count=metadata['case_count'],rollouts=metadata['rollouts'],
          executed_steps=executed_steps,reached=metadata['all_rollouts_reached'],
          primary=[r for r in summaries if r['departure_fire_s']==300],
          protected_files_verified=len(plan['protected_prior_sha256'])),ensure_ascii=True))


if __name__=='__main__':main()
