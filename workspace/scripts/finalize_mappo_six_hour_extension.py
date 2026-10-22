"""Wait for genuine jobs, use remaining time for paired noise precision, verify all artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import run_mappo_six_hour_extension as runner
from fire_evac.campaign_extension import admission_allowed,write_json
from fire_evac.mappo import FactorizedMAPPO

OUT=runner.OUT
SOURCE=runner.SOURCE


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def wait_experiments():
    while True:
        state=read(OUT/'extension_status.json')
        if state['status'] in ('experiments_complete','complete'):return
        if state['status']=='failed':raise RuntimeError('Extension failed: '+str(state.get('error')))
        launch=read(OUT/'launch.json')
        try:
            process=psutil.Process(int(launch['process_id']))
            live=process.is_running() and 'run_mappo_six_hour_extension.py' in ' '.join(process.cmdline())
        except (psutil.NoSuchProcess,psutil.AccessDenied):live=False
        if not live:raise RuntimeError('Extension process missing before verified experiment completion')
        write_json(OUT/'finalizer_status.json',{'status':'waiting','pid':os.getpid(),
            'verified_extension_pid':int(launch['process_id']),
            'phase':state.get('phase'),'verified_at_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(30)


def precision_until_deadline():
    deadline=datetime.fromisoformat(read(OUT/'extension_manifest.json')['deadline_utc']).timestamp()
    directories=[]
    for root in (SOURCE/'confirmation',OUT/'additional_primary'):
        for directory in sorted(root.glob('seed_*')):
            if all((directory/arm/'summary.json').exists() for arm in runner.base.ARMS):
                directories.append(directory)
    location=OUT/'precision'
    location.mkdir(parents=True,exist_ok=True)
    manifest_path=location/'manifest.json'
    if not manifest_path.exists():
        write_json(manifest_path,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'stopping_rule':'calendar deadline only; complete entire paired population blocks; reserve 210s reporting',
            'scenario_scope':'the same twelve development validation scenarios',
            'population_seed_start':44001,'policy_roots':[str(p) for p in directories],
            'purpose':'reduce population realization noise without inflating independent training-seed count',
            'API_requests':0,'final_test_scenarios_loaded':False})
    models=[]
    for directory in directories:
        seed=int(directory.name.split('_')[-1])
        for arm in runner.base.ARMS:
            checkpoint=directory/arm/'best.pt'
            model,_=FactorizedMAPPO.load(checkpoint)
            models.append((seed,arm,model))
    scenarios=runner.base.make_domain_randomized_trust_bank(24,20261021,'DSG')[12:]
    completed={int(p.stem.split('_')[-1]) for p in location.glob('population_*.csv')}
    population=max(completed,default=44000)+1
    durations=[]
    while admission_allowed(time.time(),deadline,max(durations[-3:] or [45.0])*1.2,210):
        started=time.monotonic();rows=[]
        for seed,arm,model in models:
            env=runner.ORIGINAL_FACTORY(arm,scenarios)
            values=runner.base.evaluate(model,env,scenarios,(population,),arm)
            env.close()
            for row in values:row['training_seed']=seed
            rows.extend(values)
        runner.base.write_csv(location/f'population_{population}.csv',rows)
        durations.append(time.monotonic()-started)
        completed.add(population)
        write_json(OUT/'finalizer_status.json',{'status':'precision_evaluation','pid':os.getpid(),
            'completed_population_blocks':len(completed),'last_population_seed':population,
            'block_seconds':durations[-1],'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat()})
        population+=1
    write_json(location/'complete.json',{'population_blocks':len(completed),
        'population_seeds':sorted(completed),'training_seeds':sorted({s for s,_,_ in models}),
        'stop_reason':'remaining calendar time reserved for verified reports',
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})


def validate_job(path,arm,seed,timesteps,expected_scenarios):
    summary=read(path/'summary.json')
    assert summary['status']=='complete',(path,'incomplete')
    assert summary['arm']==arm and int(summary['seed'])==seed,(path,'wrong identity')
    assert summary['timesteps']==timesteps and summary['training']['timesteps']==timesteps,(path,'wrong budget')
    assert summary['final_test_scenarios_loaded'] is False,(path,'test scope')
    assert summary['deepseek_api_requests']==0,(path,'API during training')
    config=summary['config']
    assert config['route_entropy_coefficient']==.01,(path,'route entropy')
    if 'e003_i002' in path.parts or 'e003_i005' in path.parts:
        expected=.02 if 'e003_i002' in path.parts else .05
        assert config['flow_entropy_coefficient']==.03 and config['plan_imitation_coefficient']==expected
    elif 'shared_raw' in path.parts or 'shared_masked' in path.parts:
        assert config['flow_entropy_coefficient']==.03 and config['plan_imitation_coefficient']==0
    if 'mechanism' in path.parts:
        assert summary.get('episode_seed_schedule')=='derived-explicit-uint32-v1',(path,'uncontrolled training RNG')
        assert read(path/'training_protocol.json')['episode_seed_schedule']=='derived-explicit-uint32-v1'
    frame=pd.read_csv(path/'episode_results.csv')
    assert len(frame)==24 and set(frame.scenario_id)==expected_scenarios,(path,'validation coverage')
    assert set(frame.population_seed)=={42001,42002},(path,'population coverage')
    assert not frame.duplicated(['scenario_id','population_seed']).any(),(path,'duplicate episodes')
    assert frame.incapacitation_rate.between(0,1).all(),(path,'invalid endpoint')
    model,metadata=FactorizedMAPPO.load(path/'best.pt')
    assert metadata['timesteps']==summary['selected_checkpoint']['timesteps'],(path,'checkpoint mismatch')
    assert model.observation_dim==(34 if arm==runner.RAW else 50),(path,'network shape')
    checkpoints=read(path/'checkpoints.json')
    assert len(checkpoints)>=5 and checkpoints[-1]['timesteps']==timesteps,(path,'checkpoint history')
    assert all('route_entropy' in c['training'] and 'flow_entropy' in c['training'] and
        'flow_action_fractions_nonwait' in c['training'] for c in checkpoints),(path,'missing diagnostics')
    scenarios=runner.base.make_domain_randomized_trust_bank(24,20261021,'DSG')[12:]
    if 'mechanism' in path.parts:
        env=runner.factory(runner.CACHED,scenarios,'masked') if arm=='mappo_50_masked_control' else runner.factory(arm,scenarios)
    else:
        env=runner.ORIGINAL_FACTORY(arm,scenarios)
    repeated=pd.DataFrame(runner.base.evaluate(model,env,scenarios,(42001,42002),arm))
    env.close()
    keys=['scenario_id','population_seed']
    old=frame.set_index(keys).sort_index();new=repeated.set_index(keys).sort_index()
    columns=['incapacitation_rate','mean_cumulative_fed','cumulative_congestion',
             'evacuation_time_seconds','wait_fraction','mean_release_fraction_nonwait','full_release_episode']
    np.testing.assert_allclose(old[columns].to_numpy(),new[columns].to_numpy(),rtol=1e-10,atol=1e-12)
    replay_error=float(np.max(np.abs(old[columns].to_numpy()-new[columns].to_numpy())))
    return {'path':str(path),'arm':arm,'seed':seed,'timesteps':timesteps,
        'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
        'evaluation_replay_rows':len(repeated),'evaluation_replay_max_abs_error':replay_error}


def verify_artifacts():
    scenarios=runner.base.make_domain_randomized_trust_bank(24,20261021,'DSG')
    ids={s.scenario_id for s in scenarios[12:]}
    original=read(SOURCE/'campaign_status.json')
    assert original['status']=='complete' and original['deepseek_api_requests']==0
    assert original['final_test_scenarios_loaded'] is False
    assert read(OUT/'full_training_reproducibility_result.json')['passed'] is True
    jobs=[]
    for cid,_,_ in runner.base.SCREEN_CONFIGS:
        for seed in runner.base.SCREEN_SEEDS:
            jobs.append(validate_job(SOURCE/'screen'/cid/f'seed_{seed}',runner.CACHED,seed,100000,ids))
    screen=[]
    for cid,_,_ in runner.base.SCREEN_CONFIGS:
        for seed in runner.base.SCREEN_SEEDS:
            row=read(SOURCE/'screen'/cid/f'seed_{seed}'/'summary.json')
            row['config_id']=cid;screen.append(row)
    chosen,_=runner.base.select_screen_config(screen)
    assert chosen==read(SOURCE/'selection.json')['selected'],'selection differs from frozen rule'
    primary_seeds=[]
    for root in (SOURCE/'confirmation',OUT/'additional_primary'):
        for path in sorted(root.glob('seed_*')):
            seed=int(path.name.split('_')[-1]);primary_seeds.append(seed)
            for arm in runner.base.ARMS:
                jobs.append(validate_job(path/arm,arm,seed,150000,ids))
    assert len(primary_seeds)>=5
    gate=read(OUT/'plan_bank'/'gate.json')
    arms=(runner.RULE,runner.CACHED,runner.EXPANDED) if gate['passed'] else (runner.RULE,runner.CACHED)
    for seed in runner.SEEDS:
        for shared,arm in [('shared_raw',runner.RAW),('shared_masked','mappo_50_masked_control')]:
            jobs.append(validate_job(OUT/'mechanism'/shared/f'seed_{seed}'/arm,arm,seed,150000,ids))
        for cid,_,_ in runner.CONFIGS:
            for arm in arms:
                jobs.append(validate_job(OUT/'mechanism'/cid/f'seed_{seed}'/arm,arm,seed,150000,ids))
    audit_counts=[]
    for cid,_,_ in runner.CONFIGS:
        for seed in runner.SEEDS:
            for arm in (runner.CACHED,runner.EXPANDED) if gate['passed'] else (runner.CACHED,):
                location=OUT/'mechanism_audit'/cid/f'seed_{seed}'/arm
                frame=pd.read_csv(location/'episode_results.csv')
                trace=pd.read_csv(location/'step_results.csv')
                assert len(frame)==96 and set(frame.provider_variant)=={'native','rule','shuffled','masked'}
                assert set(frame.scenario_id)==ids
                assert len(trace)==int((frame.evacuation_time_seconds/5).sum())
                assert set(trace.actual_joint_action)<={-1,0,1,2,3,4,5,6,7,8}
                expected=set(zip(frame.provider_variant,frame.scenario_id,frame.population_seed))
                observed=set(trace[['provider_variant','scenario_id','population_seed']].itertuples(index=False,name=None))
                assert expected==observed
                audit_counts.append({'path':str(location),'episodes':len(frame),'steps':len(trace)})
    original_audit=pd.read_csv(SOURCE/'channel_audit.csv')
    assert set(original_audit.provider_variant)=={'native','rule','shuffled','masked'}
    assert len(original_audit)==96*len(list((SOURCE/'confirmation').glob('seed_*')))
    latency=pd.read_csv(OUT/'online_latency_pilot'/'episode_results.csv')
    assert len(latency)==48 and latency.provider_variant.nunique()==8
    online_paths=list((OUT/'online_pilot').glob('*/complete.json'))
    assert len(online_paths)==(4 if gate['passed'] else 3)
    for path in online_paths:
        frame=pd.read_csv(path.parent/'episode_results.csv')
        assert len(frame)==30 and frame.provider_variant.nunique()==5
    ledger=read(OUT/'api'/'budget.json')
    assert ledger['actual_cost_upper_cny']+ledger['reserved_cny']<=2.0+1e-10
    assert ledger['attempted_requests']<=240
    requests=[json.loads(line) for line in (OUT/'api'/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(requests)==ledger['attempted_requests']
    for path in (OUT/'precision').glob('population_*.csv'):
        frame=pd.read_csv(path)
        assert len(frame)==len(primary_seeds)*3*12,(path,'incomplete precision block')
        assert set(frame.training_seed)==set(primary_seeds)
        assert set(frame.arm)==set(runner.base.ARMS)
        assert set(frame.scenario_id)==ids
        assert not frame.duplicated(['training_seed','arm','scenario_id']).any()
    secret_pattern=re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9]{24,96}(?![A-Za-z0-9])')
    examined=0
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix in {'.json','.jsonl','.log','.md','.csv'}:
            if secret_pattern.search(path.read_text(encoding='utf-8-sig',errors='replace')):
                raise RuntimeError('Credential-like content found in '+str(path))
            examined+=1
    return {'passed':True,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'unique_training_jobs':len(jobs),'primary_training_seeds':primary_seeds,
        'verified_jobs':jobs,'mechanism_audits':audit_counts,
        'API_budget_verified':ledger,'credential_scan_files':examined,
        'original_selection_unchanged':read(SOURCE/'selection.json')['selected']['config_id']=='e001_i000',
        'allowed_input_scope':'source reviewed: only the existing 24 generated development scenarios and training snapshots',
        'final_test_scenarios_loaded':False}


def command(script,*args):
    name=Path(script).stem
    log=OUT/(name+'_finalizer.log')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    with log.open('w',encoding='utf-8') as handle:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,
                       stdout=handle,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-wait',action='store_true')
    parser.add_argument('--skip-precision',action='store_true');args=parser.parse_args()
    if not args.no_wait:wait_experiments()
    if not args.skip_precision:precision_until_deadline()
    audit=verify_artifacts()
    write_json(OUT/'completion_audit.json',audit)
    command('analyze_mappo_six_hour_extension.py')
    command('plot_mappo_six_hour_results.py')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    with (OUT/'test_suite_final.log').open('w',encoding='utf-8') as handle:
        subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env=env,
                       stdout=handle,stderr=subprocess.STDOUT,check=True)
    report=OUT/'analysis'/'六小时MAPPO与DeepSeek开发实验完整报告.md'
    target=ROOT/'reports'/'MAPPO_DeepSeek六小时开发实验报告_20261022.md'
    target.write_text(report.read_text(encoding='utf-8'),encoding='utf-8')
    started=datetime.fromisoformat(read(SOURCE/'campaign_status.json')['started_at_utc'])
    completed=datetime.now(timezone.utc)
    result={'status':'complete','completed_at_utc':completed.isoformat(),
        'elapsed_hours_from_original_start':(completed-started).total_seconds()/3600,
        'report':str(target),'completion_audit':str(OUT/'completion_audit.json'),
        'API_budget':read(OUT/'api'/'budget.json'),'final_test_scenarios_loaded':False}
    write_json(OUT/'verified_completion.json',result)
    write_json(OUT/'finalizer_status.json',result)
    state=read(OUT/'extension_status.json');state.update(result)
    write_json(OUT/'extension_status.json',state)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        write_json(OUT/'finalizer_status.json',{'status':'failed','error_type':type(exc).__name__,
            'error':str(exc)[:500],'at_utc':datetime.now(timezone.utc).isoformat()})
        raise
