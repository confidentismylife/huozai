"""Run exactly one monitored continuous FDS child, then analyze its outputs."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from prepare_safestep_continuous450 import ROOT, OUT, CASE, PREFIX, EXE, sha, save


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp');save(temporary,value);os.replace(temporary,path)


def tail(path, limit=131072):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));return f.read().decode('ascii',errors='replace')


def csv_time(path):
    for line in reversed(tail(path,32768).splitlines()):
        try:return float(line.split(',',1)[0])
        except ValueError:pass
    return None


def live_csv(path):
    raw=path.read_bytes();end=raw.rfind(b'\n')
    assert end>=0
    return pd.read_csv(io.BytesIO(raw[:end+1]),skiprows=1)


def check_prefix(plan):
    old=ROOT/plan['early_prefix_check']['reference'];new=CASE
    cfg=plan['early_prefix_check'];metrics={};passed=True
    old_prefix='hotel_clean120'
    a=live_csv(old/f'{old_prefix}_devc.csv');b=live_csv(new/f'{PREFIX}_devc.csv')
    a=a[a.Time<=cfg['compare_through_s']];b=b[b.Time<=cfg['compare_through_s']]
    merged=a.merge(b,on='Time',suffixes=('_reference','_new'),validate='one_to_one')
    assert len(merged)==len(a) and len(a)>200
    gases={g:[c for c in a.columns if c.endswith('_'+g)] for g in ('CO','CO2','HCN')}
    gas_cols={c for group in gases.values() for c in group}
    temperatures=[c for c in a.columns if c!='Time' and c not in gas_cols]
    for g,cols in gases.items():
        x=merged[[c+'_reference' for c in cols]].to_numpy()*1e6
        y=merged[[c+'_new' for c in cols]].to_numpy()*1e6
        peak=max(float(np.max(x)),cfg['gas_reference_peak_floors_ppm'][g])
        err=float(np.abs(x-y).max());scaled=err/peak
        metrics[g]={'max_abs_ppm':err,'peak_scaled_error':scaled,'passed':scaled<=cfg['gas_peak_scaled_error_max']}
        passed &= metrics[g]['passed']
    delta=merged[[c+'_new' for c in temperatures]].to_numpy()-merged[[c+'_reference' for c in temperatures]].to_numpy()
    max_temp=float(np.max(np.abs(delta)));metrics['temperature_max_abs_C']=max_temp
    passed &= max_temp<=cfg['temperature_max_abs_C']
    ha=live_csv(old/f'{old_prefix}_hrr.csv');hb=live_csv(new/f'{PREFIX}_hrr.csv')
    h=ha[ha.Time<=cfg['compare_through_s']].merge(hb[hb.Time<=cfg['compare_through_s']],on='Time',suffixes=('_reference','_new'),validate='one_to_one')
    assert len(h)==len(ha[ha.Time<=cfg['compare_through_s']])
    max_hrr=float(np.max(np.abs(h.HRR_new-h.HRR_reference)/np.maximum(np.abs(h.HRR_reference),cfg['reference_HRR_floor_kW'])))
    passed &= max_hrr<=cfg['HRR_max_relative_error'];metrics['HRR_max_relative_error']=max_hrr
    exact=np.array_equal(a.to_numpy(),b.to_numpy())
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'passed':bool(passed),'matched_device_rows':len(merged),
       'compare_through_fire_s':cfg['compare_through_s'],'all_prefix_device_values_bitwise_equal_in_CSV':exact,
       'metrics':metrics,'raw_outputs_modified':False,'restart_failure_reclassified':False}
    save(OUT/'continuous_prefix_check.json',result)
    return bool(passed)


def main():
    plan=json.loads((OUT/'frozen_plan.json').read_text('utf-8'));runtime=plan['runtime']
    assert not (CASE/'process_start.json').exists() and not (CASE/f'{PREFIX}.stop').exists()
    assert sha(EXE)==plan['FDS_sha256'] and sha(CASE/f'{PREFIX}.fds')==plan['new_input_sha256']
    env=os.environ.copy();bindir=str(EXE.parent)
    env['PATH']=os.pathsep.join([bindir+'/mpi',bindir,env['PATH']]);env['I_MPI_ROOT']=bindir+'/mpi'
    env['OMP_NUM_THREADS']=str(runtime['OMP_NUM_THREADS'])
    began=time.monotonic();last_advance=began;highest=-1.;next_record=began;prefix_done=False
    reason=None;exception=None;proc=None
    try:
        with (CASE/'stdout.log').open('wb') as stdout,(CASE/'stderr.log').open('wb') as stderr:
            proc=subprocess.Popen([str(EXE),f'{PREFIX}.fds'],cwd=CASE,env=env,stdin=subprocess.DEVNULL,
                stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            start={'started_at':datetime.now(timezone.utc).isoformat(),'pid':proc.pid,'runner_pid':os.getpid(),
                'command':[str(EXE),f'{PREFIX}.fds'],'OMP_NUM_THREADS':runtime['OMP_NUM_THREADS'],
                'hard_wall_limit_s':runtime['hard_wall_limit_s'],'restart_loaded':False,'input_sha256':sha(CASE/f'{PREFIX}.fds')}
            save(CASE/'process_start.json',start);print(json.dumps({'event':'started',**start}),flush=True)
            while proc.poll() is None:
                now=time.monotonic();native=csv_time(CASE/f'{PREFIX}_devc.csv')
                log=tail(CASE/'stderr.log')
                matches=re.findall(r'Simulation Time:\s*([\d.Ee+\-]+)\s*s',log)
                logged=float(matches[-1]) if matches else None
                current=max(x for x in [native,logged,-1] if x is not None)
                if current>highest+1e-9:highest=current;last_advance=now
                if now-began>runtime['hard_wall_limit_s']:reason='hard_wall_limit'
                elif now-last_advance>runtime['no_simulation_progress_limit_s']:reason='no_simulation_progress_limit'
                elif re.search(r'(?m)^\s*ERROR(?:\(|:)',log):reason='FDS_error_detected'
                if native is not None and native>=plan['early_prefix_check']['check_after_native_time_s'] and not prefix_done:
                    prefix_done=True
                    if not check_prefix(plan):reason='unchanged_input_prefix_consistency_failed'
                if reason:
                    proc.kill();proc.wait(timeout=30);break
                if now>=next_record:
                    memory=psutil.virtual_memory()
                    record={'recorded_at':datetime.now(timezone.utc).isoformat(),'status':'RUNNING',
                        'elapsed_wall_s':round(now-began,2),'native_device_fire_s':native,'last_logged_fire_s':logged,
                        'target_fire_s':450.,'last_progress_age_s':round(now-last_advance,2),
                        'available_RAM_GiB':memory.available/2**30,'process_alive':True,'prefix_check_completed':prefix_done}
                    atomic_json(OUT/'latest_status.json',record)
                    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True);next_record=now+runtime['progress_record_interval_s']
                time.sleep(runtime['monitor_interval_s'])
            proc.wait(timeout=30)
    except Exception:
        exception=traceback.format_exc();reason='runner_exception'
        if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=30)
    text=(CASE/'stderr.log').read_text('utf-8',errors='replace') if (CASE/'stderr.log').exists() else ''
    errors=[line.strip() for line in text.splitlines() if re.match(r'\s*ERROR(?:\(|:)',line)]
    final_time=csv_time(CASE/f'{PREFIX}_devc.csv');returncode=proc.returncode if proc is not None else None
    passed=returncode==0 and 'STOP: FDS completed successfully' in text and not errors and reason is None and final_time is not None and abs(final_time-450)<1e-6
    if passed and not prefix_done:
        prefix_done=True
        if not check_prefix(plan):passed=False;reason='final_prefix_consistency_failed'
    result={'finished_at':datetime.now(timezone.utc).isoformat(),'passed':bool(passed),
        'elapsed_wall_s':time.monotonic()-began,'returncode':returncode,'termination_reason':reason,
        'runner_exception':exception,'FDS_errors':errors,'normal_completion_marker':'STOP: FDS completed successfully' in text,
        'final_native_fire_s':final_time,'prefix_checked':prefix_done,'continuous_process_count':1 if proc is not None else 0,
        'restart_loaded':False,'stop_file_created':False,'input_sha256_unchanged':sha(CASE/f'{PREFIX}.fds')==plan['new_input_sha256']}
    save(CASE/'run_result.json',result)
    atomic_json(OUT/'latest_status.json',{'status':'SIMULATION_COMPLETE_ANALYSIS_PENDING' if passed else 'SIMULATION_INCOMPLETE',**result})
    print(json.dumps({'event':'solver_finished',**result}),flush=True)
    analysis=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_safestep_continuous450.py')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    if analysis.returncode:
        atomic_json(OUT/'latest_status.json',{'status':'ANALYSIS_FAILED_RAW_OUTPUTS_PRESERVED','analysis_returncode':analysis.returncode,'solver_passed':passed})
    return analysis.returncode


if __name__=='__main__':sys.exit(main())
