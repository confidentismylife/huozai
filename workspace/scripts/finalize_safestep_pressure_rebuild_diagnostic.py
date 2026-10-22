"""Audit and freeze the completed native diagnostic, without solver reruns."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from prepare_safestep_pressure_zone_fix import ROOT, OUT, sha, save, COMMIT


def main():
    assert not (OUT/'completion_manifest.json').exists()
    plan = json.loads((OUT/'frozen_plan.json').read_text('utf-8'))
    result = json.loads((OUT/'native_diagnostic_results.json').read_text('utf-8'))
    native_plan = json.loads((OUT/'native_diagnostic_plan.json').read_text('utf-8'))
    runfiles = sorted((OUT/'native_runs').glob('*/run_result.json'))
    assert len(runfiles)==native_plan['solver_runs']==result['solver_runs']==8
    for p in runfiles:
        r=json.loads(p.read_text('utf-8'))
        assert r['passed'] and r['returncode']==0 and not r['FDS_errors'] and not r['timed_out']
        assert r['expected_marker_found'] and r['stop_injection'] is None
    assert not list((OUT/'native_runs').rglob('*.stop'))
    protected = plan['protected_previous_sha256']
    bad = [p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not bad
    direct_effects = {}
    for geom in ('dynamic','static'):
        a=pd.read_csv(OUT/'native_runs'/f'{geom}_marker_fixed_continuous'/f'{geom}_marker_fixed_devc.csv',skiprows=1)
        b=pd.read_csv(OUT/'native_runs'/f'{geom}_marker_removed_continuous'/f'{geom}_marker_removed_devc.csv',skiprows=1)
        assert np.array_equal(a.Time.to_numpy(), b.Time.to_numpy())
        delta=np.abs(a.iloc[:,1:].to_numpy()-b.iloc[:,1:].to_numpy())
        direct_effects[geom] = {'whole_continuous_max_T_change_C':float(delta[:,:2].max()),
            'whole_continuous_max_CO2_change_ppm':float(delta[:,2:4].max()*1e6),
            'whole_continuous_max_U_change_m_s':float(delta[:,4].max()),
            'interpretation':'Descriptive intervention-size check, not geometry equivalence'}
    save(OUT/'continuous_intervention_effect.json',direct_effects)
    fields=['max_T_abs_C','max_CO2_abs_ppm','max_U_abs_m_s','max_final_pressure_abs_Pa']
    tab=[]
    for name,r in result['results'].items():
        tab.append({'case':name,**{k:r[k] for k in fields}, 'checkpoint_s':r['checkpoint_s'],
            'marker_geometry_remove_s':r['marker_remove_times_s'][0] if r['marker_remove_times_s'] else None})
    pd.DataFrame(tab).to_csv(OUT/'paired_comparison.csv',index=False,encoding='utf-8-sig')
    base=result['results']['dynamic_marker_fixed']; intervention=result['results']['dynamic_marker_removed']
    reductions={k:1-intervention[k]/base[k] for k in fields}
    save(OUT/'dynamic_reduction_fractions.json',reductions)
    local=OUT/'candidate_patch_local_source/baseline'
    patterns={
        'identity_connectivity_initialization':('main.f90','ALLOCATE(CONNECTED_ZONES'),
        'restart_skips_initial_obstruction_update':('main.f90','IF (.NOT.RESTART) CALL CREATE_OR_REMOVE_OBSTRUCTIONS'),
        'divergence_calls_merge_only_for_obstruction_event':('divg.f90','IF (N_ZONE>0 .AND. OBST_CREATED_OR_REMOVED) CALL MERGE_PRESSURE_ZONES'),
        'obstruction_update_sets_rebuild_flag':('init.f90','OBST_CREATED_OR_REMOVED = .TRUE.'),
        'rebuild_global_transitive_connections':('main.f90','CONNECTED_ZONES = MATMUL(CONNECTED_ZONES,CONNECTED_ZONES)'),
    }
    evidence={}
    for key,(file,pattern) in patterns.items():
        hits=[i+1 for i,line in enumerate((local/file).read_text('utf-8').splitlines()) if pattern in line]
        assert hits
        evidence[key]={'file':(local/file).relative_to(ROOT).as_posix(),'lines':hits,
            'official_url':f'https://github.com/firemodels/fds/blob/{COMMIT}/Source/{file}#L{hits[0]}'}
    dump=(local/'dump.f90').read_text('utf-8')
    for sub in ['DUMP_RESTART','READ_RESTART']:
        match=re.search(r'(?m)^SUBROUTINE '+sub+r'\b.*?^END SUBROUTINE '+sub+r'\b',dump,re.S|re.M)
        assert match and 'CONNECTED_ZONES' not in match.group()
    evidence['connection_matrix_not_explicitly_serialized_in_pinned_restart_subroutines']=True
    evidence['source_inference_boundary']='Runtime connectivity matrix was not instrumented directly. Source and intervention support the mechanism; source patch has not executed.'
    save(OUT/'source_mechanism_evidence.json',evidence)
    threshold=native_plan['screen']
    absolute_passes={
        'temperature':intervention['max_T_abs_C']<=threshold['temperature_max_abs_C'],
        'CO2':intervention['max_CO2_abs_ppm']<=threshold['CO2_max_abs_ppm'],
        'velocity':intervention['max_U_abs_m_s']<=threshold['interface_velocity_max_abs_m_s'],
        'final_pressure':intervention['max_final_pressure_abs_Pa']<=threshold['zone_final_pressure_max_abs_Pa']}
    report=ROOT/'reports/SafeStep压力分区续算机制干预实验_2026-10-17.md'
    assert report.exists()
    links=re.findall(r'\]\(([^)]+)\)',report.read_text('utf-8'))
    local_links=[]
    for target in links:
        if target.startswith(('http://','https://','#')):continue
        path=Path(target.split('#')[0]);assert path.is_absolute() and path.exists(),target
        local_links.append(target)
    verification={
        'verified_at':datetime.now(timezone.utc).isoformat(),
        'status':'COMPLETE_DIAGNOSTIC_WITH_UNCOMPILED_CANDIDATE_PATCH',
        'native_solver_runs_passed':len(runfiles),'native_solver_wall_s':result['total_solver_wall_s'],
        'protected_previous_files_verified':len(protected),'mismatches':bad,
        'strict_precheckpoint_exact_rows_each_pair':{k:v['prefix_rows_exact'] for k,v in result['results'].items()},
        'checkpoint_boundary_missing_rows_reported':True,
        'appended_geometry_events_verified_independently_of_copied_future_events':True,
        'raw_outputs_edited':False,'checkpoint_bytes_edited':False,'stop_files_created':0,
        'thresholds_changed_after_results':False,'dynamic_absolute_screens':absolute_passes,
        'dynamic_all_absolute_screens_pass':all(absolute_passes.values()),
        'dynamic_T_and_final_pressure_error_reductions_exceed_90_percent':all(reductions[k]>=.9 for k in ['max_T_abs_C','max_final_pressure_abs_Pa']),
        'local_report_links_verified':len(local_links),
        'candidate_patch_compiled':False,'candidate_patch_executed':False,
        'solver_fix_validated':False,'hotel_restart_validated':False,'new_hotel_runs':0,
        'installed_FDS_path':'C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe',
        'installed_FDS_sha256_at_completion':sha('C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe'),
        'system_update_retried_after_Docker_WSL_failure':False,
        '300_or900s_runs':0,'route_FED_computed':False,'training_updates':0,'LLM_requests':0,'sealed_final48_access':False,
    }
    save(OUT/'final_verification.json',verification)
    artifacts=[p for p in OUT.rglob('*') if p.is_file()]
    artifacts += [report,ROOT/'scripts/prepare_safestep_pressure_zone_fix.py',
         ROOT/'scripts/run_safestep_pressure_rebuild_diagnostic.py',Path(__file__).resolve()]
    manifest={'status':verification['status'],'created_at':datetime.now(timezone.utc).isoformat(),
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(artifacts)},
        'protected_previous_files_verified':len(protected),'native_solver_runs_passed':8,
        'native_solver_wall_s':result['total_solver_wall_s'],
        'source_patch_validated':False,'hotel_restart_validated':False,
        '300_or900s_runs':0,'training_updates':0,'LLM_requests':0,'sealed_final48_access':False}
    save(OUT/'completion_manifest.json',manifest)
    print(json.dumps({k:v for k,v in verification.items() if 'sha256' not in k}),flush=True)
    print(json.dumps({'frozen_artifact_count':len(manifest['artifact_sha256'])}),flush=True)


if __name__=='__main__':main()
