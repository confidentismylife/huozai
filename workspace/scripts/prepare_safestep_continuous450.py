"""Freeze a new continuous-only fire experiment after restart diagnostics.

Prior restart failures retain their status. User authorized a separate strategy
using an uninterrupted native FDS process, not a relaxation of restart gates.
"""
import ast
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

from prepare_safestep_pressure_zone_fix import sha, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/safestep_continuous450_20261017'
PREFIX = 'hotel_continuous450'
CASE = OUT/'run'
OLD = ROOT/'outputs/safestep_clean_hotel120_20261017'
EXE = Path('C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe')
AUTHOR = ROOT/'outputs/safestep_paper_numerics_20261015/author_repository'


def main():
    OUT.mkdir(exist_ok=False)
    source_plan = ROOT/'outputs/safestep_pressure_zone_fix_20261017/frozen_plan.json'
    protected = json.loads(source_plan.read_text('utf-8'))['protected_previous_sha256']
    for folder in ['safestep_pressure_zone_fix_20261017']:
        m=ROOT/'outputs'/folder/'completion_manifest.json'
        protected.update(json.loads(m.read_text('utf-8'))['artifact_sha256'])
        protected[m.relative_to(ROOT).as_posix()]=sha(m)
    protected[source_plan.relative_to(ROOT).as_posix()]=sha(source_plan)
    bad=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not bad,bad
    original_path=OLD/'runs/hotel_clean120/hotel_clean120.fds'
    original=original_path.read_text('utf-8')
    assert original.count("CHID='hotel_clean120'")==1 and original.count('&TIME T_END=120.0 /')==1
    text=original.replace("CHID='hotel_clean120'",f"CHID='{PREFIX}'").replace('&TIME T_END=120.0 /','&TIME T_END=450.0 /')
    assert text.replace(f"CHID='{PREFIX}'","CHID='hotel_clean120'").replace('&TIME T_END=450.0 /','&TIME T_END=120.0 /')==original
    assert 'RESTART=.TRUE.' not in text and 'DIAGNOSTIC_MARKER' not in text
    CASE.mkdir();inp=CASE/f'{PREFIX}.fds';inp.write_text(text,'utf-8')
    (OUT/'input_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
        fromfile=original_path.name,tofile=inp.name)),'utf-8')
    results_path=ROOT/'outputs/safestep_paper_numerics_20261015/numeric_run/results.json'
    result=json.loads(results_path.read_text('utf-8'))
    parsed=ast.parse((AUTHOR/'coordinates2areanumber.py').read_text('utf-8-sig'))
    mapping=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='mapping' for t in n.targets))
    routes=[];dt=2/.3
    for item in result['fixed_routes']:
        path=item['path'];segments=[]
        assert len(path)-1==item['steps']
        for k,cell in enumerate(path[:-1]):
            region=mapping[tuple(cell)];assert region.startswith('R')
            segments.append({'cell':cell,'region':region,'start_fire_s':300+k*dt,'end_fire_s':300+(k+1)*dt})
        assert mapping[tuple(path[-1])].startswith('EXIT_') and segments[-1]['end_fire_s']<450
        routes.append({'id':item['id'],'path':path,'steps':item['steps'],'exit':mapping[tuple(path[-1])],
           'travel_time_s':item['steps']*dt,'segments':segments,
           'archived_author_compatibility_FED':item['FED'],
           'reference_value_use':'Display only; never used to fit gas parameters or select route'})
    save(OUT/'frozen_routes.json',{'source':results_path.relative_to(ROOT).as_posix(),
        'source_sha256':sha(results_path),'walking_speed_m_s':.3,'grid_step_m':2.,'evacuation_start_fire_s':300.,
        'primary_integration':'Departure-cell residence over each movement interval; no pre-300s exposure; exposure stops on arrival at exit',
        'routes':routes,'author_mapping':[{ 'cell':list(k),'region':v} for k,v in mapping.items()]})
    parameters_path=ROOT/'outputs/safestep_physics_time_contract_20261016/exploratory_human_reference_NOT_AUTHOR_ORIGINAL.json'
    parameters=json.loads(parameters_path.read_text('utf-8'))
    parameters['use_for_candidate_continuous450_route_FED']=True
    parameters['scope_note']='New authorized candidate reconstruction only; original 120s pilot remains ineligible for evacuation FED. Parameters are NOT author-original verified.'
    save(OUT/'frozen_candidate_FER_parameters.json',parameters)
    shutil.copyfile(ROOT/'outputs/safestep_hotel_fds_candidate_20261016/region_mapping_candidate.csv',OUT/'region_mapping_candidate.csv')
    plan={
       'frozen_at':datetime.now(timezone.utc).isoformat(),
       'purpose':'One continuous native FDS 0..450s candidate trajectory and four fixed-route exposure calculations',
       'strategy_change':'User authorized continuous-only alternative after restart diagnostics; prior failed restart screens remain failed and are not reclassified.',
       'FDS_executable':str(EXE),'FDS_version':'FDS-6.11.1-0-gff928db-release','FDS_sha256':sha(EXE),
       'parent_input':original_path.relative_to(ROOT).as_posix(),'parent_input_sha256':sha(original_path),
       'new_input_sha256':sha(inp),'input_changes':{'CHID':PREFIX,'T_END_s':450.},
       'physical_parameters_and_geometry_changed':False,'source_solver_modified':False,
       'time_s':{'start':0.,'end':450.,'evacuation_start':300.,'longest_frozen_route_end':max(r['segments'][-1]['end_fire_s'] for r in routes)},
       'runtime':{'hard_wall_limit_s':10800,'no_simulation_progress_limit_s':900,'monitor_interval_s':10,
          'progress_record_interval_s':30,'OMP_NUM_THREADS':2,'MPI_processes':1,'stop_file_creation_permitted':False,
          'restart_loading_permitted':False,'automatic_retry_permitted':False,
          'timeout_action':'Terminate only this child process; preserve partial outputs, classify incomplete, no .stop/restart'},
       'early_prefix_check':{'reference':(OLD/'runs/hotel_clean120').relative_to(ROOT).as_posix(),
          'check_after_native_time_s':121.,'compare_through_s':119.,
          'temperature_max_abs_C':2.,'gas_peak_scaled_error_max':.05,
          'gas_reference_peak_floors_ppm':{'CO':1.,'HCN':.2,'CO2':20.},
          'HRR_max_relative_error':.05,'reference_HRR_floor_kW':1.,
          'purpose':'Execution consistency of unchanged initial physical inputs, not independent physical validation',
          'failure_action':'Stop this continuous candidate and report failure; no tuning or in-place rerun'},
       'analysis':{'require_normal_termination_at450_for_full_stage_success':True,
          'native_gap_max_s':1.,'native_gas_regions':58,'gas_devices':174,'units':'wet volume fraction mol/mol',
          'FER_rate_basis':'Formula per minute converted explicitly to per second; compute at native nodes',
          'primary_route_FED':'Exact piecewise-linear rate integral over frozen residence segments',
          'author_compatibility_FED':'Separate legacy 5s floor-sampled, fixed-step calculation; never substitutes primary integration',
          'author_hazard_reference':'Same primary integration and frozen segments using archived rates; descriptive cross-input comparison only',
          'no_extrapolation':True,'no_raw_smoothing_or_spike_deletion':True,'no_parameter_fit_to_author_results':True,
          'no_success_threshold_on_route_ranking_or_FED_agreement':True,
          'candidate_confounds':['inferred geometry and region XY mapping','transferred HCN yield','fuel properties assumptions',
              'gas-temperature window proxy','post-326s constant-peak fire assumption','unverified author V/D and gas-to-FER implementation']},
       'resource_scope':{'planned_solver_runs':1,'training_updates':0,'LLM_requests':0,'final48_scenarios_accessed':False,
          '900s_runs':0,'solver_patch_builds':0,'Docker_or_WSL_updates':0},
       'protected_previous_sha256':protected,
    }
    save(OUT/'frozen_plan.json',plan)
    save(OUT/'preflight.json',{'previous_files_verified':len(protected),'mismatches':bad,
        'new_input_sha256':sha(inp),'solver_sha256':sha(EXE),'route_count':len(routes),
        'longest_route_covered':True,'disk_free_GiB':shutil.disk_usage(OUT).free/2**30,'ready':True})
    print(json.dumps({'status':'FROZEN_READY','protected_files':len(protected),'input_only_changes':plan['input_changes'],
         'max_wall_h':3,'route_count':len(routes)}),flush=True)


if __name__=='__main__':main()
