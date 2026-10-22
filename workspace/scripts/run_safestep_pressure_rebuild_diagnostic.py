"""Native-FDS mechanism intervention after an unavailable isolated build.

The intervention changes geometry by one cell. It is a diagnostic, NOT a solver
patch, and NOT a physically equivalent replacement for the hotel input.
"""
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_safestep_pressure_zone_fix import ROOT, OUT, OLD, COMMIT, sha, save
from run_safestep_restart_microfixture import run
from analyze_safestep_restart_hrr_audit import checkpoint_metadata, read_csv
from analyze_safestep_fds_smokecheck import parse_smv


def prepare_patch():
    """Use already pinned local source; keep network failure evidence separate."""
    target = OUT/'candidate_patch_local_source'; target.mkdir(exist_ok=False)
    original = ROOT/'outputs/safestep_restart_hrr_audit_20261017/materials'
    provenance = []
    for source in list(original.glob('*.f90')) + [OLD/'pressure_zone_source/divg.f90']:
        dest = target/'baseline'/source.name; dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        provenance.append({'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source),
             'official_url': f'https://github.com/firemodels/fds/blob/{COMMIT}/Source/{source.name}'})
    source = target/'baseline/main.f90'; before = source.read_text('utf-8')
    anchor = '! Compute divergence just in case the flow field is not initialized to ambient\n\nCALL INITIALIZE_DIVERGENCE_INTEGRALS'
    replacement = ('! Compute divergence just in case the flow field is not initialized to ambient\n\n'
        '! Reconstruct pressure-zone connectivity from restored wall/cell state.\n'
        '! CONNECTED_ZONES is initialized to identity and is not in the restart file.\n'
        'IF (RESTART .AND. N_ZONE>0) OBST_CREATED_OR_REMOVED = .TRUE.\n\n'
        'CALL INITIALIZE_DIVERGENCE_INTEGRALS')
    assert before.count(anchor) == 1
    after = before.replace(anchor, replacement)
    (target/'patched').mkdir(); (target/'patched/main.f90').write_text(after, 'utf-8', newline='\n')
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
           fromfile='a/Source/main.f90', tofile='b/Source/main.f90'))
    (OUT/'restart_pressure_connectivity_candidate.patch').write_text(diff, 'utf-8')
    save(OUT/'candidate_patch_provenance.json', {'source_commit': COMMIT, 'source_files': provenance,
        'baseline_main_sha256': sha(source), 'patched_main_sha256': sha(target/'patched/main.f90'),
        'diff': diff, 'compiled': False, 'executed': False, 'restart_format_changed': False})


def main():
    assert (OUT/'frozen_plan.json').exists()
    if not (OUT/'candidate_patch_provenance.json').exists():
        prepare_patch()
    limitation = {
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'native_compilers': 'No ifx/ifort/gfortran found in PATH or inspected runtime directories',
        'Docker_start_attempt': 'Docker Desktop tried WSL update; screenshot reports elevated update failed: operating system denied access to specified file',
        'Docker_engine': 'Unavailable: dockerDesktopLinuxEngine pipe does not exist',
        'action': 'Discontinued Docker/WSL build route; no further update or elevation attempted',
        'full_source_fetch': 'Failed with TLS UNEXPECTED_EOF; no full source build completed',
        'candidate_patch': 'Prepared from nine previously retrieved pinned source files; UNCOMPILED AND UNVALIDATED',
        'installed_FDS_modified': False,
    }
    if not (OUT/'build_environment_limitation.json').exists():
        save(OUT/'build_environment_limitation.json', limitation)
    plan = {
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'reason_for_plan_adjustment': 'Native compiler absent; Docker/WSL build unavailable. Test mechanism using installed FDS without claiming solver repair.',
        'question': 'Does a small scheduled geometry event that forces pressure connectivity recomputation reduce dynamic-zone restart drift?',
        'solver_runs': 8,
        'cases': ['dynamic_marker_fixed', 'dynamic_marker_removed', 'static_marker_fixed', 'static_marker_removed'],
        'marker': {'XB': [0, .2, 0, .2, 0, .2], 'volume_m3': .008, 'cells': 1,
                   'remove_setpoint_s': 4.24, 'fixed_control_setpoint_s': 100.0},
        'restart': 'Same native periodic checkpoint per continuous/restart pair; same model input except MISC RESTART=T; no byte edits',
        'screen': json.loads((OUT/'frozen_plan.json').read_text('utf-8'))['micro_engineering_screens'],
        'claim_boundary': 'Geometry intervention changes model. Positive outcome supports mechanism only; does not validate candidate source patch or hotel restart.',
        'hotel_followup_this_adjusted_stage': False,
        'source_patch_to_installed_executable': False,
        '300_or900s_runs': 0, 'training_updates': 0, 'LLM_requests': 0, 'sealed_final48_access': False,
    }
    if (OUT/'native_diagnostic_plan.json').exists():
        plan = json.loads((OUT/'native_diagnostic_plan.json').read_text('utf-8'))
    else:
        save(OUT/'native_diagnostic_plan.json', plan)
    base = OUT/'native_runs'; base.mkdir(exist_ok=True)
    results = {}; curves = {}; wall = 0.
    for geometry, action in [(g, a) for g in ('dynamic','static') for a in ('fixed','removed')]:
        name = f'{geometry}_marker_{action}'
        old_prefix = 'sealed_dynamic_door' if geometry == 'dynamic' else 'sealed_static_open'
        old_input = OLD/'sealed_zone_diagnostic'/f'{old_prefix}_continuous'/f'{old_prefix}_continuous.fds'
        text = old_input.read_text('utf-8').replace(f"CHID='{old_prefix}'", f"CHID='{name}'")
        at = 4.24 if action == 'removed' else 100.
        marker = (f"&DEVC ID='REBUILD_TRIGGER', QUANTITY='TIME', XYZ=0.4,0.4,0.4, SETPOINT={at}, INITIAL_STATE=.TRUE., LATCH=.TRUE. /\n"
                  "&OBST ID='DIAGNOSTIC_MARKER', XB=0.0,0.2,0.0,0.2,0.0,0.2, SURF_ID='INERT', DEVC_ID='REBUILD_TRIGGER' /\n")
        text = text.replace('&TAIL /', marker+'&TAIL /')
        parent = base/f'{name}_continuous'; parent.mkdir(exist_ok=True)
        parent_input = parent/f'{parent.name}.fds'
        if parent_input.exists():
            assert parent_input.read_text('utf-8') == text
        else:
            parent_input.write_text(text, 'utf-8')
        if (parent/'run_result.json').exists():
            assert json.loads((parent/'run_result.json').read_text('utf-8'))['passed']
        else:
            run(parent, name)
        metas = [checkpoint_metadata(parent/f'{name}_{i}.restart', (10,8,10), .02) for i in (1,2)]
        cp = metas[0]['fire_time_s']; assert all(abs(m['fire_time_s']-4.2)<1e-8 for m in metas)
        resumed = base/f'{name}_resumed'
        if not resumed.exists():
            resumed.mkdir(); copied = {}
            for p in parent.iterdir():
                if p.is_file() and p.name.startswith(name):
                    shutil.copyfile(p, resumed/p.name); copied[p.name] = sha(p)
            (resumed/f'{resumed.name}.fds').write_text(text.replace('&TAIL /','&MISC RESTART=.TRUE. /\n&TAIL /'), 'utf-8')
            save(resumed/'derivation.json', {'parent': parent.relative_to(ROOT).as_posix(), 'copied_sha256': copied,
                 'input_change': 'MISC RESTART=T only', 'checkpoint_bytes_modified': False})
        if (resumed/'run_result.json').exists():
            assert json.loads((resumed/'run_result.json').read_text('utf-8'))['passed']
        else:
            run(resumed, name)
        checkpoint_unchanged = all(sha(resumed/f'{name}_{i}.restart') == metas[i-1]['sha256'] for i in (1,2))
        assert checkpoint_unchanged
        a = read_csv(parent/f'{name}_devc.csv'); b = read_csv(resumed/f'{name}_devc.csv')
        retained = a[a.Time<cp-1e-8]
        assert np.array_equal(retained.to_numpy(), b[b.Time<cp-1e-8].to_numpy())
        missing_prefix = np.setdiff1d(a[a.Time<=cp+1e-6].Time, b[b.Time<=cp+1e-6].Time)
        assert len(missing_prefix)<=1 and np.all(np.abs(missing_prefix-cp)<1e-6)
        grid = np.arange(4.22, 6.500001, .02)
        assert grid.min()>=max(a.Time.min(),b.Time.min()) and grid.max()<=min(a.Time.max(),b.Time.max())+1e-9
        channels = ['LEFT_T','RIGHT_T','LEFT_CO2','RIGHT_CO2','INTERFACE_U']
        ref = np.column_stack([np.interp(grid,a.Time,a[c]) for c in channels])
        res = np.column_stack([np.interp(grid,b.Time,b[c]) for c in channels])
        error = np.abs(ref-res)
        metrics = {c: {'max_abs_difference': float(error[:,i].max()),
                     'final_abs_difference': float(error[-1,i])} for i,c in enumerate(channels)}
        ha = read_csv(parent/f'{name}_hrr.csv'); hb = read_csv(resumed/f'{name}_hrr.csv')
        zones = [c for c in ha if c.startswith('ZONE_')]
        pressure = {z: {'continuous_final_Pa': float(ha.iloc[-1][z]), 'resumed_final_Pa': float(hb.iloc[-1][z]),
                    'final_abs_difference_Pa': float(abs(ha.iloc[-1][z]-hb.iloc[-1][z]))} for z in zones}
        geom_events = {}
        for kind, d in [('continuous', parent), ('resumed',resumed)]:
            _, events = parse_smv(d/f'{name}.smv')
            geom_events[kind] = [e for e in events if e['ID'] in ('DOOR','DIAGNOSTIC_MARKER')]
            wall += json.loads((d/'run_result.json').read_text('utf-8'))['elapsed_s']
        marker_times = [e['geometry_event_time_s'] for e in geom_events['continuous'] if e['ID']=='DIAGNOSTIC_MARKER' and e['geometry_event_time_s']>0]
        if action == 'removed':
            assert marker_times and all(cp<t<4.4 for t in marker_times)
        else:
            assert not marker_times
        # Full completed-parent SMV was copied, so its future events remain in
        # the prefix. Verify newly appended events independently; do not treat
        # copied future entries as evidence that restart executed the action.
        parent_events = geom_events['continuous']
        assert geom_events['resumed'][:len(parent_events)] == parent_events
        appended_events = geom_events['resumed'][len(parent_events):]
        expected_appended = [e for e in parent_events if e['geometry_event_time_s']>cp]
        assert appended_events == expected_appended
        curves[name] = {'grid_s': grid.tolist(), 'channels':channels, 'reference':ref.tolist(), 'resumed':res.tolist(),
            'pressure_native_continuous': ha[['Time',*zones]].to_dict('records'),
            'pressure_native_resumed': hb[['Time',*zones]].to_dict('records')}
        results[name] = {'checkpoint_s': cp, 'checkpoint_metadata': metas, 'prefix_rows_exact': len(retained),
            'rounded_checkpoint_boundary_rows_removed_by_native_restart': missing_prefix.tolist(),
            'checkpoints_unchanged': checkpoint_unchanged, 'device_metrics': metrics, 'pressure_metrics':pressure,
            'geometry_events':geom_events, 'marker_remove_times_s': marker_times,
            'newly_appended_restart_geometry_events': appended_events,
            'copied_future_geometry_entries_not_counted_as_restart_evidence': True,
            'max_T_abs_C': float(error[:,:2].max()), 'max_CO2_abs_ppm':float(error[:,2:4].max()*1e6),
            'max_U_abs_m_s':float(error[:,4].max()), 'max_final_pressure_abs_Pa':max(v['final_abs_difference_Pa'] for v in pressure.values())}
        save(OUT/'native_partial_results.json', {'completed_pairs': len(results), 'results':results})
        print(json.dumps({'case': name, **{k:results[name][k] for k in ['max_T_abs_C','max_CO2_abs_ppm','max_U_abs_m_s','max_final_pressure_abs_Pa']}}), flush=True)
    thresholds = plan['screen']; screens={}
    for geometry in ('dynamic','static'):
        fixed=results[f'{geometry}_marker_fixed']; removed=results[f'{geometry}_marker_removed']
        screens[geometry] = {
            'T_error_reduction_fraction': 1-removed['max_T_abs_C']/fixed['max_T_abs_C'],
            'final_pressure_error_reduction_fraction': 1-removed['max_final_pressure_abs_Pa']/fixed['max_final_pressure_abs_Pa'],
            'temperature_screen': removed['max_T_abs_C'] <= thresholds['temperature_max_abs_C'],
            'CO2_screen':removed['max_CO2_abs_ppm']<=thresholds['CO2_max_abs_ppm'],
            'velocity_screen':removed['max_U_abs_m_s']<=thresholds['interface_velocity_max_abs_m_s'],
            'final_pressure_screen':removed['max_final_pressure_abs_Pa']<=thresholds['zone_final_pressure_max_abs_Pa'],
        }
    save(OUT/'native_curve_data.json', curves)
    summary = {'status':'EIGHT_NATIVE_DIAGNOSTIC_RUNS_COMPLETED_NOT_SOLVER_FIX',
        'completed_at':datetime.now(timezone.utc).isoformat(), 'results':results, 'screens':screens,
        'solver_runs':8,'total_solver_wall_s':wall,'candidate_patch_compiled':False,
        'solver_fix_validated':False, 'hotel_restart_validated':False,
        'new_hotel_runs':0,'300_or900s_runs':0,'route_FED_computed':False,
        'training_updates':0,'LLM_requests':0,'sealed_final48_access':False}
    save(OUT/'native_diagnostic_results.json', summary)
    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,geom in enumerate(('dynamic','static')):
        for action, color in [('fixed','#b33b36'),('removed','#176b48')]:
            name=f'{geom}_marker_{action}'; c=curves[name]; g=np.array(c['grid_s'])
            delta=np.abs(np.array(c['reference'])-np.array(c['resumed']))
            axes[row,0].plot(g,delta[:,:2].max(axis=1),label=f'marker {action}',color=color)
            for kind,style in [('continuous','-'),('resumed','--')]:
                h=pd.DataFrame(c[f'pressure_native_{kind}']);h=h[h.Time>4.2]
                gap=h['ZONE_1']-h['ZONE_2'] if 'ZONE_2' in h else h['ZONE_1']
                axes[row,1].plot(h.Time,gap,label=f'{action}, {kind}',color=color,linestyle=style)
        axes[row,0].set(title=f'{geom}: paired temperature drift',xlabel='Fire time (s)',ylabel='Max probe difference (C)')
        axes[row,1].set(title=f'{geom}: '+('zone pressure difference' if geom=='dynamic' else 'zone pressure'),xlabel='Fire time (s)',ylabel='Pa')
        event_time = results[f'{geom}_marker_removed']['marker_remove_times_s'][0]
        for ax in axes[row]: ax.grid(alpha=.2);ax.legend(fontsize=8);ax.axvline(event_time,color='gray',alpha=.5,linestyle=':')
    fig.suptitle('Diagnostic geometry intervention, not an equivalent solver repair')
    fig.savefig(OUT/'native_pressure_rebuild_diagnostic.png',dpi=180);plt.close(fig)
    print(json.dumps({'status':summary['status'],'screens':screens,'solver_wall_s':wall}),flush=True)


if __name__ == '__main__':
    main()
