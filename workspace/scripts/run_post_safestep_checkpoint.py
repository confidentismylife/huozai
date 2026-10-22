"""Validate passive recovery, then run new bounded fine-grid attempts sequentially."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil

from prepare_post_safestep_checkpoint import ROOT, OLD, OUT, read, sha, save
sys.path.insert(0, str(ROOT/'src'))
from fire_evac.fds_checkpoint import (
    read_passive_checkpoint, score_mesh_routes, managed_child,
    inspect_native_output, complete_run,
)
from fire_evac.post_safestep_physics import fed_rate_per_s, integrate_rate, read_device_csv


def native(folder, chid):
    result = inspect_native_output(folder/(chid+'_devc.csv'))
    if not result['valid']:
        raise ValueError(result['error'])
    return result['data']


def rates(data):
    ids = [int(re.match(r'N(\d+)_CO$', key)[1]) for key in data if re.match(r'N\d+_CO$', key)]
    return {i: fed_rate_per_s(*(np.clip(data[f'N{i:03d}_{s}'], 0, 1) for s in ('CO', 'CO2', 'O2'))) for i in ids}


def owned_termination(proc, identity, folder, input_file):
    try:
        check = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        proc.wait(timeout=15)
        return
    assert check.create_time() == identity['process_creation_epoch_s']
    assert check.name().lower() == 'fds_openmp.exe' and Path(check.cwd()).resolve() == folder.resolve()
    assert Path(check.cmdline()[-1]).name == input_file.name and sha(input_file) == identity['input_sha256']
    proc.kill()  # Deliberately no .stop: preserve passive chemistry states.
    proc.wait(timeout=15)


def run(job):
    plan = read(OUT/'execution_plan_frozen.json')
    folder = OUT/'runs'/job['name']
    inp = ROOT/job['input']
    assert sha(inp) == job['input_sha256'] and not (folder/'process_start.json').exists()
    assert not any((p.info['name'] or '').lower() == 'fds_openmp.exe' for p in psutil.process_iter(['name']))
    assert sha(Path(plan['solver']['executable'])) == plan['solver']['sha256']
    env = os.environ.copy()
    binary = str(Path(plan['solver']['executable']).parent)
    env['PATH'] = os.pathsep.join([binary+'/mpi', binary, env['PATH']])
    env['I_MPI_ROOT'] = binary+'/mpi'
    env['OMP_NUM_THREADS'] = str(plan['solver']['threads'])
    fine = job['kind'] == 'fine_mesh'
    budget = plan['limits']['fine_case_wall_s' if fine else 'fixture_wall_s']
    baseline = routes = None
    if fine:
        assert read(OUT/'recovery_validation.json')['passed']
        baseline_name = f"{job['layout']}_F2_G150_{job['vent']}_dx0p25_adaptive"
        baseline = native(OLD/'runs'/baseline_name, baseline_name)
        baseline_rates = rates(baseline)
        routes = read(OLD/'diagnostic_routes_frozen.json')[job['layout']]
    began = last_change = last_eval = time.monotonic()
    previous_bytes, peak, last_native, transient_reads = 0, 0, None, 0
    next_log, cause, witness = began+30, None, None
    with ExitStack() as resources:
        stdout = resources.enter_context((folder/'stdout.log').open('xb'))
        stderr = resources.enter_context((folder/'stderr.log').open('xb'))
        proc = resources.enter_context(managed_child([plan['solver']['executable'], inp.name], cwd=folder,
            env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            failure_record=folder/'monitor_failure.json', creationflags=subprocess.CREATE_NO_WINDOW))
        identity = {'pid': proc.pid, 'process_creation_epoch_s': psutil.Process(proc.pid).create_time(),
            'started_UTC': datetime.now(timezone.utc).isoformat(), 'input_sha256': sha(inp),
            'threads': plan['solver']['threads'], 'wall_budget_s': budget, 'native_stop_file_used': False}
        save(folder/'process_start.json', identity)
        print(json.dumps({'event': 'started', 'case': job['name'], **identity}), flush=True)
        while proc.poll() is None:
            now = time.monotonic()
            try:
                memory = psutil.Process(proc.pid).memory_info().rss
                peak = max(peak, memory)
            except psutil.NoSuchProcess:
                break
            size = sum(p.stat().st_size for p in folder.glob('*.csv')) + (folder/'stderr.log').stat().st_size
            if size != previous_bytes:
                previous_bytes, last_change = size, now
            if now-began >= budget:
                cause = 'WALL_TIMEOUT'
            elif memory > plan['limits']['resident_memory_GiB']*1024**3:
                cause = 'MEMORY_LIMIT'
            elif now-last_change >= plan['limits']['no_progress_wall_s']:
                cause = 'NO_PROGRESS_TIMEOUT'
            if now-last_eval >= 5:
                last_eval = now
                if (folder/(job['chid']+'_devc.csv')).exists():
                    try:
                        data = native(folder, job['chid'])
                    except (ValueError, OSError, IndexError, StopIteration):
                        transient_reads += 1  # In-progress writes are not used for decisions.
                    else:
                        last_native = float(data['Time'][-1])
                        if fine:
                            score = score_mesh_routes(data['Time'], rates(data), baseline['Time'], baseline_rates, routes)
                            if score['witness']:
                                snapshot = folder/'witness_native_snapshot.npz'
                                np.savez_compressed(snapshot, **data)
                                witness = {**score['witness'], 'layout': job['layout'], 'case': job['name'],
                                    'native_last_sample_s': last_native, 'threshold': .10,
                                    'status': 'EARLY_REJECT_COMPLETED_ROUTE_WITNESS',
                                    'recorded_UTC': datetime.now(timezone.utc).isoformat(),
                                    'full180s_complete': False, 'physical_model_validated': False,
                                    'parsed_native_snapshot': snapshot.name, 'snapshot_sha256': sha(snapshot),
                                    'snapshot_basis': 'Lossless arrays parsed from complete native CSV before termination; no resampling'}
                                save(folder/'mesh_failure_witness.json', witness)
                                cause = 'MESH_NECESSARY_CONDITION'
            if cause:
                if proc.poll() is None:
                    owned_termination(proc, identity, folder, inp)
                break
            if now >= next_log:
                print(json.dumps({'event': 'progress', 'case': job['name'], 'wall_s': round(now-began, 1),
                    'native_fire_s': last_native, 'rss_MiB': round(memory/1024**2, 1),
                    'passive_checkpoint_files_present': len(list(folder.glob('*.restart')))}), flush=True)
                next_log = now+30
            time.sleep(.25)
        proc.wait(timeout=15)
    logs = (folder/'stderr.log').read_text(errors='replace') + (folder/'stdout.log').read_text(errors='replace')
    errors = [line for line in logs.splitlines() if re.match(r'\s*ERROR(?:\(|:)', line)]
    output = inspect_native_output(folder/(job['chid']+'_devc.csv'))
    data = output['data']
    native_end = float(data['Time'][-1]) if output['valid'] else None
    reported = re.findall(r'Simulation Time:\s*([\d.Ee+-]+)\s*s', logs)
    solver_end = float(reported[-1]) if reported else None
    normal = complete_run(proc.returncode, 'FDS completed successfully' in logs, errors,
        cause, solver_end, native_end, job['end_s'])
    checkpoints = []
    for path in folder.glob('*.restart'):
        try:
            meta = read_passive_checkpoint(path, job['IJK'])
            valid = output['valid'] and meta['Q_nonzero_cells'] > 0 and meta['fire_time_s'] <= native_end+1.6
            checkpoints.append({'file': path.name, 'sha256': sha(path), **meta,
                'complete_passive_checkpoint_candidate': bool(valid)})
        except (ValueError, OSError) as error:
            checkpoints.append({'file': path.name, 'complete_passive_checkpoint_candidate': False, 'error': str(error)})
    result = {**identity, 'case': job['name'], 'normal_completion': bool(normal),
        'returncode': proc.returncode, 'termination_cause': cause, 'FDS_errors': errors,
        'native_output_valid': output['valid'], 'native_output_error': output['error'],
        'solver_last_reported_fire_s': solver_end,
        'native_start_s': float(data['Time'][0]) if output['valid'] else None, 'native_end_s': native_end,
        'native_max_gap_s': float(np.diff(data['Time']).max()) if output['valid'] else None,
        'elapsed_wall_s': time.monotonic()-began,
        'peak_rss_MiB': peak/1024**2, 'transient_live_read_failures': transient_reads,
        'passive_checkpoints': checkpoints, 'witness': witness, 'log_tail': logs[-1800:]}
    save(folder/'run_result.json', result)
    print(json.dumps({'event': 'finished', 'case': job['name'], 'normal': normal,
        'cause': cause, 'native_end_s': result['native_end_s'], 'wall_s': result['elapsed_wall_s'],
        'checkpoint_candidates': sum(c['complete_passive_checkpoint_candidate'] for c in checkpoints)}), flush=True)
    return result


def resumed_job(reference, result):
    source = OUT/'runs'/reference['name']
    good = [c for c in result['passive_checkpoints'] if c['complete_passive_checkpoint_candidate']]
    assert len(good) == 1 and 0 < good[0]['fire_time_s'] < reference['end_s']-1
    checkpoint = source/good[0]['file']
    assert sha(checkpoint) == good[0]['sha256']
    fresh = read_passive_checkpoint(checkpoint, reference['IJK'])
    assert fresh['terminal_scope_verified'] and fresh['Q_nonzero_cells'] > 0
    assert fresh['fire_time_s'] == good[0]['fire_time_s']
    folder = OUT/'runs'/(reference['layout']+'_recovery_resumed_v2')
    folder.mkdir(exist_ok=False)
    copied = {}
    for path in source.iterdir():
        if path.is_file() and path.name.startswith(reference['chid']) and not path.name.endswith('.stop'):
            shutil.copyfile(path, folder/path.name)
            copied[path.name] = sha(path)
    text = (ROOT/reference['input']).read_text(encoding='utf-8')
    assert 'OVERWRITE=.FALSE.' in text
    text = text.replace('OVERWRITE=.FALSE.', 'OVERWRITE=.TRUE.')
    text, count = re.subn(r'(&MISC[^/]+)(/)', r'\1, RESTART=.TRUE. /', text)
    assert count == 1
    inp = folder/(folder.name+'.fds')
    inp.write_text(text, encoding='utf-8', newline='\n')
    job = {**reference, 'name': folder.name, 'kind': 'recovery_resumed',
        'input': inp.relative_to(ROOT).as_posix(), 'input_sha256': sha(inp),
        'checkpoint_time_s': good[0]['fire_time_s']}
    save(folder/'input_manifest.json', job)
    save(folder/'derivation.json', {'parent': source.relative_to(ROOT).as_posix(),
        'copied_solver_files_sha256': copied, 'same_CHID': reference['chid'],
        'input_changes': ['RESTART=.TRUE.', 'OVERWRITE=.TRUE. only in this new copied-output directory'],
        'parent_output_files_modified': False, 'binary_checkpoint_bytes_modified': False})
    return job


def validate_recovery(reference, resumed):
    limits = read(OUT/'execution_plan_frozen.json')['recovery_validation']
    a = native(OUT/'runs'/reference['name'], reference['chid'])
    b = native(OUT/'runs'/resumed['name'], resumed['chid'])
    cp = resumed['checkpoint_time_s']
    pre_a, pre_b = a['Time'] < cp-1e-6, b['Time'] < cp-1e-6
    identical = set(a) == set(b) and all(np.array_equal(a[k][pre_a], b[k][pre_b]) for k in a)
    end = min(float(a['Time'][-1]), float(b['Time'][-1]))
    grid = b['Time'][(b['Time'] > cp) & (b['Time'] <= end)]
    checks = []
    for key in a:
        gas = re.match(r'N\d+_(CO|CO2|O2)$', key)
        temperature = re.match(r'N\d+_T$', key)
        if not (gas or temperature):
            continue
        av, bv = np.interp(grid, a['Time'], a[key]), np.interp(grid, b['Time'], b[key])
        error = float(np.max(np.abs(av-bv)))
        scale = float(np.max(np.abs(av if gas else av-20.)))
        allowed = (limits['gas_absolute_floor_mol_per_mol'] if gas else limits['temperature_absolute_floor_C']) + .01*scale
        checks.append({'channel': key, 'max_absolute_error': error, 'allowed_error': allowed,
            'passed': error <= allowed})
    ra, rb = rates(a), rates(b)
    doses = []
    for index in ra:
        da, db = (integrate_rate(t['Time'], r[index], cp, end) for t, r in ((a, ra), (b, rb)))
        relative = abs(db-da)/da if da >= limits['dose_denominator_floor'] else None
        doses.append({'node': index, 'reference_FED': da, 'resumed_FED': db,
            'absolute_difference': abs(da-db), 'relative_difference': relative,
            'passed': relative is None or relative <= limits['dose_relative_limit']})
    coverage = (abs(end-reference['end_s']) <= 1e-6 and a['Time'][0] <= 0 and b['Time'][0] <= 0
        and max(np.diff(a['Time']).max(), np.diff(b['Time']).max()) <= limits['maximum_native_gap_s'])
    passed = bool(identical and len(grid) and coverage and all(c['passed'] for c in checks+doses))
    result = {'layout': reference['layout'], 'passed': passed, 'checkpoint_s': cp,
        'pre_checkpoint_rows_identical': identical, 'native_coverage_passed': bool(coverage),
        'post_checkpoint_samples': len(grid), 'gas_temperature_checks': checks, 'point_dose_checks': doses,
        'scope': 'Two coarse-grid 100s representative recovery diagnostics; does not validate 180s fine-grid physics'}
    save(OUT/'runs'/resumed['name']/'recovery_comparison.json', result)
    print(json.dumps({'event': 'recovery_checked', 'layout': reference['layout'], 'passed': passed,
        'failed_fields': [c['channel'] for c in checks if not c['passed']],
        'failed_doses': sum(not d['passed'] for d in doses)}), flush=True)
    return result


def fixtures(continue_completed=False):
    comparisons = []
    for job in read(OUT/'planned_jobs.json'):
        if job['kind'] != 'recovery_reference':
            continue
        previous = OUT/'runs'/job['name']/'run_result.json'
        if continue_completed and previous.is_file():
            result = read(previous)
            assert result['normal_completion'] and result['input_sha256'] == job['input_sha256']
            folder = OUT/'runs'/job['name']
            assert sha(ROOT/job['input']) == job['input_sha256']
            logs = (folder/'stderr.log').read_text(errors='replace')
            last = re.findall(r'Simulation Time:\s*([\d.Ee+-]+)\s*s', logs)
            actual = native(folder, job['chid'])
            assert complete_run(result['returncode'], 'FDS completed successfully' in logs,
                result['FDS_errors'], result['termination_cause'], float(last[-1]), float(actual['Time'][-1]), job['end_s'])
            print(json.dumps({'event': 'use_explicitly_completed_reference', 'case': job['name']}), flush=True)
        else:
            result = run(job)
        assert result['normal_completion'], 'Preserve failed reference, no automatic retry'
        recovery = resumed_job(job, result)
        assert run(recovery)['normal_completion'], 'Preserve failed recovery, no automatic retry'
        comparisons.append(validate_recovery(job, recovery))
    summary = {'passed': all(c['passed'] for c in comparisons),
        'layouts': [{'layout': c['layout'], 'passed': c['passed']} for c in comparisons],
        'native_stop_file_used': False, 'all_passive_recovery_sources': True}
    save(OUT/'recovery_validation.json', summary)
    assert summary['passed'], 'Recovery failed; do not relax frozen tolerances or launch fine grids'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['fixtures', 'fine'])
    parser.add_argument('--continue-completed', action='store_true', help='Explicitly reuse completed references; never retry a solver attempt')
    args = parser.parse_args()
    if args.phase == 'fixtures':
        fixtures(args.continue_completed)
    else:
        assert read(OUT/'recovery_validation.json')['passed']
        for job in read(OUT/'planned_jobs.json'):
            if job['kind'] == 'fine_mesh':
                run(job)  # Both planned layouts run, even if the first is rejected.
