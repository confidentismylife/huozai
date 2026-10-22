"""Isolated paired-observation diagnostic; never launches policy evaluation."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/post_safestep_physics_20261019'
PARENT = ROOT/'outputs/post_safestep_checkpoint_20261019'
OUT = ROOT/'outputs/post_safestep_sampling_20261019'
sys.path.insert(0, str(ROOT/'src'))
from fire_evac.fds_checkpoint import managed_child, inspect_native_output, complete_run
from fire_evac.fds_sampling_diagnostic import route_breakdown, gas_first_dose
from fire_evac.post_safestep_physics import static_mixture_fixture, read_device_csv


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def native(folder, chid):
    output = inspect_native_output(folder/(chid+'_devc.csv'))
    if not output['valid']:
        raise ValueError(output['error'])
    return output['data']


def extra_probes(text, nodes):
    lines = []
    for node in sorted(set(nodes)):
        for species in ('CO', 'CO2', 'O2', 'T'):
            name = f'N{node:03d}_{species}'
            found = [line for line in text.splitlines() if f"ID='{name}'" in line]
            assert len(found) == 1, name
            line = found[0].replace(f"ID='{name}'", f"ID='I{node:03d}_{species}'")
            assert line.rstrip().endswith('/') and 'SPATIAL_STATISTIC' not in line
            lines.append(line.rstrip()[:-1]+", SPATIAL_STATISTIC='INTERPOLATION' /")
    assert text.count('&TAIL /') == 1
    return text.replace('&TAIL /', '\n'.join(lines)+'\n&TAIL /')


def prepare():
    assert not OUT.exists(), 'Preserve existing diagnostic attempts'
    parent_manifest = read(PARENT/'completion_manifest.json')
    protected = {**read(PARENT/'protected_parent_hashes.json'),
        **parent_manifest['artifact_sha256'],
        (PARENT/'completion_manifest.json').relative_to(ROOT).as_posix(): sha(PARENT/'completion_manifest.json')}
    for rel, expected in protected.items():
        assert sha(ROOT/rel) == expected, rel
    OUT.mkdir()
    (OUT/'runs').mkdir()
    save(OUT/'protected_parent_hashes.json', protected)
    parent_plan = read(PARENT/'execution_plan_frozen.json')
    witnesses = read(PARENT/'mesh_results.json')['early_witnesses_replayed']
    jobs = []
    for w in witnesses:
        lid = w['layout']
        vent = 'V0' if lid == 'L1' else 'V1'
        for dx in ('0p25', '0p125'):
            parent_name = f'{lid}_F2_G150_{vent}_dx{dx}_'+('adaptive' if dx == '0p25' else 'checkpoint')
            parent_dir = (OLD if dx == '0p25' else PARENT)/'runs'/parent_name
            original = (parent_dir/(parent_name+'.fds')).read_text(encoding='utf-8')
            chid = f'{lid}_dx{dx}_dual_probe'
            text = original.replace(f"CHID='{parent_name}'", f"CHID='{chid}'")
            text, count = re.subn(r'\bT_END=180\b', 'T_END=70', text)
            assert count == 1
            text = re.sub(r'\bDT_RESTART=[\d.]+', 'DT_RESTART=30.', text)
            if 'DT_FLUSH=' not in text:
                text = text.replace('DT_RESTART=30.', 'DT_RESTART=30., DT_FLUSH=1.')
            # Both operators read the same field, with existing devices unchanged.
            text = extra_probes(text, w['route']['route'][:-1])
            folder = OUT/'runs'/chid
            folder.mkdir()
            inp = folder/(chid+'.fds')
            inp.write_text(text, encoding='utf-8', newline='\n')
            jobs.append({'name': chid, 'kind': 'paired_probe', 'layout': lid, 'dx_m': float(dx.replace('p', '.')),
                'input': inp.relative_to(ROOT).as_posix(), 'input_sha256': sha(inp),
                'parent': parent_dir.relative_to(ROOT).as_posix(), 'parent_chid': parent_name,
                'parent_input_sha256': sha(parent_dir/(parent_name+'.fds')), 'end_s': 70.,
                'route': w['route'], 'wall_budget_s': 10800.})
    uniform = static_mixture_fixture('uniform_probe', .2)
    add = []
    for name in ('CO', 'CO2', 'O2'):
        line = next(x for x in uniform.splitlines() if f"ID='{name}'" in x)
        add.append(line.replace(f"ID='{name}'", f"ID='I_{name}'").rstrip()[:-1]+", SPATIAL_STATISTIC='INTERPOLATION' /")
    uniform = uniform.replace('&TAIL /', '\n'.join(add)+'\n&TAIL /')
    step = """&HEAD CHID='temperature_step', TITLE='Analytic t0 observation-operator check' /
&MESH IJK=10,10,10, XB=0,1,0,1,0,1 /
&TIME T_END=0.2, DT=0.01 /
&MISC TMPA=20., HUMIDITY=0., OVERWRITE=.FALSE. /
&DUMP DT_DEVC=0.1, SIG_FIGS=12 /
&INIT XB=0.5,1,0,1,0,1, TEMPERATURE=40. /
&DEVC ID='N_T', XYZ=0.5,0.5,0.5, QUANTITY='TEMPERATURE', TIME_AVERAGED=.FALSE. /
&DEVC ID='I_T', XYZ=0.5,0.5,0.5, QUANTITY='TEMPERATURE', TIME_AVERAGED=.FALSE., SPATIAL_STATISTIC='INTERPOLATION' /
&TAIL /
"""
    fixtures = []
    for name, text in [('uniform_probe', uniform), ('temperature_step', step)]:
        folder = OUT/'runs'/name
        folder.mkdir()
        inp = folder/(name+'.fds')
        inp.write_text(text, encoding='utf-8', newline='\n')
        fixtures.append({'name': name, 'kind': 'fixture', 'input': inp.relative_to(ROOT).as_posix(),
            'input_sha256': sha(inp), 'end_s': .2, 'wall_budget_s': 300.})
    save(OUT/'jobs_frozen.json', fixtures+jobs)
    save(OUT/'protocol_frozen.json', {'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'spec': 'docs/2026-10-19-fds-sampling-diagnostic.md', 'solver': parent_plan['solver'],
        'selection': 'Post-hoc two previous failure witnesses, exploratory only',
        'changes': ['CHID', 'T_END=70', 'DT_RESTART=30', 'DT_FLUSH=1', 'append passive interpolation probes'],
        'limits': {'RSS_GiB': 4, 'no_progress_wall_s': 600, 'one_solver': True},
        'default_prefix_parity': {'gas_absolute_mol_per_mol': 1e-9, 'temperature_absolute_C': 1e-5, 'native_time_absolute_s': 1e-8},
        'temperature_step_t0_expected_C': {'default': 40, 'interpolation': 30, 'absolute_tolerance': 1e-7},
        'uniform_probe_max_absolute_difference_mol_per_mol': 1e-10,
        'old_mesh_failure_unchanged': True, 'formal24_started': False, '900s_started': False,
        'new_RL_updates': 0, 'LLM_requests': 0, 'sealed_final48_accessed': False,
        'implementation_sha256': {p: sha(ROOT/p) for p in ['scripts/run_post_safestep_sampling.py',
            'src/fire_evac/fds_sampling_diagnostic.py', 'tests/test_fds_sampling_diagnostic.py',
            'docs/2026-10-19-fds-sampling-diagnostic.md']}})
    print(json.dumps({'prepared_jobs': len(jobs)+len(fixtures), 'protected_files': len(protected)}))


def cached():
    records = []
    for w in read(PARENT/'mesh_results.json')['early_witnesses_replayed']:
        lid = w['layout']
        graph = read(OLD/(lid+'_graph.json'))['graph']['coordinates_xy']
        pair = {}
        for dx, basis in [('0.25', 'candidate'), ('0.125', 'fine')]:
            vent = 'V0' if lid == 'L1' else 'V1'
            parent_name = f'{lid}_F2_G150_{vent}_dx'+('0p25_adaptive' if basis == 'candidate' else '0p125_checkpoint')
            folder = (OLD if basis == 'candidate' else PARENT)/'runs'/parent_name
            data = native(folder, parent_name)
            parts = route_breakdown(data, w['route'])
            expected = w['candidate_dx0p25_FED' if basis == 'candidate' else 'reference_dx0p125_FED']
            assert np.isclose(parts['FED'], expected, rtol=1e-12, atol=1e-16)
            for s in parts['segments']:
                s['xy_m'] = graph[s['node']]
            q1 = gas_first_dose(data, w['route'], max_step_s=.05)
            q2 = gas_first_dose(data, w['route'], max_step_s=.025)
            pair[basis] = {'dx_m': float(dx), **parts, 'gas_first_FED_h0p05': q1,
                'gas_first_FED_h0p025': q2, 'quadrature_halving_absolute_difference': abs(q1-q2),
                'gas_first_vs_frozen_rate_first_absolute_difference': abs(q2-parts['FED']),
                'gas_first_vs_frozen_rate_first_relative_difference': abs(q2-parts['FED'])/parts['FED']}
        deltas = []
        for a, b in zip(pair['candidate']['segments'], pair['fine']['segments']):
            assert a['node'] == b['node']
            deltas.append({'node': a['node'], 'xy_m': a['xy_m'], 'candidate_minus_fine_FED': a['FED']-b['FED'],
                'component_deltas': {k: a['components'][k]-b['components'][k] for k in a['components']}})
        assert np.isclose(sum(s['candidate_minus_fine_FED'] for s in deltas),
            w['candidate_dx0p25_FED']-w['reference_dx0p125_FED'], rtol=1e-12, atol=1e-16)
        records.append({'layout': lid, 'route': w['route'], 'pair': pair, 'segment_deltas': deltas})
    save(OUT/'cached_route_diagnosis.json', {'records': records, 'new_solver_runs': 0,
        'interpretation': 'Additive bookkeeping and temporal quadrature sensitivity, not independent causal gas effects'})
    print(json.dumps({'cache_replays': len(records), 'cache_diagnosis_saved': True}))


def run(job):
    protocol = read(OUT/'protocol_frozen.json')
    for rel, expected in protocol['implementation_sha256'].items():
        assert sha(ROOT/rel) == expected, rel
    folder = OUT/'runs'/job['name']
    inp = ROOT/job['input']
    assert not (folder/'process_start.json').exists() and sha(inp) == job['input_sha256']
    solver = protocol['solver']
    assert sha(solver['executable']) == solver['sha256']
    assert not any((p.info['name'] or '').lower() == 'fds_openmp.exe' for p in psutil.process_iter(['name']))
    env = os.environ.copy()
    binary = str(Path(solver['executable']).parent)
    env['PATH'] = os.pathsep.join([binary+'/mpi', binary, env['PATH']])
    env['I_MPI_ROOT'] = binary+'/mpi'
    env['OMP_NUM_THREADS'] = str(solver['threads'])
    began = last_change = time.monotonic()
    previous_size, peak, cause, native_end = 0, 0, None, None
    next_log = began+30
    with ExitStack() as stack:
        stdout = stack.enter_context((folder/'stdout.log').open('xb'))
        stderr = stack.enter_context((folder/'stderr.log').open('xb'))
        proc = stack.enter_context(managed_child([solver['executable'], inp.name], cwd=folder, env=env,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            failure_record=folder/'monitor_failure.json', creationflags=subprocess.CREATE_NO_WINDOW))
        identity = {'pid': proc.pid, 'process_creation_epoch_s': psutil.Process(proc.pid).create_time(),
            'started_UTC': datetime.now(timezone.utc).isoformat(), 'input_sha256': sha(inp),
            'case': job['name'], 'target_fire_s': job['end_s'], 'wall_budget_s': job['wall_budget_s']}
        save(folder/'process_start.json', identity)
        print(json.dumps({'event': 'started', **identity}), flush=True)
        while proc.poll() is None:
            now = time.monotonic()
            try:
                peak = max(peak, psutil.Process(proc.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                break
            size = sum(p.stat().st_size for p in folder.glob('*.csv'))+(folder/'stderr.log').stat().st_size
            if size != previous_size:
                previous_size, last_change = size, now
            if now-began >= job['wall_budget_s']:
                cause = 'WALL_TIMEOUT'
            elif peak > protocol['limits']['RSS_GiB']*1024**3:
                cause = 'MEMORY_LIMIT'
            elif now-last_change >= protocol['limits']['no_progress_wall_s']:
                cause = 'NO_PROGRESS_TIMEOUT'
            if cause:
                proc.kill()  # Owned Popen handle; no .stop and no arbitrary PID kill.
                break
            if now >= next_log:
                info = inspect_native_output(folder/(job['name']+'_devc.csv'))
                native_end = float(info['data']['Time'][-1]) if info['valid'] else None
                print(json.dumps({'event': 'progress', 'case': job['name'], 'wall_s': round(now-began, 1),
                    'native_fire_s': native_end, 'peak_RSS_MiB': round(peak/1024**2, 1)}), flush=True)
                next_log = now+30
            time.sleep(.25)
        proc.wait(timeout=15)
    logs = (folder/'stdout.log').read_text(errors='replace')+(folder/'stderr.log').read_text(errors='replace')
    errors = [line for line in logs.splitlines() if re.match(r'\s*ERROR(?:\(|:)', line)]
    info = inspect_native_output(folder/(job['name']+'_devc.csv'))
    native_end = float(info['data']['Time'][-1]) if info['valid'] else None
    reported = re.findall(r'Simulation Time:\s*([\d.Ee+-]+)\s*s', logs)
    solver_end = float(reported[-1]) if reported else None
    normal = complete_run(proc.returncode, 'FDS completed successfully' in logs, errors, cause, solver_end, native_end, job['end_s'])
    result = {**identity, 'normal_completion': normal, 'returncode': proc.returncode, 'termination_cause': cause,
        'FDS_errors': errors, 'native_output_valid': info['valid'], 'native_output_error': info['error'],
        'native_end_s': native_end, 'solver_last_reported_fire_s': solver_end,
        'native_max_gap_s': float(np.diff(info['data']['Time']).max()) if info['valid'] else None,
        'elapsed_wall_s': time.monotonic()-began, 'peak_RSS_MiB': peak/1024**2,
        'native_stop_file_used': False, 'actually_restarted': False}
    save(folder/'run_result.json', result)
    print(json.dumps({'event': 'finished', **result}), flush=True)
    return result


def fixtures():
    results = []
    for job in read(OUT/'jobs_frozen.json'):
        if job['kind'] != 'fixture':
            continue
        result = run(job)
        if not result['normal_completion']:
            save(OUT/'fixture_validation.json', {'passed': False, 'failed_case': job['name'], 'results': results})
            raise RuntimeError('Fixture did not complete; no new fire run allowed')
        data = native(OUT/'runs'/job['name'], job['name'])
        if job['name'] == 'uniform_probe':
            diff = max(float(np.max(np.abs(data[s]-data['I_'+s]))) for s in ('CO', 'CO2', 'O2'))
            checks = {'max_gas_pair_difference': diff, 'passed': diff <= 1e-10}
        else:
            checks = {'initial_default_T_C': float(data['N_T'][0]), 'initial_interpolated_T_C': float(data['I_T'][0])}
            checks['passed'] = bool(abs(checks['initial_default_T_C']-40) <= 1e-7 and abs(checks['initial_interpolated_T_C']-30) <= 1e-7)
        results.append({'case': job['name'], **checks})
        if not checks['passed']:
            save(OUT/'fixture_validation.json', {'passed': False, 'results': results})
            raise RuntimeError('Observation fixture mismatch; no new fire run allowed')
    save(OUT/'fixture_validation.json', {'passed': True, 'results': results})


def fields():
    assert read(OUT/'fixture_validation.json')['passed']
    for job in read(OUT/'jobs_frozen.json'):
        if job['kind'] == 'paired_probe':
            result = run(job)
            if not result['normal_completion']:
                save(OUT/'fields_stopped.json', {'failed_case': job['name'], 'reason': 'No automatic retry or budget extension'})
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cached', 'fixtures', 'fields'])
    action = parser.parse_args().action
    {'prepare': prepare, 'cached': cached, 'fixtures': fixtures, 'fields': fields}[action]()
