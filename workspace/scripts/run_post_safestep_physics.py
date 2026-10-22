"""Bounded single-child FDS runs; no old input/output mutation or automatic retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/post_safestep_physics_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='+')
    args = parser.parse_args()
    params = read(OUT / 'physical_parameters_frozen.json')
    lookup = {j['name']: j for j in read(OUT / 'planned_jobs.json')}
    if (OUT / 'adaptive_timestep_amendment.json').is_file():
        lookup.update({j['name']: j for j in read(OUT / 'adaptive_timestep_amendment.json')['jobs']})
    for name in args.case:
        job = lookup[name]
        case = OUT / 'runs' / name
        input_file = ROOT / job['input_path']
        assert hashlib.sha256(input_file.read_bytes()).hexdigest() == job['input_sha256']
        assert not (case / 'process_start.json').exists(), 'No automatic retry or output overwrite'
        if job['kind'] == 'mesh_screen':
            unit_result = read(OUT / 'runs/dose_units/run_result.json')
            assert unit_result['normal_completion']
            assert (OUT / 'dose_units_verification.json').is_file(), 'Verify dose units before fire runs'
            assert read(OUT / 'dose_units_verification.json')['passed']
        env = os.environ.copy()
        env['PATH'] = os.pathsep.join(['C:/Program Files/firemodels/FDS6/bin/mpi',
            'C:/Program Files/firemodels/FDS6/bin', env['PATH']])
        env['I_MPI_ROOT'] = 'C:/Program Files/firemodels/FDS6/bin/mpi'
        env['OMP_NUM_THREADS'] = str(params['solver']['threads'])
        began, last_progress, last_bytes = time.monotonic(), time.monotonic(), 0
        limit = params['runtime_limits']
        cause = None
        with (case / 'stdout.log').open('xb') as stdout, (case / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen([params['solver']['executable'], input_file.name],
                cwd=case, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW)
            identity = {'pid': child.pid, 'created_UTC': datetime.now(timezone.utc).isoformat(),
                'executable': params['solver']['executable'], 'input': input_file.name,
                'input_sha256': job['input_sha256'], 'threads': params['solver']['threads']}
            (case / 'process_start.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
            print(json.dumps({'event': 'started', 'case': name, 'pid': child.pid}), flush=True)
            next_log = began + 20
            peak_memory = 0
            while child.poll() is None:
                now = time.monotonic()
                try:
                    mem = psutil.Process(child.pid).memory_info().rss
                    peak_memory = max(peak_memory, mem)
                except psutil.Error:
                    mem = 0
                count = sum(p.stat().st_size for p in case.glob('*_hrr.csv'))
                count += (case / 'stderr.log').stat().st_size
                if count != last_bytes:
                    last_progress, last_bytes = now, count
                if now - began > limit['case_wall_timeout_s']:
                    cause = 'WALL_TIMEOUT'
                elif now - last_progress > limit['no_progress_timeout_s']:
                    cause = 'NO_PROGRESS_TIMEOUT'
                elif mem > limit['maximum_resident_memory_GiB'] * 1024 ** 3:
                    cause = 'MEMORY_LIMIT'
                if cause:
                    child.kill()
                    break
                if now >= next_log:
                    tail = (case / 'stderr.log').read_text(errors='replace')[-2200:]
                    times = re.findall(r'Simulation Time:\s*([\d.Ee+-]+)\s*s', tail)
                    print(json.dumps({'event': 'progress', 'case': name, 'wall_s': round(now-began, 1),
                        'last_reported_fire_s': times[-1] if times else None,
                        'rss_MiB': round(mem / 1024 ** 2, 1)}), flush=True)
                    next_log = now + 20
                time.sleep(.25)
            child.wait(timeout=15)
        logs = (case / 'stderr.log').read_text(errors='replace') + (case / 'stdout.log').read_text(errors='replace')
        normal = ('FDS completed successfully' in logs) if job['kind'] != 'setup' else ('Set-up only' in logs)
        errors = [line for line in logs.splitlines() if re.match(r'\s*ERROR(?:\(|:)', line)]
        result = {**identity, 'returncode': child.returncode, 'termination_cause': cause,
            'normal_completion': bool(normal and not errors and not cause), 'FDS_errors': errors,
            'elapsed_wall_s': time.monotonic()-began, 'peak_rss_MiB': peak_memory / 1024 ** 2,
            'requested_end_s': job['end_s'], 'log_tail': logs[-3000:]}
        (case / 'run_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'finished', 'case': name, 'normal': result['normal_completion'],
            'cause': cause, 'wall_s': round(result['elapsed_wall_s'], 2), 'errors': errors}), flush=True)
        if not result['normal_completion']:
            raise SystemExit('FDS case did not finish normally; preserve outputs and inspect')


if __name__ == '__main__':
    main()
