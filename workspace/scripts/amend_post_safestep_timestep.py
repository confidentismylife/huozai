"""Preserve initial capped-DT attempts and prepare versioned adaptive-DT jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/post_safestep_physics_20261019'


def write_json(path, obj):
    with path.open('x', encoding='utf-8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def main():
    original = json.loads((OUT / 'planned_jobs.json').read_text())
    parents = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        argv = process.info['cmdline'] or []
        if (process.info['name'] or '').lower() == 'python.exe' and any(
                arg.replace('\\', '/').endswith('scripts/run_post_safestep_physics.py') for arg in argv):
            parents.append(process)
    if len(parents) > 1:
        raise RuntimeError('Ambiguous current task supervisor; no termination performed')
    identities = []
    if parents:
        supervisor = parents[0]
        children = supervisor.children(recursive=False)
        for child in children:
            argv = child.cmdline()
            assert child.name().lower() == 'fds_openmp.exe'
            chid = Path(argv[-1]).stem
            assert chid in {j['name'] for j in original}
            assert Path(child.cwd()).resolve() == (OUT / 'runs' / chid).resolve()
            identities.append({'supervisor_pid': supervisor.pid, 'child_pid': child.pid, 'case': chid})
        # Stop this task's supervisor first so it cannot launch another queued job.
        supervisor.terminate()
        supervisor.wait(timeout=15)
        for child in children:
            try:
                child.terminate()
                child.wait(timeout=15)
            except psutil.NoSuchProcess:
                pass
        for identity in identities:
            write_json(OUT / 'runs' / identity['case'] / 'interrupted_by_numeric_amendment.json',
                {**identity, 'status': 'INCOMPLETE_INITIAL_DT_CAP', 'included_in_mesh_effects': False})
    amendment = {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'reason': 'Explicit DT=0.01 was also the default upper time-step bound; remove it to use documented automatic CFL initialization and adjustment',
        'source': 'FDS6.11.1 User Guide: Initial Time Step Size; default RESTRICT_TIME_STEP=T',
        'terminated_only_identified_current_task_processes': identities,
        'old_attempts_preserved': True, 'old_attempts_in_main_mesh_comparison': False,
        'fuel_geometry_and_dose_changed': False, 'policy_results_seen': False,
        'native_output_contract': 'DT_DEVC=1 is a requested cadence. Retain actual sample times; do not relabel to integer seconds. Native gaps may exceed1 by up to a time step.',
        'mesh_screen_max_output_gap_s': 1.6,
        'coverage': 'Require normal solver termination at180 and native last sample>=179; fixed routes finish before179',
        'jobs': [],
    }
    for job in original:
        if job['kind'] != 'mesh_screen':
            continue
        old = ROOT / job['input_path']
        name = job['name'] + '_adaptive'
        folder = OUT / 'runs' / name
        folder.mkdir(exist_ok=False)
        text = old.read_text().replace("CHID='" + job['name'] + "'", "CHID='" + name + "'")
        assert ', DT=0.01' in text
        text = text.replace(', DT=0.01', '')
        p = folder / (name + '.fds')
        p.write_text(text, encoding='utf-8', newline='\n')
        revised = {**job, 'name': name, 'input_path': str(p.relative_to(ROOT)),
            'input_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'numerical_revision': 'adaptive_initial_dt'}
        write_json(folder / 'input_manifest.json', revised)
        amendment['jobs'].append(revised)
    write_json(OUT / 'adaptive_timestep_amendment.json', amendment)
    print(json.dumps({'adaptive_jobs': len(amendment['jobs']), 'terminated': identities,
        'fuel_geometry_and_dose_changed': False}))


if __name__ == '__main__':
    main()
