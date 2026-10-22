"""Prepare an isolated, pinned FDS restart connectivity experiment.

No installed solver, checkpoint, or previous experiment is modified.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_pressure_zone_fix_20261017'
OLD = ROOT / 'outputs/safestep_clean_hotel120_20261017'
COMMIT = 'ff928dbf68a2623ced3a007be75391c6dc6c9f90'
SOURCES = ('prec cons chem prop devc type data mesh func gsmv smvv rcal turb soot '
           'pois geom ccib radi part vege ctrl hvac mass imkl wall fire velo pres '
           'init dump read divg main').split()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    protected = {}
    for folder in ['safestep_physics_time_contract_20261016',
                   'safestep_restart_hrr_audit_20261017', 'safestep_clean_hotel120_20261017']:
        manifest = ROOT / 'outputs' / folder / 'completion_manifest.json'
        protected.update(json.loads(manifest.read_text('utf-8'))['artifact_sha256'])
        protected[manifest.relative_to(ROOT).as_posix()] = sha(manifest)
    old_plan = json.loads((OLD / 'frozen_plan.json').read_text('utf-8'))
    protected.update(old_plan['protected_previous_sha256'])
    bad = [p for p, h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    save(OUT/'previous_integrity_before.json', {'checked_files': len(protected), 'mismatches': bad})
    assert not bad
    plan = {
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'question': 'Does reconstructing CONNECTED_ZONES during restart initialization restore sealed dynamic-door pressure coupling?',
        'source_commit': COMMIT,
        'patch': 'Set OBST_CREATED_OR_REMOVED true on RESTART with N_ZONE>0 immediately before INITIALIZE_DIVERGENCE_INTEGRALS',
        'build': 'Isolated Linux GNU/OpenMPI/OpenMP baseline and patched binaries; same compiler/options/libraries; no installed FDS changes',
        'build_fallback_boundary': 'If a compatible local build cannot execute, preserve source patch and record build limitation; no binary checkpoint edits',
        'primary_micro_cases': ['sealed_dynamic_door', 'sealed_static_open'],
        'primary_micro_runs': 8,
        'micro_comparison_interval': 'Original fixed 0.02s grid strictly after native 4.2s checkpoint through 6.5s; no extrapolation',
        'micro_engineering_screens': {
            'temperature_max_abs_C': 0.1,
            'CO2_max_abs_ppm': 10.0,
            'interface_velocity_max_abs_m_s': 0.005,
            'zone_final_pressure_max_abs_Pa': 10.0,
            'dynamic_temperature_and_pressure_error_reduction_fraction_min': 0.90,
            'note': 'Exploratory consistency thresholds, not physical or safety validation'
        },
        'controls': ['unpatched versus patched same-build continuous/restart pairs',
                     'sealed static-open negative control',
                     'continuous physical inputs identical between solver variants'],
        'hotel_followup_if_micro_passes': {
            'max_fire_time_s': 120.0, 'comparison_end_s': 110.0,
            'original_checkpoint_cross_compiler_compatibility': 'Must be demonstrated, never assumed',
            'same_build_fresh_checkpoint_option': True,
            'engineering_screen': old_plan['predeclared_engineering_screen'],
            'new_continuous_wall_limit_s': 3000, 'restart_wall_limit_s': 900,
        },
        'forbidden': ['installed executable replacement', 'restart byte editing',
                      '300/900s simulations', 'new training', 'LLM/API requests',
                      'sealed final48 access', 'claiming physical or author-input validation'],
        'protected_previous_sha256': protected,
    }
    save(OUT/'frozen_plan.json', plan)
    original = OUT/'source_baseline'; (original/'Source').mkdir(parents=True)
    paths = [f'Source/{s}.f90' for s in SOURCES] + ['CMakeLists.txt', 'Build/makefile']

    def retrieve(path):
        url = f'https://raw.githubusercontent.com/firemodels/fds/{COMMIT}/{path}'
        r = requests.get(url, timeout=60); r.raise_for_status()
        target = original/path; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        return {'path': path, 'url': url, 'bytes': len(r.content), 'sha256': sha(target)}

    sources = list(ThreadPoolExecutor(max_workers=4).map(retrieve, paths))
    comparisons = []
    for f in (ROOT/'outputs/safestep_restart_hrr_audit_20261017/materials').glob('*.f90'):
        other = original/'Source'/f.name
        if other.exists():
            comparisons.append({'name': f.name, 'matches_previous_pinned_source': sha(f) == sha(other)})
    f = OLD/'pressure_zone_source/divg.f90'
    comparisons.append({'name': 'divg.f90', 'matches_previous_pinned_source': sha(f) == sha(original/'Source/divg.f90')})
    assert all(v['matches_previous_pinned_source'] for v in comparisons)
    save(OUT/'source_provenance.json', {'commit': COMMIT, 'files': sources, 'previous_source_checks': comparisons})
    patched = OUT/'source_patched'; shutil.copytree(original, patched)
    mainfile = patched/'Source/main.f90'
    before = mainfile.read_text('utf-8')
    anchor = '! Compute divergence just in case the flow field is not initialized to ambient\n\nCALL INITIALIZE_DIVERGENCE_INTEGRALS'
    replacement = ('! Compute divergence just in case the flow field is not initialized to ambient\n\n'
                   '! Reconstruct pressure-zone connectivity from restored wall/cell state.\n'
                   '! CONNECTED_ZONES is initialized to identity and is not in the restart file.\n'
                   'IF (RESTART .AND. N_ZONE>0) OBST_CREATED_OR_REMOVED = .TRUE.\n\n'
                   'CALL INITIALIZE_DIVERGENCE_INTEGRALS')
    assert before.count(anchor) == 1
    after = before.replace(anchor, replacement)
    mainfile.write_text(after, 'utf-8', newline='\n')
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile='a/Source/main.f90', tofile='b/Source/main.f90'))
    (OUT/'restart_pressure_connectivity.patch').write_text(diff, 'utf-8')
    changes = [p.relative_to(original).as_posix() for p in original.rglob('*')
               if p.is_file() and sha(p) != sha(patched/p.relative_to(original))]
    assert changes == ['Source/main.f90']
    save(OUT/'patch_verification.json', {'changed_files': changes, 'diff': diff,
         'baseline_main_sha256': sha(original/'Source/main.f90'), 'patched_main_sha256': sha(mainfile),
         'restart_format_changed': False, 'installed_solver_modified': False})
    print(json.dumps({'status': 'PREPARED', 'protected_files_verified': len(protected),
                      'source_files': len(sources), 'patch_changed_files': changes}), flush=True)


if __name__ == '__main__':
    main()
