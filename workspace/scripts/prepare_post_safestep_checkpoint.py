"""Freeze a new passive-checkpoint stage without modifying the previous stage."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/post_safestep_physics_20261019'
OUT = ROOT/'outputs/post_safestep_checkpoint_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(text):
    text = re.sub(r"\bCHID='[^']+'", 'CHID=<CASE>', text)
    text = re.sub(r'\bT_END=[\d.]+', 'T_END=<END>', text)
    text = re.sub(r'\bDT_RESTART=[\d.]+', 'DT_RESTART=<INTERVAL>', text)
    text = re.sub(r',\s*DT_FLUSH=[\d.]+', '', text)
    return text


def main():
    assert not OUT.exists(), 'Preserve every attempt; do not overwrite an existing stage'
    previous = read(OLD/'completion_manifest.json')['artifact_sha256']
    protected = {**read(ROOT/'outputs/safestep_closure_plan_20261019/source_hashes.json'), **previous,
        (OLD/'completion_manifest.json').relative_to(ROOT).as_posix(): sha(OLD/'completion_manifest.json')}
    for relative, expected in protected.items():
        assert sha(ROOT/relative) == expected, relative
    OUT.mkdir()
    (OUT/'runs').mkdir()
    params = read(OLD/'physical_parameters_frozen.json')
    plan = {
        'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'parent_parameters_sha256': sha(OLD/'physical_parameters_frozen.json'),
        'parent_routes_sha256': sha(OLD/'diagnostic_routes_frozen.json'),
        'parent_mesh_rules_sha256': sha(OLD/'mesh_screen_rules_frozen.json'),
        'parent_early_rejection_sha256': sha(OLD/'mesh_early_rejection_rule.json'),
        'solver': params['solver'], 'checkpoint_interval_fire_s': 30.,
        'limits': {'fixture_wall_s': 300., 'fine_case_wall_s': 10800.,
            'no_progress_wall_s': 600., 'resident_memory_GiB': 4., 'one_solver_at_a_time': True},
        'recovery_validation': {'reference_end_s': 100., 'gas_relative_limit': .01,
            'gas_absolute_floor_mol_per_mol': 1e-10, 'temperature_relative_limit': .01,
            'temperature_absolute_floor_C': .1, 'dose_relative_limit': .01,
            'dose_denominator_floor': 1e-8, 'maximum_native_gap_s': 1.6,
            'pre_checkpoint_rows_must_be_identical': True,
            'basis': 'Paired new-layout coarse-grid recovery diagnostics, not full fine-grid physical validation'},
        'mesh_route_relative_limit': .10, 'mesh_dose_denominator_floor': 1e-8,
        'changes_allowed': ['CHID', 'DT_RESTART=30', 'DT_FLUSH=1', 'T_END=100 for recovery fixtures only'],
        'physics_changed': False, 'native_stop_file_used': False,
        'stop_file_exclusion_reason': 'Existing fixed-version 2026-10-17 source and microfixture evidence of stop-induced chemistry disturbance',
        'recovery_source': 'Passive periodic checkpoints only; copy after solver exit, never edit binary bytes',
        'forced_budget_stop': 'Terminate owned child without .stop; retain outputs, validate any passive checkpoint before reuse; no automatic retry',
        'mesh_rejection': 'Existing predeclared complete-route necessary condition; no early acceptance',
        'formal_24_started': False, '900s_checks_require_180s_pass': True,
        'new_RL_updates': 0, 'LLM_requests': 0, 'sealed_final48_accessed': False,
    }
    jobs = []
    for lid, vent in (('L1', 'V0'), ('L2', 'V1')):
        for kind, token, end in (('recovery_reference', '0p5', 100.), ('fine_mesh', '0p125', 180.)):
            parent = OLD/f'runs/{lid}_F2_G150_{vent}_dx{token}_adaptive'
            parent_input = parent/(parent.name+'.fds')
            assert sha(parent_input) == read(parent/'input_manifest.json')['input_sha256']
            name = f'{lid}_recovery_reference' if kind == 'recovery_reference' else f'{lid}_F2_G150_{vent}_dx0p125_checkpoint'
            folder = OUT/'runs'/name
            folder.mkdir()
            source = parent_input.read_text(encoding='utf-8')
            text, n = re.subn(r"\bCHID='[^']+'", f"CHID='{name}'", source)
            assert n == 1
            text, n = re.subn(r'\bT_END=[\d.]+', f'T_END={end:g}', text)
            assert n == 1
            text, n = re.subn(r'\bDT_RESTART=[\d.]+', 'DT_RESTART=30., DT_FLUSH=1.', text)
            assert n == 1 and canonical(text) == canonical(source)
            inp = folder/(name+'.fds')
            inp.write_text(text, encoding='utf-8', newline='\n')
            ijk = [int(x) for x in re.search(r'IJK=(\d+),(\d+),(\d+)', text).groups()]
            job = {'name': name, 'kind': kind, 'layout': lid, 'vent': vent, 'chid': name,
                'input': inp.relative_to(ROOT).as_posix(), 'input_sha256': sha(inp),
                'parent_input': parent_input.relative_to(ROOT).as_posix(), 'parent_input_sha256': sha(parent_input),
                'IJK': ijk, 'end_s': end, 'canonical_physics_sha256': hashlib.sha256(canonical(source).encode()).hexdigest()}
            save(folder/'input_manifest.json', job)
            jobs.append(job)
    save(OUT/'protected_parent_hashes.json', protected)
    save(OUT/'execution_plan_frozen.json', plan)
    save(OUT/'planned_jobs.json', jobs)
    print(json.dumps({'prepared': len(jobs), 'protected_files': len(protected), 'physics_changed': False}))


if __name__ == '__main__':
    main()
