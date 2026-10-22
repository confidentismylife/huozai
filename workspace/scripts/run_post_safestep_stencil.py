"""Explicit cell-center reconstruction after native interpolation failed checks."""
import argparse
from datetime import datetime, timezone
import re

import numpy as np

import run_post_safestep_sampling as backend
from run_post_safestep_sampling import ROOT, OLD, PARENT, read, sha, save, native
from fire_evac.fds_cell_stencil import support_centers, reconstruct
from fire_evac.post_safestep_physics import rectangle_union_contains

A = ROOT/'outputs/post_safestep_sampling_20261019'
OUT = ROOT/'outputs/post_safestep_stencil_20261019'
# Reuse the unchanged bounded runner in this dedicated Python process. Only its
# output-root configuration changes; none of the earlier files are modified.
backend.OUT = OUT


def append_centers(text, coordinates, dx, layout):
    lines, metadata = [], []
    for node, xyz in sorted(coordinates.items()):
        points = support_centers(xyz, dx)
        for corner, (x, y, z) in enumerate(points):
            if layout:
                assert 0 < z < layout['domain_xyz'][2]
                assert rectangle_union_contains(x, y, layout['open_rectangles_xy'])
            metadata.append({'node': node, 'corner': corner, 'xyz_m': [x, y, z], 'weight': .125})
            for label, quantity, species in [('CO', 'VOLUME FRACTION', 'CARBON MONOXIDE'),
                    ('CO2', 'VOLUME FRACTION', 'CARBON DIOXIDE'), ('O2', 'VOLUME FRACTION', 'OXYGEN'),
                    ('T', 'TEMPERATURE', None)]:
                spec = f", SPEC_ID='{species}'" if species else ''
                lines.append(f"&DEVC ID='C{node:03d}_{corner}_{label}', XYZ={x:g},{y:g},{z:g}, QUANTITY='{quantity}'{spec}, TIME_AVERAGED=.FALSE. /")
    assert text.count('&TAIL /') == 1
    return text.replace('&TAIL /', '\n'.join(lines)+'\n&TAIL /'), metadata


def prepare():
    assert not OUT.exists()
    manifest = read(A/'completion_manifest.json')
    protected = {**read(A/'protected_parent_hashes.json'), **manifest['artifact_sha256'],
        (A/'completion_manifest.json').relative_to(ROOT).as_posix(): sha(A/'completion_manifest.json')}
    for rel, expected in protected.items():
        assert sha(ROOT/rel) == expected, rel
    OUT.mkdir()
    (OUT/'runs').mkdir()
    save(OUT/'protected_parent_hashes.json', protected)
    a_jobs = read(A/'jobs_frozen.json')
    jobs = []
    for name in ('uniform_probe', 'temperature_step'):
        job = next(j for j in a_jobs if j['name'] == name)
        text = (ROOT/job['input']).read_text(encoding='utf-8')
        # Remove failed native interpolation devices; keep the default devices.
        text = '\n'.join(line for line in text.splitlines() if 'SPATIAL_STATISTIC' not in line)+'\n'
        if name == 'uniform_probe':
            text = text.replace('IJK=5,5,5', 'IJK=10,10,10')
        chid = name+'_centers'
        text = text.replace(f"CHID='{name}'", f"CHID='{chid}'")
        # The temperature fixture also records ambient O2 and zero CO/CO2.
        text, centers = append_centers(text, {0: [.5, .5, .5]}, .1, None)
        if name == 'temperature_step':
            # Explicit passive species are needed for gas output in this no-fire fixture.
            text = text.replace('&TIME ', "&SPEC ID='CARBON MONOXIDE' /\n&SPEC ID='CARBON DIOXIDE' /\n&SPEC ID='OXYGEN' /\n&TIME ")
        folder = OUT/'runs'/chid
        folder.mkdir()
        inp = folder/(chid+'.fds')
        inp.write_text(text, encoding='utf-8', newline='\n')
        jobs.append({'name': chid, 'kind': 'fixture', 'input': inp.relative_to(ROOT).as_posix(),
            'input_sha256': sha(inp), 'end_s': .2, 'wall_budget_s': 300., 'centers': centers})
    w = next(w for w in read(PARENT/'mesh_results.json')['early_witnesses_replayed'] if w['layout'] == 'L1')
    graph = read(OLD/'L1_graph.json')
    xy = graph['graph']['coordinates_xy']
    coordinates = {n: [*xy[n], 1.5] for n in w['route']['route'][:-1]}
    for source in [j for j in a_jobs if j.get('layout') == 'L1']:
        parent_dir = ROOT/source['parent']
        text = (parent_dir/(source['parent_chid']+'.fds')).read_text(encoding='utf-8')
        chid = source['name'].replace('dual_probe', 'eight_centers')
        text = text.replace(f"CHID='{source['parent_chid']}'", f"CHID='{chid}'")
        text = re.sub(r'\bT_END=180\b', 'T_END=70', text)
        text = re.sub(r'\bDT_RESTART=[\d.]+', 'DT_RESTART=30.', text)
        if 'DT_FLUSH=' not in text:
            text = text.replace('DT_RESTART=30.', 'DT_RESTART=30., DT_FLUSH=1.')
        text, centers = append_centers(text, coordinates, source['dx_m'], graph['layout'])
        # Every original device stays in the same order, with identical coordinates/statistics.
        original_devices = [line for line in (parent_dir/(source['parent_chid']+'.fds')).read_text(encoding='utf-8').splitlines() if line.startswith('&DEVC')]
        assert [line for line in text.splitlines() if line.startswith('&DEVC')][:len(original_devices)] == original_devices
        folder = OUT/'runs'/chid
        folder.mkdir()
        inp = folder/(chid+'.fds')
        inp.write_text(text, encoding='utf-8', newline='\n')
        jobs.append({**source, 'name': chid, 'kind': 'center_observation', 'input': inp.relative_to(ROOT).as_posix(),
            'input_sha256': sha(inp), 'centers': centers})
    save(OUT/'jobs_frozen.json', jobs)
    a_protocol = read(A/'protocol_frozen.json')
    implementations = ['scripts/run_post_safestep_stencil.py', 'scripts/run_post_safestep_sampling.py',
        'src/fire_evac/fds_cell_stencil.py', 'src/fire_evac/fds_sampling_diagnostic.py',
        'src/fire_evac/fds_checkpoint.py', 'src/fire_evac/post_safestep_physics.py',
        'tests/test_fds_cell_stencil.py', 'tests/test_fds_sampling_diagnostic.py']
    save(OUT/'protocol_frozen.json', {'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'solver': a_protocol['solver'], 'limits': a_protocol['limits'],
        'implementation_sha256': {rel: sha(ROOT/rel) for rel in implementations},
        'selected_layout': 'L1', 'selection_reason': 'Larger previous failure, exploratory post-hoc diagnostic only',
        'observation': 'Equal 1/8 mean of eight explicitly observed gas-cell centers about nominal Cartesian vertex',
        'all_supports_in_open_gas': True, 'time_output_s': 1, 'field_end_s': 70,
        'fixture_temperatures_t0_C': {'default': 40, 'reconstructed': 30, 'tolerance': 1e-7},
        'fixture_uniform_gas_max_absolute_difference': 1e-10,
        'default_prefix_parity': a_protocol['default_prefix_parity'],
        'new_RL_updates': 0, 'LLM_requests': 0, 'sealed_final48_accessed': False,
        'old_mesh_failure_unchanged': True, 'formal24_started': False, '900s_started': False,
        'native_interpolation_failed_checks_preserved': True, 'no_L2_extension_this_stage': True,
        'no_automatic_retries': True})
    print({'prepared': len(jobs), 'protected': len(protected)})


def fixtures():
    records = []
    for job in read(OUT/'jobs_frozen.json'):
        if job['kind'] != 'fixture':
            continue
        result = backend.run(job)
        if not result['normal_completion']:
            save(OUT/'fixture_validation.json', {'passed': False, 'records': records, 'failed_case': job['name']})
            raise RuntimeError('Cell-center fixture failed; no fire run')
        data = native(OUT/'runs'/job['name'], job['name'])
        rec = reconstruct(data, [0])
        if job['name'].startswith('uniform'):
            diff = max(float(np.max(np.abs(data[s]-rec['S000_'+s]))) for s in ('CO', 'CO2', 'O2'))
            item = {'case': job['name'], 'max_gas_difference': diff, 'passed': diff <= 1e-10}
        else:
            item = {'case': job['name'], 'default_t0_C': float(data['N_T'][0]),
                'reconstructed_t0_C': float(rec['S000_T'][0])}
            item['passed'] = bool(abs(item['default_t0_C']-40) <= 1e-7 and abs(item['reconstructed_t0_C']-30) <= 1e-7)
        records.append(item)
        if not item['passed']:
            save(OUT/'fixture_validation.json', {'passed': False, 'records': records})
            raise RuntimeError('Known-value reconstruction did not pass')
    save(OUT/'fixture_validation.json', {'passed': True, 'records': records})


def fields():
    assert read(OUT/'fixture_validation.json')['passed']
    for job in read(OUT/'jobs_frozen.json'):
        if job['kind'] == 'center_observation':
            result = backend.run(job)
            if not result['normal_completion']:
                save(OUT/'fields_stopped.json', {'failed_case': job['name']})
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'fixtures', 'fields'])
    action = parser.parse_args().action
    {'prepare': prepare, 'fixtures': fixtures, 'fields': fields}[action]()
