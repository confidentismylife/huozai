"""Pre-specified independent-route check of the explicit eight-cell observation."""

import argparse
from datetime import datetime, timezone
import re

import run_post_safestep_sampling as backend
from run_post_safestep_sampling import ROOT, OLD, PARENT, read, sha, save
from fire_evac.fds_cell_stencil import support_centers
from fire_evac.fds_independent_validation import select_independent_routes
from fire_evac.post_safestep_physics import rectangle_union_contains


PREVIOUS = ROOT / 'outputs/post_safestep_stencil_20261019'
OUT = ROOT / 'outputs/post_safestep_independent_stencil_20261020'
ROUTES_SOURCE = OLD / 'diagnostic_routes_frozen.json'
backend.OUT = OUT


def append_centers(text, coordinates, dx, layout):
    lines, metadata = [], []
    for node, xyz in sorted(coordinates.items()):
        points = support_centers(xyz, dx)
        for corner, (x, y, z) in enumerate(points):
            assert 0 < z < layout['domain_xyz'][2]
            assert rectangle_union_contains(x, y, layout['open_rectangles_xy'])
            metadata.append({'node': node, 'corner': corner,
                             'xyz_m': [x, y, z], 'weight': .125})
            for label, quantity, species in [
                ('CO', 'VOLUME FRACTION', 'CARBON MONOXIDE'),
                ('CO2', 'VOLUME FRACTION', 'CARBON DIOXIDE'),
                ('O2', 'VOLUME FRACTION', 'OXYGEN'),
                ('T', 'TEMPERATURE', None),
            ]:
                spec = f", SPEC_ID='{species}'" if species else ''
                lines.append(
                    f"&DEVC ID='C{node:03d}_{corner}_{label}', "
                    f"XYZ={x:g},{y:g},{z:g}, QUANTITY='{quantity}'{spec}, "
                    "TIME_AVERAGED=.FALSE. /"
                )
    assert text.count('&TAIL /') == 1
    return text.replace('&TAIL /', '\n'.join(lines) + '\n&TAIL /'), metadata


def verified_previous_artifacts():
    completion_path = PREVIOUS / 'completion_manifest.json'
    completion = read(completion_path)
    protected = {
        **read(PREVIOUS / 'protected_parent_hashes.json'),
        **completion['artifact_sha256'],
        completion_path.relative_to(ROOT).as_posix(): sha(completion_path),
    }
    revision_path = PREVIOUS / 'report_revision_manifest.json'
    revision = read(revision_path)
    protected.update(revision['artifact_sha256'])
    protected[revision_path.relative_to(ROOT).as_posix()] = sha(revision_path)
    for rel, expected in protected.items():
        assert sha(ROOT / rel) == expected, rel
    return protected


def prepare():
    assert not OUT.exists(), 'Preserve any existing independent validation attempt'
    protected = verified_previous_artifacts()
    parent_protocol = read(PREVIOUS / 'protocol_frozen.json')
    route_pool = read(ROUTES_SOURCE)
    witness_records = read(PARENT / 'mesh_results.json')['early_witnesses_replayed']
    excluded = {layout: [] for layout in route_pool}
    for record in witness_records:
        excluded[record['layout']].append(record['route'])
    coverage_selected = select_independent_routes(route_pool, excluded, field_end_s=70.0)
    selected, geometry_excluded = {}, {}
    for layout_name, routes in coverage_selected.items():
        graph = read(OLD / f'{layout_name}_graph.json')
        layout = graph['layout']
        coordinates_xy = graph['graph']['coordinates_xy']
        selected[layout_name], geometry_excluded[layout_name] = [], []
        for route in routes:
            unsupported = []
            for node in route['route'][:-1]:
                for dx in (.25, .125):
                    points = support_centers([*coordinates_xy[node], 1.5], dx)
                    if not all(rectangle_union_contains(x, y, layout['open_rectangles_xy'])
                               for x, y, _ in points):
                        unsupported.append({'node': node, 'dx_m': dx})
            if unsupported:
                geometry_excluded[layout_name].append({
                    'route': route,
                    'reason': 'At least one eight-cell support lies outside open gas',
                    'unsupported': unsupported,
                })
            else:
                selected[layout_name].append(route)
    # Counts are frozen properties of the already sealed route library. A mismatch
    # means the source changed and preparation must stop rather than silently adapt.
    assert {layout: len(routes) for layout, routes in selected.items()} == {'L1': 4, 'L2': 2}

    OUT.mkdir()
    (OUT / 'runs').mkdir()
    save(OUT / 'protected_parent_hashes.json', protected)
    selection_record = {
        'selection_rule': (
            'All routes in the pre-existing frozen library that finish by 70 s, '
            'excluding the exact L1 and L2 routes used in the prior post-hoc witness diagnostic, '
            'and requiring all eight support cells at every non-exit node to be open gas '
            'at both 0.25 m and 0.125 m resolution'
        ),
        'selection_uses_outcomes': False,
        'segment_time_s': 1.25,
        'field_end_s': 70.0,
        'excluded_prior_witnesses': excluded,
        'geometry_excluded': geometry_excluded,
        'selected': selected,
        'selected_counts': {layout: len(routes) for layout, routes in selected.items()},
        'source': ROUTES_SOURCE.relative_to(ROOT).as_posix(),
        'source_sha256': sha(ROUTES_SOURCE),
    }
    save(OUT / 'independent_routes_frozen.json', selection_record)

    source_jobs = [job for job in read(
        ROOT / 'outputs/post_safestep_sampling_20261019/jobs_frozen.json'
    ) if job.get('kind') == 'paired_probe']
    jobs = []
    for source in source_jobs:
        layout_name = source['layout']
        graph = read(OLD / f'{layout_name}_graph.json')
        layout = graph['layout']
        coordinates_xy = graph['graph']['coordinates_xy']
        nodes = sorted({node for route in selected[layout_name]
                        for node in route['route'][:-1]})
        coordinates = {node: [*coordinates_xy[node], 1.5] for node in nodes}
        parent_dir = ROOT / source['parent']
        original_path = parent_dir / f"{source['parent_chid']}.fds"
        original = original_path.read_text(encoding='utf-8')
        dx_tag = str(source['dx_m']).replace('.', 'p')
        chid = f'{layout_name}_dx{dx_tag}_independent_centers'
        text = original.replace(f"CHID='{source['parent_chid']}'", f"CHID='{chid}'")
        text, count = re.subn(r'\bT_END=180\b', 'T_END=70', text)
        assert count == 1
        text = re.sub(r'\bDT_RESTART=[\d.]+', 'DT_RESTART=30.', text)
        if 'DT_FLUSH=' not in text:
            text = text.replace('DT_RESTART=30.', 'DT_RESTART=30., DT_FLUSH=1.')
        text, centers = append_centers(text, coordinates, source['dx_m'], layout)
        original_devices = [line for line in original.splitlines() if line.startswith('&DEVC')]
        new_devices = [line for line in text.splitlines() if line.startswith('&DEVC')]
        assert new_devices[:len(original_devices)] == original_devices
        folder = OUT / 'runs' / chid
        folder.mkdir()
        input_path = folder / f'{chid}.fds'
        input_path.write_text(text, encoding='utf-8', newline='\n')
        jobs.append({
            'name': chid,
            'kind': 'independent_center_observation',
            'layout': layout_name,
            'dx_m': source['dx_m'],
            'input': input_path.relative_to(ROOT).as_posix(),
            'input_sha256': sha(input_path),
            'parent': source['parent'],
            'parent_chid': source['parent_chid'],
            'parent_input_sha256': sha(original_path),
            'end_s': 70.0,
            'wall_budget_s': 10800.0,
            'routes': selected[layout_name],
            'observed_nodes': nodes,
            'centers': centers,
        })
    assert len(jobs) == 4
    save(OUT / 'jobs_frozen.json', jobs)

    implementation = [
        'scripts/run_post_safestep_independent_stencil.py',
        'scripts/run_post_safestep_sampling.py',
        'src/fire_evac/fds_cell_stencil.py',
        'src/fire_evac/fds_sampling_diagnostic.py',
        'src/fire_evac/fds_independent_validation.py',
        'src/fire_evac/fds_checkpoint.py',
        'src/fire_evac/post_safestep_physics.py',
        'tests/test_fds_independent_validation.py',
        'tests/test_fds_cell_stencil.py',
        'tests/test_fds_sampling_diagnostic.py',
    ]
    protocol = {
        'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Independent-route descriptive replication of the explicit eight-cell observation effect',
        'experimental_unit': 'One independently solved layout-by-grid gas field',
        'repeated_measurement': 'Routes sharing a field are correlated and are not independent replicates',
        'design_blocks': ['L1', 'L2'],
        'route_selection': selection_record['selection_rule'],
        'selection_uses_outcomes': False,
        'observation': parent_protocol['observation'],
        'all_supports_in_open_gas': True,
        'solver': parent_protocol['solver'],
        'limits': parent_protocol['limits'],
        'field_end_s': 70.0,
        'time_output_s': 1,
        'dose_denominator_floor_FED': 1e-8,
        'gap_denominator_floor_FED': 1e-8,
        'default_prefix_parity': parent_protocol['default_prefix_parity'],
        'descriptive_replication_rule': (
            'Both layout medians must show a smaller absolute coarse/fine gap, '
            'with at least 3/4 improved L1 routes and 2/2 improved L2 routes. '
            'This is a pre-specified descriptive rule, not an inferential test.'
        ),
        'fixture_evidence': {
            'source': 'outputs/post_safestep_stencil_20261019/fixture_validation.json',
            'source_sha256': sha(PREVIOUS / 'fixture_validation.json'),
            'passed': read(PREVIOUS / 'fixture_validation.json')['passed'],
            'rerun': False,
        },
        'old_mesh_failure_unchanged': True,
        'strict_prefix_failure_unchanged': True,
        'formal24_started': False,
        '900s_started': False,
        'new_RL_updates': 0,
        'LLM_requests': 0,
        'sealed_final48_accessed': False,
        'no_automatic_retries': True,
        'implementation_sha256': {rel: sha(ROOT / rel) for rel in implementation},
    }
    assert protocol['fixture_evidence']['passed']
    save(OUT / 'protocol_frozen.json', protocol)
    print({
        'prepared_jobs': len(jobs),
        'selected_routes': selection_record['selected_counts'],
        'observed_nodes': {layout: len({node for route in routes for node in route['route'][:-1]})
                           for layout, routes in selected.items()},
        'protected_files': len(protected),
    })


def fields():
    protocol = read(OUT / 'protocol_frozen.json')
    assert protocol['fixture_evidence']['passed']
    for job in read(OUT / 'jobs_frozen.json'):
        result = backend.run(job)
        if not result['normal_completion']:
            save(OUT / 'fields_stopped.json', {
                'failed_case': job['name'],
                'reason': 'No automatic retry or budget extension',
            })
            raise RuntimeError(f"Field failed: {job['name']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'fields'])
    action = parser.parse_args().action
    {'prepare': prepare, 'fields': fields}[action]()
