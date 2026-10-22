"""Prepare traceable inputs without starting FDS or scoring any method."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.post_safestep_physics import build_graph, case_text, static_mixture_fixture

OUT = ROOT / 'outputs/post_safestep_physics_20261019'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def prepare():
    protocol_path = ROOT / 'schemas/post_safestep_physical_design_20261019.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    runroot = OUT / 'runs'
    runroot.mkdir(exist_ok=False)
    executable = Path('C:/Program Files/firemodels/FDS6/bin/fds_openmp.exe')
    assert sha(executable) == 'dfaf04f1d46dbad77969b1cdd17baaa02a7aa6d63f46e96e93b9d9e635452b55'
    sources = []
    urls = {
        'FDS_User_Guide_6.11.1.tex': 'Manuals/FDS_User_Guide/FDS_User_Guide.tex',
        'func_6.11.1.f90': 'Source/func.f90', 'cons_6.11.1.f90': 'Source/cons.f90',
        'ATF_Corridors_240_kW.fds': 'Validation/ATF_Corridors/FDS_Input_Files/ATF_Corridors_240_kW.fds',
        'FED_CO_HCN.fds': 'Verification/Species/FED_CO_HCN.fds',
        'FED_FIC_SMIX.fds': 'Verification/Species/FED_FIC_SMIX.fds',
    }
    for name, source in urls.items():
        path = OUT / 'sources' / name
        assert path.is_file()
        sources.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'url': 'https://raw.githubusercontent.com/firemodels/fds/FDS-6.11.1/' + source})
    params = {
        'status': 'FROZEN_BEFORE_NEW_FDS_OR_POLICY_RESULTS',
        'parent_protocol': str(protocol_path.relative_to(ROOT)), 'parent_sha256': sha(protocol_path),
        'source_evidence': sources,
        'solver': {'executable': str(executable), 'sha256': sha(executable),
            'expected_version': 'FDS-6.11.1-0-gff928db-release',
            'threads': 4, 'mpi_ranks': 1},
        'fuel': {'name': 'N-HEPTANE', 'C': 7, 'H': 16,
            'CO_yield_kg_per_kg': .008, 'soot_yield_kg_per_kg': .015,
            'HCN_yield_kg_per_kg': 0., 'HCN_modeling': 'not included; N-free fuel in the chosen simple chemistry',
            'heat_of_combustion': 'Built-in heptane species thermochemistry and specified yields; record computed solver value before formal use',
            'radiative_fraction': .40,
            'source': 'WTC_05 example in fixed User Guide for CO/soot; heptane radiation fraction table',
            'applicability': 'Transferred surrogate assumptions, not measured yields for a real building or new independent validation'},
        'walls': {'all_solid_faces': 'GYPSUM', 'conductivity_W_per_m_K': .17,
            'specific_heat_kJ_per_kg_K': 1.09, 'density_kg_per_m3': 650,
            'thickness_m': .013, 'backing': 'INSULATED',
            'source': 'ATF_Corridors_240_kW material values; all-face lining and insulated backing are our simplifications'},
        'ambient': {'temperature_C': 20., 'relative_humidity_percent': 40., 'source': 'design assumptions'},
        'gas_to_rate': {'identifier': 'FDS_6_11_1_CO_CO2_O2_activity2_per_second',
            'input_basis': 'wet VOLUME FRACTION mol/mol', 'CO_to_ppm': 1e6,
            'CO2_O2_to_volume_percent': 100., 'minute_to_second_divisor': 60.,
            'CO_coefficient_per_minute': 2.7641667e-5, 'CO_exponent': 1.036,
            'HV': 'exp(0.1903*CO2_vol_percent+2.0004)/7.1 when CO2>0, otherwise 1',
            'hypoxia': 'exp(-8.13+0.54*(20.9-O2_vol_percent)) per minute, ONLY when O2 mol/mol<0.20',
            'activity': 'light work (2)', 'other_species': 'HCN, NOx and irritants excluded by this chosen chemistry; no universal hazard claim',
            'integration': 'Compute rate at native concentration nodes, piecewise-linear rate integration over actual residence intervals; no extrapolation',
            'relation_to_SafeStep': 'A different explicitly defined gas model; no absolute-FED equality assumed'},
        'pre_scoring_geometry_amendments': ['Navigation excludes the 1 m2 burner footprint; shared by all future methods',
            'All solid faces have the same pinned gypsum lining and insulated backing',
            'Vent and burner coordinates share the common 0.5 m geometry lattice'],
        'runtime_limits': {'case_wall_timeout_s': 1200, 'no_progress_timeout_s': 240,
            'maximum_resident_memory_GiB': 4., 'one_solver_process_at_a_time': True,
            'mesh_screen_end_s': 180., 'mesh_screen_scope': '0..180 s only; not a validation of the entire planned 900 s horizon'},
        'no_policy_scoring_before_physics_checks': True, 'sealed_final48_accessed': False,
    }
    jobs = []
    def job(name, text, meta):
        folder = runroot / name
        folder.mkdir()
        p = folder / (name + '.fds')
        p.write_text(text, encoding='utf-8', newline='\n')
        info = {'name': name, 'input_path': str(p.relative_to(ROOT)), 'input_sha256': sha(p), **meta}
        save(folder / 'input_manifest.json', info)
        jobs.append(info)
    job('dose_units', static_mixture_fixture('dose_units'), {'kind': 'dose_verification', 'end_s': 20.})
    graphs = {}
    for layout_id, vent in (('L1', 'V0'), ('L2', 'V1')):
        layout = protocol['geometry_design_assumptions'][layout_id]
        graph = build_graph(layout, 'F2')
        graphs[layout_id] = {'layout': layout, 'graph': graph}
        save(OUT / (layout_id + '_graph.json'), graphs[layout_id])
        for spacing in (.5, .25, .125):
            token = str(spacing).replace('.', 'p')
            name = f'{layout_id}_F2_G150_{vent}_dx{token}'
            text = case_text(name, layout, 'F2', vent, spacing, 180, graph['coordinates_xy'])
            job(name, text, {'kind': 'mesh_screen', 'layout_id': layout_id, 'ventilation_id': vent,
                'mesh_spacing_m': spacing, 'end_s': 180., 'prescribed_peak_HRR_kW': 1000.})
        name = layout_id + '_init'
        job(name, case_text(name, layout, 'F2', vent, .5, 0, graph['coordinates_xy'], setup_only=True),
            {'kind': 'setup', 'layout_id': layout_id, 'end_s': 0.})
    save(OUT / 'physical_parameters_frozen.json', params)
    save(OUT / 'planned_jobs.json', jobs)
    print(json.dumps({'prepared_jobs': len(jobs), 'source_files': len(sources),
        'graph_nodes': {k: len(v['graph']['coordinates_xy']) for k, v in graphs.items()}, 'FDS_runs': 0}))


if __name__ == '__main__':
    prepare()
