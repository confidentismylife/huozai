"""Audit known-mixture units, geometry and the prespecified 180 s mesh screen."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.post_safestep_physics import (
    fed_rate_per_s, integrate_rate, read_device_csv, solid_rectangles,
)
OUT = ROOT / 'outputs/post_safestep_physics_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    with path.open('x', encoding='utf-8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def units():
    case = OUT / 'runs/dose_units'
    assert read(case / 'run_result.json')['normal_completion']
    data, unit_names = read_device_csv(case / 'dose_units_devc.csv')
    rates = fed_rate_per_s(data['CO'], data['CO2'], data['O2'])
    t = data['Time']
    reference = data['FED'][-1] - data['FED'][0]
    calculated = integrate_rate(t, rates, t[0], t[-1])
    relative = abs(calculated-reference) / reference
    record = {'passed': bool(relative < 1e-7),
        'native_start_s': float(t[0]), 'native_end_s': float(t[-1]),
        'fixed_mixture_mol_per_mol': {name: float(data[name][0]) for name in ('CO', 'CO2', 'O2')},
        'solver_accumulated_FED': float(reference), 'offline_accumulated_FED': calculated,
        'relative_difference': float(relative), 'allowed_relative_difference': 1e-7,
        'rate_per_s': float(rates[0]), 'native_units': unit_names,
        'interpretation': 'Independent solver versus offline CO/CO2/O2 rate and seconds check; fabricated fixed mixture, not physical fire validation'}
    save(OUT / 'dose_units_verification.json', record)
    print(json.dumps(record))
    assert record['passed']


def shortest(graph, start, end):
    prev, queue = {start: None}, deque([start])
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in graph['adjacency'][cur]:
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def geometry():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plots = []
    routes = {}
    for layout_id in ('L1', 'L2'):
        record = read(OUT / (layout_id + '_graph.json'))
        layout, graph = record['layout'], record['graph']
        fig, ax = plt.subplots(figsize=(12, 6))
        w, h, _ = layout['domain_xyz']
        for a, b, c, d in solid_rectangles(layout):
            ax.add_patch(Rectangle((a, c), b-a, d-c, facecolor='#CBD1D5', edgecolor='#64747E', linewidth=.6))
        xy = np.asarray(graph['coordinates_xy'])
        for i, neighbors in enumerate(graph['adjacency']):
            for j in neighbors:
                if i < j:
                    ax.plot(xy[[i, j], 0], xy[[i, j], 1], color='#C7DADC', linewidth=.65, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=6, color='#4B8C93', label='1米路径节点', zorder=2)
        starts = np.asarray(layout['start_centers_xy'])
        ax.scatter(starts[:, 0], starts[:, 1], marker='o', s=54, color='#1B5374', label='10个固定起点', zorder=4)
        for i, (x, y) in enumerate(starts, 1):
            ax.text(x+.12, y+.17, str(i), fontsize=8)
        fx, fy = layout['fire_centers_xy']['F2']
        ax.add_patch(Rectangle((fx-.5, fy-.5), 1, 1, facecolor='#C75B18', label='1平方米火源及禁行区', zorder=5))
        exits = np.asarray(layout['exit_centers_xy'])
        ax.scatter(exits[:, 0], exits[:, 1], s=90, marker='s', color='#007E63', label='吸收出口', zorder=5)
        if layout_id == 'L2':
            wx, wy = layout['extra_window_xy_plane']
            ax.plot([wx-1, wx+1], [wy, wy], color='#9E45A1', linewidth=5, label='V1额外开窗')
        ax.set(xlim=(-.6, w+.6), ylim=(-.6, h+.6), xlabel='x / 米', ylabel='y / 米', aspect='equal',
            title=f'{layout_id}代表场景：F2，' + ('V0' if layout_id == 'L1' else 'V1') + '；路径图与实际输入共用几何')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, fontsize=9)
        fig.tight_layout()
        target = OUT / (layout_id + '_geometry_and_graph.png')
        fig.savefig(target, dpi=170, bbox_inches='tight')
        plt.close(fig)
        plots.append(str(target.relative_to(ROOT)))
        cases = []
        for start in graph['start_indices']:
            for exit_index in graph['exit_indices']:
                route = shortest(graph, start, exit_index)
                for departure in (60., 120.):
                    cases.append({'start_index': start, 'exit_index': exit_index,
                        'departure_s': departure, 'route': route})
        assert all(item['departure_s'] + (len(item['route'])-1)/.8 <= 180 for item in cases)
        routes[layout_id] = cases
    save(OUT / 'diagnostic_routes_frozen.json', routes)
    save(OUT / 'mesh_screen_rules_frozen.json', {
        'scope': 'Two representative fields only, 0..180 s, no control effect scoring',
        'pair': '0.25 m versus 0.125 m is the candidate-primary comparison; 0.5 m is a diagnostic',
        'route_selection': 'Shortest legal graph route to each exit, 10 starts, departures 60 and 120 s',
        'relative_route_difference_max_target': .1,
        'route_denominator_floor_FED': 1e-8,
        'field_errors': 'Report native curves and normalized mean differences; not a substitute for route and local-error checks',
        'overall_pass': 'All runs normal and complete; no gas invalidity; HRR realized; all finite-dose routes within10% for both layouts. Near-zero routes are disclosed separately.',
        'full_matrix_gate': 'Passing180s alone is not sufficient to validate the planned900s horizon; extend representative duration before formal effects.'})
    save(OUT / 'geometry_preflight.json', {'passed_abstract_and_parser': True,
        'FDS_setup_results': {k: read(OUT / f'runs/{k}_init/run_result.json')['normal_completion'] for k in ('L1', 'L2')},
        'plots': plots, 'route_count_per_layout': 40,
        'physical_validation': False})
    print(json.dumps({'geometry_plots': plots, 'fixed_route_scores_per_layout_mesh': 40}))


def audit_case(name):
    folder = OUT / 'runs' / name
    result = read(folder / 'run_result.json')
    data, units_map = read_device_csv(folder / (name + '_devc.csv'))
    hrr, _ = read_device_csv(folder / (name + '_hrr.csv'))
    t = data['Time']
    ids = sorted(int(re.match(r'N(\d+)_CO$', n)[1]) for n in data if re.match(r'N(\d+)_CO$', n))
    for i in ids:
        for suffix in ('CO', 'CO2', 'O2'):
            a = data[f'N{i:03d}_{suffix}']
            assert np.isfinite(a).all() and a.min() >= -1e-12 and a.max() <= 1 + 1e-12
    rates = {i: fed_rate_per_s(*(np.clip(data[f'N{i:03d}_{key}'], 0, 1) for key in ('CO', 'CO2', 'O2'))) for i in ids}
    job = read(folder / 'input_manifest.json')
    record = {'case': name, 'normal': result['normal_completion'], 'end_s': float(t[-1]),
        'native_samples': len(t), 'native_max_gap_s': float(np.diff(t).max()),
        'sensor_nodes': len(ids), 'max_CO_ppm': max(float(data[f'N{i:03d}_CO'].max()) * 1e6 for i in ids),
        'max_temperature_C': max(float(data[f'N{i:03d}_T'].max()) for i in ids),
        'max_HRR_kW': float(hrr['HRR'].max()), 'min_O2_mol_per_mol': min(float(data[f'N{i:03d}_O2'].min()) for i in ids),
        'elapsed_wall_s': result['elapsed_wall_s'], 'peak_rss_MiB': result['peak_rss_MiB'],
        'finite_values': all(bool(np.isfinite(v).all()) for v in data.values()),
        'all_native_gas_units_mol_per_mol': all(units_map[f'N{i:03d}_{k}'].strip() == 'mol/mol' for i in ids for k in ('CO','CO2','O2'))}
    record['complete_time_coverage'] = bool(t[0] <= 1e-9 and t[-1] >= job['end_s'] - 1.0 and np.diff(t).max() <= 1.6)
    record['output_time_contract'] = 'Actual instantaneous sample times retained; requested1s cadence, not fabricated uniform times'
    record['direct_solver_point_FED_checks'] = []
    for name in data:
        match = re.match(r'N(\d+)_FED$', name)
        if not match:
            continue
        index = int(match[1])
        reference = float(data[name][-1] - data[name][0])
        offline = integrate_rate(t, rates[index], t[0], t[-1])
        record['direct_solver_point_FED_checks'].append({'node': index,
            'FDS_cumulative_dose': reference, 'native_node_rate_integral': offline,
            'absolute_difference': abs(reference-offline),
            'relative_difference': abs(reference-offline)/reference if reference >= 1e-8 else None,
            'interpretation': 'Time-sampling approximation check, distinct from the exact fixed-mixture unit check'})
    return record, data, rates


def mesh():
    records, route_scores, witnesses = [], {}, []
    for layout_id, vent in (('L1', 'V0'), ('L2', 'V1')):
        routes = read(OUT / 'diagnostic_routes_frozen.json')[layout_id]
        for dx in (.5, .25, .125):
            name = f'{layout_id}_F2_G150_{vent}_dx{str(dx).replace(".", "p")}_adaptive'
            record, data, rates = audit_case(name)
            witness_file = OUT / 'runs' / name / 'mesh_failure_witness.json'
            if witness_file.is_file():
                witness = read(witness_file)
                witnesses.append(witness)
                record['status'] = 'EARLY_REJECT_MESH_RULE_NOT_NUMERICAL_INSTABILITY'
            else:
                record['status'] = 'COMPLETE' if record['normal'] else 'INCOMPLETE'
            records.append(record)
            scores = []
            for item in routes:
                route_end = item['departure_s'] + (len(item['route']) - 1) / .8
                if route_end > data['Time'][-1]:
                    scores.append(None)
                    continue
                dose = 0.
                for step, node in enumerate(item['route'][:-1]):
                    begin = item['departure_s'] + step / .8
                    dose += integrate_rate(data['Time'], rates[node], begin, begin + 1/.8)
                scores.append(dose)
            route_scores[layout_id + '_' + str(dx)] = scores
    comparisons = []
    for layout_id in ('L1', 'L2'):
        fine = np.asarray([np.nan if x is None else x for x in route_scores[layout_id + '_0.125']])
        for dx in (.5, .25):
            other = np.asarray([np.nan if x is None else x for x in route_scores[layout_id + '_' + str(dx)]])
            eligible = np.isfinite(fine) & np.isfinite(other)
            valid = eligible & (fine >= 1e-8)
            relative = np.abs(other[valid] - fine[valid]) / fine[valid]
            comparisons.append({'layout': layout_id, 'dx_m': dx, 'reference_dx_m': .125,
                'planned_routes': len(fine), 'completed_routes': int(eligible.sum()),
                'unscored_routes': int((~eligible).sum()), 'finite_dose_routes': int(valid.sum()),
                'near_zero_routes': int((eligible & ~valid).sum()),
                'max_absolute_dose_difference': float(np.max(np.abs(other[eligible]-fine[eligible]))) if eligible.any() else None,
                'maximum_relative_difference': float(relative.max()) if len(relative) else None,
                'median_relative_difference': float(np.median(relative)) if len(relative) else None,
                'routes_over_10pct': int((relative > .1).sum()),
                'passes_10pct_route_screen': bool(eligible.all() and len(relative) and np.all(relative <= .1))})
    passed = all(r['normal'] and r['complete_time_coverage'] and r['finite_values'] and r['all_native_gas_units_mol_per_mol'] for r in records)
    passed = passed and all(c['passes_10pct_route_screen'] for c in comparisons if c['dx_m'] == .25)
    incomplete = [r for r in records if not r['normal'] or not r['complete_time_coverage']]
    budget_only = bool(incomplete) and all(
        read(OUT/'runs'/r['case']/'run_result.json')['termination_cause'] == 'WALL_TIMEOUT'
        and not read(OUT/'runs'/r['case']/'run_result.json')['FDS_errors']
        for r in incomplete)
    status = ('REJECTED_BY_COMPLETED_ROUTE_WITNESS' if witnesses else
        ('PASS' if passed else
         ('INCOMPLETE_RUNTIME_BUDGET' if budget_only else 'FAILED_OR_INCOMPLETE')))
    save(OUT / 'mesh_screen_results.json', {'passed_180s_screen': passed,
        'status': status,
        'early_failure_witnesses': witnesses, 'cases': records,
        'comparisons': comparisons, 'diagnostic_route_scores': route_scores,
        'full_900s_physical_inputs_accepted': False, 'formal_24_matrix_started': False})
    print(json.dumps({'passed_180s_screen': passed, 'comparisons': comparisons}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['units', 'geometry', 'mesh'])
    args = parser.parse_args()
    {'units': units, 'geometry': geometry, 'mesh': mesh}[args.mode]()
