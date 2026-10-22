"""Stop a fine-grid screen on a completed-route witness of the frozen failure rule.

This is a necessary-condition early rejection, never an early acceptance. It
does not change the 10% threshold, select routes from results, or modify FDS.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.post_safestep_physics import fed_rate_per_s, integrate_rate, read_device_csv
OUT = ROOT / 'outputs/post_safestep_physics_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rate_table(data, node):
    return fed_rate_per_s(*(np.clip(data[f'N{node:03d}_{name}'], 0, 1) for name in ('CO', 'CO2', 'O2')))


def dose(data, item):
    total = 0.
    for step, node in enumerate(item['route'][:-1]):
        start = item['departure_s'] + step / .8
        total += integrate_rate(data['Time'], rate_table(data, node), start, start + 1/.8)
    return total


def freeze():
    for lid, vent in (('L1', 'V0'), ('L2', 'V1')):
        assert not (OUT / f'runs/{lid}_F2_G150_{vent}_dx0p125_adaptive/process_start.json').exists(), 'Freeze before any fine-grid result'
    record = {
        'frozen_UTC': datetime.now(timezone.utc).isoformat(),
        'parent_rule': 'mesh_screen_rules_frozen.json: ALL finite-dose diagnostic routes within10pct',
        'allowed_action': 'Early reject only after a prespecified route has fully completed in both grids and violates10pct; never early accept',
        'routes': 'Previously frozen40 diagnostic routes per layout, no new route selection',
        'threshold': .10, 'fine_dose_floor': 1e-8,
        'preserve_partial_outputs': True,
        'new_policy_results_seen': False,
        'meaning': 'One necessary condition already failed; partial trajectories cannot certify full180s output or any900s horizon',
    }
    with (OUT / 'mesh_early_rejection_rule.json').open('x', encoding='utf-8') as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    print(json.dumps({'early_rejection_rule_frozen': True}))


def watch():
    assert (OUT / 'mesh_early_rejection_rule.json').exists()
    began = time.monotonic()
    done = set()
    cached = {}
    while len(done) < 2 and time.monotonic() - began < 4200:
        for lid, vent in (('L1', 'V0'), ('L2', 'V1')):
            if lid in done:
                continue
            name = f'{lid}_F2_G150_{vent}_dx0p125_adaptive'
            case = OUT / 'runs' / name
            if (case / 'run_result.json').exists() or (case / 'mesh_failure_witness.json').exists():
                done.add(lid)
                continue
            if not (case / 'process_start.json').exists():
                continue
            baseline = OUT / f'runs/{lid}_F2_G150_{vent}_dx0p25_adaptive'
            if not (baseline / 'run_result.json').exists():
                continue
            if lid not in cached:
                cached[lid] = read_device_csv(baseline / (baseline.name + '_devc.csv'))[0]
            data_path = case / (name + '_devc.csv')
            if not data_path.is_file():
                continue
            try:
                fine = read_device_csv(data_path)[0]
            except (ValueError, OSError, StopIteration, IndexError):
                continue  # Ignore an in-progress append, never fabricate a row.
            if not len(fine['Time']):
                continue
            native_end = float(fine['Time'][-1])
            eligible = 0
            witness = None
            for item in read(OUT / 'diagnostic_routes_frozen.json')[lid]:
                end = item['departure_s'] + (len(item['route']) - 1) / .8
                if end > native_end or item['departure_s'] < fine['Time'][0]:
                    continue
                eligible += 1
                fine_dose = dose(fine, item)
                if fine_dose < 1e-8:
                    continue
                coarse_dose = dose(cached[lid], item)
                relative = abs(coarse_dose - fine_dose) / fine_dose
                if relative > .10:
                    witness = {'case': name, 'layout': lid, 'route': item,
                        'route_end_s': end, 'native_last_sample_s': native_end,
                        'candidate_dx0p25_FED': coarse_dose, 'reference_dx0p125_FED': fine_dose,
                        'relative_difference': relative, 'threshold': .10,
                        'eligible_routes_checked_up_to_witness': eligible,
                        'status': 'EARLY_REJECT_COMPLETED_ROUTE_WITNESS',
                        'full180s_complete': False, 'physical_model_validated': False}
                    break
            if witness:
                identity = read(case / 'process_start.json')
                try:
                    process = psutil.Process(identity['pid'])
                    argv = process.cmdline()
                    assert process.name().lower() == 'fds_openmp.exe'
                    assert Path(process.cwd()).resolve() == case.resolve()
                    assert Path(argv[-1]).name == name + '.fds'
                    assert hashlib.sha256((case / argv[-1]).read_bytes()).hexdigest() == identity['input_sha256']
                    witness['terminated_identified_pid'] = identity['pid']
                    witness['recorded_UTC'] = datetime.now(timezone.utc).isoformat()
                    with (case / 'mesh_failure_witness.json').open('x', encoding='utf-8') as file:
                        json.dump(witness, file, ensure_ascii=False, indent=2)
                    process.terminate()
                    process.wait(timeout=15)
                    print(json.dumps({'event': 'mesh_rule_failed', 'case': name,
                        'route_end_s': end, 'relative_difference': relative,
                        'reference_FED': fine_dose}), flush=True)
                    done.add(lid)
                except psutil.NoSuchProcess:
                    pass
        time.sleep(10)
    print(json.dumps({'monitor_done': sorted(done)}), flush=True)


if __name__ == '__main__':
    freeze() if '--freeze' in sys.argv else watch()
