"""Read-only review of frozen 450 s outputs; no simulation or parameter fitting.

Independent stdlib trapezoid integration of linear native rate samples.
Component accounting is conditional on the frozen formula, not causal attribution.
"""
import bisect
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/safestep_continuous450_20261017'
OUT = ROOT / 'outputs/safestep450_cached_contribution_audit_20261017'


def read_csv(path, skip=0):
    with path.open(encoding='utf-8-sig', newline='') as f:
        for _ in range(skip):
            next(f)
        return list(csv.DictReader(f))


def integral(times, values, start, end):
    assert times[0] <= start <= end <= times[-1]
    def at(t):
        i = min(bisect.bisect_right(times, t) - 1, len(times) - 2)
        weight = (t - times[i]) / (times[i + 1] - times[i])
        return values[i] + weight * (values[i + 1] - values[i])
    knots = [start] + times[bisect.bisect_right(times, start):bisect.bisect_left(times, end)] + [end]
    return math.fsum((b-a) * (at(a)+at(b)) / 2 for a,b in zip(knots, knots[1:]))


def main():
    inputs = [SOURCE / name for name in (
        'native_gas_and_FER_long.csv', 'candidate_route_residence_integrals.csv',
        'archived_rate_same_residence_integrals.csv', 'analysis_results.json',
        'run/hotel_continuous450_hrr.csv')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    native = defaultdict(list)
    formula_error = 0.0
    for row in read_csv(inputs[0]):
        row = {k: v if k == 'region' else float(v) for k,v in row.items()}
        native[row['region']].append(row)
        co = 3.317e-5 * row['CO_ppm']**1.036 * 25/30 * math.exp(row['CO2_percent']/5)/60
        hcn = row['HCN_ppm']**2.36/1.2e6 * math.exp(row['CO2_percent']/5)/60
        formula_error = max(formula_error, abs(co-row['FER_CO_per_s']),
                            abs(hcn-row['FER_HCN_per_s']), abs(co+hcn-row['FER_total_per_s']))
    for rows in native.values():
        assert all(b['fire_time_s'] > a['fire_time_s'] for a,b in zip(rows,rows[1:]))
    archived = {(r['route'],r['movement_interval']): float(r['FED_increment']) for r in read_csv(inputs[2])}
    segments = []
    for seg in read_csv(inputs[1]):
        rows = native[seg['region']]
        times = [r['fire_time_s'] for r in rows]
        start, end = float(seg['start_fire_s']), float(seg['end_fire_s'])
        parts = {component: integral(times, [r[field] for r in rows], start, end)
                 for component,field in [('CO','FER_CO_per_s'),('HCN','FER_HCN_per_s'),('total','FER_total_per_s')]}
        assert abs(parts['total'] - float(seg['FED_increment'])) < 1e-10
        assert abs(parts['CO'] + parts['HCN'] - parts['total']) < 1e-10
        # Algebraic diagnostic only: set the formula's multiplier to one at native nodes.
        unamplified = integral(times, [r['FER_total_per_s']/r['CO2_multiplier'] for r in rows], start, end)
        old = archived[(seg['route'],seg['movement_interval'])]
        segments.append(dict(route=seg['route'], interval=int(seg['movement_interval']), region=seg['region'],
                             start_fire_s=start, end_fire_s=end, **parts,
                             archived=old, delta=parts['total']-old,
                             algebraic_without_CO2_multiplier=unamplified))
    routes = []
    result = json.loads(inputs[3].read_text(encoding='utf-8'))
    expected = {r['id']:r for r in result['candidate_routes']}
    for name in expected:
        group = [s for s in segments if s['route'] == name]
        sums = {k:math.fsum(s[k] for s in group) for k in
                ('CO','HCN','total','archived','delta','algebraic_without_CO2_multiplier')}
        assert abs(sums['total'] - expected[name]['FED']) < 1e-10
        assert abs(sums['archived'] - expected[name]['archived_same_primary_FED']) < 1e-10
        routes.append(dict(route=name, **sums, HCN_fraction=sums['HCN']/sums['total'],
                           effective_CO2_formula_multiplier=sums['total']/sums['algebraic_without_CO2_multiplier'],
                           top3_excess_segments=sorted(group, key=lambda s:s['delta'], reverse=True)[:3]))
    hrr = read_csv(inputs[4], skip=1)
    pressure = []
    for key in ('ZONE_1','ZONE_12'):
        peak = max(hrr, key=lambda r:float(r[key]))
        pressure.append(dict(zone=key, peak_gauge_Pa=float(peak[key]), time_s=float(peak['Time'])))
    assert formula_error < 1e-12
    for p in inputs:
        assert hashes[str(p.relative_to(ROOT))] == hashlib.sha256(p.read_bytes()).hexdigest()
    output = dict(status='OFFLINE_DESCRIPTIVE_ACCOUNTING_NOT_CAUSAL_ATTRIBUTION',
                  formula_max_abs_error=formula_error, segment_count=len(segments),
                  route_results=routes, pressure_peaks=pressure,
                  source_hashes_before_and_after_equal=hashes,
                  simulation_runs=0, training_updates=0,
                  caveats=['HCN/CO components both include the CO2 multiplier.',
                           'Component fractions do not allocate discrepancy causes.',
                           'Removing a multiplier algebraically is not a physical fire intervention.',
                           'No pre-300 s exposure, heat, visibility or oxygen component is added.',
                           'Final crowd scenarios were not read.'])
    OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'segment_components.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(segments[0]));writer.writeheader();writer.writerows(segments)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
    manifest['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
