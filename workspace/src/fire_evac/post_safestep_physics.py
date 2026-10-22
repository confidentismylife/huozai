"""Source-pinned new FDS cases and explicitly unit-labelled gas dose integration.

Chemistry and wall values are transferred modeling assumptions, not measured
parameters for either layout. FED rates follow FDS 6.11.1 func.f90:FUNCTION FED
for CO, CO2 hyperventilation and O2 only, activity level 2. No HCN is introduced
into the nitrogen-free heptane fuel.
"""
from __future__ import annotations

from collections import deque
import csv
import math
from pathlib import Path

import numpy as np


def fed_rate_per_s(co_mol_per_mol, co2_mol_per_mol, o2_mol_per_mol):
    co, co2, o2 = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in
        (co_mol_per_mol, co2_mol_per_mol, o2_mol_per_mol)])
    for value in (co, co2, o2):
        if not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 1):
            raise ValueError('Expected finite wet mole fractions in mol/mol, not ppm or percent')
    co_ppm = co * 1e6
    co2_volume_percent = co2 * 100
    o2_volume_percent = o2 * 100
    co_rate_per_min = 2.7641667e-5 * co_ppm ** 1.036
    hv = np.where(co2 > 0, np.exp(.1903 * co2_volume_percent + 2.0004) / 7.1, 1.)
    hypoxia_per_min = np.where(o2 < .20, np.exp(-8.13 + .54 * (20.9 - o2_volume_percent)), 0.)
    return (co_rate_per_min * hv + hypoxia_per_min) / 60.


def integrate_rate(times_s, rates_per_s, start_s, end_s):
    t, r = np.asarray(times_s, float), np.asarray(rates_per_s, float)
    if t.ndim != 1 or r.shape != t.shape or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Strictly increasing one-dimensional native samples required')
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError('Rate must be finite and nonnegative')
    if start_s < t[0] - 1e-9 or end_s > t[-1] + 1e-9 or end_s < start_s:
        raise ValueError('No extrapolation permitted')
    knots = np.r_[start_s, t[(t > start_s) & (t < end_s)], end_s]
    return float(np.trapezoid(np.interp(knots, t, r), knots))


def rectangle_union_contains(x, y, rectangles):
    return any(a <= x <= b and c <= y <= d for a, b, c, d in rectangles)


def solid_rectangles(layout):
    """Represent the complement on a common 0.5 m grid; merge identical rows."""
    w, h, _ = layout['domain_xyz']
    rows = []
    for iy in range(round(h * 2)):
        blocked = [not rectangle_union_contains((ix + .5) / 2, (iy + .5) / 2,
                    layout['open_rectangles_xy']) for ix in range(round(w * 2))]
        spans, ix = [], 0
        while ix < len(blocked):
            if not blocked[ix]:
                ix += 1
                continue
            begin = ix
            while ix < len(blocked) and blocked[ix]:
                ix += 1
            spans.append((begin / 2, ix / 2))
        rows.append(spans)
    active, result = {}, []
    for iy in range(len(rows) + 1):
        spans = set(rows[iy]) if iy < len(rows) else set()
        for span in list(active):
            if span not in spans:
                start = active.pop(span)
                result.append([span[0], span[1], start / 2, iy / 2])
        for span in spans:
            active.setdefault(span, iy)
    return sorted(result)


def build_graph(layout, fire_position=None):
    w, h, _ = layout['domain_xyz']
    rectangles = layout['open_rectangles_xy']
    burner = layout['fire_centers_xy'][fire_position] if fire_position else None
    def clear(x, y):
        probes = [(x + dx, y + dy) for dx, dy in ((0, 0), (.1, 0), (-.1, 0), (0, .1), (0, -.1))]
        return all(rectangle_union_contains(a, b, rectangles) for a, b in probes)
    nodes = [(x, y) for y in range(1, round(h)) for x in range(1, round(w)) if clear(x, y)]
    if burner is not None:
        nodes = [(x, y) for x, y in nodes
                 if not (abs(x - burner[0]) <= .50000001 and abs(y - burner[1]) <= .50000001)]
    exits = [tuple(p) for p in layout['exit_centers_xy']]
    nodes += exits
    lookup = {p: i for i, p in enumerate(nodes)}
    adjacency = [[] for _ in nodes]
    for i, (x, y) in enumerate(nodes):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (x + dx, y + dy)
            if q in lookup and rectangle_union_contains(x + dx / 2, y + dy / 2, rectangles):
                adjacency[i].append(lookup[q])
    starts = [lookup[tuple(p)] for p in layout['start_centers_xy']]
    reachable = []
    for end in exits:
        seen, queue = {lookup[end]}, deque([lookup[end]])
        while queue:
            i = queue.popleft()
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        reachable.append(all(i in seen for i in starts))
    if not all(reachable):
        raise ValueError('Some prescribed starts cannot reach both exits')
    return {'coordinates_xy': [list(p) for p in nodes], 'adjacency': adjacency,
            'start_indices': starts, 'exit_indices': [lookup[p] for p in exits],
            'all_starts_reach_both_exits': all(reachable),
            'burner_patch_excluded_from_navigation': burner is not None}


def case_text(chid, layout, fire_position, ventilation, spacing_m, end_s, coordinates,
              time_to_peak_s=150, setup_only=False):
    w, h, z = layout['domain_xyz']
    fx, fy = layout['fire_centers_xy'][fire_position]
    lines = [
        f"&HEAD CHID='{chid}', TITLE='New prescribed heptane case, not SafeStep reconstruction' /",
        f"&MESH IJK={round(w/spacing_m)},{round(h/spacing_m)},{round(z/spacing_m)}, XB=0,{w},0,{h},0,{z} /",
        f"&TIME T_END={0 if setup_only else end_s}, DT=0.01 /",
        "&MISC TMPA=20., HUMIDITY=40., OVERWRITE=.FALSE. /",
        "&DUMP DT_DEVC=1., DT_HRR=1., SIG_FIGS=12, DT_RESTART=100000. /",
        "! Yields from official WTC example, transferred modeling assumptions only.",
        "&REAC FUEL='N-HEPTANE', C=7., H=16., CO_YIELD=0.008, SOOT_YIELD=0.015, RADIATIVE_FRACTION=0.40 /",
        "! Wall data transferred from official ATF corridor example. All solid faces use this lining.",
        "&MATL ID='GYPSUM', CONDUCTIVITY=0.17, SPECIFIC_HEAT=1.09, DENSITY=650. /",
        "&SURF ID='WALL', DEFAULT=.TRUE., MATL_ID='GYPSUM', THICKNESS=0.013, BACKING='INSULATED' /",
        f"&SURF ID='BURNER', HRRPUA=1000., TAU_Q=-{time_to_peak_s}, COLOR='RED' /",
        f"&VENT ID='FIRE', XB={fx-.5:g},{fx+.5:g},{fy-.5:g},{fy+.5:g},0,0, SURF_ID='BURNER' /",
        "&PROP ID='LIGHT_WORK', FED_ACTIVITY=2 /",
    ]
    for index, (a, b, c, d) in enumerate(solid_rectangles(layout)):
        lines.append(f"&OBST ID='SOLID_{index:03d}', XB={a:g},{b:g},{c:g},{d:g},0,{z}, SURF_ID='WALL' /")
    for index, (ex, ey) in enumerate(layout['exit_centers_xy']):
        if ex not in (0, w):
            raise ValueError('Exit must be on an x boundary')
        lines.append(f"&VENT ID='EXIT_{index}', XB={ex:g},{ex:g},{ey-1:g},{ey+1:g},0,2, SURF_ID='OPEN' /")
    if ventilation == 'V1':
        wx, wy = layout['extra_window_xy_plane']
        lines.append(f"&VENT ID='WINDOW', XB={wx-1:g},{wx+1:g},{wy:g},{wy:g},1.5,2.5, SURF_ID='OPEN' /")
    for index, (x, y) in enumerate(coordinates):
        if x <= 0 or x >= w or y <= 0 or y >= h:
            continue
        for label, species in (('CO', 'CARBON MONOXIDE'), ('CO2', 'CARBON DIOXIDE'), ('O2', 'OXYGEN')):
            lines.append(f"&DEVC ID='N{index:03d}_{label}', XYZ={x:g},{y:g},1.5, QUANTITY='VOLUME FRACTION', SPEC_ID='{species}', TIME_AVERAGED=.FALSE. /")
        for label, quantity in (('T', 'TEMPERATURE'), ('K', 'EXTINCTION COEFFICIENT')):
            lines.append(f"&DEVC ID='N{index:03d}_{label}', XYZ={x:g},{y:g},1.5, QUANTITY='{quantity}', TIME_AVERAGED=.FALSE. /")
    # Independent direct solver cumulative FED at a small fixed probe subset.
    for index, (x, y) in list(enumerate(coordinates))[:4]:
        lines.append(f"&DEVC ID='N{index:03d}_FED', XYZ={x:g},{y:g},1.5, QUANTITY='FED', PROP_ID='LIGHT_WORK', TIME_AVERAGED=.FALSE. /")
    lines.append('&TAIL /')
    return '\n'.join(lines) + '\n'


def static_mixture_fixture(chid, end_s=20):
    # Exact mass fractions in the official FED_FIC_SMIX verification input.
    return f"""&HEAD CHID='{chid}', TITLE='Official-mixture CO CO2 O2 dose-unit fixture' /
&MESH IJK=5,5,5, XB=0,1,0,1,0,1 /
&TIME T_END={end_s}, DT=0.05 /
&MISC TMPA=20., HUMIDITY=0., OVERWRITE=.FALSE. /
&DUMP DT_DEVC=0.1, DT_HRR=0.1, SIG_FIGS=12 /
&SPEC ID='OXYGEN', LUMPED_COMPONENT_ONLY=.TRUE. /
&SPEC ID='CARBON DIOXIDE', LUMPED_COMPONENT_ONLY=.TRUE. /
&SPEC ID='CARBON MONOXIDE', LUMPED_COMPONENT_ONLY=.TRUE. /
&SPEC ID='NITROGEN', LUMPED_COMPONENT_ONLY=.TRUE. /
&SPEC ID='O2_CO2_CO', BACKGROUND=.TRUE., SPEC_ID='OXYGEN','CARBON DIOXIDE','CARBON MONOXIDE','NITROGEN', MASS_FRACTION=0.10806110,0.05215865,0.00313658,0.83664366 /
&PROP ID='LIGHT_WORK', FED_ACTIVITY=2 /
&DEVC ID='CO', XYZ=0.5,0.5,0.5, QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE', TIME_AVERAGED=.FALSE. /
&DEVC ID='CO2', XYZ=0.5,0.5,0.5, QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON DIOXIDE', TIME_AVERAGED=.FALSE. /
&DEVC ID='O2', XYZ=0.5,0.5,0.5, QUANTITY='VOLUME FRACTION', SPEC_ID='OXYGEN', TIME_AVERAGED=.FALSE. /
&DEVC ID='FED', XYZ=0.5,0.5,0.5, QUANTITY='FED', PROP_ID='LIGHT_WORK', TIME_AVERAGED=.FALSE. /
&TAIL /
"""


def read_device_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        units = next(reader)
        names = [item.strip().strip('"') for item in next(reader)]
        records = [[float(item) for item in row] for row in reader if row]
    values = np.array(records, dtype=float)
    return {name: values[:, i] for i, name in enumerate(names)}, dict(zip(names, units))
