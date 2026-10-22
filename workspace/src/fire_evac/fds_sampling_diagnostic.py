"""Bounded observation and quadrature diagnostics for frozen FDS routes."""
import math

import numpy as np

from .post_safestep_physics import fed_rate_per_s, integrate_rate


def _segments(data, route, speed, prefix):
    times = np.asarray(data['Time'], dtype=float)
    begin = float(route['departure_s'])
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or not np.isfinite(speed) or speed <= 0
            or not np.isfinite(begin) or len(route['route']) < 2):
        raise ValueError('Finite native times, nonempty route and positive speed required')
    end = begin + (len(route['route'])-1)/speed
    if begin < times[0] or end > times[-1]:
        raise ValueError('The complete moving route must have native time support')
    for step, node in enumerate(route['route'][:-1]):
        gas = [np.asarray(data[f'{prefix}{node:03d}_{s}'], dtype=float) for s in ('CO', 'CO2', 'O2')]
        if any(v.shape != times.shape for v in gas):
            raise ValueError('Gas and time arrays must have equal native shape')
        # Strict validation: do not conceal invalid gas units or values by clipping.
        fed_rate_per_s(*gas)
        yield times, begin+step/speed, begin+(step+1)/speed, node, gas


def route_breakdown(data, route, speed=.8, prefix='N'):
    """Reconstruct the unchanged rate-first, left-node route integral exactly.

    CO2_increment includes its interaction with CO. It is an additive accounting
    convention, not a causal allocation of independent gas effects.
    """
    segments = []
    totals = {'CO_base': 0., 'CO2_increment': 0., 'O2_deficit': 0.}
    for times, begin, end, node, (co, co2, o2) in _segments(data, route, speed, prefix):
        base = fed_rate_per_s(co, np.zeros_like(co2), np.full_like(o2, .209))
        full_co = fed_rate_per_s(co, co2, np.full_like(o2, .209))
        hypoxia = fed_rate_per_s(np.zeros_like(co), np.zeros_like(co2), o2)
        rates = {'CO_base': base, 'CO2_increment': full_co-base, 'O2_deficit': hypoxia}
        components = {key: integrate_rate(times, value, begin, end) for key, value in rates.items()}
        dose = integrate_rate(times, fed_rate_per_s(co, co2, o2), begin, end)
        for key, value in components.items():
            totals[key] += value
        segments.append({'node': int(node), 'begin_s': begin, 'end_s': end,
            'FED': dose, 'components': components})
    return {'FED': sum(s['FED'] for s in segments), 'components': totals, 'segments': segments}


def gas_first_dose(data, route, speed=.8, prefix='N', max_step_s=.025):
    """Diagnostic only: integrate FED(linearly interpolated native gases).

    Finer quadrature does not recover unrecorded subsecond FDS fluctuations.
    """
    if not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError('Positive quadrature spacing required')
    total = 0.
    for times, begin, end, _, gas in _segments(data, route, speed, prefix):
        n = max(1, math.ceil((end-begin)/max_step_s))
        knots = np.unique(np.r_[np.linspace(begin, end, n+1), times[(times > begin) & (times < end)]])
        values = [np.interp(knots, times, v) for v in gas]
        total += float(np.trapezoid(fed_rate_per_s(*values), knots))
    return total
