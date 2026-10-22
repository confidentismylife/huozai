"""Causal aggregate response features and development-only paired-gain ridge.

No simulator individual, behavior identity, checkpoint identity or outcome is
read by this module. History contains completed public transitions only.
"""
from collections import Counter
import math
import numpy as np

VERSION = 'crowd-response-gain-v1'
VARIANTS = ('current', 'public_history', 'response')
POPULATION = 240
ENTRY_CEILING = 12
TOLERANCE = .001


def public_projection(s):
    names = ('step', 'occupancy_fraction', 'smoke_increment_per_step',
             'exit_capacity_persons_per_step', 'exit_closed', 'forecast_source',
             'forecast_growth', 'forecast_onset_step', 'forecast_reopen_steps',
             'actual_sign', 'sign_hold_remaining_steps')
    return {k: s[k] for k in names}


def context():
    return dict(history=[], offers=0, overrides=0)


def observe_completed(ctx, public, action, detail, sensors):
    """Append after the action finishes; never accept FED or hidden attributes."""
    h = ctx['history']
    assert public['step'] == len(h)
    r = dict(before=public_projection(public), action=int(action),
             sign=int(sensors['actual_sign']), release_cap=int(sensors['actual_room_capacity']),
             released=int(sensors['actual_room_count']),
             counts_after=list(map(int, sensors['zone_counts_after'])),
             entries=list(map(float, sensors['branch_entries'])))
    assert len(r['entries']) == 2 and len(r['counts_after']) == 5
    assert 0 <= r['released'] <= r['release_cap'] and all(x >= 0 for x in r['entries'])
    h.append(r)
    ctx['offers'] += int(detail['eligible'])
    ctx['overrides'] += int(action != detail['base_action'])


def feature_names(variant):
    assert variant in VARIANTS
    names = [f'obs_{i}' for i in range(34)] + [f'base_{a}' for a in range(9)] + [f'candidate_{a}' for a in range(9)]
    names += ['past_offers', 'past_overrides']
    if variant == 'current': return names
    names += ['past_steps', 'no_history']
    names += [f'occupancy_change_{i}' for i in range(5)]
    names += [f'last3_mean_occupancy_{i}' for i in range(5)]
    names += ['past_north_sign_fraction', 'past_south_sign_fraction', 'last3_release_mean',
              'last3_release_cap_mean', 'last3_sign_switch_fraction']
    if variant == 'public_history': return names
    names += ['last3_north_entries', 'last3_south_entries', 'no_junction_flow',
              'past_north_entries', 'past_south_entries', 'past_sign_follow_share',
              'no_signed_flow', 'last3_sign_follow_share', 'last3_entry_ceiling_fraction',
              'last3_junction_clearance', 'last3_release_use', 'last3_exit_available',
              'last3_north_exits', 'last3_south_exits', 'last3_north_exit_use',
              'last3_south_exit_use', 'public_fed_upper_bound']
    return names


def features(observation, public, base, candidate, ctx, variant):
    """Read only decision-time sensors and the strictly earlier history."""
    x = list(map(float, observation))
    assert len(x) == 34 and np.isfinite(x).all()
    x[28:33] = [0.] * 5
    x += [float(base == a) for a in range(9)] + [float(candidate == a) for a in range(9)]
    x += [float(ctx['offers']), float(ctx['overrides'])]
    h = ctx['history']
    assert int(public['step']) == len(h)
    assert all(r['before']['step'] == i for i, r in enumerate(h))
    if variant != 'current':
        w = h[-3:]; n = len(w)
        before = h[-1]['before']['occupancy_fraction'] if h else public['occupancy_fraction']
        x += [len(h) / 120., float(not h)]
        x += [float(public['occupancy_fraction'][i]) - float(before[i]) for i in range(5)]
        x += [sum(r['before']['occupancy_fraction'][i] for r in w) / max(n, 1) for i in range(5)]
        x += [sum(r['sign'] == b for r in h) / max(len(h), 1) for b in range(2)]
        x += [sum(r['released'] for r in w) / max(n, 1) / 20.,
              sum(r['release_cap'] for r in w) / max(n, 1) / 20.,
              sum(r['before']['actual_sign'] != r['sign'] for r in w) / max(n, 1)]
        if variant == 'response':
            total = [sum(r['entries'][b] for r in h) for b in range(2)]
            recent = [sum(r['entries'][b] for r in w) for b in range(2)]
            signed = [r for r in h if r['sign'] in (0, 1)]
            signed_w = [r for r in w if r['sign'] in (0, 1)]
            def follows(seq):
                den = sum(sum(r['entries']) for r in seq)
                return sum(r['entries'][r['sign']] for r in seq) / den if den else 0.
            x += [v / max(n, 1) / ENTRY_CEILING for v in recent] + [float(sum(recent) == 0)]
            x += [v / POPULATION for v in total] + [follows(signed), float(not sum(sum(r['entries']) for r in signed)), follows(signed_w)]
            x += [sum(any(v >= ENTRY_CEILING for v in r['entries']) for r in w) / max(n, 1)]
            # Departed junction fraction is capacity-confounded, not compliance.
            denom = sum(round(r['before']['occupancy_fraction'][2] * POPULATION) for r in w)
            x += [sum(recent) / max(denom, 1), sum(r['released'] for r in w) / max(sum(r['release_cap'] for r in w), 1)]
            # Counts close only when public smoke bounds prove FED<1 for everyone.
            # This excludes hidden incapacitation counts from the inference.
            bound = 0.; exits = []; capacities = []
            start = len(h) - len(w)
            for i, r in enumerate(h):
                bound += max(r['before']['smoke_increment_per_step'])
                if i >= start and bound < 1.:
                    values = [round(r['before']['occupancy_fraction'][3+b] * POPULATION) + r['entries'][b] - r['counts_after'][3+b] for b in range(2)]
                    assert all(-1e-9 <= v <= r['before']['exit_capacity_persons_per_step'][b] + 1e-9 for b, v in enumerate(values))
                    exits.append(values); capacities.append(r['before']['exit_capacity_persons_per_step'])
            available = bool(w) and len(exits) == len(w)
            x += [float(available)]
            x += [sum(v[b] for v in exits) / max(n, 1) / 12. if available else 0. for b in range(2)]
            x += [sum(v[b] for v in exits) / max(sum(v[b] for v in capacities), 1) if available else 0. for b in range(2)]
            x += [bound]
    assert len(x) == len(feature_names(variant)) and np.isfinite(x).all()
    return x


def fit_ridge(rows, variant, alpha=1.):
    """Each scene/population family has weight one; two separate targets."""
    names = feature_names(variant)
    X = np.asarray([r['features'][variant] for r in rows], dtype=float)
    Y = np.asarray([[r['incap_delta'], r['fed_delta']] for r in rows], dtype=float)
    counts = Counter(r['family'] for r in rows)
    w = np.asarray([1. / counts[r['family']] for r in rows])
    mean = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-10] = 1.
    Z = (X - mean) / scale
    intercept = np.average(Y, axis=0, weights=w)
    coef = np.linalg.solve(Z.T @ (w[:, None] * Z) + alpha * np.eye(X.shape[1]), Z.T @ (w[:, None] * (Y - intercept)))
    return dict(version=VERSION, variant=variant, names=names, alpha=alpha,
                mean=mean.tolist(), scale=scale.tolist(), intercept=intercept.tolist(), coef=coef.tolist(),
                fit_states=len(rows), fit_families=len(counts))


def predict(fitted, x):
    assert fitted['version'] == VERSION and len(x) == len(fitted['names'])
    return (((np.asarray(x) - fitted['mean']) / fitted['scale']) @ np.asarray(fitted['coef']) + fitted['intercept']).tolist()


def calibrate(fitted, rows, quantile=.9):
    residuals = {}
    for r in rows:
        p = predict(fitted, r['features'][fitted['variant']])
        residual = [r['incap_delta'] - p[0], r['fed_delta'] - p[1]]
        prior = residuals.setdefault(r['family'], [-math.inf, -math.inf])
        residuals[r['family']] = [max(a, b) for a, b in zip(prior, residual)]
    assert residuals
    band = np.quantile(list(residuals.values()), quantile, axis=0, method='higher')
    return dict(upper_adverse_residual=np.maximum(band, 0).tolist(), quantile=quantile,
                calibration_families=len(residuals), family_max_residual=residuals,
                claim='Empirical development residual margin, not a calibrated probability or safety bound.')


def accepts(prediction, margin=None):
    values = [a + b for a, b in zip(prediction, margin or [0., 0.])]
    inc, fed = values
    # Even a small predicted incapacitation increase cannot be traded for FED.
    return bool(inc <= 1e-12 and (inc <= -TOLERANCE or fed <= -TOLERANCE))
