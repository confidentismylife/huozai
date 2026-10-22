"""Fixed, development-only representation comparison; preserves the v1 draft.

History and response share the SAME completed sensor records. The latter
appends deterministic response summaries. No online identity or outcome input.
This module is restricted to the audited early cache (at most three steps).
"""
import copy
from collections import Counter
import numpy as np
from . import crowd_response_gain as draft

VERSION = 'crowd-response-comparison-v2'
VARIANTS = draft.VARIANTS
WINDOW = 3


def public_projection(public):
    return copy.deepcopy(draft.public_projection(public))


def context():
    return draft.context()


def observe_completed(ctx, public, action, detail, sensors):
    draft.observe_completed(ctx, public_projection(public), action, detail, sensors)


def raw_record_names():
    return (['present', 'step'] + [f'occupancy_{i}' for i in range(5)]
            + [f'smoke_{i}' for i in range(5)] + ['capacity_n', 'capacity_s',
               'closed_n', 'closed_s', 'forecast_south', 'forecast_growth',
               'forecast_onset', 'forecast_reopen', 'before_sign', 'hold_remaining',
               'action', 'actual_sign', 'release_cap', 'released']
            + [f'counts_after_{i}' for i in range(5)] + ['entries_n', 'entries_s'])


def encode_record(record):
    p = record['before']
    values = ([1., p['step'] / 120.] + list(p['occupancy_fraction'])
              + list(p['smoke_increment_per_step'])
              + [v / 12. for v in p['exit_capacity_persons_per_step']]
              + list(map(float, p['exit_closed']))
              + [float(p['forecast_source'] == 'south'), p['forecast_growth'],
                 p['forecast_onset_step'] / 120., p['forecast_reopen_steps'] / 20.,
                 p['actual_sign'] / 2., p['sign_hold_remaining_steps'] / 2.,
                 record['action'] / 8., record['sign'] / 2.,
                 record['release_cap'] / 20., record['released'] / 20.]
              + [v / 240. for v in record['counts_after']]
              + [v / 12. for v in record['entries']])
    assert len(values) == len(raw_record_names())
    return list(map(float, values))


def feature_names(variant):
    names = draft.feature_names('current')
    if variant == 'current':
        return names
    assert variant in VARIANTS
    names = draft.feature_names('public_history') + [
        f'lag_{lag}_{name}' for lag in range(1, WINDOW + 1) for name in raw_record_names()]
    if variant == 'response':
        names += draft.feature_names('response')[len(draft.feature_names('public_history')):]
    return names


def features(observation, public, base, candidate, ctx, variant):
    # Refuse later states rather than silently give response a longer window.
    assert len(ctx['history']) <= WINDOW
    if variant == 'current':
        return draft.features(observation, public, base, candidate, ctx, variant)
    x = draft.features(observation, public, base, candidate, ctx, 'public_history')
    for lag in range(1, WINDOW + 1):
        x += encode_record(ctx['history'][-lag]) if len(ctx['history']) >= lag else [0.] * len(raw_record_names())
    if variant == 'response':
        full = draft.features(observation, public, base, candidate, ctx, 'response')
        x += full[len(draft.feature_names('public_history')):]
    assert len(x) == len(feature_names(variant)) and np.isfinite(x).all()
    return x


def fit_ridge(rows, variant, alpha=1.):
    assert rows and alpha == 1.
    X = np.asarray([r['features'][variant] for r in rows], dtype=float)
    Y = np.asarray([[r['incap_delta'], r['fed_delta']] for r in rows], dtype=float)
    counts = Counter(r['family'] for r in rows)
    w = np.asarray([1. / counts[r['family']] for r in rows])
    mean = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-10] = 1.
    Z = (X - mean) / scale
    intercept = np.average(Y, axis=0, weights=w)
    coef = np.linalg.solve(Z.T @ (w[:, None] * Z) + np.eye(X.shape[1]),
                           Z.T @ (w[:, None] * (Y - intercept)))
    return dict(version=VERSION, variant=variant, names=feature_names(variant),
                alpha=alpha, mean=mean.tolist(), scale=scale.tolist(),
                intercept=intercept.tolist(), coef=coef.tolist(),
                fit_states=len(rows), fit_families=len(counts))


def predict(fitted, x):
    assert fitted['version'] == VERSION and len(x) == len(fitted['names'])
    return (((np.asarray(x) - fitted['mean']) / fitted['scale'])
            @ np.asarray(fitted['coef']) + fitted['intercept']).tolist()


def calibrate(fitted, rows):
    residuals = {}
    for r in rows:
        prediction = predict(fitted, r['features'][fitted['variant']])
        residual = np.asarray([r['incap_delta'], r['fed_delta']]) - prediction
        old = residuals.setdefault(r['family'], [-np.inf, -np.inf])
        residuals[r['family']] = np.maximum(old, residual).tolist()
    assert residuals
    margin = np.maximum(np.quantile(list(residuals.values()), .9, axis=0, method='higher'), 0.)
    return dict(upper_adverse_residual=margin.tolist(), quantile=.9,
                calibration_families=len(residuals), family_max_residual=residuals,
                claim='Empirical one-sided development residual margin, not a safety bound or probability.')


accepts = draft.accepts
