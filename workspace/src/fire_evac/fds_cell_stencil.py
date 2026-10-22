"""Independent eight-cell observation on interior Cartesian mesh vertices."""
from itertools import product

import numpy as np


def support_centers(xyz, dx):
    point = np.asarray(xyz, dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(dx) or dx <= 0
            or not np.allclose(point/dx, np.round(point/dx), rtol=0, atol=1e-9)):
        raise ValueError('This equal-weight stencil requires an interior Cartesian grid vertex')
    return [(point+np.array(sign)*dx/2).tolist() for sign in product((-1, 1), repeat=3)]


def reconstruct(data, nodes):
    times = np.asarray(data['Time'], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError('Complete finite native time support required')
    result = {'Time': times.copy()}
    for node in nodes:
        for label in ('CO', 'CO2', 'O2', 'T'):
            keys = [f'C{node:03d}_{corner}_{label}' for corner in range(8)]
            if any(key not in data for key in keys):
                raise ValueError('All eight support cells must be observed; no filling allowed')
            values = np.array([data[key] for key in keys], dtype=float)
            if values.shape != (8, len(times)) or not np.isfinite(values).all():
                raise ValueError('Invalid native support arrays')
            if label != 'T' and (values.min() < 0 or values.max() > 1):
                raise ValueError('Expected physical mole fractions, not ppm or percent')
            result[f'S{node:03d}_{label}'] = np.mean(values, axis=0)
    return result
