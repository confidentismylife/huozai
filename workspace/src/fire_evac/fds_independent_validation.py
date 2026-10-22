"""Pre-frozen route selection and descriptive paired mesh comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def route_key(route):
    return (
        route['start_index'],
        route['exit_index'],
        float(route['departure_s']),
        tuple(route['route']),
    )


def route_end_s(route, segment_time_s=1.25):
    return float(route['departure_s']) + segment_time_s * (len(route['route']) - 1)


def select_independent_routes(route_pool, excluded_routes, field_end_s=70.0,
                              segment_time_s=1.25):
    """Select every route completed within the field, excluding prior witnesses.

    Selection uses only frozen route definitions and time coverage. It never reads
    dose, concentration, mesh-gap, or sampling-comparison outcomes.
    """
    excluded = {layout: {route_key(route) for route in routes}
                for layout, routes in excluded_routes.items()}
    selected = {}
    for layout, routes in route_pool.items():
        items = []
        for route in routes:
            if route_key(route) in excluded.get(layout, set()):
                continue
            end_s = route_end_s(route, segment_time_s)
            if end_s <= field_end_s + 1e-12:
                items.append({**route, 'route_end_s': end_s})
        selected[layout] = items
    return selected


def compare_operators(medium: Mapping[str, float], fine: Mapping[str, float],
                      dose_denominator_floor=1e-8,
                      gap_denominator_floor=1e-8):
    """Compare default and eight-cell gaps without hiding near-zero doses."""
    result = {}
    for operator in ('default', 'eight_cell'):
        candidate = float(medium[operator])
        reference = float(fine[operator])
        signed = candidate - reference
        result[operator] = {
            'candidate_FED': candidate,
            'fine_FED': reference,
            'signed_gap_FED': signed,
            'absolute_gap_FED': abs(signed),
            'relative_gap': (abs(signed) / reference
                             if reference >= dose_denominator_floor else None),
        }
    old_gap = result['default']['absolute_gap_FED']
    new_gap = result['eight_cell']['absolute_gap_FED']
    result['absolute_gap_change_FED'] = new_gap - old_gap
    result['absolute_gap_reduced'] = new_gap < old_gap
    result['stable_reduction_fraction'] = (
        (old_gap - new_gap) / old_gap if old_gap >= gap_denominator_floor else None
    )
    result['signed_gap_reversed'] = (
        result['default']['signed_gap_FED'] * result['eight_cell']['signed_gap_FED'] < 0
    )
    return result


def descriptive_layout_summary(route_records: Sequence[Mapping]):
    """Summarize repeated routes without treating them as independent replicates."""
    if not route_records:
        raise ValueError('At least one route record is required')
    changes = sorted(float(record['comparison']['absolute_gap_change_FED'])
                     for record in route_records)
    middle = len(changes) // 2
    median = (changes[middle] if len(changes) % 2
              else (changes[middle - 1] + changes[middle]) / 2)
    return {
        'routes': len(route_records),
        'absolute_gap_reduced_routes': sum(
            bool(record['comparison']['absolute_gap_reduced']) for record in route_records
        ),
        'relative_gap_comparable_routes': sum(
            record['comparison']['default']['relative_gap'] is not None
            and record['comparison']['eight_cell']['relative_gap'] is not None
            for record in route_records
        ),
        'median_absolute_gap_change_FED': median,
        'median_absolute_gap_reduced': median < 0,
    }
