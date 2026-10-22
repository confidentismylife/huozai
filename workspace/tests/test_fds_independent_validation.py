import pytest

from fire_evac.fds_independent_validation import (
    compare_operators,
    descriptive_layout_summary,
    select_independent_routes,
)


def test_selection_excludes_witness_and_uses_coverage_only():
    pool = {'L1': [
        {'start_index': 1, 'exit_index': 9, 'departure_s': 60, 'route': [1, 2, 9]},
        {'start_index': 3, 'exit_index': 9, 'departure_s': 60, 'route': [3, 4, 5, 9]},
        {'start_index': 6, 'exit_index': 9, 'departure_s': 68, 'route': [6, 7, 9]},
    ]}
    witness = {'L1': [pool['L1'][0]]}
    selected = select_independent_routes(pool, witness, field_end_s=65)
    assert [route['start_index'] for route in selected['L1']] == [3]
    assert selected['L1'][0]['route_end_s'] == 63.75


def test_comparison_preserves_near_zero_and_gap_direction():
    result = compare_operators(
        {'default': 4e-7, 'eight_cell': 1.7e-7},
        {'default': 7e-7, 'eight_cell': 1.5e-7},
    )
    assert result['default']['relative_gap'] == pytest.approx(3 / 7)
    assert result['eight_cell']['relative_gap'] == pytest.approx(2 / 15)
    assert result['absolute_gap_reduced'] is True
    assert result['stable_reduction_fraction'] is not None
    assert result['signed_gap_reversed'] is True

    near_zero = compare_operators(
        {'default': 2e-9, 'eight_cell': 1e-9},
        {'default': 1e-10, 'eight_cell': 2e-10},
    )
    assert near_zero['default']['relative_gap'] is None
    assert near_zero['eight_cell']['relative_gap'] is None
    assert near_zero['stable_reduction_fraction'] is None


def test_layout_summary_is_descriptive():
    records = [
        {'comparison': compare_operators(
            {'default': 4e-7, 'eight_cell': 2e-7},
            {'default': 7e-7, 'eight_cell': 2.1e-7})},
        {'comparison': compare_operators(
            {'default': 5e-7, 'eight_cell': 4e-7},
            {'default': 6e-7, 'eight_cell': 4.2e-7})},
    ]
    summary = descriptive_layout_summary(records)
    assert summary['routes'] == 2
    assert summary['absolute_gap_reduced_routes'] == 2
    assert summary['median_absolute_gap_reduced'] is True
