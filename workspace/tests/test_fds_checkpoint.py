"""Integrity and eligibility checks: catch corrupt recovery and premature scoring."""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fire_evac.fds_checkpoint import (
    read_passive_checkpoint, score_mesh_routes, managed_child,
    inspect_native_output, complete_run,
)


def fixture_bytes(*, active=True):
    # Independent miniature instance of the pinned three-species record layout.
    records = [b'\0' * 8 for _ in range(10)]
    cell = struct.pack('<27d', *([0.] * 27))
    records += [cell, cell * 3, cell, cell]
    records += [struct.pack('<27d', *([4. if active else 0.] * 27)), cell, cell]
    timing = bytearray(96)
    struct.pack_into('<d', timing, 0, 60.)
    struct.pack_into('<i', timing, 8, 420)
    struct.pack_into('<d', timing, 24, .1)
    struct.pack_into('<d', timing, 48, 59.)
    records += [bytes(timing), struct.pack('<17d', *([1.] * 17)),
        struct.pack('<2i', 0, 10), struct.pack('<i', 0), struct.pack('<3i', 0, 50, 1)]
    return b''.join(struct.pack('<i', len(r)) + r + struct.pack('<i', len(r)) for r in records)


def test_reads_time_and_active_state_only_from_framed_file(tmp_path):
    path = tmp_path / 'case_1.restart'
    path.write_bytes(fixture_bytes())
    info = read_passive_checkpoint(path, (1, 1, 1), hrr_interval=1.)
    assert info['fire_time_s'] == 60.
    assert info['DT_s'] == .1
    assert info['records_checked'] == 22
    assert info['Q_nonzero_cells'] == 27


def test_truncated_tail_rejected_even_after_valid_time_record(tmp_path):
    path = tmp_path / 'truncated.restart'
    path.write_bytes(fixture_bytes()[:-2])
    with pytest.raises(ValueError, match='record|framing|truncated'):
        read_passive_checkpoint(path, (1, 1, 1), hrr_interval=1.)


def test_zero_reaction_checkpoint_is_reported_as_suspect(tmp_path):
    path = tmp_path / 'zeroQ.restart'
    path.write_bytes(fixture_bytes(active=False))
    assert read_passive_checkpoint(path, (1, 1, 1), hrr_interval=1.)['Q_nonzero_cells'] == 0


def test_complete_record_boundary_truncation_is_not_valid_checkpoint(tmp_path):
    path = tmp_path/'missing_terminal.restart'
    path.write_bytes(fixture_bytes()[:-20])
    with pytest.raises(ValueError, match='terminal|tail|complete'):
        read_passive_checkpoint(path, (1, 1, 1))


def test_extra_record_after_terminal_is_rejected(tmp_path):
    path = tmp_path/'extra.restart'
    path.write_bytes(fixture_bytes()+struct.pack('<i', 4)+b'extr'+struct.pack('<i', 4))
    with pytest.raises(ValueError, match='terminal|tail|complete'):
        read_passive_checkpoint(path, (1, 1, 1))


def test_monitor_exception_reaps_the_actual_owned_child(tmp_path):
    record = tmp_path/'monitor_error.json'
    with pytest.raises(RuntimeError, match='injected monitor failure'):
        with managed_child([sys.executable, '-c', 'import time; time.sleep(60)'],
                failure_record=record, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as proc:
            raise RuntimeError('injected monitor failure')
    assert proc.poll() is not None
    assert record.exists()


def test_partial_final_csv_is_an_invalid_result_not_an_unhandled_exception(tmp_path):
    path = tmp_path/'partial_devc.csv'
    raw = 's,mol/mol\nTime,N000_CO\n0,0\n1,0.001\n2,1.0E-'
    path.write_text(raw)
    result = inspect_native_output(path)
    assert result['valid'] is False
    assert result['data'] is None
    assert result['error']
    assert path.read_text() == raw


@pytest.mark.parametrize('solver_end,native_end,expected', [
    (180., 180., True), (180., 179., False), (179., 180., False),
    (180., None, False),
])
def test_success_marker_cannot_substitute_for_target_time(solver_end, native_end, expected):
    result = complete_run(0, True, [], None, solver_end, native_end, 180.)
    assert result is expected


ROUTES = [{'start_index': 0, 'exit_index': 1, 'departure_s': 0., 'route': [0, 1]}]


def test_no_scoring_before_full_route_is_supported():
    result = score_mesh_routes(np.array([0., 1.24]), {0: np.array([.02, .02])},
        np.array([0., 2.]), {0: np.array([.03, .03])}, ROUTES)
    assert result['completed_routes'] == 0
    assert result['witness'] is None


def test_completed_route_can_fail_necessary_condition():
    result = score_mesh_routes(np.array([0., 2.]), {0: np.array([.02, .02])},
        np.array([0., 2.]), {0: np.array([.03, .03])}, ROUTES)
    assert result['completed_routes'] == 1
    assert result['witness']['reference_dx0p125_FED'] == pytest.approx(.025)
    assert result['witness']['candidate_dx0p25_FED'] == pytest.approx(.0375)
    assert result['witness']['relative_difference'] == pytest.approx(.5)


def test_near_zero_reference_does_not_create_percentage_witness():
    result = score_mesh_routes(np.array([0., 2.]), {0: np.array([1e-12, 1e-12])},
        np.array([0., 2.]), {0: np.array([.03, .03])}, ROUTES)
    assert result['near_zero_routes'] == 1
    assert result['witness'] is None
    assert result['scores'][0]['absolute_difference'] == pytest.approx(.0375 - 1.25e-12)
