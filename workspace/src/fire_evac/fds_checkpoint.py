"""Integrity and complete-route prerequisites for the new checkpoint stage."""
from pathlib import Path
from collections import deque
from contextlib import contextmanager
import csv
import json
import re
import struct
import subprocess

import numpy as np

from .post_safestep_physics import integrate_rate, read_device_csv


@contextmanager
def managed_child(args, *, failure_record=None, **kwargs):
    proc = subprocess.Popen(args, **kwargs)
    try:
        yield proc
    except BaseException as error:
        if failure_record is not None:
            try:
                with Path(failure_record).open('x', encoding='utf-8') as file:
                    json.dump({'termination_cause': 'MONITOR_EXCEPTION', 'owned_pid': proc.pid,
                        'exception_type': type(error).__name__, 'exception': str(error),
                        'native_stop_file_used': False}, file)
            except OSError:
                pass  # Cleanup must still happen if the diagnostic write fails.
        raise
    finally:
        if proc.poll() is None:
            proc.kill()  # Popen holds the owned process handle, not an arbitrary PID.
        proc.wait(timeout=15)


def inspect_native_output(path):
    try:
        data, units = read_device_csv(Path(path))
        if len(data['Time']) < 2 or not np.all(np.diff(data['Time']) > 0):
            raise ValueError('Strict native timestamps required; no rows removed')
        if not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError('Nonfinite native output')
        for key in data:
            if re.match(r'N\d+_(CO|CO2|O2)$', key):
                if units[key].strip() != 'mol/mol' or data[key].min() < -1e-12 or data[key].max() > 1+1e-12:
                    raise ValueError('Invalid gas fraction or unit')
        return {'valid': True, 'data': data, 'error': None}
    except (ValueError, OSError, csv.Error, IndexError, KeyError, StopIteration) as error:
        return {'valid': False, 'data': None, 'error': f'{type(error).__name__}: {error}'}


def complete_run(returncode, success_marker, errors, cause, solver_end, native_end, requested_end):
    if solver_end is None or native_end is None:
        return False
    return bool(returncode == 0 and success_marker and not errors and cause is None
        and np.isfinite(solver_end) and np.isfinite(native_end)
        and abs(solver_end-requested_end) <= 1e-5
        and abs(native_end-requested_end) <= 1e-6)


def read_passive_checkpoint(path, ijk, hrr_interval=1.):
    """Read the pinned FDS6.11.1 Cartesian, 3-tracked-species layout, through EOF.

    Q and timing offsets follow the prior source-grounded 2026-10-17 reader.
    Framing success alone does not establish provenance or physical validity.
    Only closed solver files are accepted as recovery artifacts by the caller.
    """
    path = Path(path)
    total = path.stat().st_size
    base_bytes = int(np.prod(np.asarray(ijk) + 2)) * 8
    previous, timing, nonzero, index = None, None, None, 0
    tail = deque(maxlen=3)
    with path.open('rb') as stream:
        while marker := stream.read(4):
            if len(marker) != 4:
                raise ValueError('truncated record marker')
            size = struct.unpack('<i', marker)[0]
            if size < 0 or size >= 2**30 or size > total - stream.tell() - 4:
                raise ValueError('invalid or truncated record length')
            data, trailer = stream.read(size), stream.read(4)
            if len(data) != size or len(trailer) != 4 or struct.unpack('<i', trailer)[0] != size:
                raise ValueError('record framing mismatch')
            tail.append(data)
            if index == 10 and size != base_bytes:
                raise ValueError('unexpected allocated D_SOURCE record')
            if index == 11 and size != base_bytes * 3:
                raise ValueError('unexpected tracked-species record')
            if index == 14:
                if size != base_bytes:
                    raise ValueError('unexpected Q record')
                q = np.frombuffer(data, dtype='<f8')
                if not np.isfinite(q).all():
                    raise ValueError('nonfinite Q record')
                nonzero = int(np.count_nonzero(q))
            if size == 17 * 8 and abs(struct.unpack_from('<d', data, 6 * 8)[0] - hrr_interval) < 1e-12:
                if timing is not None or previous is None or len(previous) < 80:
                    raise ValueError('ambiguous timing record')
                timing = {'fire_time_s': struct.unpack_from('<d', previous)[0],
                    'iteration': struct.unpack_from('<i', previous, 8)[0],
                    'DT_s': struct.unpack_from('<d', previous, 24)[0],
                    'last_HRR_dump_time_s': struct.unpack_from('<d', previous, len(previous)-48)[0]}
                if (not np.isfinite(timing['fire_time_s']) or timing['fire_time_s'] < 0
                        or not np.isfinite(timing['DT_s']) or timing['DT_s'] <= 0):
                    raise ValueError('invalid timing record')
            previous, index = data, index + 1
    if timing is None or nonzero is None:
        raise ValueError('required checkpoint records missing')
    # dump.f90:3648..3710, only this stage's fixed no-thin-wall/no-CFACE/
    # no-particle/no-HVAC/no-levelset/no-BNDF scope has these terminal records.
    if [len(r) for r in tail] != [8, 4, 12]:
        raise ValueError('mandatory checkpoint terminal records missing or extra tail')
    thin, cface, particles = (struct.unpack('<2i', tail[0]),
        struct.unpack('<i', tail[1]), struct.unpack('<3i', tail[2]))
    if (thin[0] != 0 or thin[1] < 0 or cface[0] != 0 or particles[0] != 0
            or particles[1] < 0 or particles[2] < 1):
        raise ValueError('unsupported checkpoint terminal scope or incomplete tail')
    return {**timing, 'Q_nonzero_cells': nonzero, 'records_checked': index,
            'bytes': total, 'terminal_scope_verified': True,
            'format_scope': 'FDS6.11.1 Cartesian/3species, no thin walls/CFACE/particles/HVAC/levelset/BNDF'}


def score_mesh_routes(fine_times, fine_rates, candidate_times, candidate_rates, routes,
                      speed=.8, floor=1e-8, threshold=.1):
    tf, tc = np.asarray(fine_times), np.asarray(candidate_times)
    for times in (tf, tc):
        if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError('strict finite native time support required')
    scores, witness = [], None
    for item in routes:
        begin = item['departure_s']
        end = begin + (len(item['route'])-1) / speed
        if begin < max(tf[0], tc[0]) or end > min(tf[-1], tc[-1]):
            continue
        doses = []
        for times, rates in ((tf, fine_rates), (tc, candidate_rates)):
            doses.append(sum(integrate_rate(times, rates[node], begin + step/speed,
                begin + (step+1)/speed) for step, node in enumerate(item['route'][:-1])))
        reference, candidate = doses
        relative = abs(candidate-reference)/reference if reference >= floor else None
        score = {'route': item, 'route_end_s': end,
            'reference_dx0p125_FED': reference, 'candidate_dx0p25_FED': candidate,
            'absolute_difference': abs(candidate-reference), 'relative_difference': relative}
        scores.append(score)
        if witness is None and relative is not None and relative > threshold:
            witness = score
    finite = [s for s in scores if s['relative_difference'] is not None]
    return {'planned_routes': len(routes), 'completed_routes': len(scores),
        'unscored_routes': len(routes)-len(scores), 'finite_dose_routes': len(finite),
        'near_zero_routes': len(scores)-len(finite), 'witness': witness, 'scores': scores,
        'maximum_relative_difference': max((s['relative_difference'] for s in finite), default=None)}
