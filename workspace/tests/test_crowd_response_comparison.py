import copy
import pytest
from fire_evac import crowd_response_comparison as m
from test_crowd_response_gain import public


def history(entries=(3, 2)):
    ctx = m.context()
    p = public()
    p['occupancy_fraction'] = [0., 0., 10/240, 0., 0.]
    m.observe_completed(ctx, p, 1, dict(eligible=False, base_action=1),
                        dict(actual_sign=0, actual_room_capacity=10, actual_room_count=0,
                             zone_counts_after=[0, 0, 5, 3, 2], branch_entries=list(entries)))
    current = public(1)
    current['occupancy_fraction'] = [0, 0, 5/240, 3/240, 2/240]
    return ctx, p, current


def test_history_and_response_both_receive_raw_flow():
    ctx, _, p = history()
    changed = copy.deepcopy(ctx)
    changed['history'][0]['entries'] = [2., 3.]
    changed['history'][0]['counts_after'] = [0, 0, 5, 2, 3]
    for variant in ('public_history', 'response'):
        assert m.features([0.]*34, p, 1, 5, ctx, variant) != m.features([0.]*34, p, 1, 5, changed, variant)
    h = m.features([0.]*34, p, 1, 5, ctx, 'public_history')
    assert m.features([0.]*34, p, 1, 5, ctx, 'response')[:len(h)] == h


def test_completed_public_records_are_copied():
    ctx, old, p = history()
    old['occupancy_fraction'][0] = 999
    assert ctx['history'][0]['before']['occupancy_fraction'][0] == 0


def test_future_and_hidden_fields_do_not_change_features():
    ctx, _, p = history()
    dirty = copy.deepcopy(ctx)
    dirty['history'][0].update(behavior='bad', future_fed=999, hidden_people=[999])
    dirty['history'][0]['before'].update(episode_id='bad', future_outcome=999)
    for v in m.VARIANTS:
        assert m.features([0.]*34, p, 1, 5, ctx, v) == m.features([0.]*34, p, 1, 5, dirty, v)


def test_frozen_early_window_refuses_longer_history():
    ctx, _, p = history()
    ctx['history'] *= 4
    with pytest.raises(AssertionError):
        m.features([0.]*34, p, 1, 5, ctx, 'response')
