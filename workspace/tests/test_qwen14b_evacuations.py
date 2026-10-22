from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_qwen14b_evacuations import grade_response, cached_proposal
from fire_evac.timestamped_pipeline import Snapshot, CommonActionConstraints


def response(content, reason='stop'):
    return {'done': True, 'done_reason': reason, 'message': {'content': content}}


def snapshot(step=0):
    return Snapshot('q14unit', step, step * 5., 'unit', (.5, .5, 0., 0., 0.),
                    (.001, .001, .001, .001, .001), (8, 8), (False, False),
                    0, .01, 5, 0., 2, 0, 20, 240)


def test_strict_response_and_truncation_keep_failure_visible():
    good, status = grade_response(response('{"target":"south","confidence":0.8}'))
    assert good == {'target_branch': 1, 'confidence': .8} and status == 'valid'
    for text in ('{"target":"south","confidence":true}', '{"target":"south","confidence":NaN}',
                 '{"target":"south","confidence":0.8,"extra":1}', 'not json'):
        assert grade_response(response(text))[0] is None
    assert grade_response(response('{"target":"south","confidence":0.8}', 'length'))[1] == 'truncated_or_incomplete'


def test_measured_arrival_preserves_observation_timestamp_and_staleness():
    record = {'advice': {'target_branch': 1, 'confidence': .8}, 'latency_seconds': 22.1}
    p = cached_proposal(record, snapshot(), True)
    assert p.observation_time_s == 0 and p.received_time_s == 22.1
    decision = CommonActionConstraints().validate_proposal(p, snapshot(5), 25.)
    assert not decision.accepted and 'stale' in decision.reasons
    instant = cached_proposal(record, snapshot())
    assert CommonActionConstraints().validate_proposal(instant, snapshot(), 0.).accepted


def test_invalid_response_falls_back_in_both_modes():
    record = {'advice': None, 'latency_seconds': 20.}
    assert cached_proposal(record, snapshot()) is None
    assert cached_proposal(record, snapshot(), True) is None
