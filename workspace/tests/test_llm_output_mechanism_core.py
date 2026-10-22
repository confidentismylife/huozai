from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llm_output_mechanism_core import cases, CONDITIONS, payload, grade, prompt


def answer(content, done_reason='stop', thinking=''):
    return {'message': {'content': content, 'thinking': thinking}, 'done_reason': done_reason}


def test_payload_only_varies_two_interventions_and_references_stay_private():
    case = cases()[0]
    modified = dict(case, reference_exit='north', case_id='PRIVATE', pair_id='PRIVATE')
    assert prompt(case) == prompt(modified)
    requests = [payload(case, c) for c in CONDITIONS]
    for r, c in zip(requests, CONDITIONS):
        assert ('format' in r) == c['schema'] and r['think'] == c['think']
        assert r['options']['num_predict'] == 1024 and r['options']['num_ctx'] == 4096
        assert r['messages'] == requests[0]['messages']
        reduced = {k: v for k, v in r.items() if k not in ('think', 'format')}
        assert reduced == {k: v for k, v in requests[0].items() if k not in ('think', 'format')}


def test_format_failure_can_still_have_correct_semantic_direction():
    result = grade(answer('```json\n{"recommended_exit":"south","confidence":0.9}\n```'), 'south')
    assert result['category'] == 'invalid_format' and result['semantic_correct'] is True
    assert not result['usable_correct']


def test_truncation_is_not_counted_as_wrong_direction():
    result = grade(answer('{"recommended_exit":"north","confidence":0.8}', 'length'), 'south')
    assert result['category'] == 'truncated' and result['semantic_correct'] is False
    assert not result['usable_correct']


def test_thinking_is_never_used_as_final_answer():
    result = grade(answer('', thinking='{"recommended_exit":"south","confidence":0.9}'), 'south')
    assert result['semantic_direction'] is None and result['thinking_present']


@pytest.mark.parametrize('content', [
    '{"recommended_exit":"south","confidence":true}',
    '{"recommended_exit":"south","confidence":NaN}',
    '{"recommended_exit":"south","confidence":1.1}',
    '{"recommended_exit":"north","recommended_exit":"south","confidence":0.9}',
])
def test_invalid_confidence_or_duplicate_keys_never_becomes_usable(content):
    assert not grade(answer(content), 'south')['usable_correct']


def test_mirrors_keep_all_non_location_inputs_identical():
    records = cases()
    assert len(records) == 8
    for left, right in zip(records[::2], records[1::2]):
        assert left['reference_exit'] != right['reference_exit']
        assert left['pair_id'] == right['pair_id']
        for key in left['inputs'].keys() - {'confirmed_fire_location'}:
            assert left['inputs'][key] == right['inputs'][key]
