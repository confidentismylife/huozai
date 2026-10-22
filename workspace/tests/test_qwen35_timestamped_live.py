import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_qwen35_timestamped_live import score, strict_json, relation_summary, observable_arithmetic_check
from run_timestamped_development_campaign import confirmation_requests


def response(label, confidence=.9):
    import json
    return {'done': True, 'message': {'content': json.dumps({'recommended_exit': label, 'confidence': confidence})}}


def test_abstract_labels_are_scored_by_physical_mapping():
    req = {'candidate_order': ['B', 'A'], 'private_reference': {'reference_exit': 'B', 'label_to_physical': {'A': 0, 'B': 1}}}
    result = score(response('B'), req)
    assert result['correct'] and result['selected_physical'] == 1 and result['selected_first']
    assert not score(response('B', True), req)['format_valid']
    assert not score({'done': True, 'message': {'content': '{"recommended_exit":"B","confidence":0.9,"confidence":0.7}'}}, req)['format_valid']


def test_separate_confirmation_balanced_and_rule_is_not_raw_accuracy():
    requests = confirmation_requests()
    assert len(requests) == 16 and sum(r['private_reference']['reference_physical_exit'] for r in requests) == 8
    for req in requests:
        ref = req['private_reference']['reference_exit']
        assert ref not in req['payload']['messages'][0]['content'].split('Data: ')[0].split('Arithmetic check:')[1].split('objective;')[1]
        wrong = 'south' if ref == 'north' else 'north'
        result = observable_arithmetic_check(response(wrong), req)
        assert not result['raw_score']['correct']
        assert result['rule_checked_correct'] and result['rule_replaced_or_rejected_raw']


def test_pair_metrics_count_missing_and_invalid():
    requests = []
    for order in (False, True):
        requests.append(dict(request_id=str(order), mirror=0, role='hazard', objective='minimize', naming='abstract', reversed_order=order))
    rows = [dict(request_id='False', correct=True, format_valid=True, selected_physical=1)]
    relation = relation_summary(requests, rows)['relations']['reversed_order']
    assert relation['planned_pairs'] == 1 and relation['missing_or_invalid_pairs'] == 1
    assert relation['both_correct_pairs'] == 0
