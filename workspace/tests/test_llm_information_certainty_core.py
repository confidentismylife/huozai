import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from llm_information_certainty_core import CONDITIONS, cases, expected_exposure, reference, payload, private_reference


@pytest.mark.parametrize('location,q,expected_north,expected_south,answer', [
    ('north', 1.0, 1.0, 0.0, 'south'),
    ('south', 0.9, 0.1, 0.9, 'north'),
    ('north', 0.6, 0.6, 0.4, 'south'),
    ('north', 0.4, 0.4, 0.6, 'north'),
])
def test_hand_calculated_expected_costs(location, q, expected_north, expected_south, answer):
    assert expected_exposure(location, q, 'north') == pytest.approx(expected_north)
    assert expected_exposure(location, q, 'south') == pytest.approx(expected_south)
    assert reference(location, q) == answer


def test_unknown_reference_for_uninformative_report():
    with pytest.raises(ValueError, match='unique'):
        reference('north', 0.5)


def test_only_predeclared_information_fields_change():
    import json
    from llm_information_certainty_core import inputs
    for case in cases():
        bodies = [payload(case, c) for c in CONDITIONS]
        normalized = [{k: v for k, v in b.items() if k != 'messages'} for b in bodies]
        assert all(v == normalized[0] for v in normalized)
        prefixes = [b['messages'][0]['content'].split('Data: ')[0] for b in bodies]
        assert len(set(prefixes)) == 1
        data = [json.loads(b['messages'][0]['content'].split('Data: ')[1]) for b in bodies]
        assert data == [inputs(case, c) for c in CONDITIONS]
        for value in data:
            assert not set(private_reference(case, CONDITIONS[0])) & set(value)
        assert {k for k in data[0] if data[0][k] != data[1][k]} == {'information_type'}
        assert {k for k in data[1] if data[1][k] != data[2][k]} == {'probability_report_correct'}
        assert {k for k in data[1] if data[1][k] != data[3][k]} == {'probability_report_correct'}
