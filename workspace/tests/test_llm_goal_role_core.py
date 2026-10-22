from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_goal_role_core import CONDITIONS, cases, inputs, decoded_costs, choose, regret, private_reference, payload


@pytest.mark.parametrize('anchor,q', [('north',0.9),('south',0.9),('north',0.6),('south',0.6)])
def test_equivalent_encodings_and_goal_reversal(anchor,q):
    other = 'south' if anchor == 'north' else 'north'
    hazard = decoded_costs(anchor,'hazard',q)
    clear = decoded_costs(other,'clear',q)
    assert hazard == pytest.approx(clear)
    assert hazard[anchor] == pytest.approx(q)
    assert hazard[other] == pytest.approx(1-q)
    assert choose(hazard,'minimize') == other
    assert choose(clear,'minimize') == other
    assert choose(hazard,'maximize') == anchor
    assert choose(clear,'maximize') == anchor


def test_inverted_objective_regret_and_ambiguous_reference():
    assert regret(0.9,0.1,'minimize') == pytest.approx(0.8)
    assert regret(0.1,0.9,'maximize') == pytest.approx(0.8)
    for objective in ('minimize','maximize'):
        with pytest.raises(ValueError,match='unique'):
            choose({'north':0.5,'south':0.5},objective)


def test_predeclared_control_changes_and_no_private_answer_in_data():
    import json
    for case in cases():
        data = {c['id']: inputs(case,c) for c in CONDITIONS}
        for role in ('hazard','clear'):
            a,b = data[f'minimize_{role}'],data[f'maximize_{role}']
            assert {k for k in a if a[k]!=b[k]} == {'objective'}
        for objective in ('minimize','maximize'):
            a,b = data[f'{objective}_hazard'],data[f'{objective}_clear']
            assert {k for k in a if a[k]!=b[k]} == {'report_role','reported_location'}
            assert decoded_costs(a['reported_location'],a['report_role'],a['probability_report_correct']) == pytest.approx(
                decoded_costs(b['reported_location'],b['report_role'],b['probability_report_correct']))
        bodies = [payload(case,c) for c in CONDITIONS]
        assert len({b['messages'][0]['content'].split('Data: ')[0] for b in bodies}) == 1
        normalized = [{k:v for k,v in b.items() if k!='messages'} for b in bodies]
        assert all(b==normalized[0] for b in normalized)
        for condition,body in zip(CONDITIONS,bodies):
            actual = json.loads(body['messages'][0]['content'].split('Data: ')[1])
            assert not {'hazard_anchor','reference_exit','expected_cost_by_exit','optimal_expected_cost'} & set(actual)
            costs = decoded_costs(actual['reported_location'],actual['report_role'],actual['probability_report_correct'])
            ref = private_reference(case,condition)
            assert costs == pytest.approx(ref['expected_cost_by_exit'])
