"""Test prompt privacy and failure handling with fake responses, no LLM calls."""
from pathlib import Path
import sys,json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import auto_evacuation_llm_independent as branch


def setup_fake(tmp_path,monkeypatch,responses):
    monkeypatch.setattr(branch,'OUT',tmp_path);monkeypatch.setattr(branch.campaign,'OUT',tmp_path)
    branch.campaign.write(tmp_path/'budget.json',{'llm_requests':0})
    branch.campaign.write(tmp_path/'campaign_status.json',{'state':'ready'})
    scene={'scenario_id':'FAKE001','source_branch':1,'forecast_source_branch':0,'growth_rate':.01,'onset_step':6,
        'background_hazard':.002,'north_exit_capacity':5,'south_exit_capacity':10,'blocked_branch':-1,'blockage_end_step':0,'conflict_type':'secret_type'}
    seen=[]
    def fake_api(url,payload=None,timeout=60):
        if url.endswith('/api/tags'):return {'models':[{'name':'qwen3:4b','digest':'fake-model'}]}
        seen.append(payload);return {'message':{'content':responses[min(len(seen)-1,len(responses)-1)]},'eval_count':2}
    monkeypatch.setattr(branch.campaign,'api_json',fake_api)
    return {'banks':{'llm':[scene]}},seen


def test_clear_prompt_excludes_source_and_freezes_before_actual(tmp_path,monkeypatch):
    manifest,seen=setup_fake(tmp_path,monkeypatch,['{"target":"south","confidence":0.8}'])
    advice=branch.clear_advice(manifest);assert advice['FAKE001'].target_branch==1
    prompt=seen[0]['messages'][0]['content'];assert '0=north, 1=south' in prompt and '"forecast_source": "north"' in prompt
    assert 'source_branch' not in prompt and 'secret_type' not in prompt
    assert branch.campaign.read(tmp_path/'llm_advice_frozen.json')['actual_seen'] is False
    assert branch.campaign.read(tmp_path/'budget.json')['llm_requests']==1


def test_two_parse_failures_do_not_substitute_rule(tmp_path,monkeypatch):
    manifest,seen=setup_fake(tmp_path,monkeypatch,['{"target":"bad","confidence":0.8}'])
    with pytest.raises(RuntimeError,match='without rule substitution'):branch.clear_advice(manifest)
    assert len(seen)==2 and branch.campaign.read(tmp_path/'budget.json')['llm_requests']==2
    assert len(list((tmp_path/'llm_failures').glob('*.json')))==2 and not (tmp_path/'llm_advice_frozen.json').exists()
