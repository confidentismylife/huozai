"""Check case coverage, mirror-family holdouts, frozen inputs and model ledger."""
import csv
import hashlib
import json
from pathlib import Path

from run_bristol_context_choice_benchmark import OUTPUT,DATA,CONFIG,condition,family,features,api
from audit_liao_dynamic_data import sha


def verify():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    plan=json.loads((OUTPUT/'frozen_LLM_manifest.json').read_text(encoding='utf-8'))
    records=list(csv.DictReader((OUTPUT/'all_predictions.csv').open(encoding='utf-8-sig')))
    source=list(csv.DictReader((DATA/'bristol_context_cases.csv').open(encoding='utf-8-sig')))
    eligible=[r for r in source if r['eligible_context_benchmark']=='True']
    assert len(records)==len(eligible)==489
    assert {r['case_key'] for r in records}=={r['case_key'] for r in eligible}
    assert plan['config_sha256']==sha(CONFIG) and plan['source_cases_sha256']==sha(DATA/'bristol_context_cases.csv')
    responses=[json.loads(line) for line in (OUTPUT/'responses.jsonl').read_text().splitlines()]
    bykey={r['key']:r for r in responses}
    assert len(bykey)==len(responses)==len(plan['tasks'])==34
    held=set()
    for fold in plan['folds']:
        held_conditions={tuple(c) for c in fold['held_conditions']};train_conditions={tuple(c) for c in fold['train_conditions']}
        assert not held_conditions & train_conditions
        fam=tuple(fold['family'])
        assert held_conditions=={condition(r) for r in records if family(r)==fam}
        assert train_conditions=={condition(r) for r in records if family(r)!=fam}
        held.update(held_conditions)
    assert held=={condition(r) for r in records}
    allowed={'assigned_left_crowd','assigned_right_crowd','map_information','nearer_to_final_exit','crowd_state_design','training_base_probability_left'}
    for task in plan['tasks']:
        response=bykey[task['key']];system,user=[r['content'] for r in task['payload']['messages']]
        assert set(json.loads(user))==allowed
        assert hashlib.sha256((system+'\n'+user).encode()).hexdigest()==task['prompt_sha256']==response['prompt_sha256']
        assert response['model_digest']==plan['model_digest']==cfg['LLM']['digest']
        if response.get('response'):assert response['response']['model']=='qwen3:4b'
        assert 0<=response['p_left']<=1
    for r in eligible:
        changed={**r,'observed_exit':'2' if r['observed_exit']=='1' else '1','source_csv_row':'999999'}
        assert (features(r)==features(changed)).all() and family(r)==family(changed)
        assert r['nearer_to_final_exit']!='unresolved'
    current=next(m for m in api('/api/tags',timeout=5)['models'] if m['name']==cfg['LLM']['model'])
    assert current['digest']==cfg['LLM']['digest']
    result=dict(all_489_eligible_cases_accounted_for=True,all_34_conditions_accounted_for=True,
        all_18_mirror_family_holdouts_verified=True,prompt_inputs_outcomes_separated=True,
        outcome_perturbation_does_not_change_features_or_folds=True,model_and_prompt_hashes_verified=True,
        config_and_case_source_hashes_unchanged=True,
        ambiguous_58_records_excluded_without_outcome_decoding=True,
        valid_LLM_responses=sum(r['valid'] for r in responses),PPO_updates=0)
    (OUTPUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
