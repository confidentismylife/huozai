"""Verify frozen sampling, input separation, model identity and response ledger."""
import csv
import hashlib
import json
from collections import Counter

from audit_liao_dynamic_data import SOURCE, DEFAULT_OUT, sha
from run_liao_llm_motion_diagnostic import OUTPUT, BASE, EXPECTED_DIGEST, api


def verify():
    plan=json.loads((OUTPUT/'frozen_manifest.json').read_text(encoding='utf-8'))
    responses=[json.loads(line) for line in (OUTPUT/'responses.jsonl').read_text(encoding='utf-8').splitlines()]
    bykey={r['window_key']:r for r in responses}
    assert len(bykey)==len(responses)==120
    original=list(csv.DictReader((BASE/'development_out_of_fold_predictions.csv').open(encoding='utf-8-sig')))
    original_keys={r['window_key'] for r in original}
    original_positive={r['window_key'] for r in original if int(r['switch_within_horizon'])}
    selected_positive=set();negative_counts=Counter()
    allowed={'mean_step_speed_m_s','recent_half_second_speed_m_s','early_half_second_speed_m_s',
             'speed_change_m_s','absolute_turn_radians','path_inefficiency_m'}
    for task in plan['tasks']:
        key=task['window_key'];row=task['evaluation_only'];payload=task['payload'];response=bykey[key]
        assert key in original_keys and row['partition']=='proposed_development'
        assert response['model_digest']==plan['model_digest']==EXPECTED_DIGEST
        assert response['prompt_sha256']==task['prompt_sha256']
        system,user=(r['content'] for r in payload['messages'])
        assert hashlib.sha256((system+'\n'+user).encode()).hexdigest()==task['prompt_sha256']
        observation=json.loads(user)
        assert set(observation)=={'history_seconds','forecast_seconds','training_baseline_probability','motion'}
        assert set(observation['motion'])==allowed
        assert observation['training_baseline_probability']==float(row['training_frequency'])
        if int(row['switch_within_horizon']):
            selected_positive.add(key);assert row['inclusion_probability']==1
        else:
            negative_counts[row['run']]+=1
            total=sum(r['run']==row['run'] and not int(r['switch_within_horizon']) for r in original)
            assert row['inclusion_probability']==9/total
        if response.get('response'):
            assert response['response']['model']=='qwen3:4b'
        assert 0<=response['p_change']<=1
    assert selected_positive==original_positive and len(selected_positive)==30
    assert set(negative_counts.values())=={9} and len(negative_counts)==10
    current=next(m for m in api('/api/tags',timeout=5)['models'] if m['name']=='qwen3:4b')
    assert current['digest']==EXPECTED_DIGEST
    assert all(sha(BASE/name)==digest for name,digest in plan['source_sha256'].items())
    intake=json.loads((DEFAULT_OUT/'audit_summary.json').read_text(encoding='utf-8'))
    assert all(sha(SOURCE/name)==digest for name,digest in intake['source_sha256_before'].items())
    result=dict(all_120_cases_accounted_for=True,all_30_positive_windows_censused=True,
        nine_random_negative_windows_per_run=True,prompt_contains_only_allowed_motion_and_training_frequency=True,
        no_outcome_or_person_id_sent=True,model_digest_and_response_model_verified=True,
        prompt_hashes_verified=True,source_and_baseline_hashes_unchanged=True,
        valid_response_count=sum(r['valid'] for r in responses),old_14B_cache_reused=False)
    (OUTPUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
