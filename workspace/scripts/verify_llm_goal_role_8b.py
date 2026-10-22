"""Independent role decoding, goal references and paired raw-output audit."""
from pathlib import Path
import csv
import hashlib
import json
import random
import statistics

from verify_llm_output_mechanism import classify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/llm_goal_role_8b_20261006'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    return a is None and b is None or a is not None and b is not None and abs(a-b)<1e-10


def verify_base():
    m,budget,summary = read(OUT/'manifest.json'),read(OUT/'budget.json'),read(OUT/'summary.json')
    assert len(m['cases'])==8 and len(m['request_order'])==32
    conditions = {f'{objective}_{role}':(objective,role) for role in ('hazard','clear') for objective in ('minimize','maximize')}
    assert {c['id']:(c['objective'],c['role']) for c in m['conditions']}==conditions
    for name,digest in m['frozen_sha256'].items():
        assert sha(ROOT/name)==digest,name
    from finalize_auto_evacuation_campaign_review import ledger_check,MAIN,HISTORY
    parents = {}
    for name,digest in m['parent_completion_sha256'].items():
        path = ROOT/name
        assert sha(path)==digest,name
        parents[name] = ledger_check(path,HISTORY if path.parent==MAIN else None)
        assert parents[name]==m['parent_checks'][name]
    assert sha(ROOT/'outputs/third_round_development_status.json')==m['old_global_status_sha256']
    assert sha(OUT/'model_service_snapshot.json')==m['model_snapshot_sha256']
    assert sha(OUT/'pre_inference_checks.json')==m['pre_inference_checks_sha256']
    assert read(OUT/'pre_inference_checks.json')['all_checks_passed']
    cases = {s['case_id']:s for s in m['cases']}
    families = sorted({s['pair_id'] for s in m['cases']})
    assert len(families)==4
    for pair in families:
        a,b = [s for s in m['cases'] if s['pair_id']==pair]
        assert {a['hazard_anchor'],b['hazard_anchor']}=={'north','south'}
        assert a['physical_inputs']==b['physical_inputs'] and a['probability']==b['probability']
    rng = random.Random(2026100604)
    rng.shuffle(families)
    planned = []
    for pair in families:
        block = [(s,c) for s in m['cases'] if s['pair_id']==pair for c in m['conditions']]
        rng.shuffle(block)
        planned += [f"{s['case_id']}_{c['id']}" for s,c in block]
    assert planned==[r['request_id'] for r in m['request_order']]
    expected_schema = {'type':'object','properties':{'recommended_exit':{'type':'string','enum':['north','south']},
        'confidence':{'type':'number','minimum':0,'maximum':1}},'required':['recommended_exit','confidence'],'additionalProperties':False}
    rows,prefixes,raw_count = {},set(),0
    for req in m['request_order']:
        body,case = req['payload'],cases[req['case_id']]
        digest = hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
        assert digest==req['payload_sha256']
        assert body['model']=='qwen3:8b' and body['think'] is False and body['stream'] is False
        assert body['format']==expected_schema and body['options']=={'temperature':0,'seed':2026100601,'num_predict':1024,'num_ctx':4096}
        assert len(body['messages'])==1 and body['messages'][0]['role']=='user'
        prefix,text = body['messages'][0]['content'].split('Data: ')
        prefixes.add(prefix)
        data = json.loads(text)
        objective,role = conditions[req['condition_id']]
        other = 'south' if case['hazard_anchor']=='north' else 'north'
        location = case['hazard_anchor'] if role=='hazard' else other
        q = case['probability']
        assert q in (0.9,0.6)
        assert data=={**case['physical_inputs'],'reported_location':location,'report_role':role,
                     'probability_report_correct':q,'objective':objective}
        assert not {'hazard_anchor','reference_exit','expected_cost_by_exit','optimal_expected_cost'} & set(data)
        assert data['north_exit_capacity']==data['south_exit_capacity']
        assert data['north_route_distance']==data['south_route_distance']
        assert data['initial_room_occupancy'][0]==data['initial_room_occupancy'][1]
        assert data['measured_smoke_north']==data['measured_smoke_south']==0
        at_report = q if role=='hazard' else 1-q
        posterior = {'north':at_report if location=='north' else 1-at_report,
                     'south':at_report if location=='south' else 1-at_report}
        canonical = {'north':q if case['hazard_anchor']=='north' else 1-q,
                     'south':q if case['hazard_anchor']=='south' else 1-q}
        assert all(close(posterior[k],canonical[k]) for k in canonical)
        expected = min(posterior,key=posterior.get) if objective=='minimize' else max(posterior,key=posterior.get)
        optimum = posterior[expected]
        ref = req['private_reference']
        assert ref['reference_exit']==expected and ref['objective']==objective
        assert all(close(ref['expected_cost_by_exit'][k],posterior[k]) for k in posterior)
        assert close(ref['optimal_expected_cost'],optimum)
        result = read(OUT/'results'/f"{req['request_id']}.json")
        assert result['payload_sha256']==req['payload_sha256']
        if result['raw_path'] is not None:
            raw = read(ROOT/result['raw_path'])
            assert raw['payload']==body and raw['payload_sha256']==req['payload_sha256']
            assert not raw.get('service_error') and raw['response']['done'] is True and raw['response']['model']=='qwen3:8b'
            outcome = classify(raw['response'],expected)
            assert all(result['outcome'][k]==v for k,v in outcome.items())
            assert result['latency_seconds']==raw['latency_seconds']
            assert not (OUT/'attempts'/f"{req['request_id']}_attempt{result['attempt']+1}.started.json").exists()
            prompt_tokens = raw['response'].get('prompt_eval_count') or 0
            raw_count += 1
        else:
            outcome = result['outcome']
            assert outcome['category']=='transport_failure'
            for attempt in (1,2):
                assert (OUT/'attempts'/f"{req['request_id']}_attempt{attempt}.started.json").exists()
                assert (OUT/'transport_failures'/f"{req['request_id']}_attempt{attempt}.json").exists()
            prompt_tokens = 0
        valid = bool(outcome['strict_format_valid'] and not outcome['truncated'] and outcome.get('semantic_direction') in ('north','south'))
        cost = posterior[outcome['semantic_direction']] if valid else None
        loss = cost-optimum if valid and objective=='minimize' else optimum-cost if valid else None
        rows[(req['case_id'],req['condition_id'])] = {'id':req['request_id'],'outcome':outcome,'valid':valid,
            'follows':bool(valid and outcome['semantic_direction']==location),'cost':cost,'regret':loss,
            'latency':result['latency_seconds'],'reference':expected,'prompt_tokens':prompt_tokens}
    assert len(rows)==32 and len(prefixes)==1
    with (OUT/'case_results.csv').open(encoding='utf-8-sig',newline='') as stream:
        table = {r['request_id']:r for r in csv.DictReader(stream)}
    assert len(table)==32
    for value in rows.values():
        r = table[value['id']]
        assert r['category']==value['outcome']['category'] and r['reference_exit']==value['reference']
        assert r['valid_final']==str(value['valid']) and r['follows_reported_location']==str(value['follows'])
        for column,key in [('selected_expected_cost_if_valid','cost'),('objective_regret_if_valid','regret')]:
            assert r[column]=='' if value[key] is None else close(float(r[column]),value[key])
    passes = []
    for cid in conditions:
        values = [v for (case,cond),v in rows.items() if cond==cid]
        g = summary['groups'][cid]
        for category in ('usable_correct','valid_wrong_direction','invalid_format','truncated','transport_failure'):
            assert g[category]==sum(v['outcome']['category']==category for v in values)
        swaps = 0
        for pair in families:
            a,b = [s for s in m['cases'] if s['pair_id']==pair]
            da,db = rows[(a['case_id'],cid)]['outcome']['semantic_direction'],rows[(b['case_id'],cid)]['outcome']['semantic_direction']
            swaps += da is not None and db is not None and da!=db
        passed = sum(v['outcome']['usable_correct'] for v in values)==8 and swaps==4
        assert g['mirror_pairs_swapped']==swaps and g['basic_function_passed']==passed
        passes.append(passed)
        valid = [v for v in values if v['valid']]
        times = [v['latency'] for v in values if v['latency'] is not None]
        assert g['valid_final']==len(valid) and g['follows_reported_location']==sum(v['follows'] for v in values)
        assert close(g['mean_expected_cost_if_valid'],statistics.mean(v['cost'] for v in valid) if valid else None)
        assert close(g['mean_objective_regret_if_valid'],statistics.mean(v['regret'] for v in valid) if valid else None)
        assert close(g['mean_latency_seconds'],statistics.mean(times) if times else None)
        assert close(g['total_latency_seconds'],sum(times))
        assert g['thinking_present']==sum(v['outcome']['thinking_present'] for v in values)
        assert g['generated_tokens']==sum(v['outcome'].get('generated_tokens') or 0 for v in values)
        assert g['prompt_tokens']==sum(v['prompt_tokens'] for v in values)
    independent_pairs = {}
    for role in ('hazard','clear'):
        pairs = [(rows[(s['case_id'],f'minimize_{role}')],rows[(s['case_id'],f'maximize_{role}')]) for s in m['cases']]
        delivered = [(a,b) for a,b in pairs if a['valid'] and b['valid']]
        independent_pairs[f'goal_flip_{role}'] = {'comparison':f'goal_flip_{role}','kind':'goal_flip','pairs':8,
            'fully_delivered_pairs':len(delivered),'expected_relation_observed':sum(a['outcome']['semantic_direction']!=b['outcome']['semantic_direction'] for a,b in delivered),
            'both_answers_correct':sum(a['outcome']['usable_correct'] and b['outcome']['usable_correct'] for a,b in pairs)}
    for objective in ('minimize','maximize'):
        pairs = [(rows[(s['case_id'],f'{objective}_hazard')],rows[(s['case_id'],f'{objective}_clear')]) for s in m['cases']]
        delivered = [(a,b) for a,b in pairs if a['valid'] and b['valid']]
        independent_pairs[f'role_invariance_{objective}'] = {'comparison':f'role_invariance_{objective}','kind':'role_invariance','pairs':8,
            'fully_delivered_pairs':len(delivered),'expected_relation_observed':sum(a['outcome']['semantic_direction']==b['outcome']['semantic_direction'] for a,b in delivered),
            'both_answers_correct':sum(a['outcome']['usable_correct'] and b['outcome']['usable_correct'] for a,b in pairs)}
    assert {p['comparison']:p for p in summary['paired_diagnostics']}==independent_pairs
    with (OUT/'paired_goal_role_results.csv').open(encoding='utf-8-sig',newline='') as stream:
        pairs_table = list(csv.DictReader(stream))
    assert len(pairs_table)==4
    for row in pairs_table:
        expected = independent_pairs[row['comparison']]
        assert row['kind']==expected['kind']
        for key in ('pairs','fully_delivered_pairs','expected_relation_observed','both_answers_correct'):
            assert int(row[key])==expected[key]
    follows = sum(v['follows'] for v in rows.values())
    assert summary['reported_location_match_count']==follows
    assert summary['always_follows_reported_location_in_valid_answers']==(follows==32)
    assert summary['all_cells_passed']==all(passes)
    assert summary['all_primary_pairs_correct']==all(p['both_answers_correct']==8 and p['expected_relation_observed']==8 for p in independent_pairs.values())
    assert budget['api_attempts_started']==len(list((OUT/'attempts').glob('*.started.json')))<=64
    assert budget['successful_http_responses']==raw_count
    assert budget['logical_requests_completed']==len(list((OUT/'results').glob('*.json')))==32
    assert budget['simulations']==budget['ppo_updates']==budget['fits']==0
    reserved = ROOT/'outputs/llm_semantic_channel_diagnostic_20261005'
    assert not (reserved/'fresh_actual').exists() and not (reserved/'fresh_advice_frozen.json').exists()
    return {'all_checks_passed':True,'logical_requests_rechecked':32,'raw_responses_reclassified':raw_count,
        'role_encodings_same_underlying_risk':True,'goal_references_reconstructed':32,
        'same_prompt_schema_options_and_single_message':True,'randomized_order_reconstructed':True,
        'only_predeclared_goal_and_role_fields_change':True,'private_answers_not_sent_to_model':True,
        'paired_goal_flip_and_role_invariance_reconstructed':True,'cost_and_signed_objective_regret_reconstructed':True,
        'no_loss_imputed_for_missing_answers':True,'ancestor_records_checked':parents,
        'old_global_status_unchanged':True,'reserved48_scenarios_untouched':True,
        'new_inference_for_verification':0,'new_simulations':0,'new_fits':0,'new_ppo_updates':0,'verifier_sha256':sha(__file__)}



def verify():
    m=read(OUT/'manifest.json')
    baseline=ROOT/m['base_manifest']
    assert sha(baseline)==m['base_manifest_sha256']
    base=read(baseline)
    assert m['cases']==base['cases'] and m['conditions']==base['conditions']
    assert m['request_order_seed']==base['request_order_seed']
    assert len(m['request_order'])==len(base['request_order'])==32
    for before,after in zip(base['request_order'],m['request_order']):
        assert {k for k in before if before[k]!=after[k]}=={'payload','payload_sha256'}
        assert {k for k in before['payload'] if before['payload'][k]!=after['payload'][k]}=={'model'}
        assert before['payload']['model']=='qwen3:4b' and after['payload']['model']=='qwen3:8b'
    snap=read(OUT/'model_service_snapshot.json')
    oldsnap=read(baseline.parent/'model_service_snapshot.json')
    assert snap['version']==oldsnap['version']
    tag=next(v for v in snap['tags']['models'] if v['name']=='qwen3:8b')
    assert tag['digest']==m['model_digest'] and tag['digest'].startswith('500a1f067a9f')
    assert snap['show']['details']['quantization_level']=='Q4_K_M'
    assert 8_000_000_000<=snap['show']['model_info']['general.parameter_count']<=8_500_000_000
    result=verify_base()
    current=read(OUT/'summary.json')
    previous=read(baseline.parent/'summary.json')
    expected=[]
    for name,g in current['groups'].items():
        b=previous['groups'][name]
        expected.append({'condition':name,'baseline_correct':b['usable_correct'],'8b_correct':g['usable_correct'],
                        'correct_count_change':g['usable_correct']-b['usable_correct'],
                        'baseline_mean_latency_seconds':b['mean_latency_seconds'],'8b_mean_latency_seconds':g['mean_latency_seconds']})
    comp=read(OUT/'comparison.json')
    assert comp['groups']==expected and comp['not_parameter_count_only_effect'] is True
    assert comp['independent_evacuation_effect_established'] is False and comp['held_out_replication'] is False
    with (OUT/'model_comparison.csv').open(encoding='utf-8-sig',newline='') as stream:
        table=list(csv.DictReader(stream))
    assert len(table)==4
    for row,want in zip(table,expected):
        assert row['condition']==want['condition']
        for field in ['baseline_correct','8b_correct','correct_count_change']:
            assert int(row[field])==want[field]
        for field in ['baseline_mean_latency_seconds','8b_mean_latency_seconds']:
            assert row[field]=='' if want[field] is None else close(float(row[field]),want[field])
    result.update({'api_model_only_change_from_sealed_baseline_verified':True,
                   'model_parameter_count_and_quantization_verified':True,
                   'same_ollama_version':True,'cross_model_comparison_recomputed':True,
                   'scale_only_causal_claim_permitted':False,
                   'baseline_4b_thinking_variant':oldsnap['show']['model_info'].get('general.finetune')=='Thinking',
                   'internal_templates_equal':snap['show']['template']==oldsnap['show']['template'],
                   'verifier_base_source_sha256':sha(ROOT/'scripts/verify_llm_goal_role.py')})
    return result


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False))
