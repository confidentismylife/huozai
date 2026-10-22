"""Read-only independent result audit; no simulator or controller imports.

Recompute trace accounting and paired/factorial statistics with the standard
library. This audit writes a new artifact and never changes frozen results.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
import argparse
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
ZONES = ('room_a', 'room_b', 'junction', 'north_corridor', 'south_corridor')
METRICS = ('mean_cumulative_fed', 'incapacitation_rate', 'evacuation_rate',
           'cumulative_congestion', 'episode_end_seconds',
           *(f'fed_{z}' for z in ZONES), *(f'person_seconds_{z}' for z in ZONES))
PROTECTED = {
    'outputs/local_llm_selection.json': '302c6db2c857811dd41d1ed36248d1d9a71f3e32108cd870ecfe02aa520ff721',
    'outputs/third_round_development_status.json': 'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0',
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def csv_read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, label, atol=1e-10):
    a, b = float(a), float(b)
    assert math.isfinite(a) and math.isfinite(b), label
    assert abs(a - b) <= atol, (label, a, b)


def classify(incap, fed):
    if incap > .001:
        return 'harmful'
    if incap < -.001:
        return 'beneficial'
    if fed > .001:
        return 'harmful'
    if fed < -.001:
        return 'beneficial'
    return 'small_or_neutral'


def run(full, continuations, analysis):
    full, continuations, analysis = map(lambda x: Path(x).resolve(), (full, continuations, analysis))
    prep = ROOT / 'outputs/direction_release_mechanism_preparation_20261007'
    plan, summary = read(prep / 'manifest.json'), read(analysis / 'summary.json')
    hashes = {}
    for p, expected in {**plan['sources'], **PROTECTED}.items():
        assert digest(ROOT / p) == expected, p
        hashes[p] = expected
    for folder in (full, continuations, analysis):
        assert folder.parent == ROOT / 'outputs'
        manifest = read(folder / 'completion_manifest.json')
        assert manifest['state'] == 'complete'
        for p, expected in manifest['files'].items():
            assert digest(ROOT / p) == expected, p
            hashes[p] = expected
        hashes[(folder / 'completion_manifest.json').relative_to(ROOT).as_posix()] = digest(folder / 'completion_manifest.json')
    for p, expected in summary['source_manifests'].items():
        assert digest(ROOT / p) == expected, p
    assert summary['protected_sha256'] == PROTECTED
    assert not summary['candidate_adopted'] and not summary['qualification_claim']

    rows = csv_read(full / 'episodes.csv')
    index = {r['case_id']: r for r in rows}
    assert len(rows) == len(index) == 1920
    expected_cases = set()
    for group, prefix in plan['population_groups'].items():
        for rec in plan['scene_records']:
            for rep in range(2):
                pop = prefix + rec['scene_index'] * 10 + rep
                for behavior in plan['behaviors']:
                    for mode in plan['modes']:
                        for hold in plan['hold_settings'][group]:
                            for seed in plan['training_seeds']:
                                for policy in plan['policies']:
                                    expected_cases.add('|'.join(map(str,(group,rec['scene']['scenario_id'],behavior,pop,seed,mode,hold,policy))))
                            expected_cases.add('|'.join(map(str,(group,rec['scene']['scenario_id'],behavior,pop,0,mode,hold,'pure_rule'))))
    assert set(index) == expected_cases
    assert Counter(r['policy'] for r in rows) == Counter(no_advice=600, direction_hint_only=600, preserve_release_hint=600, pure_rule=120)
    assert Counter(r['population_group'] for r in rows) == Counter(replay=1280, fresh_frozen=640)

    with (full / 'selected_states.jsonl').open(encoding='utf-8') as f:
        selected = {r['state_id']: r for r in map(json.loads, f)}
    assert len(selected) == 44
    accum = defaultdict(lambda: dict(steps=0, fed=[0.] * 5, people=[0] * 5, incap=[0] * 5, counts=Counter(), congestion=0.))
    changed_ids, actual_steps = set(), 0
    constraints = Counter(unavailable_entries=0, release_violations=0, direction_violations=0, action_mismatches=0)
    with (full / 'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for t in map(json.loads, f):
            actual_steps += 1
            a = accum[t['case_id']]
            assert t['source_step'] == a['steps']
            assert len(t['observation']) == 34 and not any(t['observation'][28:33])
            snap, action, base, hint = t['public_snapshot'], t['requested_action'], t['base_action'], t['original_hint_action']
            eligible = [not snap['exit_closed'][b] and snap['smoke_increment_per_step'][3+b] < .15 for b in range(2)]
            limit = max(0, min(20, sum(snap['exit_capacity_persons_per_step'][b] for b in range(2) if eligible[b]), 45 - round(snap['occupancy_fraction'][2]*240)))
            mask = [eligible[i//4] and (not snap['sign_hold_remaining_steps'] or i//4 == snap['actual_sign']) and (i%4+1)*5 <= limit for i in range(8)] + [True]
            assert mask[action] and mask[base] and mask[hint], (t['case_id'], t['source_step'])
            constraints['action_mismatches'] += not (action == t['applied_action'] == t['actual_joint_action'])
            constraints['release_violations'] += not (0 <= t['actual_room_count'] <= t['actual_room_capacity'] <= limit)
            constraints['direction_violations'] += action != 8 and (t['actual_sign'] != action//4 or not eligible[action//4])
            constraints['unavailable_entries'] += sum(int(t['branch_entries'][b]) for b in range(2) if not eligible[b])
            r = index[t['case_id']]
            if r['policy'] == 'preserve_release_hint' and base != action:
                if 8 not in (base, hint):
                    assert action//4 == hint//4 and action%4 == base%4
            if r['population_group'] == 'replay' and r['policy'] == 'direction_hint_only' and int(r['hold_steps']) == 2 and action != base:
                sid = f"{t['case_id']}|{t['source_step']}"
                s = selected[sid]
                assert t['gate_status'] == 'accepted' and (base,hint) == (s['base_action'],s['hint_action'])
                assert s['legal_actions'] == [i for i,v in enumerate(mask) if v]
                assert s['first_changed_state'] == (a['counts']['action_changed_steps'] == 0)
                assert s['observation'] == t['observation']
                for z in range(5):
                    close(s['past_zone_fed'][z], a['fed'][z], (sid,'past_fed',z))
                    close(s['past_zone_person_steps'][z], a['people'][z], (sid,'past_people',z))
                    close(s['past_zone_incapacitations'][z], a['incap'][z], (sid,'past_incap',z))
                changed_ids.add(sid)
            a['counts']['action_changed_steps'] += action != base
            a['counts']['same_direction_release_changed_steps'] += 8 not in (action,base) and action//4 == base//4 and action%4 != base%4
            a['counts']['release_preserve_wait_exceptions'] += t['release_matching'] == 'waiting_not_factorizable'
            a['counts']['release_preserve_infeasible_fallbacks'] += t['release_matching'] == 'matched_release_infeasible_keep_base'
            for z in range(5):
                assert t['zone_fed_increments'][z] >= 0
                assert t['zone_person_steps'][z] == round(snap['occupancy_fraction'][z]*240)
                a['fed'][z] += t['zone_fed_increments'][z]
                a['people'][z] += t['zone_person_steps'][z]
                a['incap'][z] += t['zone_new_incapacitations'][z]
            a['congestion'] += sum(max(0., n/45 - 1) for n in t['zone_counts_after'])
            a['steps'] += 1
            a['last_remaining'] = sum(t['zone_counts_after'])
    assert set(accum) == set(index) and changed_ids == set(selected)
    assert not any(constraints.values())
    for cid, a in accum.items():
        r = index[cid]
        for z, name in enumerate(ZONES):
            close(r[f'fed_{name}'], a['fed'][z]/240, (cid,name,'fed'))
            close(r[f'person_seconds_{name}'], a['people'][z]*5/240, (cid,name,'person_seconds'))
            close(r[f'incap_count_{name}'], a['incap'][z], (cid,name,'incap'))
        for k in ('action_changed_steps','same_direction_release_changed_steps','release_preserve_wait_exceptions','release_preserve_infeasible_fallbacks'):
            assert int(r[k]) == a['counts'][k]
        close(r['mean_cumulative_fed'], sum(a['fed'])/240, (cid,'total_fed'))
        close(r['incapacitation_rate'], sum(a['incap'])/240, (cid,'incap'))
        close(r['remaining'], a['last_remaining'], (cid,'remaining'))
        close(r['evacuation_rate'], 1-(sum(a['incap'])+a['last_remaining'])/240, (cid,'evacuated'))
        close(r['cumulative_congestion'], a['congestion'], (cid,'congestion'))
        close(r['episode_end_seconds'], a['steps']*5, (cid,'time'))
        if r['all_evacuated_seconds']:
            assert not sum(a['incap']) and not a['last_remaining']
            close(r['all_evacuated_seconds'], r['episode_end_seconds'], (cid,'success_time'))
        else:
            assert sum(a['incap']) or a['last_remaining']
    assert actual_steps == 94742

    old = csv_read(ROOT / 'outputs/direction_hint_fresh_population_confirmation_20261007T052903182504Z/episodes.csv')
    old_index = {(r['scenario_id'],r['behavior'],r['population_seed'],r['training_seed'],r['source_mode'],r['variant']):r for r in old}
    replay_count = 0
    for r in rows:
        if r['population_group'] != 'replay' or int(r['hold_steps']) != 2 or r['policy'] not in ('no_advice','direction_hint_only'):
            continue
        b = old_index[(r['scenario_id'],r['behavior'],r['population_seed'],r['training_seed'],r['source_mode'],r['policy'])]
        for m in ('mean_cumulative_fed','incapacitation_rate','evacuation_rate','cumulative_congestion','remaining','episode_end_seconds','action_changed_steps','same_direction_release_changed_steps','unavailable_entries','release_violations','direction_violations'):
            close(r[m],b[m],('old_replay',r['case_id'],m),atol=1e-12)
        assert r['all_evacuated_seconds'] == b['all_evacuated_seconds']
        replay_count += 1
    assert replay_count == 400

    pair_checks, fresh_pairs = 0, []
    for e in summary['full_effects']:
        pairs = []
        scene_pairs = defaultdict(list)
        for b in rows:
            if (b['population_group'],b['source_mode'],int(b['hold_steps']),b['policy']) != (e['population_group'],e['mode'],e['hold_steps'],e['baseline']):
                continue
            parts = b['case_id'].split('|')
            parts[-1] = e['candidate']
            if e['candidate'] == 'pure_rule':
                parts[4] = '0'
            r = index['|'.join(parts)]
            d = {m:float(r[m])-float(b[m]) for m in METRICS}
            pairs.append(d)
            scene_pairs[b['scenario_id']].append(d)
            if e['population_group'] == 'fresh_frozen' and e['candidate'] == 'preserve_release_hint':
                fresh_pairs.append(dict(case_id=r['case_id'],scenario_id=r['scenario_id'],behavior=r['behavior'],training_seed=int(r['training_seed']),population_seed=int(r['population_seed']),mode=r['source_mode'],baseline=e['baseline'],differences=d,outcome_class=classify(d['incapacitation_rate'],d['mean_cumulative_fed'])))
        assert len(pairs) == e['paired_comparisons'] == 100 and len(scene_pairs) == 5
        for m in METRICS:
            close(fmean(d[m] for d in pairs), e['mean_differences'][m], ('paired_mean',m))
            for scene, ds in scene_pairs.items():
                close(fmean(d[m] for d in ds),e['scene_mean_differences'][scene][m],('paired_scene',scene,m))
        assert Counter(classify(d['incapacitation_rate'],d['mean_cumulative_fed']) for d in pairs) == e['individual_pair_outcome_classes']
        pair_checks += 1
    assert pair_checks == 24

    cf = csv_read(continuations / 'terminal_rollouts.csv')
    assert len(cf) == 990
    cf_unique, groups = set(), defaultdict(list)
    actual_terminal_checks = 0
    for r in cf:
        sid = r['state_id']; s = selected[sid]
        key = (sid,int(r['forced_action']),r['future_policy'],int(r['hold_steps']))
        unique = (*key,int(r['future_seed']))
        assert unique not in cf_unique
        cf_unique.add(unique); groups[key].append(r)
        assert not any(int(r[k]) for k in ('unavailable_entries','release_violations','direction_violations'))
        close(sum(float(r[f'fed_{z}']) for z in ZONES),r['mean_cumulative_fed'],('cf_fed',unique))
        close(sum(int(r[f'incap_count_{z}']) for z in ZONES)/240,r['incapacitation_rate'],('cf_incap',unique))
        if r['future_kind'] == 'realized' and key[1:] == (s['hint_action'],'direction_hint_only',2):
            parent = index[s['case_id']]
            for m in (*METRICS,*(f'incap_count_{z}' for z in ZONES),'remaining'):
                close(r[m],parent[m],('cf_actual_parent',sid,m),atol=1e-12)
            actual_terminal_checks += 1
    assert actual_terminal_checks == 44
    suffix_steps = sum(accum[s['case_id']]['steps']-s['source_step'] for s in selected.values())
    assert suffix_steps == summary['exact_suffix_control_steps'] == 2106
    means = {k:{m:fmean(float(r[m]) for r in rs) for m in METRICS} for k,rs in groups.items()}
    state_rows = csv_read(analysis / 'state_effects.csv')
    assert len(state_rows) == 44
    factored = 0
    for r in state_rows:
        sid = r['state_id']; s = selected[sid]
        b,h,matched = s['base_action'],s['hint_action'],s['matched_release_action']
        expected_futures = {s['population_seed']+770000}
        if s['behavior'] != 'rule':
            expected_futures.update(plan['fresh_future_seeds'])
        for key, rs in groups.items():
            if key[0] == sid:
                assert {int(x['future_seed']) for x in rs} == expected_futures
        values = {a:means[(sid,a,'no_advice',2)] for a in s['legal_actions']}
        def diff(x,y): return {m:x[m]-y[m] for m in METRICS}
        d = diff(values[h],values[b]); dm = diff(values[matched],values[b])
        resumed = diff(means[(sid,h,'direction_hint_only',2)],means[(sid,b,'direction_hint_only',2)])
        nohold = diff(means[(sid,h,'no_advice',0)],means[(sid,b,'no_advice',0)])
        effects = dict(common_ppo=d,matched_release=dm,resume_hint=resumed,no_hold=nohold,
                       continuation_policy_interaction=diff(resumed,d),hold_interaction=diff(nohold,d))
        if 8 not in (b,h):
            direction,released = (h//4)*4+b%4,(b//4)*4+h%4
            if direction in values and released in values:
                effects['direction_at_base_release'] = diff(values[direction],values[b])
                effects['release_at_base_direction'] = diff(values[released],values[b])
                effects['direction_release_interaction'] = {m:values[h][m]-values[direction][m]-values[released][m]+values[b][m] for m in METRICS}
                effects['full_difference'] = d
                factored += 1
                for m in METRICS:
                    close(sum(effects[k][m] for k in ('direction_at_base_release','release_at_base_direction','direction_release_interaction')),d[m],('factor_identity',sid,m))
        for name, ds in effects.items():
            for m,value in ds.items():
                close(r[f'{name}_{m}'],value,('cf_effect',sid,name,m))
        assert r['outcome_class'] == classify(d['incapacitation_rate'],d['mean_cumulative_fed'])
        assert r['matched_outcome_class'] == classify(dm['incapacitation_rate'],dm['mean_cumulative_fed'])
    for g in summary['cf_groups']:
        rs = [r for r in state_rows if r['source_mode']==g['mode'] and (g['selection']=='all_changed_states' or r['first_changed_state']=='True')]
        assert len(rs) == g['states']
        assert Counter(r['outcome_class'] for r in rs) == g['actual_hint_classes']
        assert Counter(r['matched_outcome_class'] for r in rs) == g['matched_release_classes']
    for e in summary['hold_interactions']:
        settings = {x['hold_steps']:x for x in summary['full_effects'] if x['population_group']=='replay' and x['mode']==e['mode'] and x['candidate']==e['policy'] and x['baseline']=='no_advice'}
        for m in METRICS:
            close(settings[0]['mean_differences'][m]-settings[2]['mean_differences'][m],e['difference_of_differences_hold0_minus_hold2'][m],('hold_did',m))

    fresh_strata = []
    for mode in plan['modes']:
        for baseline in ('no_advice','direction_hint_only'):
            for scene in [x['scene']['scenario_id'] for x in plan['scene_records']]:
                for behavior in plan['behaviors']:
                    for seed in plan['training_seeds']:
                        pairs = [r for r in fresh_pairs if (r['mode'],r['baseline'],r['scenario_id'],r['behavior'],r['training_seed'])==(mode,baseline,scene,behavior,seed)]
                        assert len(pairs)==2
                        fresh_strata.append(dict(mode=mode,baseline=baseline,scenario_id=scene,behavior=behavior,training_seed=seed,populations=2,
                            mean_differences={m:fmean(r['differences'][m] for r in pairs) for m in METRICS},
                            outcome_classes=dict(Counter(r['outcome_class'] for r in pairs))))
    improved = [r for r in fresh_pairs if r['baseline']=='direction_hint_only' and any(abs(v)>1e-12 for v in r['differences'].values())]
    concentration = dict(changed_pairs=len(improved),scenes=sorted({r['scenario_id'] for r in improved}),training_seeds=sorted({r['training_seed'] for r in improved}),
                         pairs_per_mode=dict(Counter(r['mode'] for r in improved)),
                         note='Two timing modes reuse paired populations; these are not independent people or buildings.')
    assert concentration['changed_pairs']==8 and concentration['scenes']==['AULL0010'] and concentration['training_seeds']==[605]
    result = dict(state='passed',full_episodes=len(rows),actual_control_steps=actual_steps,old_exact_replays=replay_count,
        terminal_counterfactuals=len(cf),selected_actual_changed_states=len(selected),realized_suffixes=actual_terminal_checks,
        realized_suffix_control_steps=suffix_steps,factored_states=factored,paired_effect_groups_recomputed=pair_checks,
        physical_violations_and_action_mismatches=dict(constraints),
        wait_exceptions=sum(int(r['release_preserve_wait_exceptions']) for r in rows),infeasible_fallbacks=sum(int(r['release_preserve_infeasible_fallbacks']) for r in rows),
        fresh_improvement_concentration=concentration,protected_sha256=PROTECTED,
        hashes_checked=len(hashes),candidate_adopted=False,new_training=0,new_llm_requests=0,reserved_final_scenarios_opened=0,
        audit_method='Independent standard-library calculations from frozen output; no simulator, PPO or LLM invocation.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'outputs'/f'direction_release_mechanism_independent_audit_{stamp}'
    out.mkdir(exist_ok=False)
    for name,obj in [('summary.json',result),('fresh_behavior_weight_strata.json',fresh_strata),('fresh_paired_differences.json',fresh_pairs),('checked_source_hashes.json',hashes)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    completion=dict(state='complete',files={p.relative_to(ROOT).as_posix():digest(p) for p in [*out.iterdir(),Path(__file__)] if p.is_file()})
    (out/'completion_manifest.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),**result),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',required=True)
    parser.add_argument('--continuations',required=True)
    parser.add_argument('--analysis',required=True)
    args=parser.parse_args()
    run(args.full,args.continuations,args.analysis)
