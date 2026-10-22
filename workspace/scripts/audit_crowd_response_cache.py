"""Read only the named development cache; no env, weights, or final-bank loads."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fire_evac import crowd_response_comparison as m
from fire_evac.repeat_override_gate import evidence_class

OUT = ROOT/'outputs/research_convergence_20261011'
FULL = ROOT/'outputs/repeat_override_confirmation_full_20261007T144941461333Z'
CF = ROOT/'outputs/repeat_override_confirmation_continuations_20261007T145327415464Z'
ANALYSIS = ROOT/'outputs/repeat_override_confirmation_analysis_20261007T145724470047Z'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify():
    snapshot = read(OUT/'preservation_before.json')
    mismatches = [p for p, h in snapshot['files'].items()
                  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
    assert not mismatches, mismatches
    return len(snapshot['files'])


def collapse(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['offline_identity']].append(r)
    result = []
    for identity, copies in sorted(groups.items()):
        first = copies[0]
        for r in copies[1:]:
            assert r['features'] == first['features']
            assert np.allclose([r['incap_delta'],r['fed_delta']],
                               [first['incap_delta'],first['fed_delta']], rtol=0., atol=1e-12)
        result.append(dict(first, copies=[r['state_id'] for r in copies], offline_identity=identity))
    return result


def main():
    verify()
    selected = [json.loads(x) for x in (FULL/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    needs = {r['case_id']: max(s['source_step'] for s in selected if s['case_id']==r['case_id']) for r in selected}
    traces = defaultdict(dict)
    count = 0
    with (FULL/'actual_step_traces.jsonl').open(encoding='utf-8') as f:
        for t in map(json.loads, f):
            count += 1
            if t['case_id'] in needs and t['source_step'] <= needs[t['case_id']]:
                assert t['source_step'] not in traces[t['case_id']]
                traces[t['case_id']][t['source_step']] = t
    cfs = {}
    with (CF/'terminal_rollouts.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['future_policy'] == 'common_ppo':
                key = (r['state_id'], int(r['future_seed']), int(r['forced_action']))
                assert key not in cfs
                cfs[key] = r
    old_effects = {r['state_id']: r for r in read(ANALYSIS/'event_effects.json')}
    rows = []
    for s in selected:
        ctx = m.context()
        prefix = []
        for step in range(s['source_step']):
            t = traces[s['case_id']][step]
            # Only strictly completed past steps, no selected-step flows.
            m.observe_completed(ctx, t['public_snapshot'], t['requested_action'], t, t)
            prefix.append([t['observation'], t['requested_action'], t['actual_joint_action'],
                           ctx['history'][-1]])
        t = traces[s['case_id']][s['source_step']]
        assert ctx['offers'] == s['offers_before'] and ctx['overrides'] == s['overrides_before']
        assert t['source_step'] == t['public_snapshot']['step'] == len(ctx['history'])
        base, candidate = s['base_action'], s['candidate_action']
        assert base != candidate and base in s['legal_actions'] and candidate in s['legal_actions']
        assert base != 8 and candidate != 8 and base % 4 == candidate % 4
        feat = {v: m.features(t['observation'], t['public_snapshot'], base, candidate, ctx, v) for v in m.VARIANTS}
        seeds = sorted(seed for sid, seed, action in cfs if sid == s['state_id'] and action == base)
        assert len(seeds) == (1 if s['behavior']=='rule' else 4)
        paired = []
        for seed in seeds:
            a = cfs[(s['state_id'],seed,candidate)]
            b = cfs[(s['state_id'],seed,base)]
            paired.append(dict(future_seed=seed, incap_delta=float(a['incapacitation_rate'])-float(b['incapacitation_rate']),
                               fed_delta=float(a['mean_cumulative_fed'])-float(b['mean_cumulative_fed'])))
        inc = float(np.mean([p['incap_delta'] for p in paired]))
        fed = float(np.mean([p['fed_delta'] for p in paired]))
        old = old_effects[s['state_id']]['common_ppo']
        assert abs(inc-old['incapacitation_rate']) < 1e-12 and abs(fed-old['mean_cumulative_fed']) < 1e-12
        offline = [s['scenario_id'],s['behavior'],s['population_seed'],s['training_seed'],
                   s['source_step'],base,candidate,prefix,t['observation'],s['past_zone_fed'],ctx['offers'],ctx['overrides']]
        public = [m.public_projection(t['public_snapshot']), ctx, base, candidate,
                  feat['current']]
        rows.append(dict(state_id=s['state_id'],scenario=s['scenario_id'],behavior=s['behavior'],
                         population=s['population_seed'],checkpoint=s['training_seed'],step=s['source_step'],
                         source_mode=s['source_mode'],policy=s['policy'],family=f"{s['scenario_id']}|{s['population_seed']}",
                         offline_identity=digest(offline),public_history_identity=digest(public),
                         feature_identity={v:digest(feat[v]) for v in m.VARIANTS},
                         base=base,candidate=candidate,offers=ctx['offers'],overrides=ctx['overrides'],
                         features=feat,incap_delta=inc,fed_delta=fed,label=evidence_class(inc,fed),
                         paired_futures=paired,completed_public_history=ctx['history']))
    all_unique = collapse(rows)
    primary = collapse([r for r in rows if r['policy']=='preserve_always'])
    cells = defaultdict(list)
    for r in primary:
        cells[r['public_history_identity']].append(r)
    collisions = []
    for key, rs in cells.items():
        classes = set(r['label'] for r in rs)
        if 'beneficial' in classes and 'harmful' in classes:
            collisions.append(dict(identity=key, states=len(rs), classes=dict(Counter(r['label'] for r in rs)),
                                   state_ids=[r['state_id'] for r in rs],
                                   fed_range=[min(r['fed_delta'] for r in rs),max(r['fed_delta'] for r in rs)],
                                   incap_range=[min(r['incap_delta'] for r in rs),max(r['incap_delta'] for r in rs)]))
    reversal = [r for r in primary if r['scenario']=='AULL0019' and r['checkpoint']==604 and r['step']==3
                and ((r['population']==1497101301 and r['behavior']=='calibrated_logit')
                     or (r['population']==1497101302 and r['behavior']=='rule'))]
    assert len(reversal)==2
    a,b = reversal
    paired_report = dict(records=[{k:r[k] for k in ('state_id','behavior','population','incap_delta','fed_delta','label')}
                                  for r in reversal],
                         identical_full_recorded_public_history=a['public_history_identity']==b['public_history_identity'],
                         identical_representations={v:a['features'][v]==b['features'][v] for v in m.VARIANTS},
                         completed_steps=a['step'],completed_branch_entries=[[r['entries'] for r in x['completed_public_history']] for x in reversal],
                         fed_gap=abs(a['fed_delta']-b['fed_delta']))
    before = verify()
    summary = dict(source_records=len(rows),all_policy_unique_states=len(all_unique),
                   common_path_source_records=sum(r['policy']=='preserve_always' for r in rows),
                   primary_unique_states=len(primary),families=len({r['family'] for r in primary}),
                   scenarios=dict(Counter(r['scenario'] for r in primary)),steps=dict(Counter(r['step'] for r in primary)),
                   class_counts=dict(Counter(r['label'] for r in primary)),
                   repeat_class_counts=dict(Counter(r['label'] for r in primary if r['offers']>0)),
                   feature_dimensions={v:len(m.feature_names(v)) for v in m.VARIANTS},
                   unique_feature_cells={v:len({r['feature_identity'][v] for r in primary}) for v in m.VARIANTS},
                   exact_public_history_cells=len(cells),conflicting_cells=collisions,
                   known_reversal=paired_report,scanned_cached_steps=count,
                   preserved_files=before,new_ppo_training=0,new_llm_requests=0,new_simulations=0,reserved_final_opened=0,
                   justified_next='One frozen descriptive representation comparison only; five same-graph scenes cannot validate transport to new buildings.')
    save(OUT/'audit_summary.json',summary)
    save(OUT/'primary_rows.json',primary)
    save(OUT/'all_policy_deduplicated_rows.json',all_unique)
    save(OUT/'feature_names.json',{v:m.feature_names(v) for v in m.VARIANTS})
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
