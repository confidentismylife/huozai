"""Read-only cached observation/label reversal diagnosis; never runs an environment."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/five_ppo_fresh_family_student_development_v1'
OUT = ROOT / 'outputs/observation_reversal_cache_audit_20261005'
STATUS = ROOT / 'outputs/third_round_development_status.json'
HISTORY = ROOT / 'outputs/third_round_status_history/before_observation_reversal_audit_20261005.json'
PROTOCOL = ROOT / 'reports/相似初始观测收益反转缓存审计固定方案_2026-10-05.md'
REPORT = ROOT / 'reports/相似初始观测收益反转缓存审计报告_2026-10-05.md'
PAIR = ('F5TR0006', 'F5TE0018')
FLOOR = .001


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def stamp(): return datetime.now(timezone.utc).isoformat()
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def csv_write(name, rows):
    assert rows
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def cls(v): return 'positive' if v > FLOOR else 'negative' if v < -FLOOR else 'neutral_or_small'
def reversal(a, b): return (a > FLOOR and b < -FLOOR) or (b > FLOOR and a < -FLOOR)


def scores(losses, supported):
    """Pool fixed populations and futures BEFORE either nonlinear world operation."""
    a, s = np.asarray(losses, float), np.asarray(supported)
    if a.ndim != 4 or a.shape[1:3] != (2, 5) or s.shape != (2,) or s.dtype != bool or not s.any():
        raise ValueError('population,world2,plan5,future array with boolean support required')
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)): raise ValueError('fraction losses required')
    l = a.mean(axis=(0, 3)); active = l[s]
    r = active[:, 0].max() - active.max(axis=0)
    m = (active[:, 0, None] - active).min(axis=0)
    r[0] = m[0] = 0.
    return l, np.stack([r, m])


def kernel_decomposition(feature, model):
    """Algebraic centered-kernel terms; not weights on labels or causal attribution."""
    x = (np.asarray(feature, float)-model['mean']) / model['std']
    d = ((model['train']-x)**2).sum(axis=1)
    k = np.exp(-float(model['gamma'])*d)
    centered = k - k@model['weights'] - model['kernel_row_mean'] + float(model['kernel_grand_mean'])
    terms = centered[:, None]*model['coefficient']
    prediction = np.r_[0., terms.sum(axis=0)+model['intercept']]
    return d, k, centered, terms, prediction


def features():
    names = ['room_a_occupancy','room_b_occupancy','junction_occupancy','north_occupancy','south_occupancy',
             'room_a_smoke','room_b_smoke','junction_smoke','north_smoke','south_smoke',
             'last_sign_north','last_sign_south','last_sign_neutral','active_fraction',
             'live_north_capacity','live_south_capacity','north_unavailable','south_unavailable',
             'blockage_remaining','forecast_source_north','forecast_source_south','forecast_growth','forecast_onset',
             'nominal_north_capacity','nominal_south_capacity','forecast_blocked_north','forecast_blocked_south','forecast_blockage_end',
             'advice_north','advice_south','advice_confidence','advice_valid','advice_contradiction','previous_release_capacity',
             'elapsed','sign_hold_remaining','actual_sign_north','actual_sign_south','actual_sign_neutral',
             'room_a_smoke_integral','room_b_smoke_integral','junction_smoke_integral','north_smoke_integral','south_smoke_integral',
             'room_a_smoke_difference','room_b_smoke_difference','junction_smoke_difference','north_smoke_difference','south_smoke_difference']
    labels = ['A房人数/240','B房人数/240','汇合点人数/240','北廊人数/240','南廊人数/240',
              'A房烟气/FED增量代理','B房烟气/FED增量代理','汇合点烟气/FED增量代理','北廊烟气/FED增量代理','南廊烟气/FED增量代理',
              '上次北指示','上次南指示','上次中性指示','尚在场人数比例',
              '当前北出口容量/8','当前南出口容量/8','北出口当前容量<=0','南出口当前容量<=0',
              '封闭剩余步数/20(裁剪)','预测火源北','预测火源南','火势增长率/0.05(裁剪)','起火步数/20(裁剪)',
              '名义北出口容量/8','名义南出口容量/8','预报封闭北','预报封闭南','封闭结束步数/20(裁剪)',
              '建议北','建议南','建议置信度','建议有效','建议与烟气矛盾','上次房间放行容量/20',
              '已过步数/120','指示保持剩余步数/2','执行北指示','执行南指示','执行中性指示',
              'A房历史烟气和/120','B房历史烟气和/120','汇合点历史烟气和/120','北廊历史烟气和/120','南廊历史烟气和/120',
              'A房烟气差分','B房烟气差分','汇合点烟气差分','北廊烟气差分','南廊烟气差分']
    assert len(names) == len(labels) == 49
    rows = []
    for i in range(49):
        if i < 14: origin, line = 'src/fire_evac/env.py', 91
        elif i < 18: origin, line = 'src/fire_evac/dynamic_conflicts.py', 300
        elif i < 34: origin, line = 'src/fire_evac/confirmatory.py', 143
        else: origin, line = 'src/fire_evac/terminal_shield.py', 49
        group = ('initial_smoke_background' if 5 <= i < 10 else
                 'north_capacity_duplicate' if i in (14,23) else
                 'south_capacity_duplicate' if i in (15,24) else
                 'growth' if i == 21 else 'onset' if i == 22 else 'other')
        rows.append({'index':i,'name':names[i],'meaning':labels[i],'source':origin,'source_line':line,'distance_group':group,'masked_advice':28 <= i < 33})
    return rows


def preserve(status_path):
    from finalize_set_aware_distillation import check_completion
    hd = ROOT / 'outputs/third_round_status_history'
    mappings = [
        ('five_ppo_fresh_family_student_development_v1',status_path),
        ('ppo_candidate_label_interface_development_v1',hd/'before_five_ppo_fresh_student_20261004.json'),
        ('candidate_library_coverage_audit_development_v1',hd/'before_ppo_candidate_label_interface_20261003.json'),
        ('proposal_gain_factorial_development_v1',hd/'before_candidate_library_coverage_20261003.json'),
        ('gain_target_alignment_diagnostic_development_v1',hd/'before_proposal_gain_factorial_20261003.json'),
        ('override_worth_evidence_development_v1',hd/'before_gain_target_alignment_diagnostic_20261003.json'),
        ('consistent_gain_crossfit_development_v1',hd/'before_override_worth_evidence_20261003.json'),
        ('set_aware_distillation_development_v1',hd/'before_consistent_gain_crossfit_20261003.json'),
        ('crossed_teacher_rng_diagnostic_development_v1',hd/'before_set_aware_distillation_20261003.json'),
        ('observable_teacher_label_audit_development_v1',hd/'before_crossed_teacher_rng_diagnostic_20261003.json'),
        ('teacher_factorial_pilot_development_v1',hd/'before_observable_teacher_label_audit_20261003.json'),
        ('research_value_quantification_20261003',status_path)]
    counts = {name: check_completion(name,path) for name,path in mappings}
    assert counts['five_ppo_fresh_family_student_development_v1'] == 272
    m = read(SOURCE/'development_manifest.json')
    for p,h in m['source_input_sha256'].items(): assert sha(p) == h, p
    for p,h in read(ROOT/'outputs/research_value_quantification_20261003/completion_manifest.json')['sources_unchanged'].items(): assert sha(ROOT/p) == h
    hist = read(ROOT/'artifacts/full_history_report_assets/experiment_preservation_audit.json')
    unique = {Path(r['path']).resolve():r['current_sha256'] for r in hist['records']}
    for p,h in unique.items(): assert sha(hd/'before_teacher_diagnosis_and_factorial_pilot.json' if p == STATUS else p) == h
    for r in read(ROOT/'artifacts/full_history_report_assets/report_qa_manifest.json')['artifacts']: assert sha(r['path']) == r['sha256']
    core = read(ROOT/'outputs/set_aware_distillation_development_v1/packaging_verification.json')['legacy_core_checks']
    for r in core: assert sha(r['file']) == r['sha256']
    return {'completion_files_checked':counts,'frozen_sources_unchanged':len(m['source_input_sha256']),
            'historical_records':len(hist['records']),'historical_unique_files':len(unique),'legacy_core_files':len(core)}


def prepare():
    assert not OUT.exists() and not HISTORY.exists()
    checks = preserve(STATUS)
    OUT.mkdir()
    (OUT/'previous_status_exact_bytes.json').write_bytes(STATUS.read_bytes())
    sourcepaths = set(SOURCE.rglob('*'))
    sourcepaths = {p for p in sourcepaths if p.is_file()}
    sourcepaths.update(ROOT / p for p in features_source_files())
    write(OUT/'development_manifest.json', {'role':'post_outcome_development_cache_diagnosis_not_confirmation',
        'created_at_utc':stamp(),'date':'2026-10-05','priority_pair':list(PAIR),'all_bank_families':60,'test_nearest_pairs':24,
        'all_pair_collision_comparisons':1770,'minimum_gain':FLOOR,'nearest_neighbor_count_displayed':6,
        'protocol':PROTOCOL.relative_to(ROOT).as_posix(),'protocol_sha256':sha(PROTOCOL),
        'previous_status_sha256':sha(STATUS),'source_completion_sha256':sha(SOURCE/'completion_manifest.json'),
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sourcepaths)},
        'preservation_before':checks,'new_simulations':0,'new_model_fits':0,'new_ppo_updates':0,'new_llm_calls':0,
        'qualification_thresholds_changed':False,'no_model_or_margin_changes':True,'cache_only':True})
    print(json.dumps({'prepared':True,'checks':checks,'source_files':len(sourcepaths)},ensure_ascii=False),flush=True)


def features_source_files():
    return ['src/fire_evac/env.py','src/fire_evac/dynamic_conflicts.py','src/fire_evac/hazard.py',
            'src/fire_evac/confirmatory.py','src/fire_evac/terminal_shield.py','src/fire_evac/config.py',
            'src/fire_evac/behavior.py','src/fire_evac/source_world_planning.py','src/fire_evac/ppo_reference_planning.py',
            'scripts/run_source_world_planning.py','scripts/five_ppo_student.py','scripts/candidate_label_interface.py']


def check_inputs():
    m = read(OUT/'development_manifest.json')
    assert sha(PROTOCOL) == m['protocol_sha256'] and sha(STATUS) == m['previous_status_sha256']
    assert sha(OUT/'previous_status_exact_bytes.json') == sha(STATUS)
    for p,h in m['source_sha256'].items(): assert sha(ROOT/p) == h, p


def analyze():
    check_inputs(); assert not (OUT/'summary.json').exists()
    m = read(SOURCE/'development_manifest.json'); train = load(SOURCE/'training_dataset.npz')
    models = {target:load(SOURCE/f'models/{target}_kernel.npz') for target in ('R','M')}
    cache, scenarios, registry = {}, {}, {e['family']:e for e in m['seed_registry']}
    for block, bank in m['banks'].items():
        for scene in bank:
            family = scene['scenario_id']; scenarios[family] = scene
            z = load(SOURCE/f'{block}_planning/{family}.npz'); cache[family] = z
            np.testing.assert_array_equal(z['observations'][0],z['observations'][1])
            np.testing.assert_array_equal(z['observation'],z['observations'][0])
            f = z['observation'].copy(); f[28:33] = 0; np.testing.assert_array_equal(f,z['features'])
            l,y = scores(z['planning_risks'][...,0],z['supported'])
            np.testing.assert_allclose(l,z['pooled_world_losses'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(y,z['pooled_targets'],atol=1e-12,rtol=0)
    a,b = (cache[k] for k in PAIR); model = models['M']
    diff = b['features'].astype(float)-a['features'].astype(float); distance_contrib = (diff/model['std'])**2
    feature_rows = []
    for f in features():
        i = f['index']; feature_rows.append({**f,'train_raw':float(a['observation'][i]),'test_raw':float(b['observation'][i]),
            'train_feature':float(a['features'][i]),'test_feature':float(b['features'][i]),'test_minus_train_raw':float(b['observation'][i]-a['observation'][i]),
            'training_std':float(model['std'][i]),'squared_standardized_distance':float(distance_contrib[i]),
            'distance_share':float(distance_contrib[i]/distance_contrib.sum()) if distance_contrib.sum() else 0.})
    csv_write('feature_comparison.csv',feature_rows)
    grouped = {}
    for r in feature_rows: grouped[r['distance_group']] = grouped.get(r['distance_group'],0.)+r['squared_standardized_distance']
    obsclass = {'scenario_id':('metadata_only','family id; not an input'),
        'source_branch':('latent_at_reset','initial smoke does not identify source; each planning world sets this'),
        'growth_rate':('direct_unclipped_in_this_bank','index21 = growth/.05; generated growth <.05'),
        'onset_step':('direct_unclipped_in_this_bank','index22 = onset/20; generated onset <20'),
        'background_hazard':('recoverable_at_reset','all five initial smoke indices5:10 equal background'),
        'conflict_type':('metadata_only','type is not an input; does not drive dynamics once physical parameters fixed'),
        'north_exit_capacity':('direct','index23 nominal; index14 current capacity when unblocked'),
        'south_exit_capacity':('direct','index24 nominal; index15 current capacity when unblocked'),
        'blocked_branch':('direct','indices25:27 plus live availability; -1 means no blockage'),
        'blockage_start_step':('not_explicit_but_fixed_zero','all60 scenes =0'),
        'blockage_end_step':('direct_unclipped_in_this_bank','index27 = end/20; generated end <20'),
        'blockage_capacity':('not_explicit_but_fixed_zero','all60 scenes =0; live capacity observed during blockage'),
        'forecast_source_branch':('direct','indices19:21 onehot; supplied forecast can disagree with source')}
    scenario_rows = []
    for key in scenarios[PAIR[0]]:
        typ, note = obsclass[key]
        scenario_rows.append({'parameter':key,'train_value':scenarios[PAIR[0]][key],'test_value':scenarios[PAIR[1]][key],
            'different':scenarios[PAIR[0]][key] != scenarios[PAIR[1]][key],'observability':typ,'evidence':note})
    csv_write('scenario_comparison.csv',scenario_rows)
    assert all(s['blockage_start_step']==s['blockage_capacity']==0 and s['growth_rate']<.05 and s['onset_step']<20 and s['blockage_end_step']<20 for s in scenarios.values())

    worldrows, slicerows, gainrows, snapshot = [], [], [], {}
    for family in PAIR:
        z = cache[family]; raw = z['planning_risks']; losses = raw[...,0]; l,y = scores(losses,z['supported'])
        snapshot[family] = {'scenario':scenarios[family],'registry':registry[family],'supported':z['supported'].tolist(),
            'world_errors':z['world_errors'].tolist(),'world_losses_pp':(l*100).tolist(),'R_pp':(y[0]*100).tolist(),'M_pp':(y[1]*100).tolist(),
            'initial_observations_equal_across_populations':True,'true_source_world':scenarios[family]['source_branch']}
        scopes = [('pooled',slice(None),slice(None)),('population0',slice(0,1),slice(None)),('population1',slice(1,2),slice(None)),
            ('first3_futures',slice(None),slice(0,3)),('last3_futures',slice(None),slice(3,6)),
            ('population0_first3',slice(0,1),slice(0,3)),('population0_last3',slice(0,1),slice(3,6)),
            ('population1_first3',slice(1,2),slice(0,3)),('population1_last3',slice(1,2),slice(3,6))]
        for scope,ps,fs in scopes:
            sl, sy = scores(losses[ps,:,:,fs],z['supported'])
            for j in range(5):
                slicerows.append({'family':family,'scope':scope,'candidate':m['candidate_names'][j],'candidate_index':j,
                    'north_loss_pp':float(sl[0,j]*100),'south_loss_pp':float(sl[1,j]*100),'R_pp':float(sy[0,j]*100),'M_pp':float(sy[1,j]*100),
                    'R_class':cls(sy[0,j]),'M_class':cls(sy[1,j])})
        for w in range(2):
            paired = losses[:,w,0,None,:] - losses[:,w,:,:]
            for j in range(5):
                row = {'family':family,'world':w,'world_direction':('north','south')[w],'supported':bool(z['supported'][w]),
                    'actual_source_world':w==scenarios[family]['source_branch'],'candidate':m['candidate_names'][j],'candidate_index':j,
                    'baseline_loss_pp':float(l[w,0]*100),'candidate_loss_pp':float(l[w,j]*100),'gain_pp':float((l[w,0]-l[w,j])*100),
                    'gain_min_pp':float(paired[:,j,:].min()*100),'gain_max_pp':float(paired[:,j,:].max()*100),
                    'gain_positive_count':int((paired[:,j,:]>FLOOR).sum()),'gain_negative_count':int((paired[:,j,:]<-FLOOR).sum()),
                    'gain_neutral_or_small_count':int((np.abs(paired[:,j,:])<=FLOOR).sum()),
                    'R_baseline_worst_world':int(np.argmax(l[:,0])),'R_candidate_worst_world':int(np.argmax(l[:,j])),
                    'M_binding_world':int(np.argmin(l[:,0]-l[:,j]))}
                for oi,name in enumerate(m['outcome_fields']):row['mean_'+name] = float(raw[:,w,j,:,:].mean(axis=(0,1))[oi])
                worldrows.append(row)
                for p in range(2):
                    for f in range(6):
                        gainrows.append({'family':family,'world':w,'candidate_index':j,'population':p,'future_index':f,
                            'population_seed':int(z['population_seeds'][p]),'planning_seed':int(z['planning_seeds'][p,f]),
                            'baseline_loss_pp':float(losses[p,w,0,f]*100),'candidate_loss_pp':float(losses[p,w,j,f]*100),
                            'gain_pp':float(paired[p,j,f]*100),'class':cls(paired[p,j,f])})
    csv_write('pair_world_outcomes.csv',worldrows); csv_write('pair_fixed_splits.csv',slicerows); csv_write('pair_seed_gains.csv',gainrows)
    write(OUT/'pair_snapshot.json',snapshot)

    neighbors, termrows, predictions = [], [], {}
    for target, mm in models.items():
        d,k,c,t,pred = kernel_decomposition(b['features'],mm); choices=read(SOURCE/f'test_choices/{PAIR[1]}.json')
        np.testing.assert_allclose(pred,choices['predictions'][target+'_kernel'],atol=1e-12,rtol=0)
        order=np.argsort(d); ranks=np.empty(len(d),int);ranks[order]=np.arange(1,len(d)+1)
        predictions[target] = {'prediction_pp':(pred*100).tolist(),'intercept_pp':(mm['intercept']*100).tolist(),
            'term_sum_pp':(t.sum(axis=0)*100).tolist(),'gamma':float(mm['gamma']),
            'closest_family':str(train['families'][order[0]]),'closest_similarity':float(k[order[0]]),'closest_distance':float(d[order[0]]),
            'closest_candidate603_term_pp':float(t[order[0],1]*100)}
        for i in order:
            for j in range(1,5):
                termrows.append({'target':target,'family':str(train['families'][i]),'rank':int(ranks[i]),'candidate_index':j,
                    'squared_standardized_distance':float(d[i]),'kernel_similarity':float(k[i]),'centered_kernel':float(c[i]),
                    'coefficient':float(mm['coefficient'][i,j-1]),'contribution_pp':float(t[i,j-1]*100),
                    'training_label_pp':float(train['targets'][i,('R','M').index(target),j]*100)})
            if ranks[i] <= 6:
                neighbors.append({'target':target,'rank':int(ranks[i]),'family':str(train['families'][i]),'distance':float(d[i]),
                    'similarity':float(k[i]),**{f'label_{601+j}_pp':float(train['targets'][i,('R','M').index(target),j]*100) for j in range(1,5)}})
    csv_write('nearest_six_training_families.csv',neighbors);csv_write('kernel_contributions.csv',termrows)

    actual = load(SOURCE/f'test_actual/{PAIR[1]}.npz'); outcome=actual['evaluation_outcomes']; trueworld=scenarios[PAIR[1]]['source_branch']
    actualmeans=outcome.mean(axis=(0,2)); actualgain=actualmeans[0,0]-actualmeans[:,0]
    actualrows=[]
    for j in range(5):
        ga=outcome[:,0,:,0]-outcome[:,j,:,0]
        actualrows.append({'family':PAIR[1],'candidate_index':j,'candidate':m['candidate_names'][j],
            **{name:float(actualmeans[j,k]) for k,name in enumerate(m['outcome_fields'])},
            'actual_loss_pp':float(actualmeans[j,0]*100),'actual_gain_pp':float(actualgain[j]*100),
            'true_world_planning_loss_pp':float(b['pooled_world_losses'][trueworld,j]*100),
            'true_world_planning_gain_pp':float((b['pooled_world_losses'][trueworld,0]-b['pooled_world_losses'][trueworld,j])*100),
            'actual_minus_planning_loss_pp':float((actualmeans[j,0]-b['pooled_world_losses'][trueworld,j])*100),
            'actual_gain_min_pp':float(ga.min()*100),'actual_gain_max_pp':float(ga.max()*100),
            'actual_gain_positive_count':int((ga>FLOOR).sum()),'actual_gain_negative_count':int((ga<-FLOOR).sum())})
    csv_write('test18_actual_outcomes.csv',actualrows)
    episodetrace, steprows, byplan = [], [], {j:[] for j in range(5)}
    for i,key in enumerate(actual['trace_keys']):
        p,j,f=map(int,key);tr=actual['execution_trace'][actual['trace_offsets'][i]:actual['trace_offsets'][i+1]]
        np.testing.assert_array_equal(tr[:,0],np.arange(len(tr)))
        assert len(tr)==outcome[p,j,f,5] and tr[0,1]==outcome[p,j,f,9] and tr[0,2]==outcome[p,j,f,10]
        np.testing.assert_allclose(tr[:,4].mean(),outcome[p,j,f,6],atol=1e-12,rtol=0)
        assert (tr[:,1]==8).sum()==outcome[p,j,f,7]
        np.testing.assert_allclose(tr[:,5].mean(),outcome[p,j,f,8],atol=1e-12,rtol=0)
        byplan[j].append(tr)
        row={'population':p,'candidate_index':j,'future_index':f,'steps':len(tr),'first_requested':int(tr[0,1]),'first_actual':int(tr[0,2]),
             'north_sign_steps':int((tr[:,3]==0).sum()),'south_sign_steps':int((tr[:,3]==1).sum()),'neutral_sign_steps':int((tr[:,3]==2).sum()),
             'mean_capacity_fraction':float(tr[:,4].mean()),'wait_requests':int((tr[:,1]==8).sum()),'hold_blocks':int(tr[:,5].sum()),
             'neutral_positive_release_steps':int(((tr[:,3]==2)&(tr[:,4]>0)).sum()),
             'unrepresented_joint_steps':int((tr[:,2]==-1).sum()),'actual_loss_pp':float(outcome[p,j,f,0]*100)}
        episodetrace.append(row)
    traceagg=[]
    for j,traces in byplan.items():
        tr=np.concatenate(traces); onset=scenarios[PAIR[1]]['onset_step']
        for phase in ('all','before_growth','growth_started'):
            st=tr if phase=='all' else tr[tr[:,0]<=onset] if phase=='before_growth' else tr[tr[:,0]>onset]
            traceagg.append({'candidate_index':j,'phase':phase,'step_count':len(st),'north_sign_steps':int((st[:,3]==0).sum()),
                'south_sign_steps':int((st[:,3]==1).sum()),'neutral_sign_steps':int((st[:,3]==2).sum()),
                'mean_capacity_fraction_step_weighted':float(st[:,4].mean()),'wait_requests':int((st[:,1]==8).sum()),
                'hold_blocks':int(st[:,5].sum()),'neutral_positive_release_steps':int(((st[:,3]==2)&(st[:,4]>0)).sum()),
                'note':'capacity command, not actual persons released; elapsed0 applies hazard at step0'})
        for elapsed in range(int(tr[:,0].max())+1):
            st=tr[tr[:,0]==elapsed]
            steprows.append({'candidate_index':j,'elapsed':elapsed,'episodes_still_running':len(st),
                'north_sign_count':int((st[:,3]==0).sum()),'south_sign_count':int((st[:,3]==1).sum()),'neutral_sign_count':int((st[:,3]==2).sum()),
                'mean_capacity_fraction':float(st[:,4].mean()),'wait_requests':int((st[:,1]==8).sum()),'hold_blocks':int(st[:,5].sum())})
    csv_write('test18_actual_trace_episodes.csv',episodetrace);csv_write('test18_actual_trace_phases.csv',traceagg);csv_write('test18_actual_trace_by_step.csv',steprows)

    broadrows=[]
    for e in m['seed_registry']:
        if e['block']!='test':continue
        fam=e['family'];z=cache[fam];d,k,_,_,_=kernel_decomposition(z['features'],models['M']);ix=int(d.argmin());near=str(train['families'][ix])
        ac=load(SOURCE/f'test_actual/{fam}.npz')['evaluation_outcomes'][...,0].mean(axis=(0,2));g=ac[0]-ac
        pred=read(SOURCE/f'test_choices/{fam}.json')['predictions']
        for ti,target in enumerate(('R','M')):
            for j in range(1,5):
                broadrows.append({'family':fam,'nearest_training_family':near,'target':target,'candidate_index':j,
                    'similarity':float(k[ix]),'distance':float(d[ix]),'nearest_label_pp':float(train['targets'][ix,ti,j]*100),
                    'test_label_pp':float(z['pooled_targets'][ti,j]*100),'strict_sign_reversal':bool(reversal(train['targets'][ix,ti,j],z['pooled_targets'][ti,j])),
                    'prediction_pp':float(pred[target+'_kernel'][j]*100),'actual_gain_pp':float(g[j]*100)})
    csv_write('all_test_nearest_candidate_pairs.csv',broadrows)
    collisions, paircomparisons=[],[]; families=sorted(cache)
    for i,fa in enumerate(families):
        for fb in families[i+1:]:
            delta=cache[fa]['features'].astype(float)-cache[fb]['features'].astype(float);distance=float(((delta/models['M']['std'])**2).sum())
            row={'family_a':fa,'family_b':fb,'exact_equal':bool(np.array_equal(cache[fa]['features'],cache[fb]['features'])),
                 'max_absolute_difference':float(np.abs(delta).max()),'within_1e_minus6':bool(np.abs(delta).max()<=1e-6),
                 'squared_standardized_distance':distance,'kernel_similarity':float(np.exp(-float(models['M']['gamma'])*distance)),
                 'max_M_label_difference_pp':float(np.abs(cache[fa]['pooled_targets'][1]-cache[fb]['pooled_targets'][1]).max()*100)}
            paircomparisons.append(row)
            if row['exact_equal'] or row['within_1e_minus6']:collisions.append(row)
    csv_write('all_bank_pair_comparisons.csv',paircomparisons)
    write(OUT/'exact_observation_collision_audit.json',{'family_count':60,'pair_comparisons':len(paircomparisons),
        'exact_equal_count':sum(r['exact_equal'] for r in paircomparisons),'within_1e_minus6_count':sum(r['within_1e_minus6'] for r in paircomparisons),
        'collisions':collisions,'population_initial_observation_equal_families':60,
        'source_world_initial_observations_equal_families':60,'source_world_equality_evidence':'frozen collection line130 asserts wx==x for both source worlds',
        'small_random_bank_no_collisions_does_not_establish_general_identifiability':True})
    opposite = {'R':[],'M':[]}
    for target in opposite:
        opposite[target]=[r for r in broadrows if r['target']==target and r['strict_sign_reversal']]
    pairreversal={target:[r['candidate_index'] for r in opposite[target] if r['family']==PAIR[1]] for target in ('R','M')}
    split_sign={}
    for fam in PAIR:
        split_sign[fam]={str(j):{'positive':sum(r['M_class']=='positive' for r in slicerows if r['family']==fam and r['candidate_index']==j and r['scope']!='pooled'),
            'negative':sum(r['M_class']=='negative' for r in slicerows if r['family']==fam and r['candidate_index']==j and r['scope']!='pooled'),
            'neutral_or_small':sum(r['M_class']=='neutral_or_small' for r in slicerows if r['family']==fam and r['candidate_index']==j and r['scope']!='pooled')} for j in range(1,5)}
    summary={'role':'descriptive_cached_reversal_audit_not_causal_or_independent_confirmation','pair':list(PAIR),
        'pair_exact_observation_equal':bool(np.array_equal(a['features'],b['features'])),
        'changed_feature_indices':[r['index'] for r in feature_rows if r['train_raw']!=r['test_raw']],
        'distance_groups':grouped,'kernel':predictions,'pair_world_snapshot':snapshot,'pair_strict_sign_reversal_candidates':pairreversal,
        'fixed_split_M_class_counts':split_sign,'test18_actual':actualrows,'test18_trace_steps':int(len(actual['execution_trace'])),
        'test18_trace_episodes':len(episodetrace),'nearest_pair_reversal_counts':{t:len(v) for t,v in opposite.items()},
        'nearest_pair_reversal_families':{t:sorted({r['family'] for r in v}) for t,v in opposite.items()},
        'nearest_pair_candidate_rows_per_target':96,'all_bank_collision_audit':read(OUT/'exact_observation_collision_audit.json'),
        'all_bank_planning_labels_verified':60,'new_simulations':0,'new_model_fits':0,'new_ppo_updates':0,'new_llm_calls':0,
        'all_scene_varying_physical_fields_except_source_observed_at_reset_in_this_bank':True,
        'growth_onset_background_not_hidden':True,'no_relevant_forecast_clipping_in_this_bank':True,
        'planning_source_variation_does_not_change_forecast_source':True,
        'causal_mechanism_identified':False,'individual_latent_population_and_dynamics_not_fully_observed':True,
        'planning_per_step_traces_available':False,'actual_train_outcomes_available':False,
        'qualification_thresholds_changed':False,'full_gate_requalified':False,
        'prior_experiment_budget':read(SOURCE/'summary.json')['budget']}
    write(OUT/'summary.json',summary)
    plot(summary,worldrows)
    write(OUT/'analysis_manifest.json',{'created_at_utc':stamp(),'script_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),
        'summary_sha256':sha(OUT/'summary.json'),'files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.csv','.png','.svg','.pdf')},
        'new_simulations':0,'new_model_fits':0})
    print(json.dumps({'analyzed':True,'feature_differences':summary['changed_feature_indices'],'kernel':predictions,
        'reversals':summary['nearest_pair_reversal_counts'],'split_M':split_sign,'actual':[(r['candidate_index'],r['actual_loss_pp'],r['actual_gain_pp']) for r in actualrows]},ensure_ascii=False),flush=True)


def plot(s,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8))
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#777777']; x=np.arange(5)
    for ax,fam in zip(axes,PAIR):
        for w,offset in ((0,-.18),(1,.18)):
            losses=[next(r['candidate_loss_pp'] for r in rows if r['family']==fam and r['world']==w and r['candidate_index']==j) for j in range(5)]
            ax.bar(x+offset,losses,.34,label=('North-source world','South-source world')[w],color=colors[w])
        ax.set_xticks(x,[str(i) for i in range(601,606)]); ax.set_ylim(0,max(r['candidate_loss_pp'] for r in rows)*1.18)
        ax.set_title(fam+' (cached planning)');ax.set_ylabel('Incapacitation (%)');ax.set_xlabel('Frozen PPO training seed');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Similar initial inputs, different absolute policy losses',fontsize=13)
    fig.tight_layout(rect=(0,.08,1,.94))
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'pair_world_losses.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--analyze',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.analyze:analyze()
    else:p.error('choose --prepare or --analyze')
