"""Verify report/derived evidence, preserve prior status, seal cache-only stage."""
from collections import Counter
from pathlib import Path
import csv
import json
import re
import sys
import numpy as np
from observation_reversal_audit import ROOT,SOURCE,OUT,STATUS,HISTORY,REPORT,PROTOCOL,PAIR,read,write,sha,load,stamp,preserve


def table(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)


def main():
    assert not (OUT/'completion_manifest.json').exists() and not HISTORY.exists()
    manifest=read(OUT/'development_manifest.json');s=read(OUT/'summary.json');v=read(OUT/'verification.json');assets=read(OUT/'report_assets_manifest.json')
    assert v['all_checks_passed'] and v['verifier_sha256']==sha(ROOT/'scripts/verify_observation_reversal_audit.py')
    assert sha(REPORT)==assets['report_sha256'] and sha(OUT/'summary.json')==assets['summary_sha256']
    assert sha(OUT/'verification.json')==assets['verification_sha256'] and sha(ROOT/'scripts/write_observation_reversal_report.py')==assets['writer_sha256']
    assert assets['figure_visually_inspected'] and not assets['essential_data_clipped']
    assert sha(PROTOCOL)==manifest['protocol_sha256'] and sha(STATUS)==manifest['previous_status_sha256']==sha(OUT/'previous_status_exact_bytes.json')
    for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,p
    am=read(OUT/'analysis_manifest.json');assert am['script_sha256']==sha(ROOT/'scripts/observation_reversal_audit.py')
    for p,h in am['files'].items():assert sha(OUT/p)==h,p
    text=REPORT.read_text(encoding='utf-8');links=re.findall(r'\]\((C:/[^)]+)\)',text)
    for raw in links:
        rawpath=re.sub(r':\d+$','',raw);path=Path(rawpath);assert path.is_file(),raw
        line=re.search(r':(\d+)$',raw)
        if line:assert int(line.group(1))<=len(path.read_text(encoding='utf-8').splitlines())
    assert len(links)>=25
    wr=table('pair_world_outcomes.csv')
    for r in wr:
        if int(r['candidate_index']) in (2,3,4):
            for key in ('candidate_loss_pp','baseline_loss_pp'):assert f"{float(r[key]):.6f}" in text
            assert f"{float(r['gain_pp']):+.6f}" in text
    # Independently check all nearest-pair actual gains omitted by the first verifier.
    broad=table('all_test_nearest_candidate_pairs.csv');byfamily={}
    for r in broad:
        if r['family'] not in byfamily:
            actual=load(SOURCE/f"test_actual/{r['family']}.npz")['evaluation_outcomes']
            byfamily[r['family']]=actual[...,0].mean(axis=(0,2))
        losses=byfamily[r['family']];j=int(r['candidate_index']);close(float(r['actual_gain_pp']),(losses[0]-losses[j])*100)
    # Ensure scope is not silently upgraded after development outcomes were viewed.
    assert s['changed_feature_indices']==[5,6,7,8,9,14,15,21,22,23,24]
    assert not s['pair_exact_observation_equal'] and not s['causal_mechanism_identified']
    assert s['all_bank_collision_audit']['exact_equal_count']==s['all_bank_collision_audit']['within_1e_minus6_count']==0
    assert s['nearest_pair_reversal_counts']=={'R':4,'M':4}
    snapshot=s['pair_world_snapshot'];train,test=(snapshot[k] for k in PAIR)
    assert train['scenario']['source_branch']==test['scenario']['source_branch']==1
    assert train['scenario']['forecast_source_branch']==test['scenario']['forecast_source_branch']==0
    for family in PAIR:
        rs=[r for r in wr if r['family']==family and r['candidate_index']=='2']
        assert len(rs)==2 and all(int(r['gain_negative_count' if family==PAIR[0] else 'gain_positive_count'])==12 for r in rs)
    model=load(SOURCE/'models/M_kernel.npz');dataset=load(SOURCE/'training_dataset.npz')
    assert int((np.var(model['train'],axis=0)>1e-12).sum())==19
    assert int((dataset['targets'][:,1,2]>.001).sum())==2
    close(dataset['targets'][:,1,2].max()*100,5.208333333333334)
    for j in (2,3,4):assert test['M_pp'][j]>dataset['targets'][:,1,j].max()*100
    termrows=table('kernel_contributions.csv');other=sum(float(r['contribution_pp']) for r in termrows if r['target']=='M' and r['candidate_index']=='2' and r['family']!=PAIR[0])
    close(other,-5.274587290081273)
    for actual in s['test18_actual']:
        j=actual['candidate_index'];assert f"{actual['actual_loss_pp']:.6f}" in text and f"{actual['actual_gain_pp']:+.6f}" in text
        if j in (2,3,4):assert actual['actual_gain_positive_count']==12 and actual['actual_gain_negative_count']==0
    trace=table('test18_actual_trace_phases.csv');allrows=[r for r in trace if r['phase']=='all']
    assert sum(int(r['step_count']) for r in allrows)==2225
    assert allrows[0]['north_sign_steps']=='521' and allrows[2]['south_sign_steps']=='456'
    assert allrows[4]['hold_blocks']=='249' and allrows[4]['neutral_positive_release_steps']=='8'
    assert '−18.607240' in text and '32.739185' in text and '34.753074' in text
    assert all(x in text for x in ('新增模拟0','拟合0','PPO更新0','LLM调用0','本轮没有执行这192条','不属于盲法预注册','不是因果分解','1家庭变差'))
    failure=read(OUT/'verification_first_attempt_failure.json');assert failure['exit_code']==1 and not failure['experimental_data_lost']
    checks=preserve(STATUS)
    test_record={'command':'D:/miniconda3/python.exe -m pytest -q tests/test_observation_reversal_audit.py',
        'observed_result':'5 passed in 0.30s','exit_code':0,'tests_passed':5,'full_project_tests_rerun':False,
        'test_sha256':sha(ROOT/'tests/test_observation_reversal_audit.py')}
    write(OUT/'targeted_test_verification.json',test_record)
    write(OUT/'packaging_verification.json',{'all_checks_passed':True,'preservation':checks,'report_links_valid':len(links),
        'report_numbers_cross_checked':True,'all_nearest_pair_actual_gains_independently_checked':192,
        'core_reversal_same_world_and_all_seed_signs_verified':True,'model_training_support_and_contributions_verified':True,
        'descriptive_not_causal_limit_verified':True,'new_simulations':0,'new_model_fits':0,'failure_preserved':True,
        'finalizer_sha256':sha(__file__)})
    old=read(STATUS);HISTORY.write_bytes(STATUS.read_bytes());assert sha(HISTORY)==manifest['previous_status_sha256']
    status={'stage':'similar_initial_observation_reversal_cache_audit_complete_causal_factors_unresolved',
        'updated_date':'2026-10-05','updated_at_utc':stamp(),'cache_only':True,'freeze_allowed':False,
        'hard_shield_qualification_passed':False,'formal_round3_training_started':False,'formal_round3_final_bank_generated':False,
        'new_simulations':0,'new_planning_rollouts':0,'new_evaluation_rollouts':0,'new_fits':0,'new_supervised_training_runs':0,
        'new_training_mean_references':0,'new_ppo_training_seeds':0,'new_llm_calls':0,
        'observable_student_closed_loop_advantage_established':False,'teacher_generalization_advantage_established':False,
        'llm_control_contribution_established':False,'qualification_thresholds_changed':False,
        'actual_control_efficacy_evaluated_in_current_stage':False,'existing_actual_cache_reanalyzed':True,
        'report':REPORT.relative_to(ROOT).as_posix(),'report_sha256':sha(REPORT),
        'summary':(OUT/'summary.json').relative_to(ROOT).as_posix(),'summary_sha256':sha(OUT/'summary.json'),
        'manifest':(OUT/'development_manifest.json').relative_to(ROOT).as_posix(),
        'independent_verification':(OUT/'verification.json').relative_to(ROOT).as_posix(),
        'completion_manifest':(OUT/'completion_manifest.json').relative_to(ROOT).as_posix(),
        'previous_stage_status':HISTORY.relative_to(ROOT).as_posix(),'previous_stage_status_sha256':sha(HISTORY),
        'previous_five_ppo_stage':old,'diagnostic_scope':{'priority_pair':list(PAIR),'planning_families':60,
            'input_dimensions':49,'world_outcome_rows':20,'fixed_split_rows':90,'paired_seed_rows':240,
            'kernel_contribution_rows':192,'test18_actual_trace_episodes':60,'test18_actual_trace_steps':2225,
            'nearest_candidate_target_rows':192,'bank_collision_pairs':1770,'seen_development_only':True},
        'conclusion':'重点家庭初始49维中11维不同，可见容量/火势参数变化伴随601绝对损失上升、603等待由约41次变为0及两世界收益排序翻转。训练邻域与中心化核贡献导致603错误负预测。固定分割和独立实际未来支持本缓存反转方向，尚不能分离物理参数与人口的因果贡献。',
        'key_results':{'train_M_603_pp':train['M_pp'][2],'test_M_603_pp':test['M_pp'][2],
            'M_kernel_prediction_603_pp':s['kernel']['M']['prediction_pp'][2],'test_actual_gain_603_pp':s['test18_actual'][2]['actual_gain_pp'],
            'strict_nearest_reversals_per96':s['nearest_pair_reversal_counts'],'exact_feature_collision_pairs':0},
        'next_step':'另存协议进行2×2容量组与火势时序组的192条受控诊断，匹配人口并记录完整逐步状态，先定位响应转折及等待行为，暂缓新增学生/PPO训练。本轮尚未执行该计划，现有留出不得作为修改后的最终评估。',
        'causal_mechanism_identified':False,'targeted_tests':test_record,'preservation':checks}
    write(STATUS,status)
    # Check prior ledgers against saved exact prior status after the intentional mutation.
    post=preserve(HISTORY);assert post==checks
    artifacts=[REPORT,PROTOCOL,STATUS,HISTORY,ROOT/'tests/test_observation_reversal_audit.py']
    artifacts += [ROOT/'scripts'/name for name in ('observation_reversal_audit.py','verify_observation_reversal_audit.py','write_observation_reversal_report.py','finalize_observation_reversal_audit.py')]
    artifacts += sorted(p for p in OUT.rglob('*') if p.is_file())
    completion={'role':'cache_only_observation_reversal_development_diagnosis_completion','all_checks_passed':True,'created_at_utc':stamp(),
        'new_simulations':0,'new_model_fits':0,'new_ppo_updates':0,'new_llm_calls':0,'targeted_tests_passed':5,
        'preservation':checks,'previous_status_exact_bytes_saved':True,'old_sources_models_reports_preserved':True,
        'original_qualification_thresholds_unchanged':True,'causal_mechanism_claimed':False,
        'prior_completion_sha256':sha(SOURCE/'completion_manifest.json'),'files':{p.relative_to(ROOT).as_posix():sha(p) for p in artifacts}}
    write(OUT/'completion_manifest.json',completion)
    for p,h in read(OUT/'completion_manifest.json')['files'].items():assert sha(ROOT/p)==h,p
    print(json.dumps({'complete':True,'new_files_hashed':len(completion['files']),'report_links':len(links),'preservation':checks,
        'new_simulations':0,'new_model_fits':0,'next_step_proposed_rollouts_not_executed':192},ensure_ascii=False),flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
