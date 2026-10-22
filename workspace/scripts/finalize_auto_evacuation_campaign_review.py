"""Read-only scientific reanalysis after both sealed automatic branches finish.

Writes only a new review directory and report. No simulation, fit, PPO update,
threshold change, or inference is performed here.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'outputs/evacuation_auto_campaign_20261005'
LLM = ROOT / 'outputs/evacuation_auto_llm_independent_20261005'
MANAGER = ROOT / 'outputs/evacuation_auto_manager_20261005'
OUT = ROOT / 'outputs/evacuation_auto_final_review_20261005'
REPORT = ROOT / 'reports/疏散自动实验最终复核与通俗说明_2026-10-05.md'
STATUS = ROOT / 'outputs/third_round_development_status.json'
HISTORY = ROOT / 'outputs/third_round_status_history/before_automatic_llm_independent_20261005.json'
ARMS = ('pure', 'rule', 'llm', 'same_llm_actor_rule')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def ledger_check(path, previous_status=None):
    ledger = read(path)
    for name, digest in ledger['files'].items():
        target = ROOT / name
        if previous_status is not None and target.resolve() == STATUS.resolve():
            target = previous_status
        assert sha(target) == digest, f'ledger mismatch: {name}'
    return len(ledger['files'])


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def link(path, label):
    return f'[{label}]({Path(path).as_posix()})'


def run():
    assert not OUT.exists() and not REPORT.exists(), 'Preserve existing review; never overwrite it.'
    for directory, status_name in ((MAIN, 'campaign_status.json'), (LLM, 'campaign_status.json'), (MANAGER, 'manager_status.json')):
        assert read(directory / status_name)['state'] == 'complete', directory
    ledgers = {
        'main_v2_records': ledger_check(MAIN / 'completion_manifest_v2.json', HISTORY),
        'llm_records': ledger_check(LLM / 'completion_manifest.json'),
        'manager_records': ledger_check(MANAGER / 'completion_manifest.json'),
    }
    main_manifest = read(MAIN / 'manifest.json')
    manifest = read(LLM / 'manifest.json')
    assert sha(manifest['base_completion_path']) == manifest['base_completion_sha256']
    for name, digest in main_manifest['frozen_sha256'].items():
        assert sha(ROOT / name) == digest, name
    repair = read(MANAGER / 'supervisor_sealing_repair_v3.json')
    assert sha(ROOT / 'scripts/manage_auto_evacuation_campaign.py') == repair['new_manager_sha256']
    assert sha(MANAGER / 'code_history/manage_v2.py') == repair['old_manager_sha256']
    initial_manager = read(MANAGER / 'manifest.json')
    assert sha(MANAGER / 'code_history/manage_v1.py') == initial_manager['manager_sha256']

    advice_path = LLM / 'llm_advice_frozen.json'
    frozen = read(advice_path)
    advice_hash = sha(advice_path)
    scenes = manifest['banks']['llm']
    assert len(scenes) == len(frozen['advice']) == 48
    latencies, prompt_tokens, completion_tokens = [], [], []
    for scene in scenes:
        family = scene['scenario_id']
        cache = read(LLM / 'llm_cache' / f'{family}.json')
        assert cache['advice'] == frozen['advice'][family]
        raw = read(LLM / 'llm_raw_responses' / f"{family}_attempt{cache['attempts']}.json")
        assert cache['prompt'] == raw['prompt']
        assert raw['model_digest'] == cache['model_digest'] == frozen['model_digest']
        parsed = json.loads(raw['response']['message']['content'])
        assert parsed['target'] in ('north', 'south')
        assert cache['advice']['target_branch'] == ('north', 'south').index(parsed['target'])
        assert 0 <= parsed['confidence'] <= 1 and parsed['confidence'] == cache['advice']['confidence']
        data = json.loads(raw['prompt'].split('Data: ', 1)[1])
        allowed = ('growth_rate', 'onset_step', 'background_hazard', 'north_exit_capacity',
                   'south_exit_capacity', 'blocked_branch', 'blockage_end_step')
        expected = {key: scene[key] for key in allowed}
        expected['forecast_source'] = ('north', 'south')[scene['forecast_source_branch']]
        assert data == expected, family
        assert cache['created_at_utc'] <= frozen['created_at_utc']
        latencies.append(raw['latency_seconds'])
        prompt_tokens.append(raw['response']['prompt_eval_count'])
        completion_tokens.append(raw['response']['eval_count'])

    files = list((LLM / 'llm_actual').glob('*.npz'))
    assert len(files) == 960 and not any(p.name.endswith('.partial.npz') for p in files)
    rows, paired, observations = [], [], {}
    physical_indices = np.r_[np.arange(28), np.arange(33, 49)]
    for scene in scenes:
        family = scene['scenario_id']
        registry = manifest['registry'][family]
        for seed in range(601, 606):
            losses = {}
            for arm in ARMS:
                path = LLM / 'llm_actual' / f'{seed}_{arm}_{family}.npz'
                with np.load(path, allow_pickle=False) as z:
                    risks = z['risks'].copy()
                    obs = z['observations'].copy()
                    assert risks.shape == (2, 1, 1, 6, 11) and z['done'].all()
                    np.testing.assert_array_equal(z['population_seeds'], registry['population'])
                    np.testing.assert_array_equal(z['future_seeds'], registry['actual'])
                assert obs.shape == (2, 49) and np.isfinite(obs).all()
                reference = observations.setdefault(family, obs[:, physical_indices].copy())
                np.testing.assert_array_equal(obs[:, physical_indices], reference)
                assert np.isfinite(risks).all()
                np.testing.assert_allclose(risks[..., 0] + risks[..., 2] + risks[..., 3], 1, rtol=0, atol=1e-10)
                assert ((risks[..., [0, 2, 3, 6, 8]] >= 0) & (risks[..., [0, 2, 3, 6, 8]] <= 1)).all()
                assert (risks[..., 5] >= 1).all() and (risks[..., 7] <= risks[..., 5]).all()
                assert np.isin(risks[..., [9, 10]], np.arange(9)).all()
                started = read(path.with_suffix('.started.json'))
                assert started['selection_sha256'] == advice_hash
                assert frozen['created_at_utc'] < started['started_at_utc']
                assert started['scene'] == scene and started['planned_rollouts'] == 12
                if arm == 'llm':
                    value = frozen['advice'][family]
                    expected_advice = [float(value['target_branch'] == 0), float(value['target_branch'] == 1), value['confidence'], 1, 0]
                    np.testing.assert_allclose(obs[:, 28:33], [expected_advice] * 2, rtol=0, atol=1e-7)
                elif arm == 'pure':
                    np.testing.assert_array_equal(obs[:, 28:33], np.zeros((2, 5)))
                loss = float(risks[..., 0].mean() * 100)
                losses[arm] = loss
                rows.append({'seed': seed, 'arm': arm, 'family': family, 'kind': scene['conflict_type'], 'loss_pp': loss})
            paired.append({'seed': seed, 'family': family, 'kind': scene['conflict_type'],
                           'delta_pp': losses['llm'] - losses['same_llm_actor_rule']})

    published_rows = csv_rows(LLM / 'llm_family_results.csv')
    assert len(published_rows) == len(rows)
    reference_rows = {(int(r['seed']), r['arm'], r['family']): r for r in published_rows}
    for row in rows:
        other = reference_rows[(row['seed'], row['arm'], row['family'])]
        assert other['kind'] == row['kind']
        np.testing.assert_allclose(row['loss_pp'], float(other['loss_pp']), atol=1e-9, rtol=0)
    family_rows = [{'family': s['scenario_id'], 'kind': s['conflict_type'],
                    'delta_pp': float(np.mean([r['delta_pp'] for r in paired if r['family'] == s['scenario_id']]))} for s in scenes]
    seed_effects = {str(seed): float(np.mean([r['delta_pp'] for r in paired if r['seed'] == seed])) for seed in range(601, 606)}
    rng = np.random.default_rng(2026100552)
    bootstrap = np.zeros(2000)
    for kind in sorted({s['conflict_type'] for s in scenes}):
        values = np.array([r['delta_pp'] for r in family_rows if r['kind'] == kind])
        bootstrap += values[rng.integers(0, len(values), size=(2000, len(values)))].sum(axis=1)
    interval = np.quantile(bootstrap / 48, [.025, .975])
    mean_effect = float(np.mean([r['delta_pp'] for r in family_rows]))
    summary = read(LLM / 'llm_summary.json')
    np.testing.assert_allclose(mean_effect, summary['primary_mean_delta_pp'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(interval, summary['primary_ci95_pp'], rtol=0, atol=1e-9)
    for seed, value in seed_effects.items():
        np.testing.assert_allclose(value, summary['per_training_seed_delta_pp'][seed], rtol=0, atol=1e-9)
    signal = bool(interval[1] < 0 and sum(v < 0 for v in seed_effects.values()) >= 4)
    assert signal == summary['improvement_signal']

    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    models = []
    training = csv_rows(ROOT / 'outputs/guarded_confirmatory_v2/training_summary.csv')
    for seed in range(601, 606):
        for arm in ('pure_ppo', 'rule_advice_ppo', 'raw_llm_ppo'):
            path = ROOT / f'outputs/guarded_confirmatory_v2/models/seed_{seed}_{arm}.zip'
            model = PPO.load(path, device='cpu')
            assert model.action_space.n == 9 and model.observation_space.shape == (34,)
            metadata = next(r for r in training if int(r['training_seed']) == seed and r['arm'] == arm)
            assert int(metadata['requested_timesteps']) == 50000 and int(metadata['actual_timesteps']) == model.num_timesteps == 50176
            models.append({'seed': seed, 'arm': arm, 'sha256': sha(path), 'input_dimensions': 34, 'actions': 9,
                           'historical_requested_timesteps': 50000, 'historical_actual_timesteps': model.num_timesteps})

    main_budget, llm_budget = read(MAIN / 'budget.json'), read(LLM / 'budget.json')
    assert main_budget['simulations_started'] == main_budget['simulations_completed'] == 21698
    assert llm_budget['simulations_started'] == llm_budget['simulations_completed'] == 11520
    assert main_budget['ppo_updates'] == llm_budget['ppo_updates'] == llm_budget['matrix_fits'] == 0
    assert main_budget['matrix_fits'] == 4 and main_budget['mean_reference_estimates'] == 1
    assert len(list((LLM / 'llm_raw_responses').glob('*.json'))) == llm_budget['llm_requests'] == 48
    assert main_budget['simulations_started'] + llm_budget['simulations_started'] == 33218 <= 44738
    review = read(MAIN / 'figure_visual_review.json')
    assert sha(MAIN / 'mechanism_response.png') == review['image_sha256']
    assert read(MAIN / 'verification.json')['all_checks_passed']
    assert read(LLM / 'verification.json')['all_checks_passed']
    assert read(MAIN / 'additional_prediction_verification_072.json')['complete_check']
    agreements = csv_rows(LLM / 'advice_agreement.csv')
    same_direction = sum(r['same_direction'] == 'True' for r in agreements)
    failure_paths = sorted((LLM / 'failures').glob('*.json'))
    failures = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'record': read(p)} for p in failure_paths]
    costs = {'model': 'qwen3:4b', 'model_digest': frozen['model_digest'], 'requests': 48,
             'total_latency_seconds': float(sum(latencies)), 'mean_latency_seconds': float(np.mean(latencies)),
             'latency_p50_seconds': float(np.quantile(latencies, .5)), 'latency_p95_seconds': float(np.quantile(latencies, .95)),
             'prompt_eval_tokens': int(sum(prompt_tokens)), 'generated_tokens': int(sum(completion_tokens)),
             'paid_api_requests': 0, 'scope': 'one initial suggestion per family; no repeated online LLM calls'}
    kind_effects = [{'kind': kind, 'families': sum(r['kind'] == kind for r in family_rows),
                     'mean_delta_pp': float(np.mean([r['delta_pp'] for r in family_rows if r['kind'] == kind]))}
                    for kind in sorted({r['kind'] for r in family_rows})]
    metrics = {'primary_mean_delta_pp': mean_effect, 'primary_ci95_pp': interval.tolist(), 'per_seed_delta_pp': seed_effects,
               'improvement_signal': signal, 'descriptive_kind_effects': kind_effects,
               'mean_loss_pp_by_arm': {arm: float(np.mean([r['loss_pp'] for r in rows if r['arm'] == arm])) for arm in ARMS},
               'improved_families_gt_0_1pp': sum(r['delta_pp'] < -.1 for r in family_rows),
               'worsened_families_gt_0_1pp': sum(r['delta_pp'] > .1 for r in family_rows),
               'same_direction_families': same_direction, 'llm_cost': costs}
    verification = {'all_checks_passed': True, **ledgers, 'llm_units_rechecked': 960, 'llm_rollouts_rechecked': 11520,
                    'primary_family_seed_pairs': 240, 'primary_bootstrap_independently_reconstructed': True,
                    'initial_physical_inputs_identical_across_arms_and_seeds': True,
                    'population_and_future_blocks_identical_across_arms_and_seeds': True,
                    'advice_prompts_and_raw_responses_checked': 48, 'historical_actor_contracts_checked': 15,
                    'combined_simulations': 33218, 'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0,
                    'frozen_image_review_hash_matches': True, 'worker_failure_records_preserved': len(failures),
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewer_sha256': sha(__file__)}

    student = read(MAIN / 'student_summary.json')
    mechanism = read(MAIN / 'mechanism_summary.json')
    north = next(r for r in mechanism['factor_effects'] if r['factor'] == 'north_exit_capacity')
    point = next(r for r in student['all_arms'] if r['arm'] == 'absolute_point')
    lines = [
        '# 火灾疏散自动实验：最终复核与通俗说明', '',
        '日期：2026-10-05。本轮按冻结方案完成全部适用分支；独立复核只读已有结果，没有增加模拟或训练。', '',
        f'共完成 **33218 条模拟**、4次学生矩阵拟合、1项均值参考和48次本地LLM请求。本轮PPO更新为0；使用601—605五个历史PPO训练种子，三个学生重采样种子不算新的PPO种子。', '',
        '这轮回答三个问题：收益为什么反转；初始观测能否可靠判断换策略的收益；同一个PPO控制器得到LLM建议后是否比得到规则建议更好。以下结论都限定于开发模拟器，原门控资格未通过。', '',
        '| 实验 | 设计和预算 | 结果 |', '| --- | --- | --- |',
        f'| 收益反转机制 | 五变量完整32组合；1536条科学模拟，另2条仪器检查 | 北容量5→2在另外四变量的16配置中全部翻转，平均M变化{north["mean_M_change_pp"]:+.4f}个百分点 |',
        '| 初始选择器 | 60训练、24校准、48留出家庭；20160条模拟 | 带余量主臂接管0/48，未达到扩展条件 |',
        '| 行为和观测噪声变化 | 前序通过才运行 | 按固定条件跳过；没有泛化通过的证据 |',
        f'| 独立真实LLM对照 | 48新家庭×5历史种子×4臂×2人口×6未来；11520条模拟 | 同控制器主差{mean_effect:+.6f}个百分点，改善信号={signal} |', '',
        '机制实验把原先同时变化的出口容量、火势增长、起火时间和背景烟气逐个控制。北出口容量改变时，策略603从频繁等待转为不请求等待，相对601的结果随之翻转。这解释了“初始看起来相近但收益相反”的重要部分。两个固定人口、两个参数端点和已有策略限制了结论；其他变量仍影响幅度和交互，不能解释成它们没有影响。', '',
        f'新选择器先预测不同火源世界下的绝对失能损失，再判断换策略是否值得。主模型收益预测误差MAE为{student["actual_gain_MAE_pp"]["absolute"]:.6f}个百分点，“永远预测零收益”的误差反而是{student["actual_gain_MAE_pp"]["zero"]:.6f}。校准要求的余量使它全部退回601，这意味着尚不能可靠识别收益；0损害来自0接管，不能据此声称门控已安全有效。', '',
        f'预先安排的去余量诊断版本接管{point["overrides"]}/48，平均失能差{point["mean_delta_pp"]:+.6f}个百分点，95%区间[{point["ci95_pp"][0]:+.6f}, {point["ci95_pp"][1]:+.6f}]；{point["harmed_families"]}个家庭变差。因此，直接降低余量没有得到改善，本轮停止该选择器的进一步扩展。', '',
        'LLM实验单独进行，避免把选择器失败误认为LLM必然无效。主比较固定同一个raw_llm PPO检查点，仅将建议输入从规则换为本地qwen3:4b。两组人口、未来随机块和初始物理输入相同；方向与置信度均可变化，文字解释不进入PPO。其他训练臂属于次要描述，不能用它们的差值单独归因于LLM。', '',
        f'主差定义为“LLM组失能率−同控制器规则组失能率”，负数更好。平均{mean_effect:+.6f}个百分点，按48家庭、冲突类别分层的2000次bootstrap给出95%区间[{interval[0]:+.6f}, {interval[1]:+.6f}]。五种子共享家庭，不能把11520条模拟当成11520个独立场景。预先约定仅当区间上界<0且至少4/5种子均值<0才报告改善信号；本次为{signal}。', '',
        '| 历史PPO训练种子 | 同控制器LLM−规则失能差（百分点） |', '| --- | --- |',
    ]
    lines += [f'| {seed} | {value:+.6f} |' for seed, value in seed_effects.items()]
    lines += ['', '| 四臂 | 平均失能率（%） |', '| --- | --- |']
    labels = {'pure': '纯PPO、无建议', 'rule': '规则建议PPO', 'llm': 'raw_llm PPO、真实LLM建议', 'same_llm_actor_rule': '相同raw_llm PPO、规则建议'}
    lines += [f'| {labels[arm]} | {metrics["mean_loss_pp_by_arm"][arm]:.6f} |' for arm in ARMS]
    lines += ['', '以下类别均值用于描述异质性，未设类别阳性判定，不能从中挑选一个子类作为总体成功。', '',
              '| 冲突类别 | 新家庭数 | 同控制器LLM−规则差（百分点） |', '| --- | --- | --- |']
    lines += [f'| {r["kind"]} | {r["families"]} | {r["mean_delta_pp"]:+.6f} |' for r in kind_effects]
    lines += ['',
              f'LLM与规则方向一致{same_direction}/48；按家庭平均，改善超过0.1个百分点的有{metrics["improved_families_gt_0_1pp"]}个，变差超过0.1个百分点的有{metrics["worsened_families_gt_0_1pp"]}个。方向一致时置信度仍可能不同，因此不能把同方向组说成输入完全相同。', '',
              f'48条建议先统一冻结，再运行实际结局。模型digest为 `{frozen["model_digest"]}`；temperature0、固定种子、JSON约束、关闭思考输出。所有提示已核对，不含真实火源、冲突标签或未来结局。总请求延迟{costs["total_latency_seconds"]:.2f}秒，平均{costs["mean_latency_seconds"]:.2f}秒，P95为{costs["latency_p95_seconds"]:.2f}秒；提示评估tokens {costs["prompt_eval_tokens"]}，生成tokens {costs["generated_tokens"]}。没有付费API请求，但本地推理仍消耗计算资源；建议是每家庭初始一次，不代表每步在线成本。', '',
              '15个冻结PPO检查点均独立载入核对：34维输入、9动作，历史每臂请求50000训练步，实际50176步。本轮没有重新训练。raw_llm历史训练臂为探索性，不能追认为此前正式确认。', '',
              f'复核验证960个LLM结果单元、11520条结局、240个家庭—训练种子配对，并独立重算主效果和区间。人数守恒、初始物理输入、人口和未来种子配对、建议先于结局冻结、原始响应与解析、模型哈希和预算均通过。主流程{ledgers["main_v2_records"]}项、LLM分支{ledgers["llm_records"]}项及管理器{ledgers["manager_records"]}项完成台账均核对通过。机制图片与已完成目检的哈希一致。', '',
              f'LLM模拟运行期间保存了{len(failures)}个失败记录，执行器从检查点恢复。已完成单元和LLM建议复用；开始数与完成数最终一致，未增加实验预算。后台汇总的封存时序及主流程两个操作日志追加问题均有版本记录；科学结果、冻结模型、原始台账和失败证据保留。', '',
              '本轮没有建立跨地图、FDS、真实火灾或新行为模型的效能证据。策略选择器是在初始时刻选择一个闭环PPO，LLM也是初始建议，因此不能把结果称为运行中动态信任门控已成功。原95%/10%/5%/95%资格标准和收益下限没有降低。已看留出今后只能作为开发诊断。', '',
              '后续研究应分开处理：选择器先查预测目标与观测支持是否一致，LLM以这里的同控制器结果决定是否值得继续。若要修改模型或提示，必须先写新方案并使用新的校准与评估银行；本轮完成后不无限追加训练追求阳性。', '',
              link(ROOT / 'reports/疏散自动实验总方案_2026-10-05.md', '冻结自动总方案') + '；' +
              link(ROOT / 'reports/疏散自动实验LLM独立分支固定补充方案_2026-10-05.md', 'LLM固定补充方案') + '；' +
              link(ROOT / 'reports/疏散自动实验阶段总报告_2026-10-05.md', '主流程详细报告') + '；' +
              link(ROOT / 'reports/LLM独立建议对照自动实验报告_2026-10-05.md', 'LLM独立详细报告') + '。', '',
              link(OUT / 'verification.json', '本次最终独立复核') + '；' + link(OUT / 'metrics.json', '复算效果与成本') + '；' +
              link(LLM / 'llm_paired_results.csv', '原始配对结果') + '；' + link(MANAGER / 'supervisor_sealing_repair_v3.json', '操作修复记录') + '。', '',
              f'![32组合机制响应]({(MAIN / "mechanism_response.png").as_posix()})', '',
              '本轮全部适用实验及复核已经完成。行为和噪声分支因前序证据未达标而跳过，失败结论完整保留。', '']
    OUT.mkdir()
    save(OUT / 'verification.json', verification)
    save(OUT / 'metrics.json', metrics)
    save(OUT / 'actor_contracts.json', models)
    save(OUT / 'preserved_worker_failures.json', failures)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    paths = [p for p in OUT.iterdir() if p.is_file()] + [REPORT, Path(__file__)]
    save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'role': 'post_campaign_read_only_review',
         'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0,
         'parent_ledgers': {str(p.relative_to(ROOT)): sha(p) for p in [MAIN / 'completion_manifest_v2.json', LLM / 'completion_manifest.json', MANAGER / 'completion_manifest.json']},
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    ledger_check(OUT / 'completion_manifest.json')
    print(json.dumps({'complete': True, 'verification': verification, 'metrics': metrics, 'report': str(REPORT)}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
