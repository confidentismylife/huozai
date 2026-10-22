"""Replay the sealed goal-role requests with a separately pinned 8B model."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import traceback

import auto_evacuation_campaign as c
import run_llm_output_mechanism as io
import llm_goal_role_analysis as analysis

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/llm_goal_role_20261006'
OUT = ROOT / 'outputs/llm_goal_role_8b_20261006'
RUNTIME = ROOT / 'outputs/llm_goal_role_8b_runtime_20261006'
PROTOCOL = ROOT / 'reports/LLM_8B模型复测固定方案_2026-10-06.md'
REPORT = ROOT / 'reports/LLM_8B模型复测实验报告_2026-10-06.md'
MODEL = 'qwen3:8b'


def replay_requests(base):
    requests = copy.deepcopy(base['request_order'])
    for request in requests:
        assert request['payload']['model'] == 'qwen3:4b'
        request['payload']['model'] = MODEL
        request['payload_sha256'] = io.hash_value(request['payload'])
    return requests


def prepare():
    assert not OUT.exists(), 'Preserve the existing experiment.'
    preflight = c.read(RUNTIME / 'pre_inference_checks.json')
    assert preflight['all_checks_passed']
    for name, digest in preflight['checked_source_sha256'].items():
        assert c.sha(ROOT / name) == digest, name
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, HISTORY
    base = c.read(BASE / 'manifest.json')
    parents = [ROOT / name for name in base['parent_completion_sha256']]
    parents += [BASE / 'completion_manifest.json', ROOT / 'outputs/llm_goal_role_runtime_20261006/completion_manifest.json']
    parent_checks = {p.relative_to(ROOT).as_posix(): ledger_check(p, HISTORY if p.parent == MAIN else None) for p in parents}
    tags = io.api('/api/tags', timeout=10)
    model = next(v for v in tags['models'] if v['name'] == MODEL)
    assert model['digest'].startswith('500a1f067a9f') and model['details']['quantization_level'] == 'Q4_K_M'
    snapshot = {'tags': tags, 'version': io.api('/api/version', timeout=10), 'show': io.api('/api/show', {'model': MODEL}, 10)}
    assert snapshot['version'] == c.read(BASE / 'model_service_snapshot.json')['version']
    assert 8_000_000_000 <= snapshot['show']['model_info']['general.parameter_count'] <= 8_500_000_000
    paths = [Path(__file__), ROOT / 'scripts/verify_llm_goal_role_8b.py', ROOT / 'tests/test_llm_goal_role_8b_replay.py', PROTOCOL,
             ROOT / 'scripts/llm_goal_role_core.py', ROOT / 'scripts/llm_goal_role_analysis.py', ROOT / 'scripts/verify_llm_goal_role.py',
             ROOT / 'scripts/run_llm_output_mechanism.py', ROOT / 'scripts/llm_output_mechanism_core.py',
             ROOT / 'scripts/verify_llm_output_mechanism.py', ROOT / 'scripts/auto_evacuation_campaign.py',
             ROOT / 'scripts/finalize_auto_evacuation_campaign_review.py', ROOT / 'tests/test_llm_goal_role_core.py']
    OUT.mkdir()
    io.save(OUT / 'pre_inference_checks.json', preflight)
    io.save(OUT / 'model_service_snapshot.json', snapshot)
    io.save(OUT / 'manifest.json', {
        'role': '8b_model_replay_of_sealed_goal_role_probe', 'created_at_utc': c.stamp(), 'local_date': '2026-10-06',
        'model_name': MODEL, 'model_digest': model['digest'], 'cases': base['cases'], 'conditions': base['conditions'],
        'request_order': replay_requests(base), 'request_order_seed': base['request_order_seed'],
        'base_manifest': (BASE / 'manifest.json').relative_to(ROOT).as_posix(), 'base_manifest_sha256': c.sha(BASE / 'manifest.json'),
        'parent_checks': parent_checks, 'parent_completion_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in parents},
        'frozen_sha256': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths},
        'model_snapshot_sha256': c.sha(OUT / 'model_service_snapshot.json'),
        'pre_inference_checks_sha256': c.sha(OUT / 'pre_inference_checks.json'),
        'old_global_status_sha256': c.sha(c.STATUS), 'logical_request_count': 32, 'api_attempt_cap': 64,
        'only_api_model_field_changed_from_baseline': True, 'not_a_parameter_count_only_comparison': True,
        'same_output_settings_as_prior_rounds': True, 'single_user_message_per_request': True,
        'wrong_invalid_empty_truncated_are_not_retried': True, 'max_retry_only_transport_errors': 1,
        'simulations': 0, 'fits': 0, 'ppo_updates': 0,
        'metadata_note': 'Baseline 4B is Thinking-2507; 8B is a different model variant and template. think=false is a request, not proof of internal reasoning mode.'})
    io.save(OUT / 'budget.json', {'api_attempts_started': 0, 'successful_http_responses': 0, 'logical_requests_completed': 0,
                                'simulations': 0, 'ppo_updates': 0, 'fits': 0})
    io.save(OUT / 'campaign_status.json', {'state': 'ready', 'stage': 'prepared', 'pid': None, 'run_attempts': 0,
                                         'created_at_utc': c.stamp(), 'completed_logical_requests': 0})
    print('32 model-only API request replays frozen; no inference yet.', flush=True)


def comparison(summary):
    old = c.read(BASE / 'summary.json')
    cells = []
    for name, value in summary['groups'].items():
        before = old['groups'][name]
        cells.append({'condition': name, 'baseline_correct': before['usable_correct'], '8b_correct': value['usable_correct'],
                      'correct_count_change': value['usable_correct'] - before['usable_correct'],
                      'baseline_mean_latency_seconds': before['mean_latency_seconds'], '8b_mean_latency_seconds': value['mean_latency_seconds']})
    io.save(OUT / 'comparison.json', {'groups': cells, 'not_parameter_count_only_effect': True,
                                    'independent_evacuation_effect_established': False, 'held_out_replication': False})
    c.csv_write(OUT / 'model_comparison.csv', cells)


def write_report(m, summary):
    old = c.read(BASE / 'summary.json')
    labels = {'minimize_hazard': '最小化 × 危险侧报告', 'maximize_hazard': '最大化 × 危险侧报告',
              'minimize_clear': '最小化 × 无烟侧报告', 'maximize_clear': '最大化 × 无烟侧报告'}
    snapshot = c.read(OUT / 'model_service_snapshot.json')
    previous = c.read(BASE / 'model_service_snapshot.json')
    info, previous_info = snapshot['show']['model_info'], previous['show']['model_info']
    lines = ['# Qwen3 8B本地复测报告', '', '日期：2026-10-06。用户授权改用8B后，下载并固定模型，完整复测上一轮32请求。', '',
             '这是已有8题、4个镜像家庭的功能复测。API请求仅将模型字段改为qwen3:8b；题目、顺序、参考答案、JSON schema、think=false请求、temperature0、seed2026100601、1024输出预算和4096上下文保持一致。不是新增独立测试集。', '',
             f"实际模型：{m['model_name']}，{info['general.parameter_count']:,}参数，Q4_K_M；digest `{m['model_digest']}`。Ollama版本{snapshot['version']['version']}。", '',
             f"模型身份审计：此前4B的general.finetune={previous_info.get('general.finetune')}，general.version={previous_info.get('general.version')}；当前8B分别为{info.get('general.finetune')}、{info.get('general.version')}。内置模板是否相同：{snapshot['show']['template']==previous['show']['template']}。", '',
             '此前模型实际为Qwen3-4B-Thinking-2507。官方模型卡说明其只支持思考模式；历史报告的“思考关”应理解为发送think=false，不能据此认定底层已关闭思考。这里另行记录这项解释更正，保留旧数据和报告原字节。8B与4B除了参数量，还存在模型版本、训练与模板差异，不能把结果差异归为纯规模效应。', '',
             '[4B官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)；[8B官方模型页](https://ollama.com/library/qwen3:8b)。', '',
             '最大化组仅比较期望暴露较大的数学标签，不能作为疏散建议。', '',
             '| 条件 | 旧4B正确/8 | 8B正确/8 | 8B合法但方向错 | 格式错 | 截断 | 传输失败 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for name, label in labels.items():
        g = summary['groups'][name]
        lines.append(f"| {label} | {old['groups'][name]['usable_correct']} | {g['usable_correct']} | {g['valid_wrong_direction']} | {g['invalid_format']} | {g['truncated']} | {g['transport_failure']} |")
    lines += ['', '| 配对检查 | 完整交付/8 | 正确方向关系/8 | 两答案均正确/8 |', '| --- | --- | --- | --- |']
    pair_labels = {'goal_flip_hazard': '危险侧：改目标应翻转', 'goal_flip_clear': '无烟侧：改目标应翻转',
                   'role_invariance_minimize': '最小化：等价信息应不变', 'role_invariance_maximize': '最大化：等价信息应不变'}
    for p in summary['paired_diagnostics']:
        lines.append(f"| {pair_labels[p['comparison']]} | {p['fully_delivered_pairs']} | {p['expected_relation_observed']} | {p['both_answers_correct']} |")
    if summary['all_cells_passed'] and summary['all_primary_pairs_correct']:
        interpretation = '8B通过本批基础功能题和全部配对检查，可以作为后续独立复测的候选模型。这不代表已经提高真实疏散效果，也未通过原PPO方案的评估门槛。'
    elif summary['always_follows_reported_location_in_valid_answers']:
        interpretation = '8B仍呈现32/32跟随报告位置的行为，本轮升级没有解决这些功能题中的目标与角色处理问题。不能单凭此结论确定内部原因或推广到所有LLM。'
    else:
        interpretation = '8B尚未全部通过基础功能与配对检查。需分别看方向、格式、截断和传输失败；一组正确率或总正确率不足以确认目标与角色处理已修复。'
    budget = c.read(OUT / 'budget.json')
    latency = sum(g['total_latency_seconds'] for g in summary['groups'].values())
    lines += ['', interpretation, '', f"8B最终答案跟随报告位置{summary['reported_location_match_count']}/32。只有最终content中的完整合法JSON计为可用；thinking不替代最终答案，缺失答案不填入回退损失。", '',
             '| 条件 | 平均期望暴露（合法答案） | 平均目标遗憾 | 平均回包秒数 |', '| --- | --- | --- | --- |']
    for name, label in labels.items():
        g = summary['groups'][name]
        show = lambda x: '未交付' if x is None else f'{x:.3f}'
        lines.append(f"| {label} | {show(g['mean_expected_cost_if_valid'])} | {show(g['mean_objective_regret_if_valid'])} | {show(g['mean_latency_seconds'])} |")
    lines += ['', '期望暴露是单步数学代价，并非实际伤亡或疏散率。最小化遗憾为所选代价减最小代价；最大化遗憾为最大代价减所选代价。自报confidence不是已经校准的风险概率。', '',
              f"完成{budget['logical_requests_completed']}个逻辑请求，实际开始{budget['api_attempts_started']}次本地聊天调用，成功服务响应{budget['successful_http_responses']}；累计回包等待{latency:.2f}秒，提示token{sum(g['prompt_tokens'] for g in summary['groups'].values())}，生成token{sum(g['generated_tokens'] for g in summary['groups'].values())}。只有传输/服务器失败允许一次重试。无付费API、模拟0、拟合0、PPO更新0。", '',
              '32回答来自4个固定镜像家庭，不作为32个独立样本，不作总体显著性结论。保留48疏散场景未启用，旧源代码、台账和全局状态保持原字节，扩大PPO训练继续暂停。', '',
              f'[固定方案]({PROTOCOL.as_posix()})；[逐题结果]({(OUT/"case_results.csv").as_posix()})；[模型比较]({(OUT/"model_comparison.csv").as_posix()})；[独立核验]({(OUT/"verification.json").as_posix()})。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


def run():
    m, status = c.read(OUT / 'manifest.json'), c.read(OUT / 'campaign_status.json')
    if status['state'] == 'complete':
        return
    if status.get('pid') != os.getpid() and c.pid_live(status.get('pid')):
        print('Existing worker preserved.', flush=True)
        return
    io.OUT, io.DIGEST, analysis.OUT = OUT, m['model_digest'], OUT
    io.check_frozen(m)
    assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == MODEL)['digest'] == m['model_digest']
    status.update(state='running', stage='inference', pid=os.getpid(), run_attempts=status['run_attempts'] + 1)
    io.save(OUT / 'campaign_status.json', status)
    try:
        for i, request in enumerate(m['request_order']):
            result = io.fetch(request, request['private_reference']['reference_exit'])
            io.reconcile_budget()
            if i == 0:
                io.save(OUT / 'loaded_model_snapshot.json', io.api('/api/ps', timeout=10))
            status.update(completed_logical_requests=i+1, last_request=request['request_id'], updated_at_utc=c.stamp())
            io.save(OUT / 'campaign_status.json', status)
            print(json.dumps({'completed': i+1, 'total': 32, 'condition': request['condition_id'], 'category': result['outcome']['category']}), flush=True)
        io.check_frozen(m)
        assert next(v for v in io.api('/api/tags', timeout=10)['models'] if v['name'] == MODEL)['digest'] == m['model_digest']
        summary = analysis.summarize(m)
        comparison(summary)
        from verify_llm_goal_role_8b import verify
        io.save(OUT / 'verification.json', verify())
        write_report(m, summary)
        paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('completion_manifest.json', 'campaign_status.json') and not p.name.endswith('.tmp')]
        paths += [ROOT / name for name in m['frozen_sha256']] + [REPORT]
        io.save(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(), 'budget': c.read(OUT / 'budget.json'),
                'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in sorted(set(paths))}, 'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0})
        from finalize_auto_evacuation_campaign_review import ledger_check
        ledger_check(OUT / 'completion_manifest.json')
        status.update(state='complete', stage='complete', completed_at_utc=c.stamp())
        io.save(OUT / 'campaign_status.json', status)
        print('8B goal-role replay sealed and complete.', flush=True)
    except Exception:
        status.update(state='failed', failed_at_utc=c.stamp())
        io.save(OUT / 'campaign_status.json', status)
        io.save(OUT / 'failures' / f"run_{status['run_attempts']:03d}.json", {'traceback': traceback.format_exc(), 'budget': c.read(OUT / 'budget.json')})
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
