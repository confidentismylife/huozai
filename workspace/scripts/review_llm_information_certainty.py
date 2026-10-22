"""Read-only interpretation of a sealed probability experiment."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import sys

import auto_evacuation_campaign as c
from finalize_auto_evacuation_campaign_review import ledger_check

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'outputs/llm_information_certainty_20261006'
OUT = ROOT/'outputs/llm_information_certainty_review_20261006'
REPORT = ROOT/'reports/LLM信息确定性结果复核与下一步_2026-10-06.md'


def parameters(value):
    result = {}
    for line in value.splitlines():
        key, child = line.strip().split(None, 1)
        result.setdefault(key, []).append(child.strip())
    return result


def run():
    assert not OUT.exists() and not REPORT.exists()
    status = c.read(STUDY/'campaign_status.json')
    assert status['state'] == 'complete' and not c.pid_live(status['pid'])
    records = ledger_check(STUDY/'completion_manifest.json')
    verification = c.read(STUDY/'verification.json')
    assert verification['all_checks_passed']
    manifest, summary = c.read(STUDY/'manifest.json'), c.read(STUDY/'summary.json')
    assert all(g['valid_wrong_direction'] == 8 and g['usable_correct'] == 0 and g['truncated'] == 0
               and g['invalid_format'] == 0 and g['transport_failure'] == 0 for g in summary['groups'].values())
    old = c.read(ROOT/'outputs/llm_output_mechanism_20261006/model_service_snapshot.json')
    new = c.read(STUDY/'model_service_snapshot.json')
    same_params = parameters(old['show']['parameters']) == parameters(new['show']['parameters'])
    same_template = old['show']['template'] == new['show']['template']
    same_version = old['version'] == new['version']
    assert same_params and same_template and same_version
    identity_path = ROOT/'outputs/llm_information_certainty_runtime_20261006/post_server_identity.json'
    identity = c.read(identity_path)
    assert identity and all(r['Name'].lower() == 'ollama.exe' for r in identity)
    copies, confidences = 0, {}
    response_count = 0
    for req in manifest['request_order']:
        result = c.read(STUDY/'results'/f"{req['request_id']}.json")
        if result['raw_path'] is None:
            continue
        raw = c.read(ROOT/result['raw_path'])
        data = json.loads(req['payload']['messages'][0]['content'].split('Data: ')[1])
        answer = json.loads(raw['response']['message']['content'])
        response_count += 1
        copies += answer['recommended_exit'] == data['hazard_location']
        confidences.setdefault(req['condition_id'], []).append(answer['confidence'])
    assert response_count == copies == 32
    latency = sum(g['total_latency_seconds'] for g in summary['groups'].values())
    cost = {'chat_requests': c.read(STUDY/'budget.json')['api_attempts_started'], 'response_latency_seconds': latency,
        'prompt_tokens': sum(g['prompt_eval_tokens'] for g in summary['groups'].values()),
        'generated_tokens': sum(g['generated_tokens'] for g in summary['groups'].values()),
        'paid_api_requests': 0, 'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0}
    review = {'all_checks_passed': True, 'source_direction_equals_output': copies, 'responses_checked': response_count,
        'confidence_values_by_condition': {k: dict(Counter(map(str, v))) for k, v in confidences.items()},
        'normalized_model_parameters_identical': same_params, 'parameter_string_difference_is_order_only': True,
        'model_template_identical': same_template, 'service_version_identical': same_version,
        'post_run_listener_identity': identity, 'listener_identity_sha256': c.sha(identity_path),
        'sealed_study_records_checked': records, 'prior_completion_sha256': c.sha(STUDY/'completion_manifest.json'),
        'cost': cost, 'new_inference_for_review': 0, 'historical_failure_root_cause_established': False,
        'reviewer_sha256': c.sha(__file__), 'created_at_utc': c.stamp()}
    lines = ['# LLM信息确定性实验：通俗结果与下一步', '', '日期：2026-10-06（北京时间）。32请求实验已完成并封存。本文件是只读复核，没有追加模型调用。', '',
        '**本轮未通过基础出口判断：32/32回答格式合法、正常结束，却都把危险报告位置选成了疏散目的地。已确认条件也失败，所以不能把这个行为仅归因于预测不确定。**', '',
        '| 条件 | 正确出口/8 | 选危险报告方向/8 | 模型选择的期望暴露 | 理论最小期望暴露 | 多出的期望代价 |',
        '| --- | --- | --- | --- | --- | --- |',
        '| 已确认，100% | 0 | 8 | 1.0 | 0.0 | 1.0 |',
        '| 预测，100%（理想桥接） | 0 | 8 | 1.0 | 0.0 | 1.0 |',
        '| 预测，90% | 0 | 8 | 0.9 | 0.1 | 0.8 |',
        '| 预测，60% | 0 | 8 | 0.6 | 0.4 | 0.2 |', '',
        '这些代价是本轮单步数学题定义的0–1烟气暴露量，不是伤亡率、FED或真实火灾指标。题目规定恰有一侧危险、另一侧无暴露，报告正确概率q已给定且校准；走报告侧期望代价q，走另一侧1-q。没有真实火源实现或算好的答案进入提示。', '',
        '例如报告“北侧危险”的正确概率为90%，模型选择北出口时，期望暴露是0.9；选择南出口时是0.1。模型这题多付出的理论代价是0.8。北/南镜像全部互换4/4，但互换的是错误出口，所以镜像通过不能当成语义正确。', '',
        '全部回包是严格JSON，没有截断、传输失败或格式错误。原始recommended_exit字段就等于hazard_location，并不是程序把出口编号翻转，也没有从思考文本中补答案。这说明“能交付JSON”与“能给正确疏散方向”是两个不同检查。方向与输入字段32/32相同是可观察事实；“模型内部只是在复制字段”仍是需要进一步对照的解释，不能直接认定。', '',
        '三个预设配对比较的正确案例净变化均为0：已确认100%改为预测100%、预测100%改为90%、预测100%改为60%，均未产生方向改善或变差。它们共同失败，当前数据没有识别出这些信息条件造成的正确率差异；这不证明不确定程度在其他任务里没有影响。', '',
        '上一轮8/8通过的题明确描述了已确认活跃火灾和另一侧无烟；本轮换成统一的未来单步概率世界。共同提示、信息表示和目标理解仍可能影响行为，但跨轮同时改变了这些内容，不能认定某一项就是根因。旧题未给定概率，当前给定概率题的结果也不能自动替代历史题的参考定义。', '',
        '服务参数核查：原始parameters字符串有顺序差异，但逐键值归一化后相同；模型digest、服务版本和模板一致。端口11434的监听进程记录为本机ollama.exe。不能把字符串排序误判成采样参数变更。LLM自报confidence分布另存于复核JSON，没有把它当作经过校准的策略正确概率。', '',
        f"成本：32次本地聊天调用，累计回包等待{latency:.2f}秒，提示评估token{cost['prompt_tokens']}、生成token{cost['generated_tokens']}。没有付费API、模拟、PPO更新或拟合。6项新测试通过，原始请求与结果、32项概率参考和全部祖先台账已核验；48个保留疏散场景仍未使用。", '',
        '下一步应继续暂停扩大PPO训练。更有价值的小实验是：保持同一概率世界和输入，预先固定“最小化暴露/最大化暴露”的任务目标对照，以及字段角色表达对照，检验输出是否会随真正的目标改变。所有新比较需提前冻结，不能看到答案后换提示挑通过结果。尚未找到正确方向为什么会在不同表述下反转的完整机制，也没有新的LLM疏散效果提升证据。', '',
        f'[原始实验报告]({(ROOT/"reports/LLM信息确定性对照实验报告_2026-10-06.md").as_posix()})；[逐题结果]({(STUDY/"case_results.csv").as_posix()})；[独立验证]({(STUDY/"verification.json").as_posix()})；[复核明细]({(OUT/"review.json").as_posix()})。', '']
    OUT.mkdir()
    c.write(OUT/'review.json', review)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    paths = [OUT/'review.json', REPORT, Path(__file__)]
    c.write(OUT/'completion_manifest.json', {'all_checks_passed': True, 'created_at_utc': c.stamp(),
        'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths}, 'new_inference': 0,
        'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0})
    ledger_check(OUT/'completion_manifest.json')
    print(json.dumps({'review_complete': True, 'source_direction_equals_output': copies, 'cost': cost,
        'parameters_identical_after_normalization': same_params, 'study_records': records}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
