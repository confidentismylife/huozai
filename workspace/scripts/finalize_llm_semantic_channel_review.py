"""Post-completion descriptive interpretation; never fits or simulates."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import auto_evacuation_campaign as c

ROOT = c.ROOT
BASE = ROOT / 'outputs/llm_semantic_channel_diagnostic_20261005'
OUT = ROOT / 'outputs/llm_semantic_channel_review_20261005'
REPORT = ROOT / 'reports/LLM建议语义诊断结果与下一步_2026-10-05.md'


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def run():
    assert not OUT.exists() and not REPORT.exists()
    status = c.read(BASE / 'campaign_status.json')
    assert status['state'] == 'complete' and not c.pid_live(status['pid'])
    ledger = c.read(BASE / 'completion_manifest.json')
    for name, digest in ledger['files'].items():
        assert c.sha(ROOT / name) == digest, name
    from finalize_auto_evacuation_campaign_review import ledger_check, MAIN, LLM, MANAGER, HISTORY
    parent_counts = {'main': ledger_check(MAIN / 'completion_manifest_v2.json', HISTORY),
                     'llm': ledger_check(LLM / 'completion_manifest.json'),
                     'manager': ledger_check(MANAGER / 'completion_manifest.json'),
                     'review': ledger_check(ROOT / 'outputs/evacuation_auto_final_review_20261005/completion_manifest.json')}
    from verify_llm_semantic_channel_diagnostic import verify
    verification = verify()
    assert verification == c.read(BASE / 'verification.json')
    manifest = c.read(BASE / 'manifest.json')
    budget = c.read(BASE / 'budget.json')
    rows = csv_read(BASE / 'diagnostic_family_results.csv')
    factors = csv_read(BASE / 'factorial_channel_responses.csv')
    means = {arm: float(np.mean([float(r['loss_pp']) for r in rows if r['arm'] == arm])) for arm in sorted({r['arm'] for r in rows})}
    indexed = {(r['family'], int(r['seed']), r['arm']): float(r['loss_pp']) for r in rows}
    from llm_semantic_channel_core import DIAGNOSTIC_ARMS
    for scene in manifest['development']:
        family = scene['scenario_id']
        for seed in range(601, 606):
            for arm in DIAGNOSTIC_ARMS:
                value = c.load(BASE / 'diagnostic_actual' / f'{seed}_{arm}_{family}.npz')['risks'][..., 0].mean() * 100
                np.testing.assert_allclose(value, indexed[(family, seed, arm)], rtol=0, atol=1e-10)
    contrast_arms = {'confidence_north': ('north_c50', 'north_c95'), 'confidence_south': ('south_c50', 'south_c95'),
                     'direction_c50': ('north_c50', 'south_c50'), 'direction_c95': ('north_c95', 'south_c95')}
    for row in factors:
        a, b = contrast_arms[row['factor']]
        key = (row['family'], int(row['seed']))
        np.testing.assert_allclose(float(row['loss_b_minus_a_pp']), indexed[(*key, b)] - indexed[(*key, a)], rtol=0, atol=1e-10)
    effects = {name: float(np.mean([float(r['loss_b_minus_a_pp']) for r in factors if r['factor'] == name])) for name in contrast_arms}
    advice = c.read(BASE / 'diagnostic_advice_frozen.json')
    same_direction = sum(advice['original'][k]['target_branch'] == advice['repaired'][k]['target_branch'] for k in advice['original'])
    confidence_changed = sum(advice['original'][k]['confidence'] != advice['repaired'][k]['confidence'] for k in advice['original'])
    raw = [c.read(path) for path in (BASE / 'llm_raw_responses').glob('*.json')]
    costs = {'real_local_llm_requests': budget['llm_requests'], 'total_latency_seconds': float(sum(r['latency_seconds'] for r in raw)),
             'prompt_eval_tokens': int(sum(r['response']['prompt_eval_count'] for r in raw)),
             'generated_tokens': int(sum(r['response']['eval_count'] for r in raw)), 'paid_api_calls': 0}
    semantic = c.read(BASE / 'semantic_summary.json')
    diagnostic = c.read(BASE / 'diagnostic_summary.json')
    metrics = {'descriptive_mean_loss_percent': means, 'factorial_mean_effect_pp': effects,
               'native_direction_unchanged_families': same_direction, 'native_confidence_changed_families': confidence_changed,
               'repaired_minus_original_native_pp': means['repaired_native'] - means['original_native'],
               'repaired_minus_rule_native_pp': means['repaired_native'] - means['rule_native'], 'cost': costs,
               'fresh_bank_unused': True, 'all_effects_are_development_descriptions': True}
    assert not diagnostic['proceed_to_fresh']
    assert not (BASE / 'fresh_actual').exists() and not (BASE / 'fresh_advice_frozen.json').exists()
    assert not any(p.name.startswith('fresh_') for p in (BASE / 'llm_cache').glob('*.json'))
    lines = [
        '# LLM建议语义诊断：结果与下一步', '',
        '日期：2026-10-05。用户授权的诊断已完成，并从原始缓存独立复核；未启动新的PPO训练。', '',
        '**本轮查清了：程序没有把出口方向翻转；PPO会响应建议；仅澄清提示与JSON字段没有改变该模型跟随预测火源的输出行为。** 这还不能区分模型决策倾向、提示理解与结构化生成机制等更深原因。', '',
        '| 检查 | 结果 | 可得结论 |', '| --- | --- | --- |',
        '| 旧48条原始回答→解析→建议列 | 全部一致，北0南1正确 | 未发现解析或编号翻转 |',
        '| 原提示，12个简单镜像案例 | 参考方向正确0/12，镜像互换6/6 | 输出随火源方向变化，没有通过离开火源的参考检查 |',
        '| 修订提示，相同12个案例 | 参考方向正确0/12，镜像互换6/6 | 字段与含义澄清没有修复该行为 |',
        '| 相同置信度、改变推荐方向 | 请求动作序列改变720/720 | PPO会响应方向输入 |',
        '| 相同方向、置信度0.50→0.95 | 请求动作序列改变669/720 | PPO也会响应置信度输入 |',
        '| 旧缓存一致性 | 2条完整模拟、11项指标一致 | 新记录工具没有改变原模拟结局 |', '',
        '镜像方向互换不等于语义正确：火源从北换成南时，模型也把输出从北换成南，因此镜像检查通过，但离开火源的参考方向仍为0/12。这个参考只用于开放、等容量、立即起火的简单案例，不能作为复杂场景最优标签。实际场景提示没有输入真实火源、冲突标签或计算好的出口答案。', '',
        '在6个已看开发家庭中，原与修订LLM方向6/6相同，置信度有4/6改变。因此，修订原生建议与原建议的结果差主要来自这些置信度变化；不能宣称已经修复出口判断。', '',
        '下面是同一个raw_llm PPO在不同建议输入下的平均失能率。只描述这6个开发家庭、5个冻结历史种子、匹配人口与未来块。', '',
        '| 建议输入 | 平均失能率（%） |', '| --- | --- |']
    labels = {'none': '同一控制器无建议', 'north_c50': '北出口，置信度0.50', 'north_c95': '北出口，置信度0.95',
              'south_c50': '南出口，置信度0.50', 'south_c95': '南出口，置信度0.95', 'original_native': '原LLM原生建议（旧缓存）',
              'rule_native': '规则原生建议（旧缓存）', 'repaired_native': '修订LLM原生建议'}
    lines += [f'| {labels[name]} | {value:.6f} |' for name, value in means.items()]
    lines += ['',
        f'修订减原LLM的平均失能差为{metrics["repaired_minus_original_native_pp"]:+.6f}个百分点，修订减规则为{metrics["repaired_minus_rule_native_pp"]:+.6f}个百分点。这里只是诊断描述，未作新场景显著性或资格判断。', '',
        f'固定北方向，置信度0.50→0.95的失能差为{effects["confidence_north"]:+.6f}个百分点；固定南方向为{effects["confidence_south"]:+.6f}个百分点。它说明控制器对该输入范围有响应，不能据此认定LLM报出的置信度已经校准，或把更多高置信度当成下一步优化方案。', '',
        '无建议臂是删除同一个带建议控制器的输入，不等于另行训练的纯PPO。无建议时的较高损失可能包含输入分布变化，不能据此证明LLM优于纯PPO。方向干预同时更新由方向定义的烟气矛盾特征，结果属于建议方向通道的整体响应，不能单独归因于某个输入列。复杂开发案例未左右镜像平衡，北/南均值不能解释为北出口普遍更好。动作序列改变也不等于安全改善。', '',
        f'本轮新增完整模拟2162条，其中2160条控制对照、2条仪器核查；另720条历史结局只读复用，不计为新模拟或独立样本。本地LLM请求30次，总延迟{costs["total_latency_seconds"]:.2f}秒，提示评估tokens {costs["prompt_eval_tokens"]}、生成tokens {costs["generated_tokens"]}；付费API请求0、PPO更新0、模型拟合0。', '',
        f'独立复核检查2162条结局及{verification["full_trace_steps_checked"]}个轨迹步，重算语义结果、输入配对、请求与执行动作、人数守恒、FED和放行记录。完成台账{len(ledger["files"])}项全部匹配；旧主流程{parent_counts["main"]}项、旧LLM分支{parent_counts["llm"]}项等祖先台账保持原字节。', '',
        '**48个新场景没有进入评估。** 这是预先固定的停止条件：修订提示未通过简单语义检查。新银行只保存场景与种子，没有生成建议或采集结局；不把未运行说成通过，也不在看到失败后修改提示再刷同一轮。', '',
        '后续优先级是一个独立的、很小的模型输出机制实验：分别检查自由文本出口回答与结构化JSON回答，以及当前推理设置，确定“跟随火源”在哪一环出现。需要预先冻结新对照与停止条件，使用新的简单案例；不能从多次提示尝试中挑一次通过追认有效。当前不扩大PPO训练，也不把硬编码规则包装成LLM贡献。', '',
        f'[本轮冻结方案]({(ROOT/"reports/LLM建议语义与控制通道自动诊断固定方案_2026-10-05.md").as_posix()})；'
        f'[原始阶段报告]({(ROOT/"reports/LLM建议语义与控制通道实验报告_2026-10-05.md").as_posix()})；'
        f'[独立验证]({(BASE/"verification.json").as_posix()})；[方向与置信度配对表]({(BASE/"factorial_channel_responses.csv").as_posix()})。', '',
        f'[本次复核数值]({(OUT/"metrics.json").as_posix()})；[完整实验台账]({(BASE/"completion_manifest.json").as_posix()})。', '']
    OUT.mkdir()
    c.write(OUT / 'metrics.json', metrics)
    c.write(OUT / 'verification.json', {'all_checks_passed': True, 'study_ledger_records': len(ledger['files']),
            'parent_records_checked': parent_counts, 'phase_verification_repeated_without_new_simulations': True,
            'factorial_summary_numbers_reconstructed': True, 'unused_fresh_bank_verified': True,
            'new_simulations': 0, 'new_fits': 0, 'new_ppo_updates': 0, 'reviewer_sha256': c.sha(__file__)})
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    paths = list(OUT.iterdir()) + [REPORT, Path(__file__)]
    c.write(OUT / 'completion_manifest.json', {'all_checks_passed': True, 'parent_completion_sha256': c.sha(BASE / 'completion_manifest.json'),
            'files': {p.relative_to(ROOT).as_posix(): c.sha(p) for p in paths}})
    for name, digest in c.read(OUT / 'completion_manifest.json')['files'].items():
        assert c.sha(ROOT / name) == digest, name
    print(json.dumps({'report': str(REPORT), 'metrics': metrics, 'all_checks_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
