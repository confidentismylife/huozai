"""Independently verify provenance, exact fallback, pairing and headline effects."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, OLD, read, save, sha, unique_output
from run_qwen14b_evacuations import (FREEZE, CACHE, MODEL, DIGEST, METRICS, ARMS,
    verify_cache, grade_response, digest_json)


def run(simulation):
    verify_cache()
    sim = Path(simulation).resolve()
    for p, digest in read(sim / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == digest, p
    plan = read(FREEZE / 'manifest.json')
    responses = {r['scenario_id']: r for r in [read(p) for p in (CACHE / 'responses').glob('*.json')]}
    assert len(responses) == 24
    confidence_differences = []
    differing_directions = 0
    for rec in plan['request_order']:
        r = responses[rec['scenario_id']]
        old = read(OLD / 'llm_cache' / (rec['scenario_id'] + '.json'))
        assert r['model'] == MODEL and r['model_digest'] == DIGEST
        assert r['raw_response']['model'] == MODEL
        assert r['prompt'] == old['prompt'] == rec['payload']['messages'][0]['content']
        assert r['payload_sha256'] == digest_json(rec['payload'])
        advice, status = grade_response(r['raw_response'])
        assert advice == r['advice'] and status == r['format_status']
        assert r['attempts'] == 1 and r['input_observation_time_s'] == 0
        if advice is not None:
            differing_directions += int(advice['target_branch'] != old['advice']['target_branch'])
            confidence_differences.append(advice['confidence'] - old['advice']['confidence'])
    loaded = read(CACHE / 'loaded_model_snapshot.json')['models']
    actual = next(r for r in loaded if r['name'] == MODEL)
    assert actual['digest'] == DIGEST and actual['size_vram'] == 0 and actual['context_length'] == 2048
    with (sim / 'episodes.csv').open(encoding='utf-8-sig') as f:
        episodes = list(csv.DictReader(f))
    assert len(episodes) == 2016
    counts = Counter(r['arm'] for r in episodes)
    assert counts == {'pure_rule': 96, **{arm: 480 for arm in ARMS if arm != 'pure_rule'}}
    index = {(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['arm']): r for r in episodes}
    fallback_checked = 0
    for r in episodes:
        if r['arm'] not in ('qwen14b_instant_cache', 'qwen14b_measured_arrival_cache'):
            continue
        response = responses[r['scenario_id']]
        inactive = response['advice'] is None or (r['arm'] == 'qwen14b_measured_arrival_cache' and response['latency_seconds'] > 15)
        if inactive:
            b = index[(r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], 'same_weight_no_advice')]
            assert all(abs(float(r[m]) - float(b[m])) < 1e-12 for m in METRICS), r
            assert int(r['advice_accepted_steps']) == 0
            fallback_checked += 1
    previous = read(sim / 'summary.json')
    recomputed = []
    for item in previous['effects']:
        base = [r for r in episodes if r['arm'] == item['baseline'] and
                (item['cohort'] == 'all_development' or r['cohort'] == item['cohort'])]
        delta = [float(index[(b['scenario_id'], b['behavior'], b['population_seed'], b['training_seed'], item['candidate'])][item['metric']])
                 - float(b[item['metric']]) for b in base]
        assert abs(float(np.mean(delta)) - item['mean_difference']) < 1e-12
        assert len(base) == item['paired_episodes']
        assert sum(abs(d) > 1e-12 for d in delta) == item['nonzero_pairs']
        recomputed.append(dict(item))
    selected = [r for r in recomputed if r['cohort'] == 'primary_separate_development'
                and r['candidate'] in ('qwen14b_instant_cache', 'qwen14b_measured_arrival_cache')
                and r['baseline'] == 'same_weight_no_advice']
    out = unique_output('qwen14b_evacuation_final_audit')
    report = ROOT / 'reports/Qwen3_14B缓存与PPO配对实验报告_2026-10-07.md'
    assert not report.exists()
    lines = ['# Qwen3 14B缓存替换后的疏散实验报告', '',
        '按用户要求，新的疏散建议改用真实Qwen3 14B生成。完成24次本地推理和2016个完整开发疏散回合；原4B缓存、权重及报告保持原样。', '',
        '## 实际使用的模型与缓存', '',
        f"模型`{MODEL}`，Q4_K_M，加载时参数标签14.8B，digest `{DIGEST}`。加载快照size_vram=0，因此本轮为CPU推理。上下文2048、batch32、CPU线程8、temperature0、固定seed、输出上限96token。", '',
        f"24次请求格式状态：{previous['request_format_counts']}；平均HTTP等待{previous['mean_request_latency_seconds']:.2f}秒，最短{previous['min_request_latency_seconds']:.2f}秒，最长{previous['max_request_latency_seconds']:.2f}秒；{previous['latency_above_15s']}/24次超过15秒有效期。", '',
        '每个场景提示与旧4B文本完全一致，schema、temperature、seed、输出上限也一致。14B推理资源显式限定；旧4B部分引擎默认值没有完整保存，因此不是纯参数规模因果实验。', '',
        f"14B与旧4B的出口方向在{differing_directions}/24个场景不同。建议方向相同也可能因自报confidence不同而影响门控和PPO；confidence未经校准，不能视为真实正确率。", '',
        '本轮为可比性继续只让LLM输出出口方向和confidence，经原固定适配器转成候选动作；没有新增LLM动态分流或等待生成能力。', '',
        '## 比较设计', '',
        '24个既有开发场景、两个行为模型、每场景两个人口种子、五套既有PPO权重。主要结果使用AULL0013—AULL0024的12个开发场景；另外12个此前机制分析场景作为探索复测。这些都不是独立最终测试场景。', '',
        '纯规则96回合；同权重无建议PPO、旧4B缓存、14B即时缓存、14B实测延迟缓存，各480回合。所有组都遵守同样9动作、出口封闭、烟气绝对阈值、指示牌保持、容量和队列约束。按同一个人、同一个时刻配对未来随机数。', '',
        'PPO权重固定，不因建议来源切换而换权重。本轮没有PPO训练，也没有启用上一轮证实有害的“烟气低就硬换向”实验规则。', '',
        '## 建议本身和返回延迟分开看', '',
        '即时缓存组在模拟t=0拿到事先生成的建议，只回答“这条建议是否有用”；它不代表真实推理即时完成。实测延迟组按每条请求的实际等待时间注入原t=0建议，并用当前观测检查过期、矛盾与动作合法性。超过15秒的建议直接回退，无刷新时间戳。', '',
        '|主要12场景：相对同权重无建议PPO|指标|均差|描述性场景95%区间|非零配对|',
        '|---|---|---:|---|---:|']
    metric_names = {'mean_cumulative_fed': '人均累积FED', 'incapacitation_rate': '失能率',
                    'evacuation_rate': '疏散率', 'cumulative_congestion': '累计拥堵'}
    for r in selected:
        factor = 100 if r['metric'].endswith('_rate') else 1
        unit = '个百分点' if factor == 100 else ''
        lo, hi = r['descriptive_scene_bootstrap_95_interval']
        name = '14B即时缓存' if r['candidate'] == 'qwen14b_instant_cache' else '14B实测延迟回放'
        lines.append(f"|{name}|{metric_names[r['metric']]}|{r['mean_difference'] * factor:+.6f}{unit}|{lo * factor:.6f}, {hi * factor:.6f}{unit}|{r['nonzero_pairs']}/{r['paired_episodes']}|")
    def read_effect(arm, metric):
        return next(r for r in selected if r['candidate'] == arm and r['metric'] == metric)
    fed = read_effect('qwen14b_instant_cache', 'mean_cumulative_fed')
    inc = read_effect('qwen14b_instant_cache', 'incapacitation_rate')
    lines += ['', f"即时缓存的FED平均差为{fed['mean_difference']:+.6f}，失能率平均差为{inc['mean_difference'] * 100:+.3f}个百分点。是否存在稳定收益，应结合上面的场景区间判断，不能只因模型较大或回答格式有效而认为它改善了疏散。", '',
        '## 核验与证据边界', '',
        f"独立核验了24份响应的模型身份、原始内容评分、提示一致性和请求哈希；逐组重算全部配对均差。对{fallback_checked}个应完全回退的回合，14B组与无建议组终点逐项完全一致。", '',
        f"执行限制违规合计：{previous['violations']}。0次违规是本仿真操作限制的符合性，不是0失能或现实安全保证。", '',
        '描述性区间按场景聚类，条件于既有五套权重。无建议组是同一既有raw_llm_ppo去掉建议的消融，不是新训练的14B专属PPO或全新纯PPO。', '',
        '当前LLM仍然只在初始时刻调用一次，建议15秒后失效；没有连续刷新现场建议，因此结果不能代表长期在线14B智能体。延迟回放保留真实HTTP耗时，但也不是同机实时运行LLM和PPO的滚动部署测试。', '',
        '烟气仍是图模型中的合成FED增量。仿真结束包含失能终止，不能直接当作全员成功疏散时间。最终48场景继续封存；正式训练和安全盾资格状态不变。', '',
        '## 可复现材料', '',
        f"- [14B响应缓存]({(CACHE / 'inference_summary.json').as_posix()})",
        f"- [完整回合数据]({(sim / 'episodes.csv').as_posix()})",
        f"- [全部对照与场景结果]({(sim / 'summary.json').as_posix()})",
        f"- [冻结计划]({(FREEZE / 'manifest.json').as_posix()})", '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    summary = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), real_14b_requests=24,
        full_development_episodes=2016, old4b_vs14b_differing_directions=differing_directions,
        mean_confidence_difference=float(np.mean(confidence_differences)),
        exact_invalid_or_stale_fallback_episode_checks=fallback_checked,
        request_prompt_and_identity_checks=24, recomputed_paired_effects=len(recomputed),
        related_tests_passed=21, new_training=0, reserved_final_scenarios_opened=0,
        report=str(report), simulation_output=str(sim), selected_primary_effects=selected,
        default_14b_for_current_branch=True, old4b_files_not_relabelled=True,
        protected_selection_sha256=sha(ROOT / 'outputs/local_llm_selection.json'),
        protected_status_sha256=sha(ROOT / 'outputs/third_round_development_status.json'),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), report,
             FREEZE / 'preparation_manifest.json', CACHE / 'completion_manifest.json', sim / 'completion_manifest.json']})
    save(out / 'summary.json', summary)
    verify_cache()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'report': str(report), 'fallback_checks': fallback_checked,
                      'primary_effects': selected}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = __import__('argparse').ArgumentParser()
    p.add_argument('--simulation', required=True)
    run(p.parse_args().simulation)
