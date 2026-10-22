"""Audit completed development episodes and write a bounded Chinese synthesis.

No new simulations, fitted models, LLM requests or final-bank reads.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output
from run_direction_hint_fresh_population_confirmation import PREP, MODES, VARIANTS, verify
from run_qwen14b_evacuations import METRICS


ATTRIBUTION = ROOT / 'outputs/qwen14b_action_attribution_final_audit_20261007T052306160460Z'
REPORT = ROOT / 'reports/14B动作审计与出口提示修正验证_2026-10-07.md'
PRIMARY = 'primary_separate_development'
PROTECTED = {
    'outputs/local_llm_selection.json': '302c6db2c857811dd41d1ed36248d1d9a71f3e32108cd870ecfe02aa520ff721',
    'outputs/third_round_development_status.json': 'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0',
}


def verify_files(manifest):
    for p, h in read(manifest)['files'].items():
        assert sha(ROOT / p) == h, p


def link(label, path):
    return f'[{label}]({Path(path).as_posix()})'


def run(source):
    source = Path(source).resolve()
    assert source.parent == ROOT / 'outputs'
    assert source.name.startswith('direction_hint_fresh_population_confirmation_')
    assert not REPORT.exists(), 'Preserve completed report; do not overwrite.'
    verify()
    verify_files(source / 'completion_manifest.json')
    verify_files(ATTRIBUTION / 'completion_manifest.json')
    attribution = read(ATTRIBUTION / 'summary.json')
    for p, h in attribution['sources'].items():
        assert sha(ROOT / p) == h, p
    for p, h in PROTECTED.items():
        assert sha(ROOT / p) == h, p
    plan = read(PREP / 'manifest.json')
    summary = read(source / 'summary.json')
    assert read(source / 'run_manifest.json')['preparation_sha256'] == sha(PREP / 'manifest.json')
    with (source / 'episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == summary['completed_full_episodes'] == plan['full_episode_budget'] == 2880
    identity = ('scenario_id', 'behavior', 'population_seed', 'training_seed', 'source_mode', 'variant')
    index = {tuple(r[k] for k in identity): r for r in rows}
    assert len(index) == len(rows)
    scenes = {s['scenario_id'] for s in plan['scenarios']}
    expected = set()
    for si, scene in enumerate(plan['scenarios']):
        for rep in range(2):
            pop = str(1477100000 + si * 10 + rep)
            for behavior in plan['behavior_models']:
                for seed in plan['training_seeds']:
                    for mode in MODES:
                        for variant in VARIANTS:
                            expected.add((scene['scenario_id'], behavior, pop, str(seed), mode, variant))
    assert set(index) == expected
    assert {r['scenario_id'] for r in rows} == scenes
    assert {int(r['population_seed']) for r in rows} == set(plan['population_seeds'])
    assert len(plan['population_seeds']) == 48
    violations = {k: sum(int(r[k]) for r in rows) for k in summary['violations']}
    assert violations == summary['violations']
    assert not any(violations.values())
    same_dir = sum(int(r['same_direction_release_changed_steps']) for r in rows
                   if r['variant'] == 'direction_hint_only')
    assert same_dir == summary['direction_hint_same_direction_release_changes'] == 0
    assert all(int(r['action_changed_steps']) == 0 for r in rows if r['variant'] == 'no_advice')
    assert all(int(r['remaining']) == 0 for r in rows)
    assert all(np.isfinite(float(r[m])) for r in rows for m in METRICS)
    assert all(abs(float(r['evacuation_rate']) + float(r['incapacitation_rate']) - 1) < 1e-12 for r in rows)
    # Neither timing mode can alter the no-advice control or its paired future.
    baseline_timing_pairs = 0
    for r in rows:
        if r['variant'] != 'no_advice' or r['source_mode'] != MODES[0]:
            continue
        other = index[tuple(r[k] if k != 'source_mode' else MODES[1] for k in identity)]
        for k in (*METRICS, 'remaining', 'episode_end_seconds', 'all_evacuated_seconds'):
            assert r[k] == other[k], (k, r)
        baseline_timing_pairs += 1
    assert baseline_timing_pairs == 480
    # Recompute every frozen paired effect from raw records, including intervals.
    for e in summary['paired_effects']:
        grouped = defaultdict(list)
        nonzero = 0
        pairs = 0
        for b in rows:
            if b['source_mode'] != e['mode'] or b['variant'] != e['baseline']:
                continue
            if e['cohort'] != 'all_development' and b['cohort'] != e['cohort']:
                continue
            a = index[tuple(b[k] if k != 'variant' else e['candidate'] for k in identity)]
            d = float(a[e['metric']]) - float(b[e['metric']])
            grouped[b['scenario_id']].append(d)
            pairs += 1
            nonzero += abs(d) > 1e-12
        values = np.array([np.mean(v) for v in grouped.values()])
        rng = np.random.default_rng(2026100765)
        boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        interval = np.quantile(boot, [.025, .975])
        assert pairs == e['paired_episodes'] and nonzero == e['nonzero_pairs']
        assert len(grouped) == e['scenario_clusters']
        assert abs(values.mean() - e['mean_difference']) < 1e-12
        assert np.allclose(interval, e['descriptive_scene_bootstrap_95_interval'], atol=1e-12, rtol=0)
    for record in summary['action_change_counts']:
        count = sum(int(r['action_changed_steps']) for r in rows
                    if all(r[k] == record[k] for k in ('cohort', 'variant'))
                    and r['source_mode'] == record['mode'])
        assert count == record['action_changed_steps']
    assert summary['new_llm_requests'] == summary['new_training'] == summary['reserved_final_scenarios_opened'] == 0
    # Extra breakdowns are explicitly descriptive and cannot drive a parameter sweep.
    descriptions = []
    for mode in MODES:
        for variant in VARIANTS:
            subset = [r for r in rows if r['cohort'] == PRIMARY
                      and r['source_mode'] == mode and r['variant'] == variant]
            descriptions.append(dict(mode=mode, variant=variant, episodes=len(subset),
                episodes_with_action_change=sum(int(r['action_changed_steps']) > 0 for r in subset),
                action_changed_steps=sum(int(r['action_changed_steps']) for r in subset),
                same_direction_release_changed_steps=sum(int(r['same_direction_release_changed_steps']) for r in subset),
                all_evacuated_episodes=sum(bool(r['all_evacuated_seconds']) for r in subset),
                mean_episode_end_seconds=float(np.mean([float(r['episode_end_seconds']) for r in subset])),
                means={m: float(np.mean([float(r[m]) for r in subset])) for m in METRICS}))
    primary = [e for e in summary['paired_effects'] if e['cohort'] == PRIMARY]
    def effect(mode, candidate, baseline, metric):
        return next(e for e in primary if all(e[k] == v for k, v in
                    [('mode', mode), ('candidate', candidate), ('baseline', baseline), ('metric', metric)]))
    assert all(effect(m, 'direction_hint_only', 'no_advice', 'mean_cumulative_fed')['mean_difference'] > 0
               for m in MODES)
    assert effect(MODES[0], 'direction_hint_only', 'no_advice', 'incapacitation_rate')['mean_difference'] > 0
    decision = 'Do not adopt direction_hint_only. Reduced interference versus original adapter is not an LLM gain over no_advice; no strength increase, new training, or final-bank evaluation.'
    out = unique_output('direction_hint_confirmation_final_audit')
    labels = {MODES[0]: '即时缓存', MODES[1]: '实测延迟回放'}
    variants = {'no_advice': '同权重无建议PPO', 'original_advice_adapter': '原建议适配器', 'direction_hint_only': '仅出口方向提示'}
    metrics = {'mean_cumulative_fed': '平均累计FED', 'incapacitation_rate': '失能率差（百分点）',
               'evacuation_rate': '疏散成功率差（百分点）', 'cumulative_congestion': '累计拥堵指标'}
    lines = [
        '# 14B动作审计与出口提示修正验证', '',
        '日期：2026年10月7日。结论：只保留14B出口提示，减少了旧接入方式的干扰，但没有给同权重无建议PPO带来正向增量，本轮不采用该修正。继续冻结权重和最终场景，暂停扩大训练。', '',
        '## 1. 这一轮回答什么问题', '',
        '之前只能看到“14B建议通过检查”或“疏散终点有变化”。本轮沿着三层问题展开：建议有没有改变动作；改变在同一状态下是否值得；根据机制提出的小修正，换一批人后能否改善全程结果。', '',
        '14B提供出口方向和自报confidence；它没有直接生成本轮的放行容量。原适配器把建议字段送进PPO，再对固定25%/50%候选加分，所以模型建议、PPO输入特征和容量加分必须分开解释。', '',
        '## 2. 已完成的实验量', '',
        '|阶段|实际完成|作用|', '|---|---:|---|',
        '|精确重放|960个既有回合，42,034个控制步|核对已有终点，追踪建议到实际执行|',
        '|执行器复制检查|160个动作变化步|分别检查方向、容量、人员位置和当步流量|',
        '|合法动作反事实|1,024个关键状态，9,342条完整后续轨迹|同一状态逐一比较全部当前合法动作|',
        '|修正验证|2,880个完整回合|原适配器、新方向提示、无建议PPO配对比较|', '',
        '这些数量不是独立建筑场景数。修正验证使用24个开发场景、48批新人口、两种行为模型、5套既有权重、两种时序模式和三个控制版本：24×2×2×5×2×3=2,880。主要分析为AULL0013—AULL0024这12个开发场景，每个模式、每个版本240个回合；其他12个场景用于探索性机制分析。', '',
        f"本次2,880回合运行约{summary['elapsed_seconds']:.1f}秒；缓存真实14B响应，不重新调用模型。新人口种子与此前分离，但建筑场景和初始出口建议仍复用，不能算新的最终测试。", '',
        '## 3. 动作审计发现了什么', '',
        '|主要12场景，既有人口|即时缓存|实测延迟回放|', '|---|---:|---:|',
        '|建议通过检查的控制步|796|400|',
        '|建议真正改变动作的控制步|75|47|',
        '|只改变放行档位的控制步|47|31|',
        '|第一次响应时，合法库含有益动作的状态|33/240|30/240|',
        '|其中LLM推荐方向含有益动作的状态|23/240|21/240|',
        '|首次动作变化状态：有益/有害/微小|6/24/9|2/30/3|', '',
        '通俗地说：PPO常常已经选择了同样的方向，14B未增加新的决策；真正改动时，很多变化发生在放行容量上。方向并非处处错误，合法库中的好机会也有限，但通过门控不能保证改变就有收益。', '',
        '反事实每次只强制一个控制步，随后统一回到同一无建议PPO直至终止。人、当前位置、累计暴露和历史保持一致；概率行为用4套按同一人、同一绝对时刻配对的未来随机数。先按失能率差±0.001，再按FED差±0.001分类；该开发分类阈值不是人体安全标准。首次变化状态是刻意选中的机制子集，不代表全部回合的有害比例。', '',
        '合法库的最佳动作是离线诊断上限，未接入控制器。不能把多个状态的单步收益加起来当成全程收益。更换反事实未来种子前先去除旧随机表包装；相关测试验证不同种子确实对应不同未来。', '',
        '## 4. 冻结的小范围修正', '',
        '新版本保留原时间戳检查、建议有效期、门控、物理动作约束和PPO权重；PPO建议字段保持清零。有效14B方向对该方向所有合法容量档位等量加分（0.25×confidence），从而保持该方向内部的PPO档位排序。PPO本来已选该方向时，直接保留原动作。', '',
        '没有扩大建议强度、搜索超参数、拟合收益分类器或使用未来oracle。出口方向改变时，新方向上的容量仍由同一PPO的排序决定，不保证与旧方向容量相等，也不保证有益。', '',
        '无建议对照是同一套已训练权重清除建议后的消融版本，没有另行训练纯PPO。因此本轮不能替代此前要求的同预算独立训练对照。', '',
        '## 5. 新人口验证：先看是否比无建议PPO更好', '',
        '下面是主要12个开发场景的配对终点差，均为“仅出口方向提示−同权重无建议PPO”。FED、失能率和拥堵的正差表示更差；成功率正差表示更好。率差已转换为百分点。', '',
        '|时序|指标|平均差|描述性场景95%区间|发生差异的配对|', '|---|---|---:|---|---:|',
    ]
    for mode in MODES:
        for metric in METRICS:
            e = effect(mode, 'direction_hint_only', 'no_advice', metric)
            factor = 100 if metric.endswith('_rate') else 1
            lo, hi = e['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{labels[mode]}|{metrics[metric]}|{factor*e['mean_difference']:+.7f}|[{factor*lo:.7f}, {factor*hi:.7f}]|{e['nonzero_pairs']}/240|")
    lines += ['',
        '即时模式平均FED增加0.0003578，失能率增加约0.0174个百分点；后者只有2个配对产生差异，区间触及零。实测延迟模式平均FED增加0.0002020，失能率和成功率在这批配对中完全相同。没有证据支持相对于无建议PPO的正向增量。延迟模式拥堵略低，不能抵消暴露没有改善这一事实。', '',
        '区间按场景聚类：先平均同一场景内的人口、行为和固定权重的配对差，再重采样12个场景。它是复用开发场景、条件于5套已有权重的描述性区间，没有跨所有比较做多重校正；不能据单个接近零的区间端点作普遍显著性声明。', '',
        '## 6. 它确实比旧接入方式少了一些干扰', '',
        '下面为“仅出口方向提示−原建议适配器”，用于检查修正是否减少旧版本问题。', '',
        '|时序|指标|平均差|描述性场景95%区间|', '|---|---|---:|---|',
    ]
    for mode in MODES:
        for metric in ('mean_cumulative_fed', 'incapacitation_rate', 'cumulative_congestion'):
            e = effect(mode, 'direction_hint_only', 'original_advice_adapter', metric)
            factor = 100 if metric.endswith('_rate') else 1
            lo, hi = e['descriptive_scene_bootstrap_95_interval']
            lines.append(f"|{labels[mode]}|{metrics[metric]}|{factor*e['mean_difference']:+.7f}|[{factor*lo:.7f}, {factor*hi:.7f}]|")
    lines += ['',
        '延迟模式下，新版本相对原适配器减少FED约0.0027409、减少失能率约0.1424个百分点；这些描述性区间在零以下。即时模式均差也更低，但区间跨零。结合两种对照，只能说修正减少了旧接入方式造成的干扰，仍不能说14B改善了疏散。', '',
        '|主要12场景，新人口|版本|改变动作的回合|改变动作的控制步|同方向放行档位变化步|', '|---|---|---:|---:|---:|',
    ]
    for d in descriptions:
        lines.append(f"|{labels[d['mode']]}|{variants[d['variant']]}|{d['episodes_with_action_change']}/240|{d['action_changed_steps']}|{d['same_direction_release_changed_steps']}|")
    lines += ['',
        '新版本即时/延迟模式的动作变化由旧版本76/48步降为28/8步。这些计数各自在其实际轨迹上与同状态无建议动作比较；它们不是将不同轨迹所有时刻直接对齐。全部新版本同方向档位变化为0，说明预定接口约束得到执行；少改动作本身并不证明策略更好。', '',
        '## 7. 时间、安全检查与证据边界', '',
        '全部2,880回合中，进入当前封闭或超过烟气阈值走廊、实际放行超限、实际方向非法三类执行违规均为0。所有人最终疏散或失能，终点remaining均为0；这不代表全部人员成功疏散。', '',
        'episode_end_seconds包含失能后的仿真终止，不能当作全员成功疏散时间。all_evacuated_seconds只在全员成功时记录；不同策略成功回合不完全相同，条件平均会有选择偏差，本轮不以该条件平均判定提速。', '',
        'FED来自简化图环境的合成烟气增量，并未校准真实人体剂量。0次操作违规也不等于0伤害。14B只对初始观测给过建议，实测延迟模式把已保存响应按原耗时返回并重新检查；没有现场持续刷新或新的14B调用。confidence为模型自报，尚无收益概率校准。', '',
        '现有两种行为模型、两个新人口种子和5套既有权重覆盖了一部分变动，仍不足以支持真实火场部署、跨建筑推广或所有LLM方案无效的结论。', '',
        '## 8. 本轮决定与下一轮条件', '',
        '本轮不采用仅方向提示版本，也不增加服从强度或继续扩大PPO训练。保留它作为已完成的开发消融，保留原结果作为机制证据；同权重无建议PPO继续作为比较基线，不视为已通过现实验证。全局资格台账保持原样。', '',
        '若继续研究，先利用既有动作库和缓存检查方向提示剩余负面差异出现在哪些状态，区分当前方向与未来烟气风险、拥堵、等待损失和信息重复。再明确一种在线可见、规则基线尚未利用的信息，并为它设计独立冻结的开发验证；应包含使用相同信息的规则对照。只有先证明增量信息的作用，才值得引入滚动LLM建议或新的训练。', '',
        '这些是下一轮条件，不是本轮已经完成的实验。当前不使用离线最佳动作搭建自动安全盾，也不打开48个保留最终场景来筛选修正。', '',
        '## 9. 审计与复现材料', '',
        '此前新功能及相关回归测试共31项通过。本次另从CSV核对2,880条唯一记录、全部计划组合、48个新人口种子、完整三组配对；480对无建议回合在两种时序下终点逐字段完全一致；重算全部72项冻结配对均差和bootstrap区间，与summary.json一致。', '',
        '新LLM请求0，新PPO训练0，保留最终场景读取0。原缓存、权重、冻结脚本、已有报告、模型选择与研究资格台账哈希核对通过。旧默认模型选择文件未被改写；本轮实验来源明确为真实Qwen3 14B缓存。', '',
        '- ' + link('先前动作与合法候选审计报告', ROOT / 'reports/14B建议作用与候选收益审计报告_2026-10-07.md'),
        '- ' + link('修正验证的事前冻结计划', PREP / 'manifest.json'),
        '- ' + link('2,880个完整回合原始终点', source / 'episodes.csv'),
        '- ' + link('所有分组与配对区间', source / 'summary.json'),
        '- ' + link('本轮独立核对与决策', out / 'summary.json'),
        '- ' + link('固定方向提示实现', ROOT / 'src/fire_evac/direction_hint_adapter.py'),
        '- ' + link('验证运行脚本', ROOT / 'scripts/run_direction_hint_fresh_population_confirmation.py'), '',
    ]
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    audit = dict(report=str(REPORT), source=str(source), decision=decision,
        verified_full_episodes=len(rows), unique_episode_keys=len(index), scenes=len(scenes),
        fresh_population_seeds=len(plan['population_seeds']), baseline_timing_pairs_identical=baseline_timing_pairs,
        paired_effects_recomputed=len(summary['paired_effects']), full_design_and_pairing_verified=True,
        violations=violations, direction_hint_same_direction_release_changes=same_dir,
        primary_group_descriptions=descriptions, primary_paired_effects=primary,
        extra_group_descriptions_status='Post-result descriptive audit; no tuning, additional simulations or confirmatory claims.',
        previous_feature_and_regression_tests_passed=31,
        new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        protected_sha256=PROTECTED, no_adoption=True, no_qualification_claim=True,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), REPORT,
                 PREP / 'manifest.json', source / 'completion_manifest.json', ATTRIBUTION / 'completion_manifest.json']})
    save(out / 'summary.json', audit)
    verify()
    verify_files(source / 'completion_manifest.json')
    save(out / 'completion_manifest.json', dict(state='complete', files={
        p.relative_to(ROOT).as_posix(): sha(p) for p in [out / 'summary.json', REPORT, Path(__file__)]}))
    print(json.dumps(dict(report=str(REPORT), audit=str(out), full_episodes=len(rows),
                         paired_effects_verified=len(summary['paired_effects']), decision=decision), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    run(parser.parse_args().source)
