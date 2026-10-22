"""Batch rescore existing public traces; extend cache diagnosis without stepping."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np

from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output, csv_write
from diagnose_direction_hint_cached_harm import (
    verify_all, REPLAY, CONFIRM_PREP, MODES, CACHE, CONTINUATIONS, PRIMARY, state_key, episode_key,
)
from fire_evac.cached_direction_diagnosis import no_advice_input_from_cache


DIAG = ROOT / 'outputs/direction_hint_cached_harm_diagnosis_20261007T055026505852Z'
PREP = ROOT / 'outputs/direction_hint_full_cache_trace_preparation_20261007'
REPORT = ROOT / 'reports/14B剩余负效应归因总审计_2026-10-07.md'


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def verify_sources():
    verify_all()
    for p, h in read(DIAG / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p
    if PREP.exists():
        for p, h in read(PREP / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p


def prepare():
    assert not PREP.exists()
    verify_sources()
    PREP.mkdir()
    files = [Path(__file__), DIAG / 'completion_manifest.json', REPLAY / 'completion_manifest.json',
             ROOT / 'src/fire_evac/cached_direction_diagnosis.py', CONFIRM_PREP / 'manifest.json']
    save(PREP / 'manifest.json', dict(
        scope='Outcome-informed exploratory full PUBLIC trace rescore; no trajectory simulation.',
        reason='First-response states miss later direction/release changes; inspect all existing 42034 controls.',
        selectors='All 960 old episodes, all recorded steps; report first hint-induced change on old trajectories and whether its legal terminal cell already exists.',
        implementation='Reconstruct public 34-input records; batched frozen PPO distributions. Require all old raw, masked, feature-only and prior actions equal cached actions; require every prior 1504 hint rescore matches.',
        no_new_simulations=True, new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        elapsed_seconds_cap=240, no_threshold_or_weight_changes=True,
        caveat='Shadow decisions on original adapter paths. After hint first differs from original adapter, later states cannot be called visited by hint; fresh population trajectories remain unavailable.',
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    print('Frozen all-step public-cache audit; no new simulations or model calls.', flush=True)


def run():
    import torch
    from stable_baselines3 import PPO
    verify_sources()
    assert not REPORT.exists()
    plan = read(CONFIRM_PREP / 'manifest.json')
    scenes = {r['scenario_id']: r for r in plan['scenarios']}
    conf = {s: read(CACHE / 'responses' / (s + '.json'))['advice']['confidence'] for s in scenes}
    states = csv_read(DIAG / 'old_population_state_diagnostics.csv')
    prior = {r['state_id']: r for r in [*states, *csv_read(DIAG / 'initial_policy_rescore.csv')]
             if 'state_id' in r}
    previous_initial = {
        (r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']), r['source_mode']): r
        for r in csv_read(DIAG / 'initial_policy_rescore.csv')}
    cached_state_ids = set(r['state_id'] for r in states)
    memory, records, inputs = {}, [], []
    with (REPLAY / 'action_trace.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads, f):
            key = episode_key(r)
            inputs.append(no_advice_input_from_cache(r, scenes[r['scenario_id']], memory.get(key)))
            records.append(r)
            memory[key] = int(r['actual_room_capacity'])
    assert len(records) == 42034
    x = np.stack(inputs)
    assert x.shape == (42034, 34)
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['checkpoints'][str(s)], device='cpu') for s in plan['training_seeds']}
    actions = np.zeros(len(records), dtype=int)
    start = time.monotonic()
    parity = Counter()
    for seed, model in models.items():
        indices = [i for i, r in enumerate(records) if r['training_seed'] == seed]
        for offset in range(0, len(indices), 512):
            assert time.monotonic() - start < 240
            ids = indices[offset:offset + 512]
            obs = x[ids].copy()
            rs = [records[i] for i in ids]
            masks = np.array([[a in r['legal_actions'] for a in range(9)] for r in rs])
            accepted = np.array([r['gate_accepted'] for r in rs])
            targets = np.array([r['llm_target'] for r in rs])
            bonuses = np.array([.25 * conf[r['scenario_id']] for r in rs])
            feat = obs.copy()
            for j, r in enumerate(rs):
                if accepted[j]:
                    t = targets[j]
                    feat[j, 28 + t] = 1
                    feat[j, 30:32] = (conf[r['scenario_id']], 1)
                    feat[j, 32] = float(r['smoke'][3 + t] > r['smoke'][4-t] + .01)
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(obs)
                probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
                tensor, _ = model.policy.obs_to_tensor(feat)
                fprobs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
            scores = np.log(np.maximum(probs, 1e-30))
            fscores = np.log(np.maximum(fprobs, 1e-30))
            raw = np.argmax(scores, axis=1)
            base = np.argmax(np.where(masks, scores, -np.inf), axis=1)
            fraw = np.argmax(fscores, axis=1)
            features_only = np.argmax(np.where(masks, fscores, -np.inf), axis=1)
            old_scores = fscores.copy()
            for j, t in enumerate(targets):
                if accepted[j]:
                    for a in (t * 4, t * 4 + 1):
                        if masks[j, a]:
                            old_scores[j, a] += bonuses[j]
            old = np.argmax(np.where(masks, old_scores, -np.inf), axis=1)
            for name, found, field in [('raw', raw, 'raw_no_advice'), ('base', base, 'action_no_advice'),
                                     ('feature_raw', fraw, 'raw_with_advice'), ('features', features_only, 'action_features_only'),
                                     ('full', old, 'action_with_advice')]:
                expected = np.array([r[field] for r in rs])
                assert np.array_equal(found, expected), (seed, offset, name, np.where(found != expected)[0].tolist())
                parity[name] += len(rs)
            hint_scores = scores.copy()
            for j, t in enumerate(targets):
                if accepted[j]:
                    hint_scores[j, t * 4:t * 4 + 4] += bonuses[j]
            hint = np.argmax(np.where(masks, hint_scores, -np.inf), axis=1)
            for i, r, a in zip(ids, rs, hint):
                sid = state_key(r)
                if sid in prior:
                    assert int(a) == int(prior[sid]['hint_action']), sid
                    parity['previous_selected_hint'] += 1
                if r['step'] == 0:
                    assert int(a) == int(previous_initial[episode_key(r)]['hint_action'])
                    parity['previous_initial_hint'] += 1
                actions[i] = a
    assert parity['previous_selected_hint'] == 1024 and parity['previous_initial_hint'] == 960
    episodes = {}
    trace_rows = []
    for r, a in zip(records, actions):
        key = episode_key(r)
        if key not in episodes:
            episodes[key] = dict(scenario_id=r['scenario_id'], cohort=r['cohort'], behavior=r['behavior'],
                population_seed=r['population_seed'], training_seed=r['training_seed'], source_mode=r['arm'],
                steps=0, hint_change_steps_on_original_path=0, same_direction_release_changes=0,
                first_hint_change_step=None, first_hint_change_has_cached_terminal_cell=None,
                first_divergence_from_original_adapter_step=None, whole_episode_matches_original_adapter=True)
        e = episodes[key]
        b, old = r['action_no_advice'], r['action_with_advice']
        a = int(a)
        changed = a != b
        same = changed and 8 not in (a, b) and a // 4 == b // 4
        assert not same
        was_on_hint_path = e['whole_episode_matches_original_adapter']
        if changed and e['first_hint_change_step'] is None:
            e['first_hint_change_step'] = int(r['step'])
            e['first_hint_change_has_cached_terminal_cell'] = state_key(r) in cached_state_ids
        if a != old and e['first_divergence_from_original_adapter_step'] is None:
            e['first_divergence_from_original_adapter_step'] = int(r['step'])
            e['whole_episode_matches_original_adapter'] = False
        e['steps'] += 1
        e['hint_change_steps_on_original_path'] += changed
        e['same_direction_release_changes'] += same
        trace_rows.append(dict(state_id=state_key(r), scenario_id=r['scenario_id'], cohort=r['cohort'],
            behavior=r['behavior'], population_seed=r['population_seed'], training_seed=r['training_seed'],
            source_mode=r['arm'], step=r['step'], observation_time_s=r['observation_time_s'],
            gate_accepted=r['gate_accepted'], base_action=b, hint_action=a, original_adapter_action=old,
            hint_changes_base=changed, state_on_hint_prefix=was_on_hint_path,
            has_cached_terminal_action_cells=state_key(r) in cached_state_ids))
    groups = []
    for cohort in (PRIMARY, 'exploratory_mechanism_development'):
        for mode in MODES:
            rs = [r for r in episodes.values() if r['cohort'] == cohort and r['source_mode'] == mode]
            affected = [r for r in rs if r['first_hint_change_step'] is not None]
            groups.append(dict(cohort=cohort, mode=mode, old_episodes=len(rs),
                affected_old_paths=len(affected), shadow_hint_change_steps=sum(r['hint_change_steps_on_original_path'] for r in rs),
                first_hint_change_steps=dict(Counter(str(r['first_hint_change_step']) for r in affected)),
                first_changes_with_cached_terminal_cells=sum(r['first_hint_change_has_cached_terminal_cell'] for r in affected),
                whole_old_paths_identical_to_hint=sum(r['whole_episode_matches_original_adapter'] for r in rs)))
    out = unique_output('direction_hint_full_cached_trace_audit')
    csv_write(out / 'all_step_shadow_actions.csv', trace_rows)
    csv_write(out / 'old_path_episodes.csv', list(episodes.values()))
    # Final factor evidence uses ALL selected changed states, including later
    # first-change states deliberately chosen for the original adapter.
    aull10 = [r for r in states if r['scenario_id'] == 'AULL0010' and r['hint_changes_action'] == 'True']
    assert len(aull10) == 8 and all(int(r['source_step']) == 3 for r in aull10)
    unique_cases = {}
    for r in aull10:
        key = (r['behavior'], r['population_seed'], r['training_seed'], r['source_step'])
        if key in unique_cases:
            for m in ('mean_cumulative_fed', 'incapacitation_rate'):
                for p in ('direction_at_base_release', 'release_at_base_direction', 'direction_release_interaction', 'difference'):
                    assert abs(float(r[m + '_' + p]) - float(unique_cases[key][m + '_' + p])) < 1e-12
        unique_cases[key] = r
    assert len(unique_cases) == 4
    assert all(float(r['incapacitation_rate_direction_at_base_release']) < 0 for r in unique_cases.values())
    assert all(float(r['mean_cumulative_fed_release_at_base_direction']) > 0 for r in unique_cases.values())
    assert all(float(r['mean_cumulative_fed_direction_release_interaction']) > 0 for r in unique_cases.values())
    changed = [r for r in states if r['hint_changes_action'] == 'True']
    assert len(changed) == 24
    harm = [r for r in changed if r['outcome_class'] == 'harmful']
    assert len(harm) == 9 and all(r['gate_accepted'] == 'True' for r in harm)
    assert all(abs(float(r['target_smoke']) - float(r['other_smoke'])) <= .01 for r in harm)
    assert all(float(r['gate_age_seconds']) in (0, 15) for r in harm)
    summary = dict(preparation_sha256=sha(PREP / 'manifest.json'), trace_controls=42034,
        full_action_parity=dict(parity), all_trace_groups=groups,
        selected_changed_states=24, selected_harmful_state_mode_records=9,
        aull10_unique_case_factorization=list(unique_cases.values()),
        new_simulations=0, new_llm_requests=0, new_training=0, reserved_final_scenarios_opened=0,
        no_online_controller_change=True, no_qualification_claim=True,
        scope='All-step shadow decisions along original-adapter old-population paths; do not substitute for unrecorded fresh-population paths.',
        rescore_elapsed_seconds=time.monotonic() - start)
    save(out / 'summary.json', summary)
    write_final_report(out, summary, states)
    verify_sources()
    save(out / 'completion_manifest.json', dict(state='complete', files={
        p.relative_to(ROOT).as_posix(): sha(p) for p in [*out.iterdir(), REPORT] if p.is_file()}))
    print(json.dumps(dict(output=str(out), report=str(REPORT), parity=dict(parity), groups=groups), ensure_ascii=False), flush=True)


def write_final_report(out, summary, states):
    mode_labels = {MODES[0]: '即时缓存', MODES[1]: '实测延迟'}
    lines = ['# 14B方向提示剩余负效应：原因归因总审计', '',
        '2026年10月7日。结论：残余问题包含方向选择与容量选择的耦合，以及烟气预测、吞吐能力和执行时机之间的冲突。当前证据不支持简单禁止某类出口，也不支持提高LLM服从强度。本轮全部基于旧缓存，训练和最终评估继续暂停。', '',
        '## 1. 本轮完成到哪一步', '',
        '读取960份新人口“方向提示−无建议PPO”配对终点，定位全部非零差异；还原并重新计算旧人口全部42,034个控制步；核对无建议原始动作、约束后动作、建议特征动作与原完整建议动作，五类结果全部与缓存一致。新的方向提示动作在此前1,024个关键状态和960个初始状态中也全部匹配（初始与关键状态有重合）。', '',
        '使用已保存的9,342条终止反事实，分析方向、放行档位及其交互，没有新增轨迹、LLM调用或训练。缓存重建与效应分解5项新测试及7项相关回归测试通过。', '',
        '分析由前轮结果触发，属于开发库事后诊断；48个保留最终场景未读取，旧权重、缓存、门控阈值和资格台账未改。', '',
        '## 2. 先定位全程负面差异', '',
        '主要12个场景即时模式240个配对中，FED更高10个、更低6个、相同224个；失能更多2个、更少0个。两例新增失能都在AULL0024、Rule行为、权重602的新人口1477100150/1477100151，每回合失能率差为+2.0833个百分点。这个病例差不能冒充全部240配对的平均差。', '',
        '主要子集实测延迟240个配对中，FED更高4个、更低4个、相同232个，失能率全部相同。AULL0013两个Rule病例的暴露上升主导该模式净差；不能只用“有几个变好/变坏”判断平均效果。', '',
        '探索场景AULL0010中，权重605的两批概率行为人口在即时、延迟模式均出现新增失能；时序复用同一人口，仍是两个配对人口，不是四个独立场景。全部病例清单见前份缓存诊断。', '',
        '## 3. 核心发现：转向与减小放行不能混成一个结论', '',
        'AULL0010在旧缓存第3步（第15秒）发生动作1→4：北侧50%放行变为南侧25%放行。两側当前烟气相同；南侧容量9人每步、北侧3人每步，但南侧也是公开预报的未来烟气源。仅看这些特征不能确定转向一定有害。', '',
        '缓存已有全部四个合法动作终点：北50%=1、南50%=5、北25%=0、南25%=4。每个候选只强制一步，随后统一执行同一无建议PPO。方向项=南50%−北50%；容量项=北25%−北50%；交互项=南25%−南50%−北25%+北50%。三项相加等于实际组合差。', '',
        '|旧人口病例（同权重605，第15秒）|纯转向FED差|纯减量FED差|交互FED差|转向且减量FED差|纯转向失能率差（百分点）|组合失能率差（百分点）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['aull10_unique_case_factorization']:
        names = ['mean_cumulative_fed_direction_at_base_release', 'mean_cumulative_fed_release_at_base_direction',
                 'mean_cumulative_fed_direction_release_interaction', 'mean_cumulative_fed_difference']
        values = [float(r[n]) for n in names]
        d, f = float(r['incapacitation_rate_direction_at_base_release']) * 100, float(r['incapacitation_rate_difference']) * 100
        lines.append(f"|{r['behavior']},人口{r['population_seed']}|{values[0]:+.6f}|{values[1]:+.6f}|{values[2]:+.6f}|{values[3]:+.6f}|{d:+.4f}|{f:+.4f}|")
    lines += ['',
        '这四个独立状态病例各自的两种时序结果完全重复，所以表只列一份。概率行为数字是4个配对未来的平均。保留50%放行时，四例转向的平均失能率均降低；减小到25%的容量项和方向×容量交互项均抬高FED，抵消了转向收益，三例组合被既有阈值分为有害。', '',
        '因此不能把AULL0010诊断成“LLM朝预报火源走，方向必定错误”。更具体的漏洞是：仅方向提示版本只保证“方向不变时容量不变”；一旦方向改变，目标方向内部PPO排序仍可能选更小档位。冻结代码在这里确实执行了这个设计，而设计本身不足以消除容量干扰。', '',
        '这些是终点分解的因果干预证据，条件于旧人口、固定未来和后续无建议PPO。容量减小是否主要延长房间等待、还是改变后续路径，目前没有分区域轨迹证据，仍是机制假设。', '',
        '## 4. 另外两类矛盾也存在', '',
        'AULL0024：目标南侧不是预报火源，却只有3人每步吞吐，北侧10人每步。初始无建议北50%转成南50%，档位相同；旧Rule两个人口的纯转向FED差分别为+0.000353和+0.003842，旧一步干预失能率没有变化。新人口完整控制则有两例新增失能。它支持早期吞吐与后期避烟之间的冲突，但旧一步结果不能直接解释新人口全部失能。', '',
        '同状态已有规则也选择南50%，所以该病例包含现有规则策略同样面临的容量时机问题。房间总放行10人不等于10人全部进入3人出口；人员行为和容量限制仍起作用，不能把它当操作超限。', '',
        'AULL0013：目标北侧容量8、南侧6，同时北侧是预报源。旧Rule人口初始南50%→北50%的FED差为+0.023580和+0.005429，全部来自纯方向项；概率行为对应差为−0.000475和−0.000339。新人口Rule中即时模式一个人口暴露增加、另一个减少，不能仅靠场景标签预测收益符号。', '',
        'AULL0019：新人口Rule/权重604有一例FED增加+0.005390，另一例+0.000267，失能率不变；它在旧首次响应中的同方向干预有益或微小。新旧人口、一步/重复建议同时变化，本轮无法区分是哪一种因素导致反转。AULL0020临时封闭恢复附近只有正负均小于0.001的FED差、失能率不变，单独保留为微小未解释差异，不夸大成安全损失结论。', '',
        '## 5. 为什么门控会接受这些动作', '',
        '1. 当前门控检查可执行性和明显的即时烟气矛盾；它不计算覆盖无建议动作的预期收益。',
        '2. 旧关键状态里被分类为有害的9条状态×时序记录均通过门控，烟气差都在0.01以内，建议年龄全部为0秒或15秒。它们不是“已超过TTL却漏检”；15秒恰好仍在冻结有效期内。9条记录包含重复时序和同场景人口，不能视为9个独立场景。',
        '3. 初始方向变化即便当步路口无人，也会触发随后两个控制步的指示牌保持；不能凭当步分支流量相同判断方向变化无效。但本轮没做保持时间消融，不能给保持时间单独分配损失。',
        '4. 模型自报confidence只是偏置大小。AULL0024无建议最佳方向的对数分数优势约0.06194，建议偏置0.1875，足以改变选择；AULL0010相关状态优势约0.218—0.221，偏置0.23，同样能翻转。分数差不是风险差，confidence也不是校准收益概率。', '',
        '## 6. 为什么必须扫描首次响应之后的缓存', '',
        '主要延迟模式的240个首次响应旧状态中，仅方向提示没有改变动作；这并不等于整个回合没有作用。新人口延迟模式已有8个改变动作回合。初始/首次响应的统计会漏掉后来仍在有效期内的变化。', '',
        '为检查这个盲点，本轮对全部42,034个原适配器控制步做无仿真的影子决策。下表不是新版本实际轨迹：若新动作先前已不同于原适配器，后续原路径只提供状态探针，不能冒充新版本实际访问的状态。', '',
        '|旧路径子集|时序|路径数|存在影子动作变化|影子变化步|首次变化步分布|首次变化已有终点反事实|', '|---|---|---:|---:|---:|---|---:|']
    for g in summary['all_trace_groups']:
        label = '主要12场景' if g['cohort'] == PRIMARY else '探索12场景'
        distribution = ', '.join(f"步{k}:{v}" for k, v in sorted(g['first_hint_change_steps'].items(), key=lambda p:int(p[0])))
        lines.append(f"|{label}|{mode_labels[g['mode']]}|{g['old_episodes']}|{g['affected_old_paths']}|{g['shadow_hint_change_steps']}|{distribution}|{g['first_changes_with_cached_terminal_cells']}|")
    lines += ['',
        'all_step_shadow_actions.csv明确记录每个状态是否仍处于新版本与原版本相同的前缀，以及是否已有该状态的完整合法动作终点。缺少终点反事实的状态没有被偷偷补跑或赋收益标签。', '',
        '## 7. 本轮结论和下一轮最小工作', '',
        '已定位：哪些具体人口/权重产生完整终点差异；哪些旧状态被方向提示改变；哪些问题来自纯方向、容量或交互；门控为什么接受；哪些现有规则也存在同类不足。没有证据支持把LLM出口方向一概认定为错误。', '',
        '尚未完成：新人口完整轨迹的精确逐步因果归因。保存的文件没有这些轨迹，旧人口和影子决策不能替代。具体区域等待、重复建议的累积作用、指示牌保持的独立作用仍需有限机制实验。', '',
        '下一轮先对AULL0010、AULL0013、AULL0024（另保留AULL0019反转病例）记录既有人口完整动作链，做固定方向×固定放行档位的配对消融，再独立检查保持时间；覆盖收益正负两类，不只检查有害病例。修正必须使用另行冻结的开发人口验证，同时比较同信息规则与无建议PPO。', '',
        '本轮不采用新控制器，不调强度、TTL或confidence门槛，不训练收益盾，不扩大PPO训练，不开启48个保留最终场景。这些有限机制实验尚未执行。', '',
        '## 8. 材料与执行审计', '',
        f'- [本轮分析冻结范围]({(PREP / "manifest.json").as_posix()})',
        f'- [全部旧路径影子动作]({(out / "all_step_shadow_actions.csv").as_posix()})',
        f'- [每个旧回合的前缀与覆盖情况]({(out / "old_path_episodes.csv").as_posix()})',
        f'- [四组合原始分解与核对汇总]({(out / "summary.json").as_posix()})',
        f'- [终点定位及首次响应缓存诊断]({(ROOT / "reports/14B方向提示剩余负效应缓存诊断_2026-10-07.md").as_posix()})',
        f'- [全部新人口配对终点]({(DIAG / "fresh_population_paired_endpoints.csv").as_posix()})',
        f'- [全部旧关键状态分解]({(DIAG / "old_population_state_diagnostics.csv").as_posix()})', '',
        '原缓存、权重、冻结代码、已有报告和台账哈希验证通过；全控制步五类动作逐条一致，前轮新方向提示核对通过；本轮新增轨迹0、LLM调用0、训练0、最终场景读取0。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'verify'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify_sources}[args.stage]()
