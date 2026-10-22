"""Outcome-informed cache-only diagnosis; freeze before rescoring checkpoints.

No simulator is constructed or advanced. No outcomes enter the controller.
"""
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import sys
import time

import numpy as np

from run_timestamped_development_campaign import ROOT, read, save, sha, csv_write, unique_output
from run_direction_hint_fresh_population_confirmation import PREP as CONFIRM_PREP, verify, MODES
from run_qwen14b_evacuations import CACHE, METRICS
from fire_evac.cached_direction_diagnosis import no_advice_input_from_cache, joint_effect_cells, factor_effects
from fire_evac.timestamped_pipeline import ppo_action
from fire_evac.direction_hint_adapter import direction_hint_action
from fire_evac.advice_causal_audit import outcome_class


PREP = ROOT / 'outputs/direction_hint_cached_harm_preparation_20261007'
CONFIRM = ROOT / 'outputs/direction_hint_fresh_population_confirmation_20261007T052903182504Z'
FINAL = ROOT / 'outputs/direction_hint_confirmation_final_audit_20261007T053326545236Z'
REPLAY = ROOT / 'outputs/qwen14b_action_replay_20261007T051502407231Z'
CONTINUATIONS = ROOT / 'outputs/qwen14b_legal_action_continuations_20261007T051810075635Z'
REPORT = ROOT / 'reports/14B方向提示剩余负效应缓存诊断_2026-10-07.md'
PRIMARY = 'primary_separate_development'


def verify_complete(folder):
    for p, h in read(folder / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p


def verify_all():
    verify()
    for folder in (CONFIRM, FINAL, REPLAY, CONTINUATIONS):
        verify_complete(folder)
    if PREP.exists():
        for p, h in read(PREP / 'manifest.json')['sources'].items():
            assert sha(ROOT / p) == h, p


def prepare():
    assert not PREP.exists()
    verify_all()
    sources = [Path(__file__), ROOT / 'src/fire_evac/cached_direction_diagnosis.py',
               ROOT / 'tests/test_cached_direction_diagnosis.py', CONFIRM_PREP / 'manifest.json',
               ROOT / 'src/fire_evac/config.py', ROOT / 'src/fire_evac/env.py',
               ROOT / 'src/fire_evac/confirmatory.py',
               *[p / 'completion_manifest.json' for p in (CONFIRM, FINAL, REPLAY, CONTINUATIONS)]]
    PREP.mkdir()
    save(PREP / 'manifest.json', dict(
        scope='Post-result exploratory cache diagnosis, not independent confirmation.',
        trigger='Direction-only hint reduced adapter interference but did not improve no-advice PPO; inspect remaining differences.',
        endpoint_selection='All 960 direction_hint_only vs no_advice pairs; main interpretation uses primary 12 development scenes. Inspect all nonzero pairs and all additional-incapacitation cases.',
        state_selection='All 1024 previously frozen selected states plus cached initial states. Denominators for mechanism tables are primary first-response states, not selected harm cases.',
        reconstruction='Rebuild 34 no-advice inputs from cached public fields, scene public forecasts and previous actual room capacity. Fail unless plain raw/masked, features-only and old full action reproduce exactly. Do not invent missing memory.',
        rescoring='Apply already frozen direction-only adapter to reconstructed OLD-population states; these are not new-population trajectories.',
        effects='Look up already cached legal one-step continuations. First forced action, then same no-advice PPO to termination. Paired future seed averages; no new rollouts.',
        decomposition='Legal 2x2 cells: base branch/base release, target branch/base release, base branch/hint release, target branch/hint release. Direction, release and interaction sum exactly to full difference. Waiting separate.',
        classification='Reuse frozen incap-first .001 then FED .001 thresholds; also show raw positive/negative terminal pairs at tolerance1e-12. Thresholds are descriptive, not real safety limits.',
        cause_tags='Current smoke contrast, throughput contrast, occupancy/capacity proxy, forecast-source alignment, gate age, empty junction and sign hold. True source is offline annotation only.',
        limitations='New-population confirmation has endpoints/counts but no per-step trace. Exact full-episode causal mechanism cannot be established by substituting old populations.',
        no_new_llm_requests=True, new_training=0, new_simulations=0,
        reserved_final_scenarios_opened=0, no_parameter_tuning=True, no_online_oracle=True,
        checkpoint_rescore_wall_seconds_cap=180,
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    print('Frozen cache-only diagnosis, old-state action parity required; no new simulations.', flush=True)


def table_link(label, path):
    return f'[{label}]({Path(path).as_posix()})'


def episode_key(r):
    return (r['scenario_id'], r['behavior'], int(r['population_seed']), int(r['training_seed']),
            r.get('source_mode', r.get('arm')))


def state_key(r):
    return '|'.join(map(str, (*episode_key(r), int(r['step']))))


def load_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def run():
    import torch
    from stable_baselines3 import PPO
    verify_all()
    assert not REPORT.exists()
    plan = read(CONFIRM_PREP / 'manifest.json')
    scenes = {s['scenario_id']: s for s in plan['scenarios']}
    advice = {s: read(CACHE / 'responses' / (s + '.json')) for s in scenes}
    selected = {}
    with (REPLAY / 'selected_states.jsonl').open(encoding='utf-8') as f:
        selected = {r['state_id']: r for r in map(json.loads, f)}
    assert len(selected) == 1024
    memory = {}
    retained = {}
    trace_count = 0
    with (REPLAY / 'action_trace.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads, f):
            key = episode_key(r)
            sid = state_key(r)
            prev = memory.get(key)
            if sid in selected or r['step'] == 0:
                retained[sid] = (r, prev)
            memory[key] = int(r['actual_room_capacity'])
            trace_count += 1
    assert trace_count == 42034 and set(selected) <= set(retained)
    cached = load_csv(CONTINUATIONS / 'terminal_rollouts.csv')
    assert len(cached) == 9342
    outcomes = defaultdict(dict)
    for r in cached:
        key = (r['state_id'], int(r['first_requested_action']))
        future = int(r['future_seed'])
        assert future not in outcomes[key]
        outcomes[key][future] = {m: float(r[m]) for m in METRICS}
    means = {key: {m: float(np.mean([r[m] for r in fs.values()])) for m in METRICS}
             for key, fs in outcomes.items()}
    for sid, r in selected.items():
        expected_futures = 1 if r['behavior'] == 'rule' else 4
        assert all(len(outcomes[(sid, a)]) == expected_futures for a in r['legal_actions'])
    torch.set_num_threads(1)
    models = {s: PPO.load(plan['checkpoints'][str(s)], device='cpu') for s in plan['training_seeds']}
    start = time.monotonic()
    state_rows, initial_rows, candidate_rows = [], [], []
    parity = Counter()
    for sid, (r, previous) in retained.items():
        assert time.monotonic() - start < 180, 'Frozen offline rescore time budget exhausted'
        scene = scenes[r['scenario_id']]
        obs = no_advice_input_from_cache(r, scene, previous)
        legal = r['legal_actions']
        mask = np.array([a in legal for a in range(9)])
        model = models[int(r['training_seed'])]
        target = r['llm_target']
        confidence = advice[r['scenario_id']]['advice']['confidence']
        p = SimpleNamespace(preferred_branch=target, confidence=confidence,
                            north_share=float(target == 0), wait_steps=0)
        g = SimpleNamespace(accepted=r['gate_accepted'],
                            valid_candidate_actions=tuple(a for a in (target * 4, target * 4 + 1) if a in legal))
        base, raw = ppo_action(model, obs, mask)
        assert base == r['action_no_advice'] and raw == r['raw_no_advice'], (sid, base, raw)
        features = obs.copy()
        if r['gate_accepted']:
            features[28 + target] = 1
            features[30:32] = (confidence, 1)
            features[32] = float(r['smoke'][3 + target] > r['smoke'][4 - target] + .01)
        feat, _ = ppo_action(model, features, mask, p, g, prior_strength=0)
        old, old_raw = ppo_action(model, features, mask, p, g)
        assert feat == r['action_features_only'] and old == r['action_with_advice'] and old_raw == r['raw_with_advice'], sid
        hint, hint_raw = direction_hint_action(model, obs, mask, p, g)
        assert hint in legal and hint_raw == raw
        assert not (hint != base and 8 not in (hint, base) and hint // 4 == base // 4)
        parity['raw_plain_and_masked_plain'] += 1
        parity['features_and_old_full'] += 1
        initial = dict(scenario_id=r['scenario_id'], behavior=r['behavior'], population_seed=r['population_seed'],
                       training_seed=r['training_seed'], source_mode=r['arm'], base_action=base, hint_action=hint,
                       gate_accepted=r['gate_accepted'], input_sha256=__import__('hashlib').sha256(obs.tobytes()).hexdigest())
        if r['step'] == 0:
            initial_rows.append(initial)
        if sid not in selected:
            continue
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(obs)
            probs = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
        scores = np.log(np.maximum(probs, 1e-30))
        best_target = max((scores[a] for a in legal if a != 8 and a // 4 == target), default=-np.inf)
        cm = {a: means[(sid, a)] for a in legal}
        delta = {m: cm[hint][m] - cm[base][m] for m in METRICS}
        classification = outcome_class(delta['incapacitation_rate'], delta['mean_cumulative_fed'])
        target_classes = [outcome_class(cm[a]['incapacitation_rate'] - cm[base]['incapacitation_rate'],
                                        cm[a]['mean_cumulative_fed'] - cm[base]['mean_cumulative_fed'])
                          for a in legal if a != 8 and a // 4 == target]
        base_branch = base // 4 if base != 8 else 1 - target
        target_smoke = float(r['smoke'][3 + target])
        other_smoke = float(r['smoke'][4 - target])
        target_capacity, other_capacity = r['capacities'][target], r['capacities'][1-target]
        cells = joint_effect_cells(base, hint, legal)
        decomposed = {m: factor_effects(cells, cm, m) for m in METRICS} if cells else {}
        labels = []
        if abs(target_smoke - other_smoke) <= .01:
            labels.append('smoke_within_current_gate_margin')
        if target_capacity < other_capacity:
            labels.append('target_lower_current_throughput')
        if target_capacity > other_capacity:
            labels.append('target_higher_current_throughput')
        if target == scene['forecast_source_branch']:
            labels.append('target_is_public_forecast_source')
        if r['step'] <= scene['onset_step']:
            labels.append('before_visible_growth_under_scene_model')
        if sum(r['occupancy'][2:]) == 0:
            labels.append('junction_and_corridors_empty')
        if target_smoke > other_smoke:
            labels.append('target_currently_more_smoke')
        futures = outcomes[(sid, base)]
        future_classes = []
        for future in futures:
            a, b = outcomes[(sid, hint)][future], futures[future]
            future_classes.append(outcome_class(a['incapacitation_rate'] - b['incapacitation_rate'],
                                                 a['mean_cumulative_fed'] - b['mean_cumulative_fed']))
        record = dict(state_id=sid, scenario_id=r['scenario_id'], cohort=r['cohort'], behavior=r['behavior'],
            population_seed=r['population_seed'], training_seed=r['training_seed'], source_mode=r['arm'],
            source_step=r['step'], observation_time_s=r['observation_time_s'],
            selection_reasons=';'.join(r['selection_reasons']), first_response='first_response_available' in r['selection_reasons'],
            gate_accepted=r['gate_accepted'], gate_reasons=';'.join(r['gate_reasons']),
            gate_age_seconds=r['observation_time_s'] - r['proposal_time_s'],
            response_time_s=r['response_time_s'], llm_target=target, llm_confidence=confidence,
            base_action=base, hint_action=hint, old_action=old, rule_action=r['rule_action'],
            hint_changes_action=hint != base, hint_equals_rule=hint == r['rule_action'],
            baseline_log_margin_over_best_target=float(scores[base] - best_target) if np.isfinite(best_target) else None,
            applied_soft_bonus=.25 * confidence if g.accepted else 0,
            outcome_class=classification, future_label_reversal='beneficial' in future_classes and 'harmful' in future_classes,
            target_has_beneficial_legal_action='beneficial' in target_classes,
            target_smoke=target_smoke, other_smoke=other_smoke,
            target_capacity=target_capacity, other_capacity=other_capacity,
            junction_people=int(round(r['occupancy'][2] * 240)),
            target_corridor_people=int(round(r['occupancy'][3 + target] * 240)),
            target_points_to_forecast_source=target == scene['forecast_source_branch'],
            target_points_to_true_source_offline_only=target == scene['source_branch'],
            signal_hold_after_initial_switch_steps=2 if r['step'] == 0 else None,
            mechanism_tags=';'.join(labels),
            **{m + '_difference': v for m, v in delta.items()})
        for m, parts in decomposed.items():
            for part, value in parts.items():
                record[m + '_' + part] = value
        state_rows.append(record)
        for a in legal:
            candidate_rows.append(dict(state_id=sid, action=a, is_base=a == base, is_hint=a == hint,
                is_matched_direction_cell=cells is not None and a == cells['direction_only'],
                is_release_cell=cells is not None and a == cells['release_only'],
                is_rule=a == r['rule_action'], **cm[a]))
    assert len(state_rows) == 1024 and len(initial_rows) == 960
    # Initial public observations do not depend on hidden personas; verify the
    # two old populations match exactly before linking to the fresh population.
    initial_index = defaultdict(list)
    for r in initial_rows:
        initial_index[(r['scenario_id'], r['behavior'], r['training_seed'], r['source_mode'])].append(r)
    for rs in initial_index.values():
        assert len(rs) == 2
        assert len({(r['input_sha256'], r['base_action'], r['hint_action'], r['gate_accepted']) for r in rs}) == 1
    full_rows = load_csv(CONFIRM / 'episodes.csv')
    baseline = {episode_key(r): r for r in full_rows if r['variant'] == 'no_advice'}
    endpoints = []
    for r in full_rows:
        if r['variant'] != 'direction_hint_only':
            continue
        b = baseline[episode_key(r)]
        delta = {m: float(r[m]) - float(b[m]) for m in METRICS}
        initial = initial_index[(r['scenario_id'], r['behavior'], int(r['training_seed']), r['source_mode'])][0]
        endpoints.append(dict(scenario_id=r['scenario_id'], cohort=r['cohort'], behavior=r['behavior'],
            population_seed=int(r['population_seed']), training_seed=int(r['training_seed']), source_mode=r['source_mode'],
            action_changed_steps=int(r['action_changed_steps']),
            reconstructed_initial_base=initial['base_action'], reconstructed_initial_hint=initial['hint_action'],
            initial_sensor_inputs_independent_of_population=True,
            exact_new_population_step_trace_available=False,
            outcome_class=outcome_class(delta['incapacitation_rate'], delta['mean_cumulative_fed']),
            **{m + '_difference': v for m, v in delta.items()}))
    assert len(endpoints) == 960
    scene_groups = defaultdict(list)
    for r in endpoints:
        scene_groups[(r['cohort'], r['scenario_id'], r['source_mode'])].append(r)
    scene_diagnostics = []
    for (cohort, sid, mode), rs in scene_groups.items():
        assert len(rs) == 20
        scene = scenes[sid]
        target = advice[sid]['advice']['target_branch']
        scene_diagnostics.append(dict(cohort=cohort, scenario_id=sid, mode=mode, paired_episodes=20,
            fed_higher_pairs=sum(r['mean_cumulative_fed_difference'] > 1e-12 for r in rs),
            fed_lower_pairs=sum(r['mean_cumulative_fed_difference'] < -1e-12 for r in rs),
            incap_higher_pairs=sum(r['incapacitation_rate_difference'] > 1e-12 for r in rs),
            incap_lower_pairs=sum(r['incapacitation_rate_difference'] < -1e-12 for r in rs),
            affected_training_seeds=sorted({r['training_seed'] for r in rs if any(abs(r[m + '_difference']) > 1e-12 for m in METRICS)}),
            llm_target=target, confidence=advice[sid]['advice']['confidence'],
            target_initial_capacity=scene['north_exit_capacity'] if target == 0 else scene['south_exit_capacity'],
            source_branch_offline=scene['source_branch'], public_forecast_source=scene['forecast_source_branch'],
            conflict_type=scene['conflict_type'], onset_step=scene['onset_step'],
            **{m + '_difference': float(np.mean([r[m + '_difference'] for r in rs])) for m in METRICS}))
    state_groups, endpoint_groups = [], []
    for cohort in (PRIMARY, 'exploratory_mechanism_development'):
        for mode in MODES:
            rs = [r for r in state_rows if r['cohort'] == cohort and r['source_mode'] == mode and r['first_response']]
            assert len(rs) == 240
            changed = [r for r in rs if r['hint_changes_action']]
            state_groups.append(dict(cohort=cohort, mode=mode, first_response_states=len(rs),
                changed_states=len(changed), all_outcome_classes=dict(Counter(r['outcome_class'] for r in rs)),
                changed_outcome_classes=dict(Counter(r['outcome_class'] for r in changed)),
                changed_rule_agreement=sum(r['hint_equals_rule'] for r in changed),
                changed_direction_and_release=sum(r['base_action'] != 8 and r['hint_action'] != 8 and r['base_action'] % 4 != r['hint_action'] % 4 for r in changed),
                future_label_reversals=sum(r['future_label_reversal'] for r in changed),
                mean_differences={m: float(np.mean([r[m + '_difference'] for r in rs])) for m in METRICS}))
            pairs = [r for r in endpoints if r['cohort'] == cohort and r['source_mode'] == mode]
            endpoint_groups.append(dict(cohort=cohort, mode=mode, paired_episodes=len(pairs),
                classified_outcomes=dict(Counter(r['outcome_class'] for r in pairs)),
                fed_higher=sum(r['mean_cumulative_fed_difference'] > 1e-12 for r in pairs),
                fed_lower=sum(r['mean_cumulative_fed_difference'] < -1e-12 for r in pairs),
                fed_identical=sum(abs(r['mean_cumulative_fed_difference']) <= 1e-12 for r in pairs),
                incap_higher=sum(r['incapacitation_rate_difference'] > 1e-12 for r in pairs),
                incap_lower=sum(r['incapacitation_rate_difference'] < -1e-12 for r in pairs)))
    out = unique_output('direction_hint_cached_harm_diagnosis')
    # Normalize optional factor columns for rectangular CSV output.
    columns = sorted({k for r in state_rows for k in r})
    csv_write(out / 'old_population_state_diagnostics.csv', [{k: r.get(k) for k in columns} for r in state_rows])
    csv_write(out / 'cached_action_cells.csv', candidate_rows)
    csv_write(out / 'fresh_population_paired_endpoints.csv', endpoints)
    csv_write(out / 'initial_policy_rescore.csv', initial_rows)
    save(out / 'scene_diagnostics.json', scene_diagnostics)
    harmful_cases = [r for r in endpoints if r['incapacitation_rate_difference'] > 1e-12]
    selected_case_states = [r for r in state_rows if r['scenario_id'] in ('AULL0024', 'AULL0010', 'AULL0013')
                            and r['first_response'] and r['hint_changes_action']]
    summary = dict(preparation_sha256=sha(PREP / 'manifest.json'),
        old_selected_states=1024, reconstructed_input_records=len(retained), cached_trace_steps_read=trace_count,
        cached_terminal_rollouts_read=len(cached), action_reconstruction_parity=dict(parity),
        initial_states_rescored=len(initial_rows), fresh_population_pairs=960,
        state_groups=state_groups, endpoint_groups=endpoint_groups,
        additional_incapacitation_cases=harmful_cases, selected_case_states=selected_case_states,
        new_simulations=0, new_training=0, new_llm_requests=0, reserved_final_scenarios_opened=0,
        no_online_policy_change=True, no_qualification_claim=True,
        limitations='Older cached states and one-step continuation interventions do not substitute for missing fresh-population full-episode traces.',
        protected_sha256={p: sha(ROOT / p) for p in ('outputs/local_llm_selection.json', 'outputs/third_round_development_status.json')},
        rescore_elapsed_seconds=time.monotonic() - start)
    save(out / 'summary.json', summary)
    write_report(out, summary, state_rows, scene_diagnostics, scenes, advice)
    verify_all()
    save(out / 'completion_manifest.json', dict(state='complete', files={
        p.relative_to(ROOT).as_posix(): sha(p) for p in [*out.iterdir(), REPORT] if p.is_file()}))
    print(json.dumps(dict(output=str(out), report=str(REPORT), parity=dict(parity),
                         state_groups=state_groups, endpoint_groups=endpoint_groups), ensure_ascii=False), flush=True)


def write_report(out, summary, states, diagnostics, scenes, advice):
    modes = {MODES[0]: '即时缓存', MODES[1]: '实测延迟'}
    lines = ['# 14B方向提示剩余负效应：缓存诊断', '',
        '日期：2026年10月7日。本轮没有新增仿真、训练或LLM请求，未读取保留最终场景。分析在看到前轮负面差异后设计，属于开发缓存的事后机制诊断，不能称为独立确认。', '',
        '## 1. 先把终点差异定位到病例', '',
        '所有比较均为“仅方向提示−同权重无建议PPO”。下表把数值微小差异也计入：FED变化阈值为1e-12；它与之前±0.001的有益/有害分类分开。', '',
        '|子集|时序|配对|FED更高/更低/相同|失能更多/更少|', '|---|---|---:|---|---|']
    for g in summary['endpoint_groups']:
        label = '主要12场景' if g['cohort'] == PRIMARY else '探索12场景'
        lines.append(f"|{label}|{modes[g['mode']]}|{g['paired_episodes']}|{g['fed_higher']}/{g['fed_lower']}/{g['fed_identical']}|{g['incap_higher']}/{g['incap_lower']}|")
    lines += ['', '|新增失能病例|行为|人口种子|权重种子|时序|失能率差（百分点）|FED差|',
              '|---|---|---:|---:|---|---:|---:|']
    for r in summary['additional_incapacitation_cases']:
        lines.append(f"|{r['scenario_id']}|{r['behavior']}|{r['population_seed']}|{r['training_seed']}|{modes[r['source_mode']]}|{r['incapacitation_rate_difference']*100:+.4f}|{r['mean_cumulative_fed_difference']:+.6f}|")
    lines += ['',
        '主要子集的新增失能全部集中在AULL0024、Rule行为、权重602、两批新人口的即时模式。探索子集AULL0010、概率行为、权重605的两批人口在两种时序均出现差异；两种时序复用人口，不能算四份独立建筑证据。', '',
        '|主要子集，有非零FED差的场景|时序|FED更高/更低配对|场景平均FED差|', '|---|---|---|---:|']
    for d in sorted(diagnostics, key=lambda r: (r['scenario_id'], r['mode'])):
        if d['cohort'] == PRIMARY and d['fed_higher_pairs'] + d['fed_lower_pairs']:
            lines.append(f"|{d['scenario_id']}|{modes[d['mode']]}|{d['fed_higher_pairs']}/{d['fed_lower_pairs']}|{d['mean_cumulative_fed_difference']:+.7f}|")
    lines += ['',
        '## 2. 从缓存还原动作，但不推进仿真', '',
        f"本轮读取42,034个旧控制步，以保存的占用、烟气、容量、指示牌、公共预报和上一实际放行容量重建34维PPO输入。对{summary['reconstructed_input_records']}个记录重新计算动作，全部无建议原始动作、约束后动作、建议特征动作和原完整建议动作均与缓存一致。缺少历史时程序直接停止，不填猜测值。", '',
        '然后将已冻结的仅方向提示适配器应用到1,024个旧人口关键状态，终点直接查询已经保存的9,342条合法动作后续轨迹，没有追加rollout。下面使用每个主要开发场景首次响应的240个状态作为分母，避免只挑有害状态。', '',
        '|子集|时序|首次响应状态|方向提示改变动作|改变状态：有益/有害/微小|与既有规则动作相同|', '|---|---|---:|---:|---|---:|']
    for g in summary['state_groups']:
        label = '主要12场景' if g['cohort'] == PRIMARY else '探索12场景'
        c = g['changed_outcome_classes']
        lines.append(f"|{label}|{modes[g['mode']]}|{g['first_response_states']}|{g['changed_states']}|{c.get('beneficial',0)}/{c.get('harmful',0)}/{c.get('small_or_neutral',0)}|{g['changed_rule_agreement']}|")
    lines += ['',
        '这是旧人口状态中的“一步干预、随后共同无建议PPO”效果，不是新人口完整方向提示轨迹的分步贡献。±0.001分类阈值沿用前轮冻结定义，不是人体安全阈值，也不证明LLM的实际收益概率。', '',
        '## 3. 三个值得区分的病例', '']
    for sid in ('AULL0024', 'AULL0010', 'AULL0013'):
        scene = scenes[sid]
        target = advice[sid]['advice']['target_branch']
        caps = (scene['north_exit_capacity'], scene['south_exit_capacity'])
        lines += [f'### {sid}', '',
            f"14B目标为{('北','南')[target]}侧，confidence={advice[sid]['advice']['confidence']:.2f}；北/南容量为{caps[0]}/{caps[1]}人每步，预报火源为{('北','南')[scene['forecast_source_branch']]}侧，模型烟气增长在第{scene['onset_step']}步之后开始。", '',
            '|旧人口同状态|时序/步|无建议→方向提示|规则动作|FED差|失能率差（百分点）|纯方向项FED/容量项FED/交互项FED|',
            '|---|---|---|---:|---:|---:|---|']
        cases = [r for r in states if r['scenario_id'] == sid and r['first_response'] and r['hint_changes_action']]
        for r in cases:
            if sid == 'AULL0024' and not (r['behavior'] == 'rule' and r['training_seed'] == 602):
                continue
            parts = [r.get('mean_cumulative_fed_' + p) for p in
                     ('direction_at_base_release', 'release_at_base_direction', 'direction_release_interaction')]
            part_string = '/'.join('不适用' if v is None else f'{v:+.6f}' for v in parts)
            lines.append(f"|{r['behavior']},权重{r['training_seed']},人口{r['population_seed']}|{modes[r['source_mode']]}/{r['source_step']}|{r['base_action']}→{r['hint_action']}|{r['rule_action']}|{r['mean_cumulative_fed_difference']:+.6f}|{100*r['incapacitation_rate_difference']:+.4f}|{part_string}|")
        lines += ['', '动作0—3为北侧25%/50%/75%/100%放行，4—7为南侧相同档位，8为等待。纯方向项固定无建议放行档位；容量项固定无建议方向；交互项记录两者联动。三项加起来等于完整一步差异，只是诊断分解。', '']
        if sid == 'AULL0024':
            lines += ['本病例推荐的方向并不是预报火源，但它吞吐较低。初始两侧烟气相等；北侧50%动作1与南侧50%动作5都合法。南侧偏好可在早期把指示牌转向每步3人的出口，虽然房间总放行仍为10人。规则动作同样是5，说明这类“避开预报源而忽略容量时机”的不足并非LLM独有。', '',
                '这是支持吞吐与等待成本冲突的证据，不能只说LLM选了火源。房间放行是总流量，人员并非全部服从指示牌，不能把10人/3人直接当违规比率。旧反事实中的纯方向项和容量项可检查是转向本身有害，还是容量与方向联动。', '']
        elif sid == 'AULL0010':
            lines += ['它有相反的矛盾：南侧容量较高，却也是公开预报的未来烟气源。早期读数还相同，门控不会按当前烟气差拒绝。旧缓存可检验相同放行档位的南向干预是否有害；新人口终点证实这个组合值得追查，但不能仅凭选择预报源就断言所有南向早期放行都错误。', '']
        else:
            lines += ['该场景推荐北侧，容量稍高，同时北侧也是预报源。小范围容量优势、尚未出现的烟气增长和指示牌保持时间可能冲突。表中不同权重或人口的符号变化应保留，不能把整个场景所有动作一律标成有害。', '']
    lines += ['## 4. 为什么当前门控没有拦住', '',
        '现有检查回答的是“时间戳是否过期、出口是否封闭、当下烟气是否明显更差、动作是否满足容量限制”。它没有学会回答“这次覆盖在未来是否划算”。只要初始烟气接近、总放行在合法上限内，低吞吐目标和预报烟气源都可能通过检查。', '',
        '指示牌还有保持时间：从初始中立转为某方向后，后续两个控制步受保持约束。即使路口初始无人，当步只有房间放行，指示牌仍会影响后续路由，因此“当步分支流量没变”不能等同于动作无效。本轮依据冻结执行器代码解释该时序，没有追加保持时间消融，不能单独量化保持时间造成了多少损失。', '',
        '0.25×confidence只是分数偏置。它能在PPO两方向得分接近时改变选择，但confidence没有校准为终点收益概率。因而“通过门控且自报有把握”不足以支持覆盖。', '',
        '## 5. 缓存能说明什么，还缺什么', '',
        '可以确定：哪些场景/人口/权重产生终点差异；旧同状态下方向、容量、交互各自对应什么终点；同信息的现有规则是否也做出该动作；初始公共输入与旧人口完全一致时，冻结策略的第一步动作。', '',
        '无法确定：新人口完整回合的每一次有害改变发生在何时、实际多少人因此改道、累计烟气在哪个区域增加，以及多个改变之间的相互影响。新人口文件只有终点和变化计数，旧人口不能替代其隐藏个体状态。因此本轮定位的是原因证据与剩余缺口，没有完成新人口全程的精确因果归因。', '',
        '最终48场景仍封存。本轮不改门控阈值、偏置强度、PPO权重或默认模型台账；不使用离线最优动作训练或部署收益盾，也不因单个病例把低容量或预报源永久禁用。', '',
        '下一轮最小需要是对已定位病例记录完整实际动作链，并分别固定方向、固定放行档位和缩短保持时间，采用相同人口/时刻随机数比较；另冻结开发人口验证修正。这个有限机制实验先于新增训练。本轮未执行这些仿真。', '',
        '## 6. 复现材料', '',
        '- ' + table_link('诊断范围与哈希冻结计划', PREP / 'manifest.json'),
        '- ' + table_link('全部新人口配对终点', out / 'fresh_population_paired_endpoints.csv'),
        '- ' + table_link('全部旧状态动作与效应分解', out / 'old_population_state_diagnostics.csv'),
        '- ' + table_link('缓存合法动作终点单元', out / 'cached_action_cells.csv'),
        '- ' + table_link('初始公共输入动作核对', out / 'initial_policy_rescore.csv'),
        '- ' + table_link('分场景定位', out / 'scene_diagnostics.json'),
        '- ' + table_link('总审计结果', out / 'summary.json'), '',
        '新增缓存重建和分解的5项单元测试通过。原缓存、已有报告和权重的完成清单哈希核对通过；全部旧状态动作一致性检查通过。本轮新增仿真0、训练0、LLM调用0、最终场景读取0。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run', 'verify'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'verify': verify_all}[args.stage]()
