"""Audit finished artifacts and publish a Chinese development report."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import numpy as np
from run_timestamped_development_campaign import ROOT, PREP, read, save, sha, unique_output, verify_preparation
from run_timestamped_pipeline_followup import verify as verify_followup, FIRST, FREEZE

RECOVERED = ROOT / 'outputs/timestamped_pipeline_followup_recovered_20261006T173442494950Z'
LIVE = ROOT / 'outputs/qwen35_timestamped_live_20261006T173717480909Z'
REPORT = ROOT / 'reports/27B提速与LLM_PPO时间戳流程执行报告_2026-10-07.md'


def audit_completion(folder):
    ledger = read(folder / 'completion_manifest.json')
    for p, digest in ledger['files'].items():
        assert sha(ROOT / p) == digest, p
    return len(ledger['files'])


def main():
    assert not REPORT.exists(), 'Preserve a previously published report.'
    verify_preparation()
    verify_followup()
    checked = {str(p.relative_to(ROOT)): audit_completion(p) for p in (FIRST, RECOVERED, LIVE)}
    a, b = read(FIRST / 'summary.json'), read(RECOVERED / 'summary.json')
    assert a['completed'] == 1488 and b['completed_rollouts_recovered'] == 480
    assert a['fresh_27b_calls'] == b['new_27b_calls'] == 0
    assert read(LIVE / 'summary.json')['attempts'] == 0
    assert len(read(ROOT / 'outputs/qwen35_label_order_probe_20261006/manifest.json')['request_order']) == 32
    assert len(read(PREP / 'confirmation_manifest.json')['request_order']) == 16
    with (FIRST / 'episodes.csv').open(encoding='utf-8-sig') as f:
        episodes = list(csv.DictReader(f))
    pure = [r for r in episodes if r['arm'] == 'pure_ppo']
    blocked_rate = sum(int(r['ppo_raw_masked']) for r in pure) / sum(int(r['steps']) for r in pure)
    all_mask_totals = {arm: {'raw_masked_steps': sum(int(r['ppo_raw_masked']) for r in episodes if r['arm'] == arm),
                            'control_steps': sum(int(r['steps']) for r in episodes if r['arm'] == arm)} for arm in a['arms']}
    deltas = {r['metric']: r for r in b['same_checkpoint_comparisons'] if r['candidate'] == 'cached_llm_ppo'}
    fed_relative = deltas['mean_cumulative_fed']['mean_difference'] / b['no_advice_means']['mean_cumulative_fed'] * 100
    preflight = read(LIVE / 'preflight.json')
    available_gib = preflight['available_memory_bytes'] / 1024**3
    out = unique_output('timestamped_campaign_final_audit')
    test_files = ['tests/test_timestamped_pipeline.py', 'tests/test_qwen35_timestamped_live.py',
                  'tests/test_timestamped_orchestrator.py', 'tests/test_timestamped_live_v3.py']
    audit = {'outcome': 'development_engineering_complete_live27b_resource_blocked',
             'completed_development_rollouts': 1968, 'new_ppo_updates': 0, 'new_training_seeds': 0,
             'live27b_attempts': 0, 'llm_label_diagnostic': '32_prepared_not_run', 'llm_confirmation': '16_prepared_not_run',
             'final_test_result': '21 passed', 'test_files_sha256': {p: sha(ROOT / p) for p in test_files},
             'checked_completion_ledgers': checked, 'raw_action_mask_counts': all_mask_totals,
             'protected_sources_unchanged': True, 'reserved_final_scenarios_opened': 0,
             'current_default_model': read(ROOT / 'outputs/local_llm_selection.json')['model'],
             'original_predictive_shield_qualified': False,
             'math_guard_scores_cannot_be_called_model_accuracy': True,
             'runtime_supersession': {'recommended': 'scripts/run_qwen35_timestamped_live_v3.py',
                 'v1': 'initial finite runner; v3 adds same-checkpoint invalid-response fallback',
                 'v2': 'preserved but superseded: misplaced completion guard; no v2 inference executed'},
             'followup_recovery': '480 completed rows recovered without repeating simulations after numpy int64 JSON serialization failure',
             'source_outputs': [str(FIRST), str(RECOVERED), str(LIVE)]}
    save(out / 'audit.json', audit)
    lines = [
        '# 27B提速与LLM＋PPO时间戳流程执行报告', '', '日期：2026-10-07（北京时间）。', '',
        '本轮完成了时间戳控制流程、21项功能测试和1968个开发仿真回合。Ollama目前可连接，但27B真实提速、32题诊断与16题语义确认仍受内存门槛阻塞，实际新27B推理0次。', '',
        '## 执行范围与资源结果', '',
        f"最后检查：Ollama {preflight['version']['version']}；活动模型0；可用内存{available_gib:.2f}GiB，固定启动门槛17GiB。27B纯文本权重约16.55GB，旧CPU运行占用约16.84GB；本轮未降低门槛来强行加载。", '',
        '固定计划为：6条线程/驻留提速测试 → 原32条名称/候选顺序诊断 → 另行冻结16条语义确认 → 4条异步现场提案。最多58次真实请求、55分钟，不自动重试；最终保留场景继续封存。新目录中的UTC时间字符串可能显示10月6日，换算北京时间为10月7日。', '',
        '本轮尚无新速度结果。旧逐题卸载的中位总等待49.617秒：加载22.717秒、输入19.340秒、生成7.613秒，生成约2.890token/秒。旧驻留总等待约25.825秒，但曾采样到仅26MB可用内存，不能据此认定可持续使用。', '',
        '已准备的提速对照保留27B Q4_K_M：短驻留、ctx1024、batch32、线程4/8、实际输出上限128，分别测加载/输入/输出和内存。降低最大输出上限本身不代表22token答案生成会变快。当前也没有核显加速的新证据；旧记录为CPU运行。', '',
        '## 已接入的控制流程', '',
        '现场观测形成带会话、步数、仿真观测时间和UTC采集时间的快照。LLM只能接收实时烟气、占用、开放状态、容量与标明不确定性的预测字段。真实火源、未来烟气和个体FED不进入输入。', '',
        'LLM提案包括优先出口、最多三项候选动作、最多两步等待和北出口分流偏好；采用现有9动作：北/南各25%、50%、75%、100%放行，加等待。分流比例只影响多步方向偏好，现有仿真无法精确控制同一步两个出口的独立比例。', '',
        '一次在途请求放到工作线程中，PPO继续根据最新观测行动。建议返回时重新检查当前出口、烟气、容量、会话与时间；格式错误、超时和过期都清空建议通道。修订后的执行器在请求前后使用相同PPO权重，避免换模型被误算成建议收益。', '',
        '门控默认15秒有效期、60秒逻辑超时。因此50秒延迟组的建议会被拒绝，这是明确的设计结果：目前这种时效要求下，慢27B不能直接承担5秒一轮的实时控制。未来若要使用更长时段的规划，必须另行定义有效时间窗并验证返回时的状态一致性。', '',
        '所有控制器共用执行约束：已封闭或每步烟气指标≥0.15的走廊不可进入，遵守指示牌保持时间，以允许出口容量总和和45人路口余量限制房间放行。仿真执行层也实际限制进入与放行人数；记录请求、约束动作、实际联合动作、实际指示牌和实际人数。', '',
        '这些是开发用操作约束和启发式信任门控。confidence仍未校准；旧预测盾的资格结论保持未通过。满足容量约束不能当成真实火灾安全认证。', '',
        '## 开发仿真与比较结果', '',
        '第一批1488回合：12条已使用的AULL开发场景，6类各2条；规则/校准logit两个人群模型；两个人口种子；601–605五组既有PPO权重。纯规则48回合，其他六组各240回合。使用的是10月5日Qwen3 4B历史缓存，错误/延迟/超时是故障注入。', '',
        '|控制组|回合|平均烟气累积指标|累计拥堵|疏散率|失能率|全员疏散回合|', '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ('pure_rule', 'pure_ppo', 'cached_llm_ppo', 'rule_advice_ppo'):
        r = a['arms'][arm]
        lines.append(f"|{arm}|{r['episodes']}|{r['mean_cumulative_fed']:.4f}|{r['cumulative_congestion']:.2f}|{r['evacuation_rate']:.2%}|{r['incapacitation_rate']:.2%}|{r['fully_evacuated_episodes']}|")
    lines += ['', '这张表里的纯PPO和LLM组使用不同的既有训练权重，不能直接把差异归因于LLM。开发场景复用，也没有最终评估的独立性。', '',
              '因此第二批另行冻结480回合，补了“相同LLM组PPO权重、完全不给建议”的240回合对照，并复现240回合错误建议审计。与这个相同权重对照相比：', '',
              '|指标|有缓存建议−相同权重无建议|描述性场景bootstrap 95%区间|', '|---|---:|---|']
    for metric in ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'episode_end_seconds'):
        r = deltas[metric]
        lo, hi = r['descriptive_scene_bootstrap_95_interval']
        lines.append(f"|{metric}|{r['mean_difference']:.6f}|{lo:.6f} ～ {hi:.6f}|")
    lines += ['', f'烟气指标平均变化约{fed_relative:.3f}%，只有20/240个配对回合有非零烟气差；疏散比例只提高0.015625个百分点。烟气、拥堵与疏散比例的描述性区间均跨零，当前不能说建议产生了稳定收益。', '',
              '延迟组与同权重无建议组的各项主要结果完全相同，说明过期建议没有偷偷改变控制。现有错误组也未产生相对同权重无建议的主要结果差异。', '',
              'bootstrap以12个场景为单位，条件于已有五组权重，不是最终显著性检验，也不包含完整训练不确定性。仿真结束时间包含失能后终止；全员成功疏散时间仅对真正全员疏散的回合记录，其余为空。', '',
              '## 错误建议是否被正确处理', '',
              '共9063个控制步注入了更高烟气方向建议。其中8723步有另一个开放且未超阈值的较低烟气出口，这些建议全部触发了烟气矛盾拒绝；另340步的替代出口不可用，不能仅凭烟气大小就判定该建议必错。', '',
              '在8723次矛盾拒绝中，最终仍有596步执行同一个较高烟气方向，比例约6.83%。这是已有PPO自己的选择，不代表建议通过了门控；它直接说明“拒绝LLM建议”与“PPO正确纠错”不能合并成一个成功率。', '',
              '原828次“更高烟气方向执行”中另232次属于替代出口不可用的情况，已经单列为歧义。上述596步也只能说明未遵循烟气矛盾筛查偏好，不能直接断言最终伤害一定更大；拥堵、等待和后续风险需要可行替代计划的反事实收益比较。', '',
              f'第一批所有控制组的不可用走廊进入、放行超限和实际执行方向违规计数均为0。纯PPO原始动作在{blocked_rate:.1%}的控制步被共同动作掩码排除，表明既有策略受新增约束影响很大；不能把这些结果视作原策略自然适应。', '',
              '指示牌可能在等待时保留上一步方向，记录中保留了此反馈；等待对应零放行，仿真进入容量另行拦截。实际设备是否能提供同样的物理约束仍待验证。', '',
              '## 32题诊断与16题确认状态', '',
              '32条名称/顺序诊断准备仍封存，未开始本轮推理。另行冻结的16题改变概率与物理数值，并增加明确的hazard/clear概率解码说明；原始模型正确率与独立算术校验/规则回退正确率分开记录，不能把规则修正当成LLM理解改善。', '',
              '评分器已覆盖A/B物理映射、候选首位统计、重复JSON键、非法confidence、缺失配对、数学角色与目标解码。21项最终功能测试全部通过，含异步错误响应回退、过期后清空建议、实际动作限制和请求完成后再核查响应数量。这里的功能测试不等于16题已由27B答对。', '',
              '新增输出严格分阶段；一次补充仿真在CSV保存后发生NumPy int64的JSON序列化错误，480个完整回合从CSV与原始轨迹恢复，没有重复仿真。现场执行器保留旧版本，修订版v3用于后续真实运行。', '',
              '## 后续已准备的执行入口', '',
              '`D:/miniconda3/python.exe scripts/run_timestamped_development_campaign.py verify`核对原计划；`D:/miniconda3/python.exe scripts/run_timestamped_pipeline_followup.py verify`核对补充计划。资源满足固定门槛后，`D:/miniconda3/python.exe scripts/run_qwen35_timestamped_live_v3.py`继续6→32→16→4真实阶段，每次另建目录。', '',
              '目前应保持训练暂停：先完成真实27B诊断/确认，检查时延与有效期是否匹配；再对596类矛盾执行做可行替代动作的收益确认。当前证据支持流程能记录并隔离失败，尚未证明27B提速或LLM＋PPO有稳定疏散收益。', '',
              '原默认8B配置、全局状态、旧论文实验、旧模型、标签准备台账与27B审计摘要均通过哈希核对。新训练0，保留最终场景读取0。', '',
              f'[首批详细数据]({FIRST.as_posix()}/report.md)；[同权重对照与错误审计]({RECOVERED.as_posix()}/report.md)；[最后27B预检查]({LIVE.as_posix()}/preflight.json)。', '',
              '参数参考：[Ollama FAQ](https://docs.ollama.com/faq)和[Ollama 0.35.1选项源代码](https://raw.githubusercontent.com/ollama/ollama/v0.35.1/api/types.go)。文档支持参数选择，本机速度结论仍需真实请求。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    save(out / 'completion_manifest.json', {'state': 'audit_complete_live_inference_blocked',
         'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in [out / 'audit.json', REPORT, Path(__file__)]},
         'verified_ledgers': checked, 'protected_sources_unchanged': True})
    print(json.dumps({'report': str(REPORT), 'audit': str(out), 'new_27b_calls': 0,
                      'development_rollouts': 1968, 'same_checkpoint_fed_change_percent': fed_relative,
                      'available_gib': available_gib}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
