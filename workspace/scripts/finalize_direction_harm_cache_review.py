"""Final cache review and statistical wording erratum; preserve frozen files."""
from pathlib import Path
import argparse
import csv
import json
import sys

from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output
from audit_direction_hint_full_cached_trace import verify_sources, DIAG, PRIMARY, MODES


SOURCE = ROOT / 'outputs/direction_hint_full_cached_trace_audit_20261007T055930820595Z'
OLD_REPORT = ROOT / 'reports/14B剩余负效应归因总审计_2026-10-07.md'
REPORT = ROOT / 'reports/14B剩余负效应归因总审计_校订版_2026-10-07.md'


def run():
    assert not REPORT.exists()
    verify_sources()
    for p, h in read(SOURCE / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p
    s = read(SOURCE / 'summary.json')
    with (SOURCE / 'all_step_shadow_actions.csv').open(encoding='utf-8-sig', newline='') as f:
        traces = list(csv.DictReader(f))
    assert len(traces) == len({r['state_id'] for r in traces}) == 42034
    assert all(r['hint_action'] == r['base_action'] or r['gate_accepted'] == 'True' for r in traces)
    assert all(not (r['hint_action'] != '8' and r['base_action'] != '8'
                   and int(r['hint_action']) // 4 == int(r['base_action']) // 4
                   and r['hint_action'] != r['base_action']) for r in traces)
    first = {}
    for r in traces:
        key = (r['scenario_id'], r['behavior'], r['population_seed'], r['training_seed'], r['source_mode'])
        if r['hint_changes_base'] == 'True' and key not in first:
            first[key] = r
    missing = [r for r in first.values() if r['cohort'] == PRIMARY and r['source_mode'] == MODES[1]
               and r['has_cached_terminal_action_cells'] == 'False']
    assert len(missing) == 8 and {r['scenario_id'] for r in missing} == {'AULL0013', 'AULL0020'}
    assert all(r['step'] == '3' and r['state_on_hint_prefix'] == 'True' for r in missing)
    protected = {
        'outputs/local_llm_selection.json': '302c6db2c857811dd41d1ed36248d1d9a71f3e32108cd870ecfe02aa520ff721',
        'outputs/third_round_development_status.json': 'a84021405b97bd76384fee0a0d48bc12bce09a42b40a2c3a2536224d03e48ed0',
    }
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    wording_corrections = [
        ('这四个独立状态病例各自的两种时序结果完全重复，所以表只列一份。',
         '四个去重的“行为模型×人口”状态病例各自的两种时序结果完全重复，所以表只列一份。它们共享场景、权重和部分人口，不能视为统计独立样本。'),
        ('它在旧首次响应中的同方向干预有益或微小。',
         '它在旧首次响应中同样的转向干预有益或微小。'),
    ]
    document = OLD_REPORT.read_text(encoding='utf-8')
    for before, after in wording_corrections:
        assert document.count(before) == 1
        document = document.replace(before, after)
    document += ('\n校订说明：本版只修正“独立状态病例”为去重组合的统计措辞，并澄清AULL0019的转向表述；'
                 '没有改变任何数值、策略或分类。原冻结报告和数据保留。最终复核确认，主要延迟子集缺少终点反事实的8个首次影子变化，'
                 '来自AULL0013和AULL0020，均在第15秒且此前策略前缀相同；新人口精确轨迹仍未保存，不能以旧人口替代。\n')
    REPORT.write_text(document, encoding='utf-8')
    out = unique_output('direction_harm_cache_final_review')
    summary = dict(report=str(REPORT), original_report=str(OLD_REPORT), cache_only_round_complete=True,
        original_results_preserved=True, numerical_results_changed=False,
        corrected_statistical_wording=wording_corrections,
        full_control_steps_verified=42034, five_original_action_fields_matched=s['full_action_parity'],
        paired_fresh_population_endpoints=960, cached_terminal_continuations=9342,
        tests_passed_new=5, tests_passed_regression=7,
        primary_delayed_missing_terminal_cells=missing,
        all_hint_changes_require_accepted_gate=True, same_direction_release_changes=0,
        protected_sha256=protected, new_simulations=0, new_training=0, new_llm_requests=0,
        reserved_final_scenarios_opened=0, no_online_controller_change=True,
        causal_scope='Cached old-state one-step interventions; exact fresh-population full-episode causal attribution is not complete.',
        decision='Keep training paused and final scenes sealed. Next finite mechanism work should record exact fresh-population action paths and separate direction, release level and sign hold. No policy adoption.',
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), SOURCE / 'completion_manifest.json',
                 DIAG / 'completion_manifest.json', OLD_REPORT, REPORT]})
    save(out / 'summary.json', summary)
    verify_sources()
    for p, h in read(SOURCE / 'completion_manifest.json')['files'].items():
        assert sha(ROOT / p) == h, p
    save(out / 'completion_manifest.json', dict(state='complete', files={
        p.relative_to(ROOT).as_posix(): sha(p) for p in [out / 'summary.json', REPORT, Path(__file__)]}))
    print(json.dumps(dict(report=str(REPORT), final_review=str(out), verified_steps=42034,
                         missing_delayed_terminal_cells=len(missing), new_simulations=0), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
