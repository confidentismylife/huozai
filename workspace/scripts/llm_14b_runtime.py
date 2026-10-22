"""Bounded runtime sampling and descriptive reporting for the real 14B workload."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import threading
import time
import psutil
import auto_evacuation_campaign as c
import run_llm_output_mechanism as io


class Monitor:
    def __init__(self, path):
        self.path = Path(path)
        self.samples = []
        self.finished = False
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    @staticmethod
    def sample():
        infos = [p.info for p in psutil.process_iter(['pid', 'ppid', 'name', 'memory_info'])]
        selected = {p['pid'] for p in infos if (p['name'] or '').lower().startswith('ollama')}
        while True:
            expanded = selected | {p['pid'] for p in infos if p['ppid'] in selected}
            if expanded == selected: break
            selected = expanded
        processes = [{'pid': p['pid'], 'name': p['name'], 'rss_bytes': p['memory_info'].rss}
                     for p in infos if p['pid'] in selected and p['memory_info'] is not None]
        return {'at_utc': datetime.now(timezone.utc).isoformat(),
                'available_memory_bytes': psutil.virtual_memory().available, 'ollama_processes': processes}

    def collect(self):
        while not self.event.is_set():
            try: self.samples.append(self.sample())
            except (psutil.Error, OSError): pass
            self.event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        if self.finished: return
        self.event.set()
        self.thread.join(timeout=3)
        self.samples.append(self.sample())
        io.save(self.path, {'interval_seconds': 1, 'process_scope': 'ollama_named_processes_and_descendants', 'samples': self.samples,
            'minimum_sampled_available_memory_bytes': min(x['available_memory_bytes'] for x in self.samples),
            'maximum_sampled_ollama_rss_sum_bytes': max(sum(p['rss_bytes'] for p in x['ollama_processes']) for x in self.samples),
            'rss_sum_can_double_count_shared_pages': True, 'sampling_does_not_establish_exact_peak': True})
        self.finished = True


def write_report(out, report, manifest, summary):
    root = out.parents[1]
    previous = c.read(root / 'outputs/llm_goal_role_8b_20261006/summary.json')
    labels = {'minimize_hazard': '最小化 × 危险侧报告', 'maximize_hazard': '最大化 × 危险侧报告',
              'minimize_clear': '最小化 × 无烟侧报告', 'maximize_clear': '最大化 × 无烟侧报告'}
    rows = []
    for key, g in summary['groups'].items():
        before = previous['groups'][key]
        rows.append({'condition': key, '8b_correct': before['usable_correct'], '14b_correct': g['usable_correct'],
            'correct_count_change': g['usable_correct'] - before['usable_correct'],
            '8b_mean_latency_seconds': before['mean_latency_seconds'], '14b_mean_latency_seconds': g['mean_latency_seconds']})
    c.csv_write(out / '8b_14b_comparison.csv', rows)
    def read_rows(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    old_rows = {r['request_id']: r for r in read_rows(root / 'outputs/llm_goal_role_8b_20261006/case_results.csv')}
    new_rows = {r['request_id']: r for r in read_rows(out / 'case_results.csv')}
    improved = sum(old_rows[k]['usable_correct'] != 'True' and new_rows[k]['usable_correct'] == 'True' for k in new_rows)
    worsened = sum(old_rows[k]['usable_correct'] == 'True' and new_rows[k]['usable_correct'] != 'True' for k in new_rows)
    metrics = [c.read(p) for p in sorted(out.glob('resource_run_*.json'))]
    subtree_metrics = [v for v in metrics if v.get('process_scope') == 'ollama_named_processes_and_descendants']
    loaded = c.read(out / 'loaded_model_snapshot.json')
    total = sum(g['usable_correct'] for g in summary['groups'].values())
    latency = sum(g['total_latency_seconds'] for g in summary['groups'].values())
    old_latency = sum(g['total_latency_seconds'] for g in previous['groups'].values())
    comparison = {'groups': rows, '8b_total_correct': sum(g['usable_correct'] for g in previous['groups'].values()),
        '14b_total_correct': total, 'improved_requests': improved, 'worsened_requests': worsened,
        'unchanged_requests': 32 - improved - worsened, '14b_total_latency_seconds': latency,
        '8b_total_latency_seconds': old_latency, 'descriptive_same_cases_only': True,
        'independent_evacuation_effect_established': False, 'held_out_replication': False}
    io.save(out / '8b_14b_comparison.json', comparison)
    lines = ['# Qwen3 14B本地复测实验报告', '', '日期：2026-10-06。14B已成功接入D盘模型目录，执行并核验同一批32请求。', '',
        f"实际模型qwen3:14b，14,768,307,200参数，Q4_K_M，manifest digest `{manifest['model_digest']}`。Ollama版本0.35.1。", '',
        '复用8题、4个镜像家庭及4个目标/报告角色条件，提示、JSON schema、顺序、参考、think=false请求、temperature0、seed2026100601、1024输出预算和4096上下文保持一致；API请求只更改模型名。14B与8B内置模板及默认参数相同。thinking字段是否存在单独记录，不能仅由请求字段推断内部过程。', '',
        '首题作为加载检查，同时计入32题，不额外增加一道题；后续继续剩余31题。采用传输或加载错误即停止的执行策略，不对错误答案、格式错误或截断进行重试。本次没有改变训练或疏散控制器。', '',
        '| 条件 | 8B正确/8 | 14B正确/8 | 14B方向错 | 格式错 | 截断 | 传输失败 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        g = summary['groups'][r['condition']]
        lines.append(f"| {labels[r['condition']]} | {r['8b_correct']} | {r['14b_correct']} | {g['valid_wrong_direction']} | {g['invalid_format']} | {g['truncated']} | {g['transport_failure']} |")
    lines += ['', f"8B为19/32，14B为{total}/32；逐题改善{improved}题、退步{worsened}题，其余不变。", '',
        '| 配对检查 | 方向关系正确/8 | 两答案均正确/8 |', '| --- | --- | --- |']
    for p in summary['paired_diagnostics']:
        lines.append(f"| {p['comparison']} | {p['expected_relation_observed']} | {p['both_answers_correct']} |")
    passed = summary['all_cells_passed'] and summary['all_primary_pairs_correct']
    lines += ['', ('14B通过本批全部基础功能与配对检查，可进入独立题目复测。' if passed else '14B尚未通过本批全部基础功能与配对检查，不能认为目标和字段角色处理问题已解决。'), '',
        f"累计请求等待：8B {old_latency:.2f}秒，14B {latency:.2f}秒，约为8B的{latency/old_latency:.2f}倍。等待时间包含首题加载、提示处理和生成，不是单独的生成速度比较。", '',
        f"加载后Ollama报告：{json.dumps(loaded, ensure_ascii=False)}", '',
        f"每秒资源采样所见最低系统可用内存：{min(v['minimum_sampled_available_memory_bytes'] for v in metrics)/1e9:.2f}GB；后续31题中Ollama进程及其子进程RSS合计采样最高值：{max(v['maximum_sampled_ollama_rss_sum_bytes'] for v in subtree_metrics)/1e9:.2f}GB。首题采样原先只筛选名称以Ollama开头的进程，遗漏名为runner.exe的子进程，因此首题的RSS合计不作为模型峰值使用。RSS合计可能重复统计共享页，采样不能确定精确峰值，也不能排除其他应用的影响。", '',
        '首题完成后发现报告读取CSV的辅助函数引用错误，并补全资源采样对子进程的统计。已保存原代码、原manifest和校验记录后再继续剩余题目；仅修正报告生成及运行监测，请求、评分函数及已得到的首题结果未更改。', '',
        '这32个请求属于4个固定镜像家庭，不能当作32个独立场景，不作总体显著性结论。最大化条件是反向数学标签测试，不能作为疏散建议。单步期望暴露不是伤亡或疏散率，自报confidence不是已经校准的风险概率。', '',
        '本批比较无法单独建立纯参数规模因果效应，更不能证明14B+PPO改善了火灾疏散。下一步应先检查独立题目上的目标理解和等价表述一致性；扩大PPO训练继续暂停。48个保留疏散场景未启用，模拟0、拟合0、PPO更新0。默认新实验模型选择记录仍保持8B原值，14B作为已实测候选记录。', '',
        f'[逐题结果]({(out/"case_results.csv").as_posix()})；[8B与14B比较]({(out/"8b_14b_comparison.csv").as_posix()})；[独立核验]({(out/"verification.json").as_posix()})。', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
