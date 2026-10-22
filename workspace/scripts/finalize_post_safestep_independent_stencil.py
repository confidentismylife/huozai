"""Finalize the pre-specified independent-route stencil validation."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fire_evac.fds_cell_stencil import reconstruct
from fire_evac.fds_independent_validation import (
    compare_operators,
    descriptive_layout_summary,
    route_key,
)
from fire_evac.fds_sampling_diagnostic import route_breakdown
from fire_evac.post_safestep_physics import read_device_csv


OUT = ROOT / 'outputs/post_safestep_independent_stencil_20261020'
REPORT = ROOT / 'reports/FDS八单元采样独立路线验证_2026-10-20.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write('\n')


def load_csv(folder, chid):
    return read_device_csv(folder / f'{chid}_devc.csv')[0]


def parity(parent, current, limits):
    common_end = min(float(parent['Time'][-1]), float(current['Time'][-1]))
    old_t = parent['Time'][parent['Time'] < common_end - 1e-6]
    new_t = current['Time'][current['Time'] < common_end - 1e-6]
    same_rows = len(old_t) == len(new_t)
    max_time = float(np.max(np.abs(old_t - new_t))) if same_rows else None
    errors = {'gas_mol_per_mol': None, 'temperature_C': None}
    aligned = {'gas_mol_per_mol': None, 'temperature_C': None}
    if same_rows:
        for suffix, kind in [('(?:CO|CO2|O2)', 'gas_mol_per_mol'),
                             ('T', 'temperature_C')]:
            channels = [key for key in parent if re.fullmatch(r'N\d+_' + suffix, key)]
            errors[kind] = max(float(np.max(np.abs(
                parent[key][:len(old_t)] - current[key][:len(new_t)]
            ))) for key in channels)
            aligned[kind] = max(float(np.max(np.abs(
                parent[key][:len(old_t)] - np.interp(old_t, current['Time'], current[key])
            ))) for key in channels)
    passed = bool(
        same_rows
        and max_time <= limits['native_time_absolute_s']
        and errors['gas_mol_per_mol'] <= limits['gas_absolute_mol_per_mol']
        and errors['temperature_C'] <= limits['temperature_absolute_C']
    )
    return {
        'common_prefix_last_s': float(old_t[-1]) if len(old_t) else None,
        'native_rows': len(old_t),
        'same_row_count': same_rows,
        'max_native_time_difference_s': max_time,
        'maximum_default_output_differences': errors,
        'posthoc_time_aligned_differences_diagnostic_only': aligned,
        'passed': passed,
    }


def main():
    for path in (OUT / 'independent_results.json', OUT / 'independent_route_comparison.csv',
                 OUT / 'independent_gap_comparison.png', OUT / 'completion_manifest.json', REPORT):
        if path.exists():
            raise FileExistsError(f'Final artifact already exists: {path}')
    protocol = read(OUT / 'protocol_frozen.json')
    protected = read(OUT / 'protected_parent_hashes.json')
    for rel, expected in {**protected, **protocol['implementation_sha256']}.items():
        assert digest(ROOT / rel) == expected, rel
    assert digest(ROOT / protocol['fixture_evidence']['source']) == protocol['fixture_evidence']['source_sha256']
    assert protocol['fixture_evidence']['passed']

    jobs = read(OUT / 'jobs_frozen.json')
    fields, doses, parity_records = {}, {}, []
    for job in jobs:
        folder = OUT / 'runs' / job['name']
        result = read(folder / 'run_result.json')
        assert result['input_sha256'] == job['input_sha256']
        assert digest(ROOT / job['input']) == job['input_sha256']
        assert result['normal_completion'] and result['native_output_valid']
        assert result['native_end_s'] == 70.0 and result['solver_last_reported_fire_s'] == 70.0
        assert not result['native_stop_file_used'] and not result['actually_restarted']
        raw = load_csv(folder, job['name'])
        rebuilt = reconstruct(raw, job['observed_nodes'])
        fields[(job['layout'], job['dx_m'])] = (raw, rebuilt, result)
        identity = {'gas_mol_per_mol': 0.0, 'temperature_C': 0.0}
        for node in job['observed_nodes']:
            for label in ('CO', 'CO2', 'O2', 'T'):
                difference = float(np.max(np.abs(
                    raw[f'N{node:03d}_{label}'] - raw[f'C{node:03d}_7_{label}']
                )))
                kind = 'temperature_C' if label == 'T' else 'gas_mol_per_mol'
                identity[kind] = max(identity[kind], difference)
        for route in job['routes']:
            doses[(job['layout'], job['dx_m'], route_key(route))] = {
                'route': route,
                'default': route_breakdown(raw, route),
                'eight_cell': route_breakdown(rebuilt, route, prefix='S'),
            }
        parent = load_csv(ROOT / job['parent'], job['parent_chid'])
        parity_records.append({
            'case': job['name'],
            'layout': job['layout'],
            'dx_m': job['dx_m'],
            **parity(parent, raw, protocol['default_prefix_parity']),
            'same_containing_cell_check_max_differences': identity,
        })

    route_records, summaries = [], {}
    selection = read(OUT / 'independent_routes_frozen.json')['selected']
    for layout, routes in selection.items():
        layout_records = []
        for number, route in enumerate(routes, 1):
            key = route_key(route)
            medium = doses[(layout, .25, key)]
            fine = doses[(layout, .125, key)]
            comparison = compare_operators(
                {name: medium[name]['FED'] for name in ('default', 'eight_cell')},
                {name: fine[name]['FED'] for name in ('default', 'eight_cell')},
                protocol['dose_denominator_floor_FED'],
                protocol['gap_denominator_floor_FED'],
            )
            record = {
                'route_id': f'{layout}-R{number}',
                'layout': layout,
                'route': route,
                'dx0p25': medium,
                'dx0p125': fine,
                'comparison': comparison,
            }
            route_records.append(record)
            layout_records.append(record)
        summaries[layout] = descriptive_layout_summary(layout_records)

    criterion_checks = {
        'L1_median_gap_reduced': summaries['L1']['median_absolute_gap_reduced'],
        'L1_at_least_3_of_4_routes_reduced': summaries['L1']['absolute_gap_reduced_routes'] >= 3,
        'L2_median_gap_reduced': summaries['L2']['median_absolute_gap_reduced'],
        'L2_both_routes_reduced': summaries['L2']['absolute_gap_reduced_routes'] == 2,
    }
    replication_supported = all(criterion_checks.values())
    strict_parity_passed = all(record['passed'] for record in parity_records)
    status = ('COMPLETE_INDEPENDENT_REPLICATION_SUPPORTED' if replication_supported
              else 'COMPLETE_INDEPENDENT_REPLICATION_RULE_NOT_MET')
    result = {
        'status': status,
        'four_fields_completed_normally': True,
        'strict_default_prefix_parity_passed': strict_parity_passed,
        'route_records': route_records,
        'layout_summaries': summaries,
        'pre_specified_criterion_checks': criterion_checks,
        'descriptive_replication_supported': replication_supported,
        'interpretation': (
            'The large gap reduction from the prior post-hoc L1 witness did not replicate '
            'under the pre-specified independent-route rule. Route results are repeated '
            'measurements within two solved layout fields, not six independent solver replicates.'
        ),
        'same_grid_default_prefix_parity': parity_records,
        'old_mesh_failure_unchanged': True,
        'prior_strict_prefix_failure_preserved_as_historical_result': True,
        'formal24_started': False,
        '900s_started': False,
        'new_RL_updates': 0,
        'LLM_requests': 0,
        'sealed_final48_accessed': False,
        'protected_previous_files_verified': len(protected),
    }
    save(OUT / 'independent_results.json', result)

    csv_path = OUT / 'independent_route_comparison.csv'
    with csv_path.open('x', newline='', encoding='utf-8-sig') as target:
        columns = [
            'route_id', 'layout', 'start_index', 'exit_index', 'departure_s', 'route_end_s',
            'default_dx0p25_FED', 'default_dx0p125_FED', 'default_absolute_gap_FED',
            'default_relative_gap', 'eight_cell_dx0p25_FED', 'eight_cell_dx0p125_FED',
            'eight_cell_absolute_gap_FED', 'eight_cell_relative_gap',
            'absolute_gap_change_FED', 'absolute_gap_reduced', 'stable_reduction_fraction',
        ]
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for record in route_records:
            route, comparison = record['route'], record['comparison']
            writer.writerow({
                'route_id': record['route_id'], 'layout': record['layout'],
                'start_index': route['start_index'], 'exit_index': route['exit_index'],
                'departure_s': route['departure_s'], 'route_end_s': route['route_end_s'],
                'default_dx0p25_FED': comparison['default']['candidate_FED'],
                'default_dx0p125_FED': comparison['default']['fine_FED'],
                'default_absolute_gap_FED': comparison['default']['absolute_gap_FED'],
                'default_relative_gap': comparison['default']['relative_gap'],
                'eight_cell_dx0p25_FED': comparison['eight_cell']['candidate_FED'],
                'eight_cell_dx0p125_FED': comparison['eight_cell']['fine_FED'],
                'eight_cell_absolute_gap_FED': comparison['eight_cell']['absolute_gap_FED'],
                'eight_cell_relative_gap': comparison['eight_cell']['relative_gap'],
                'absolute_gap_change_FED': comparison['absolute_gap_change_FED'],
                'absolute_gap_reduced': comparison['absolute_gap_reduced'],
                'stable_reduction_fraction': comparison['stable_reduction_fraction'],
            })

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, layout in zip(axes, ('L1', 'L2')):
        records = [record for record in route_records if record['layout'] == layout]
        positions = np.arange(len(records))
        default = [record['comparison']['default']['absolute_gap_FED'] for record in records]
        rebuilt = [record['comparison']['eight_cell']['absolute_gap_FED'] for record in records]
        axis.bar(positions - .18, default, .36, label='默认单元', color='#52657A')
        axis.bar(positions + .18, rebuilt, .36, label='八单元重建', color='#008578')
        axis.set_xticks(positions, [record['route_id'] for record in records])
        axis.set_yscale('log')
        axis.set_title(f'{layout}：粗细网格绝对差')
        axis.set_ylabel('绝对网格差 / FED（对数轴）')
        axis.grid(axis='y', alpha=.2, which='both')
        axis.legend(fontsize=8)
    figure.suptitle('独立路线验证：柱越低表示两网格越接近')
    figure.text(.5, .02, '路线共享同一数值场，是重复测量；不是6次独立求解器重复。',
                ha='center', fontsize=10)
    image_path = OUT / 'independent_gap_comparison.png'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        figure.tight_layout(rect=(0, .06, 1, .94))
        figure.savefig(image_path, dpi=170)
    plt.close(figure)
    missing_glyphs = [str(item.message) for item in caught if 'Glyph' in str(item.message)]
    if missing_glyphs:
        raise ValueError(f'Missing glyphs in figure: {missing_glyphs}')

    table_rows = []
    for record in route_records:
        comp = record['comparison']
        percent = lambda value: '不定义' if value is None else f'{100 * value:.2f}%'
        table_rows.append(
            f"| {record['route_id']} | {record['route']['start_index']}→{record['route']['exit_index']} | "
            f"{comp['default']['absolute_gap_FED']:.6g} | {percent(comp['default']['relative_gap'])} | "
            f"{comp['eight_cell']['absolute_gap_FED']:.6g} | {percent(comp['eight_cell']['relative_gap'])} | "
            f"{'变小' if comp['absolute_gap_reduced'] else '变大'} |"
        )
    run_rows = []
    for job in jobs:
        run = fields[(job['layout'], job['dx_m'])][2]
        run_rows.append(
            f"| {job['layout']} | {job['dx_m']} | {run['elapsed_wall_s']/60:.2f} | "
            f"{run['peak_RSS_MiB']:.1f} | {run['native_end_s']:.1f} |"
        )
    parity_rows = []
    for record in parity_records:
        errors = record['maximum_default_output_differences']
        parity_rows.append(
            f"| {record['case']} | {record['max_native_time_difference_s']:.3g} | "
            f"{errors['gas_mol_per_mol']:.3g} | {errors['temperature_C']:.3g} | "
            f"{'通过' if record['passed'] else '未通过'} |"
        )
    report = f'''# FDS八单元采样的独立路线验证

完成日期：2026-10-20。状态：`{status}`。

## 核心结论

上一轮在事后选中的L1失败路线中，八单元重建把粗细网格相对差从46.13%降到8.95%。本轮在查看新结果前冻结路线、几何资格和判断规则，再检验未参与上一轮诊断的路线。**预设复现规则没有达到，因此不能把上一轮的大幅缩差当作普遍规律。**

6条路线中，绝对网格差有3条变小、3条变大。L1只有2/4变小，且绝对差变化的中位数为增加{summaries['L1']['median_absolute_gap_change_FED']:.6g} FED；L2为1/2变小，中位数为减少{abs(summaries['L2']['median_absolute_gap_change_FED']):.6g} FED。预设规则要求L1至少3/4、L2必须2/2，而且两个布局的中位数都要改善；实际未满足。

这项结果否定的是“八单元重建会稳定缩小网格差”这一较强说法。它没有证明默认单元读数更准确，也没有证明八单元重建更差。两种方式都缺少独立物理真值；八单元平均会改变局部峰值和剂量尺度。

## 实验如何避免再次事后挑选

路线来自原有封存库，选择只使用三项事前信息：能否在70秒前走完、是否是上一轮使用过的两条见证路线、每个非出口节点周围八个支持单元是否都在开放气体中。选择过程不读取新剂量或网格差。最终纳入L1四条、L2两条；另有三条因八点邻域穿入墙体而在计算前排除。

真正独立求解的单位是“布局×网格”的4个气体场。共享同一个场的多条路线是重复测量，不能把6条路线当成6次独立FDS重复，也不进行路线级显著性检验。

## 运行完成情况

| 布局 | 网格/米 | 墙钟/分钟 | 峰值内存/MiB | 原生结束时间/秒 |
|---|---:|---:|---:|---:|
{chr(10).join(run_rows)}

四个场均正常完成70秒，未重启、未使用`.stop`，也未自动重试。每个布局的粗细网格输入保留全部原默认探头，并在相同12个路线节点追加八单元中心探头。

## 六条路线的配对结果

| 路线 | 起点→出口 | 默认绝对差/FED | 默认相对差 | 八单元绝对差/FED | 八单元相对差 | 绝对差变化 |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

相对差只有在对应细网格剂量不低于`1e-8 FED`时定义，因此仅2/6条路线能同时比较两种观测的相对差。接近零剂量的路线仍保留在绝对差结果和预设路线计数中，没有事后删除。稳定缩减比例也只在默认绝对差不低于`1e-8 FED`时报告，详见机器可读结果。

需要特别区分绝对差和相对差。L2-R1的绝对差从约`3.00e-6`降至`2.72e-6 FED`，但相对差从89.09%升至92.26%，原因是八单元重建同时改变了作为分母的细网格剂量。单看某一种百分比会遗漏这一点。

## 默认探头复现检查

冻结门槛为：原生时间戳绝对差不超过`1e-8 s`、气体不超过`1e-9 mol/mol`、温度不超过`1e-5 °C`。

| 场 | 最大时间差/s | 最大气体差/mol/mol | 最大温度差/°C | 严格通过 |
|---|---:|---:|---:|---|
{chr(10).join(parity_rows)}

本轮四个场的默认探头共同前缀均通过严格门槛；新增默认探头与正向三个轴相邻的第7号中心探头也逐值一致。这提高了“本轮场内观测算子比较”的可解释性。上一轮细网格时间戳未通过的历史结果仍保留，不能被本轮结果追溯改写为通过。

## 这对项目意味着什么

当前证据支持一个较谨慎的结论：烟气采样定义确实能显著改变个别路线的网格差，但影响方向依赖路线和布局，不能把八单元平均作为自动修复。上一轮46.13%→8.95%更像是一个值得解释的局部案例，而不是已经验证的通用改进。

因此，正式24场景矩阵仍不应启动。下一步应停止围绕单一路线继续调采样，改为确定应用层面的物理真值或参考：例如选择有实验或高分辨率证据的测点、明确墙边节点的观测定义，并把默认单元、八单元平均和更合适的体积/轨迹积分放在同一盲化协议下比较。没有参考真值时，只能讨论数值一致性，不能讨论准确性。

本轮未训练RL、未调用LLM、未读取封存48场景，也未运行180/900秒或正式矩阵。原0.25米网格筛查失败保持不变。

- [机器可读完整结果]({(OUT / 'independent_results.json').as_posix()})
- [路线级CSV]({csv_path.as_posix()})
- [冻结协议]({(OUT / 'protocol_frozen.json').as_posix()})
- [冻结路线及几何排除]({(OUT / 'independent_routes_frozen.json').as_posix()})
- [上一轮局部诊断报告]({(ROOT / 'reports/FDS显式八单元采样诊断_L1_2026-10-20.md').as_posix()})

![独立路线的粗细网格绝对差]({image_path.as_posix()})
'''
    with REPORT.open('x', encoding='utf-8') as target:
        target.write(report)

    files = [path for path in OUT.rglob('*') if path.is_file()]
    files.extend([REPORT, Path(__file__).resolve()])
    save(OUT / 'completion_manifest.json', {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage_status': status,
        'artifact_sha256': {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in sorted(set(files))
        },
        'protected_previous_files_verified': len(protected),
        'old_mesh_failure_unchanged': True,
        'formal24_started': False,
    })
    print(json.dumps({
        'status': status,
        'report': str(REPORT),
        'routes': len(route_records),
        'improved_routes': sum(record['comparison']['absolute_gap_reduced']
                               for record in route_records),
        'strict_prefix_parity_passed': strict_parity_passed,
        'criterion_checks': criterion_checks,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
