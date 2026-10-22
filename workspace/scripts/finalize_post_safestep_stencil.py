"""Four-dose L1 comparison, with immutable input and same-field checks."""
from datetime import datetime, timezone
import re

import numpy as np

from run_post_safestep_stencil import ROOT, OUT, OLD, PARENT, A, read, sha, save, native
from fire_evac.fds_cell_stencil import reconstruct
from fire_evac.fds_sampling_diagnostic import route_breakdown
from fire_evac.post_safestep_physics import read_device_csv


def finalize():
    protocol = read(OUT/'protocol_frozen.json')
    assert read(OUT/'fixture_validation.json')['passed']
    protected = read(OUT/'protected_parent_hashes.json')
    for rel, expected in {**protected, **protocol['implementation_sha256']}.items():
        assert sha(ROOT/rel) == expected, rel
    jobs = [j for j in read(OUT/'jobs_frozen.json') if j['kind'] == 'center_observation']
    old_witness = next(w for w in read(PARENT/'mesh_results.json')['early_witnesses_replayed'] if w['layout'] == 'L1')
    records, arrays, parity = [], {}, []
    for job in jobs:
        folder = OUT/'runs'/job['name']
        if not (folder/'run_result.json').exists():
            records.append({'case': job['name'], 'status': 'NOT_STARTED_OR_NOT_FINISHED'})
            continue
        result = read(folder/'run_result.json')
        if not result['normal_completion'] or not result['native_output_valid']:
            records.append({'case': job['name'], 'status': 'INCOMPLETE_OR_INVALID', 'run': result})
            continue
        data, units = read_device_csv(folder/(job['name']+'_devc.csv'))
        nodes = job['route']['route'][:-1]
        for node in nodes:
            for corner in range(8):
                for label in ('CO', 'CO2', 'O2'):
                    assert units[f'C{node:03d}_{corner}_{label}'].strip() == 'mol/mol'
                assert units[f'C{node:03d}_{corner}_T'].strip() == 'C'
        rec = reconstruct(data, nodes)
        default = route_breakdown(data, job['route'])
        interpolated = route_breakdown(rec, job['route'], prefix='S')
        # Post-hoc accounting: high-side point versus averaging about the nominal
        # vertex. The two axis groups are not independent physical interventions.
        axis_doses = {}
        for operator, corners in [('vertical_only', [6, 7]), ('horizontal_only', [1, 3, 5, 7])]:
            partial = {'Time': data['Time']}
            for node in nodes:
                for label in ('CO', 'CO2', 'O2'):
                    partial[f'A{node:03d}_{label}'] = np.mean(
                        [data[f'C{node:03d}_{c}_{label}'] for c in corners], axis=0)
            axis_doses[operator] = route_breakdown(partial, job['route'], prefix='A')['FED']
        n, f = default['FED'], interpolated['FED']
        z, xy = axis_doses['vertical_only'], axis_doses['horizontal_only']
        axis_doses['vertical_allocation_FED'] = .5*((z-n)+(f-xy))
        axis_doses['horizontal_allocation_FED'] = .5*((xy-n)+(f-z))
        assert np.isclose(axis_doses['vertical_allocation_FED']+axis_doses['horizontal_allocation_FED'], f-n,
            rtol=1e-12, atol=1e-16)
        # At these nominal grid vertices the original containing cell is +++.
        # This extra identity check is descriptive, not a new mesh pass rule.
        identity_diffs = {'gas_mol_per_mol': 0., 'temperature_C': 0.}
        for node in nodes:
            for label in ('CO', 'CO2', 'O2', 'T'):
                delta = float(np.max(np.abs(data[f'N{node:03d}_{label}']-data[f'C{node:03d}_7_{label}'])))
                k = 'temperature_C' if label == 'T' else 'gas_mol_per_mol'
                identity_diffs[k] = max(identity_diffs[k], delta)
        parent = native(ROOT/job['parent'], job['parent_chid'])
        common_end = min(float(parent['Time'][-1]), float(data['Time'][-1]))
        t = parent['Time'][parent['Time'] < common_end-1e-6]
        new_t = data['Time'][data['Time'] < common_end-1e-6]
        max_time_error = float(np.max(np.abs(t-new_t))) if len(t) == len(new_t) else None
        time_match = max_time_error is not None and max_time_error <= 1e-8
        errors = {'gas_mol_per_mol': None, 'temperature_C': None}
        aligned_errors = {'gas_mol_per_mol': None, 'temperature_C': None}
        if len(t) == len(new_t):
            for suffix, key in [('(?:CO|CO2|O2)', 'gas_mol_per_mol'), ('T', 'temperature_C')]:
                names = [k for k in parent if re.fullmatch('N\\d+_'+suffix, k)]
                assert names
                errors[key] = max(float(np.max(np.abs(parent[k][:len(t)]-data[k][:len(t)]))) for k in names)
                aligned_errors[key] = max(float(np.max(np.abs(parent[k][:len(t)]-
                    np.interp(t, data['Time'], data[k])))) for k in names)
        lim = protocol['default_prefix_parity']
        passed = bool(time_match and errors['gas_mol_per_mol'] <= lim['gas_absolute_mol_per_mol']
            and errors['temperature_C'] <= lim['temperature_absolute_C'])
        parity.append({'dx_m': job['dx_m'], 'native_times_match': time_match,
            'max_native_time_difference_s': max_time_error,
            'common_prefix_last_s': float(t[-1]), 'native_rows': len(t),
            'maximum_default_output_differences': errors,
            'posthoc_time_aligned_differences_diagnostic_only': aligned_errors, 'passed': passed})
        old_FED = old_witness['candidate_dx0p25_FED' if job['dx_m'] == .25 else 'reference_dx0p125_FED']
        replay_diff = {'old_default_FED': old_FED, 'new_default_FED': default['FED'],
            'absolute_difference_FED': abs(default['FED']-old_FED),
            'relative_difference': abs(default['FED']-old_FED)/old_FED,
            'basis': 'Descriptive only; does not replace frozen strict native-time check'}
        records.append({'case': job['name'], 'status': 'COMPLETE_70S_DIAGNOSTIC', 'dx_m': job['dx_m'],
            'route': job['route'], 'default': default, 'eight_cell': interpolated,
            'same_containing_cell_check_max_differences': identity_diffs,
            'posthoc_axis_group_accounting': axis_doses, 'old_witness_default_dose_replay': replay_diff, 'run': result})
        arrays[job['dx_m']] = (data, rec)
    complete = len(records) == 2 and all(r['status'] == 'COMPLETE_70S_DIAGNOSTIC' for r in records)
    supported = complete and all(p['passed'] for p in parity) and all(r['run']['native_max_gap_s'] <= 1.6 for r in records)
    comparison = None
    if complete:
        medium = next(r for r in records if r['dx_m'] == .25)
        fine = next(r for r in records if r['dx_m'] == .125)
        comparison = {}
        for operator in ('default', 'eight_cell'):
            a, b = medium[operator]['FED'], fine[operator]['FED']
            comparison[operator] = {'candidate_FED': a, 'fine_FED': b, 'signed_gap_FED': a-b,
                'absolute_gap_FED': abs(a-b), 'relative_gap': abs(a-b)/b if b >= 1e-8 else None}
        comparison['signed_gap_change_FED'] = comparison['eight_cell']['signed_gap_FED']-comparison['default']['signed_gap_FED']
        comparison['absolute_gap_reduction_fraction'] = (comparison['default']['absolute_gap_FED']-comparison['eight_cell']['absolute_gap_FED'])/comparison['default']['absolute_gap_FED']
    status = ('COMPLETE_EXPLORATORY_SAMPLING_DIAGNOSTIC' if supported else
        ('COMPLETE_70S_STRICT_PREFIX_PARITY_FAILED' if complete else 'INCOMPLETE_OR_INVALID'))
    if complete and not supported:
        decision = '两网格均完成70秒，但本轮预先冻结的新旧输出严格复现门槛未通过，须单列具体差值。四剂量仍能描述本轮各场内的观测变化，不能把它升格为原失败轨迹的完全一致复现；不放宽原门槛。'
    elif not supported:
        decision = '本轮尚不能完成观测定义的配对解释；须依据运行、数据或共同前缀复现失败的具体原因判断，不能用不完整结果宣布改善。'
    elif comparison['eight_cell']['relative_gap'] is None:
        decision = '重建后的细网格剂量低于既有百分比分母下限，仅报告绝对差；不能将百分比未定义当作通过。'
    elif comparison['eight_cell']['relative_gap'] > .1:
        decision = '显式八单元重建后，这条路线的网格差异仍超过10%。这项采样替代没有消除本路线的网格敏感性，不能以改用插值作为启动正式矩阵的依据。'
    else:
        decision = '在新观测定义下，这一条路线的相对差低于10%。这支持采样定义会显著影响本例数值比较；只有一条事后路线和70秒数据，不能翻转旧筛查或宣称全部网格通过。'
    save(OUT/'sampling_comparison.json', {'status': status, 'records': records, 'same_grid_default_prefix_parity': parity,
        'four_dose_comparison': comparison, 'interpretation_decision': decision, 'original_mesh_failure_unchanged': True,
        '180s_accepted': False, '900s_started': False, 'formal24_started': False,
        'LLM_requests': 0, 'RL_updates': 0, 'sealed_final48_accessed': False,
        'selection_is_posthoc': True, 'only_L1_witness_route': True,
        'protected_previous_files_verified': len(protected)})
    fig_path = OUT/'sampling_comparison.png'
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
        plt.rcParams['font.family'] = 'Microsoft YaHei'
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        x = np.arange(2)
        for offset, operator, label, color in [(-.18, 'default', '原默认单元读数', '#52657A'),
                (.18, 'eight_cell', '显式八单元重建', '#008578')]:
            values = [medium[operator]['FED']*1e6, fine[operator]['FED']*1e6]
            bars = axes[0].bar(x+offset, values, width=.36, label=label, color=color)
            axes[0].bar_label(bars, fmt='%.4f', fontsize=9)
        axes[0].set_xticks(x, ['0.25米', '0.125米'])
        axes[0].set_ylabel('路线气体FED（×10⁻⁶）')
        axes[0].set_title('L1既定路线：60秒出发，66.25秒结束')
        axes[0].legend(fontsize=8)
        for dx, color in [(.25, '#155B80'), (.125, '#B04A29')]:
            raw, rec = arrays[dx]
            for field, prefix, style, name in [(raw, 'N', '-', '默认'), (rec, 'S', '--', '重建')]:
                mask = (field['Time'] >= 58) & (field['Time'] <= 69)
                axes[1].plot(field['Time'][mask], field[prefix+'077_CO'][mask]*1e6,
                    linestyle=style, color=color, label=f'{dx}米 {name}')
        axes[1].set_title('原差异最大路段：[23,5]米的CO')
        axes[1].set_xlabel('火灾时间/秒')
        axes[1].set_ylabel('CO / ppm')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=170)
        plt.close(fig)
    dose_rows, run_rows, parity_rows, axis_rows, replay_lines = [], [], [], [], []
    for r in records:
        if r['status'] == 'COMPLETE_70S_DIAGNOSTIC':
            dose_rows.append(f"| {r['dx_m']} | {r['default']['FED']:.8g} | {r['eight_cell']['FED']:.8g} | "
                f"{r['eight_cell']['FED']-r['default']['FED']:.8g} |")
            run_rows.append(f"| {r['dx_m']} | {r['run']['native_end_s']:.3f} | {r['run']['elapsed_wall_s']/60:.2f} | {r['run']['peak_RSS_MiB']:.1f} |")
            axes = r['posthoc_axis_group_accounting']
            axis_rows.append(f"| {r['dx_m']} | {axes['vertical_only']:.8g} | {axes['horizontal_only']:.8g} | "
                f"{axes['vertical_allocation_FED']:.8g} | {axes['horizontal_allocation_FED']:.8g} |")
            replay = r['old_witness_default_dose_replay']
            replay_lines.append(f"- {r['dx_m']}米：本轮默认路线FED与旧证据绝对差{replay['absolute_difference_FED']:.8g}，相对差{replay['relative_difference']*100:.8g}%。")
    for p in parity:
        e = p['maximum_default_output_differences']
        parity_rows.append(f"| {p['dx_m']} | {p['common_prefix_last_s']:.3f} | {p['max_native_time_difference_s']} | {e['gas_mol_per_mol']} | {e['temperature_C']} | {p['passed']} |")
    comp_text = ''
    if comparison:
        a, b = comparison['default'], comparison['eight_cell']
        pct = lambda v: f"{v*100:.4f}%" if v is not None else '剂量不足，不定义百分比'
        comp_text = f'''默认观测下，中/细网格相对差为{pct(a['relative_gap'])}，绝对差{a['absolute_gap_FED']:.8g} FED；八单元重建后相对差为{pct(b['relative_gap'])}，绝对差{b['absolute_gap_FED']:.8g} FED。有符号网格差的变化为{comparison['signed_gap_change_FED']:.8g} FED。

相对于原绝对网格差，本次绝对差的缩减比例为{comparison['absolute_gap_reduction_fraction']*100:.3f}%（负数表示扩大）。这是两种观测定义的数值差距变化，不是误差真值、伤亡变化或独立的因果贡献比例。'''
    text = f'''# L1路线的显式八单元观测诊断

本轮开始于2026-10-19，完成UTC时间为{datetime.now(timezone.utc).isoformat()}。状态：`{status}`。

{decision}

这是已见失败路线的事后探索，仅比较L1一条路线，不能代表全部40条路线、L2或真实建筑。原0.25米网格未通过筛查的结论不变。

## 为什么追加这项诊断

缓存重放确认原差异主要是CO相关项；时间插值与剂量计算顺序的改变，在现有线性插值假设下最大仅改变0.143%的路线剂量。两个原生INTERPOLATION检查又未通过：均匀气体插值探头读零、20/40℃分界读40℃。这些反例全部保留，未以改阈值方式修正。

本轮改为明确读取名义坐标周围八个单元中心，在各坐标均为网格顶点且八个支持单元都是开放气体的条件下，以1/8等权重重建气体和温度。先平均各气体，再用原冻结公式计算FED率；没有把八个FED率平均冒充同一观测。

这是独立实现的三线性顶点观测，不是宣称已修好FDS原生插值。两个小检查通过：均匀气体差为0；20/40℃分界得到30℃。空间插值仍可能平滑局部峰值，不能自动当作更准确的真实暴露。

## 配对执行与输出

| 网格/米 | 末原生火灾秒 | 墙钟分钟 | 峰值内存/MiB |
|---|---:|---:|---:|
{chr(10).join(run_rows) or '| 未完成 | — | — | — |'}

每网格只运行一个新70秒火灾，同时保留所有旧默认探头并追加160个单元中心探头。燃料、产率、几何、火源、1秒输出、1进程×4线程保持不变。新输入从零计算，不修改旧断点，不拼接CSV。没有启动L2扩展、180/900秒或24场景矩阵。

## 两种观测定义的四个剂量

固定路线[55,56,57,77,100,102]，60秒出发，66.25秒结束，速度0.8米/秒；保持原左端节点路径积分。

| 网格/米 | 默认单元FED | 八单元重建FED | 同网格重建减默认FED |
|---|---:|---:|---:|
{chr(10).join(dose_rows) or '| 未获得配对完整结果 | — | — | — |'}

{comp_text or '尚无完整且有效的四剂量比较。'}

补充的事后轴向分解使用同一批已输出单元，未增加求解器运行或通过标准。仅平均高度方向时选6、7号单元；仅平均水平方向时选1、3、5、7号单元。每种算子均先平均气体再算剂量。

| 网格/米 | 仅高度平均FED | 仅水平平均FED | 高度组分摊的剂量变化 | 水平组分摊的剂量变化 |
|---|---:|---:|---:|---:|
{chr(10).join(axis_rows) or '| 无完整结果 | — | — | — | — |'}

分摊采用两组对称平均：高度贡献=0.5×[(仅高度−默认)+(全平均−仅水平)]，水平贡献相应对换；两项之和严格等于全平均减默认。这是对观测选择的记账约定，含有非线性交互的分摊，不是高度或水平差异的独立因果效应，也不是原网格差异的真值归因。

## 新探头是否改变了原有输出

沿用本轮事先冻结的共同前缀复现阈值：原生时间戳误差<=1e-8秒，气体绝对差<=1e-9 mol/mol，温度<=1e-5℃。只比较两个文件都支持的前缀，不外推、不补行。

| 网格/米 | 共同比较至/秒 | 最大原生时间差/秒 | 最大按行气体差/mol/mol | 最大按行温度差/℃ | 严格复现通过 |
|---|---:|---|---:|---:|---|
{chr(10).join(parity_rows) or '| 无可用配对 | — | — | — | — | — |'}

另记录各名义点的默认读数与其正向三个轴邻接单元（编号7）的读数差，核对它们是否实际读取同一单元；这是一项描述性检查，不是新的网格通过标准。逐点结果及两种观测的逐段剂量见机器可读文件。

按行值差在时间未满足配对门槛时仅为描述；机器可读结果还另报按实际时间插值配对的气体、温度差，明确标为事后诊断，不用它替代严格时间戳门槛。时间戳出现极小分岔，不应描述成求解器崩溃或显著物理差异；同样不能将它静默忽略并称严格复现通过。

另作旧失败证据的默认路线剂量重放，量化实际影响，但不用它顶替严格时间戳条件：

{chr(10).join(replay_lines) or '尚无完整重放结果。'}

## 能说明什么，不能说明什么

通过小检查说明显式八单元重建在所列已知场及开放气体支持条件下符合定义；共同前缀复现用于判断新增探头是否扰动了已有输出。四剂量能量化这一次场计算中观测定义改变带来的数值差距变化。若复现未通过，就不能把网格间变化只归于观测算子。

即使新相对差小于10%，也不能翻转旧定义下的失败，更不能宣布全部网格收敛；若仍大于10%，说明这项观测替代未消除本路线的差异。需要把本条结果与其他路线、时间覆盖、LES时变和独立物理验证区分。细网格不是火灾真值；所有数值是所选气体的FED，不是伤亡概率。

这些路线剂量较低。原10%和1e-8分母下限是先前冻结的工程初筛选择，不是安全界限；本轮保留其失败判定。当前证据尚未直接测量计划中150/300/450秒出发路线受到多大影响，因此也不能据此断言全部较晚路线都不可用。任何后续新协议都应先论证应用相关的剂量尺度和数值要求，不能按本轮结果改门槛以获得通过。

本轮9项新增关键测试均观察到功能缺失时失败，随后与既有20项一起通过。两个原生小检查及两个显式重建小检查是实现诊断，不是4个独立火灾；本轮只有L1的两个网格数值场。保护{len(protected)}项先前文件，未训练RL、调用LLM或读取封存48场景。

- [四剂量、逐段与复现结果]({(OUT/'sampling_comparison.json').as_posix()})
- [重建小检查]({(OUT/'fixture_validation.json').as_posix()})
- [冻结协议]({(OUT/'protocol_frozen.json').as_posix()})
- [缓存分解及原生插值反例报告]({(ROOT/'reports/FDS路线差异分解与原生插值检查_2026-10-19.md').as_posix()})
'''
    if complete:
        text += f'\n![四剂量与局部CO]({fig_path.as_posix()})\n'
    report = ROOT/'reports/FDS显式八单元采样诊断_L1_2026-10-19.md'
    with report.open('x', encoding='utf-8') as file:
        file.write(text)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    files += [report, ROOT/'scripts/finalize_post_safestep_stencil.py']
    files += [ROOT/rel for rel in protocol['implementation_sha256']]
    save(OUT/'completion_manifest.json', {'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage_status': status, 'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))},
        'protected_previous_files_verified': len(protected), 'original_mesh_failure_unchanged': True})
    print({'report': str(report), 'status': status, 'comparison': comparison})


if __name__ == '__main__':
    finalize()
