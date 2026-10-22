"""Audit both completed attempts and replay any saved mesh rejection evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np

from prepare_post_safestep_checkpoint import ROOT, OLD, OUT, read, sha, save
from run_post_safestep_checkpoint import native, rates
sys.path.insert(0, str(ROOT/'src'))
from fire_evac.fds_checkpoint import score_mesh_routes


def case_data(job, result):
    folder = OUT/'runs'/job['name']
    if result['native_output_valid']:
        return native(folder, job['chid']), 'FINAL_NATIVE_CSV'
    if result['witness']:
        witness = result['witness']
        snapshot = folder/witness['parsed_native_snapshot']
        assert sha(snapshot) == witness['snapshot_sha256']
        with np.load(snapshot, allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}, 'VALID_PRETERMINATION_SNAPSHOT_RAW_CSV_INVALID'
    return None, 'NO_VALID_COMPLETE_NATIVE_TABLE'


def plot_curves(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, item in enumerate(cases):
        for data, label, color in (
            (item['candidate'], '0.25米：原完整180秒参考', '#155B80'),
            (item['data'], '0.125米：本轮实际覆盖', '#008578')):
            if data is None:
                continue
            co = np.max(np.array([v for k, v in data.items() if re.match(r'N\d+_CO$', k)]), axis=0)*1e6
            temp = np.max(np.array([v for k, v in data.items() if re.match(r'N\d+_T$', k)]), axis=0)
            axes[0, col].plot(data['Time'], co, label=label, color=color)
            axes[1, col].plot(data['Time'], temp, label=label, color=color)
        axes[0, col].set_title(item['job']['layout'])
        axes[0, col].set_ylabel('名义测点最大CO / ppm')
        axes[1, col].set_ylabel('名义测点最高温度 / ℃')
        axes[1, col].set_xlabel('火灾绝对时间 / 秒')
        for row in range(2):
            axes[row, col].set_xlim(0, 180)
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=9)
    fig.suptitle('相同物理输入下的原生观测；曲线在实际数据末端停止')
    fig.tight_layout()
    fig.savefig(OUT/'fine_mesh_native_curves.png', dpi=170)
    plt.close(fig)


def main():
    jobs = [j for j in read(OUT/'planned_jobs.json') if j['kind'] == 'fine_mesh']
    assert len(jobs) == 2
    recovery = read(OUT/'recovery_validation.json')
    assert recovery['passed']
    parity = read(OUT/'coarse_output_setting_parity.json')['records']
    protected = read(OUT/'protected_parent_hashes.json')
    for relative, expected in protected.items():
        assert sha(ROOT/relative) == expected, relative
    release = read(OUT/'fine_execution_release.json')
    for relative, expected in release['implementation_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    routes = read(OLD/'diagnostic_routes_frozen.json')
    cases, comparisons, witnessed, fine_parity = [], [], [], []
    for job in jobs:
        folder = OUT/'runs'/job['name']
        result = read(folder/'run_result.json')
        data, basis = case_data(job, result)
        candidate_name = f"{job['layout']}_F2_G150_{job['vent']}_dx0p25_adaptive"
        candidate = native(OLD/'runs'/candidate_name, candidate_name)
        if data is not None:
            score = score_mesh_routes(data['Time'], rates(data), candidate['Time'], rates(candidate), routes[job['layout']])
        else:
            score = {'planned_routes': 40, 'completed_routes': 0, 'unscored_routes': 40,
                'finite_dose_routes': 0, 'near_zero_routes': 0, 'witness': None,
                'maximum_relative_difference': None, 'scores': []}
        if result['witness']:
            w = result['witness']
            snapshot = folder/w['parsed_native_snapshot']
            assert sha(snapshot) == w['snapshot_sha256']
            with np.load(snapshot, allow_pickle=False) as archive:
                original = {k: archive[k].copy() for k in archive.files}
            replay = score_mesh_routes(original['Time'], rates(original), candidate['Time'], rates(candidate), routes[job['layout']])
            rw = replay['witness']
            assert rw is not None and rw['route'] == w['route']
            for key in ('reference_dx0p125_FED', 'candidate_dx0p25_FED', 'relative_difference'):
                assert np.isclose(rw[key], w[key], rtol=1e-12, atol=1e-14)
            xy = read(OLD/(job['layout']+'_graph.json'))['graph']['coordinates_xy']
            witnessed.append({**w, 'independent_snapshot_replay_passed': True,
                'start_xy_m': xy[w['route']['start_index']], 'exit_xy_m': xy[w['route']['exit_index']]})
        comparisons.append({'layout': job['layout'], 'data_basis': basis, **score})
        cases.append({'job': job, 'result': result, 'data': data, 'candidate': candidate, 'basis': basis})
        if data is not None:
            previous_name = f"{job['layout']}_F2_G150_{job['vent']}_dx0p125_adaptive"
            previous = native(OLD/'runs'/previous_name, previous_name)
            common_end = min(float(previous['Time'][-1]), float(data['Time'][-1]))
            t = previous['Time'][previous['Time'] < common_end-1e-6]
            if len(t) > 1:
                gas_error = max(float(np.max(np.abs(np.interp(t, previous['Time'], previous[k])-
                    np.interp(t, data['Time'], data[k])))) for k in data if re.match(r'N\d+_(CO|CO2|O2)$', k))
                temp_error = max(float(np.max(np.abs(np.interp(t, previous['Time'], previous[k])-
                    np.interp(t, data['Time'], data[k])))) for k in data if re.match(r'N\d+_T$', k))
                fine_parity.append({'layout': job['layout'], 'common_end_s': float(t[-1]),
                    'max_gas_difference_mol_per_mol': gas_error, 'max_temperature_difference_C': temp_error,
                    'basis': 'Existing incomplete adaptive case versus current checkpoint case, shared prefix only; descriptive, no new acceptance threshold'})
    route_pass = all(c['completed_routes'] == 40 and c['finite_dose_routes'] > 0
        and c['maximum_relative_difference'] <= .1 for c in comparisons)
    full_pass = route_pass and all(c['result']['normal_completion'] and c['result']['native_output_valid']
        and c['result']['native_max_gap_s'] <= 1.6 for c in cases)
    rejected = any(c['witness'] is not None for c in comparisons)
    status = 'REJECTED_COMPLETED_ROUTE_RULE' if rejected else ('PASS_180S_ONLY' if full_pass else 'INCOMPLETE_OR_FAILED')
    summary = {'status': status, 'passed_180s_screen': bool(full_pass),
        'early_witnesses_replayed': witnessed, 'comparisons': comparisons,
        'fine_attempts': [{k: v for k, v in c['result'].items() if k != 'log_tail'} for c in cases],
        'formal_24_matrix_started': False, '900s_representatives_run_this_stage': False,
        'full_900s_inputs_accepted': False, 'new_RL_updates': 0, 'LLM_requests': 0,
        'sealed_final48_accessed': False, 'protected_parent_hashes_verified': len(protected)}
    save(OUT/'mesh_results.json', summary)
    save(OUT/'fine_output_setting_parity.json', fine_parity)
    plot_curves(cases)
    control_rows = []
    for lid in ('L1', 'L2'):
        r = read(OUT/f'runs/{lid}_recovery_resumed_v2/recovery_comparison.json')
        vals = [c['relative_difference'] for c in r['point_dose_checks'] if c['relative_difference'] is not None]
        control_rows.append(f"| {lid} | {r['checkpoint_s']:.3f} | {'通过' if r['pre_checkpoint_rows_identical'] else '失败'} | "
            f"{len(r['gas_temperature_checks'])} | {len(vals)}/{len(r['point_dose_checks'])} | {max(vals)*100:.3f}% |")
    case_rows = []
    for c in cases:
        r = c['result']
        end = f"{r['native_end_s']:.3f}" if r['native_end_s'] is not None else '原始CSV无效'
        cp = ', '.join(f"{p['fire_time_s']:.3f}秒" for p in r['passive_checkpoints']
            if p['complete_passive_checkpoint_candidate']) or '无通过完整性检查的断点'
        case_rows.append(f"| {c['job']['layout']} | {end} | {'正常完成' if r['normal_completion'] else r['termination_cause'] or '见错误日志'} | "
            f"{r['elapsed_wall_s']/60:.2f} | {r['peak_rss_MiB']:.1f} | {cp} |")
    route_rows = []
    for c in comparisons:
        value = c['maximum_relative_difference']
        rel = f'{value*100:.2f}%' if value is not None else '无足够非零剂量'
        route_rows.append(f"| {c['layout']} | {c['completed_routes']}/40 | {c['finite_dose_routes']} | {c['near_zero_routes']} | {rel} |")
    witness_text = '\n'.join(f"- {w['layout']}：{w['route']['departure_s']:g}秒出发、{w['route_end_s']:.2f}秒结束的固定路线；"
        f"起点{w['start_xy_m']}米，出口{w['exit_xy_m']}米；"
        f"0.25米FED={w['candidate_dx0p25_FED']:.7g}，0.125米FED={w['reference_dx0p125_FED']:.7g}，"
        f"绝对差{w['absolute_difference']:.7g} FED、相对差{w['relative_difference']*100:.2f}%。终止前的完整原生数组已另存，重放核对通过。" for w in witnessed)
    fine_parity_text = '\n'.join(
        f"- {p['layout']}：与上一轮已有细网格输出的共同前缀至{p['common_end_s']:.2f}秒，最大气体差{p['max_gas_difference_mol_per_mol']:.3g} mol/mol，最大温度差{p['max_temperature_difference_C']:.3g}℃。"
        for p in fine_parity)
    if rejected:
        decision = '当前0.25米候选未通过既有10%路线必要条件。先诊断网格、固定点采样与路径积分的敏感性；不扩大训练、不直接开始900秒或24场景矩阵，也不通过修改预测阈值补偿物理输入不确定性。'
    elif full_pass:
        decision = '两个代表场景通过180秒初筛。下一项为900秒代表场景的时间覆盖、输出与预算检查；通过此处不等于900秒输入或正式24场景矩阵已验收。'
    else:
        decision = '本轮检查仍未完整通过。依据具体运行或数据错误继续处理；预算耗尽与已证实路线差异分别报告，不将缺失路线当作通过。'
    all_results = [read(p) for p in (OUT/'runs').glob('*/run_result.json')]
    text = f'''# 被动断点验证与细网格续项结果

日期：2026-10-19（北京时间）。本轮为新建FDS模型的运行和数值诊断，不是SafeStep原建筑独立验证，也没有产生LLM/RL控制效果。

## 本轮结论

两布局的正常周期断点恢复对照已通过；本轮细网格状态为`{status}`。

{decision}

## 1. 相比上一轮改了什么

燃料、CO/烟炱产率、壁面、几何、火源和暴露公式保持冻结设置。新建任务将DT_RESTART设为30火灾秒、DT_FLUSH设为1秒；细网格仍为0.125米、T_END=180秒、1进程×4线程。单细网格墙钟预算从上一轮1200秒改为本轮10800秒，内存上限仍为4GiB，一次只运行一个求解器。这里的预算是上限，不是完成耗时承诺。

未创建.stop文件。旧固定版本源码审计曾发现用户停止过程干扰燃烧状态，因此这里只把正常计算中生成的周期断点作为恢复来源。断点二进制逐记录检查到文件末尾，并验证本轮固定输入所需的末尾计数记录；此解析器不适用于任意FDS模型。

源码版本固定为ff928dbf68a2623ced3a007be75391c6dc6c9f90，来源定位和文件哈希见[固定版本断点源码]({(OUT/'checkpoint_source_evidence.json').as_posix()})。

本轮使用独立目录，上一轮未算完的66秒、55秒CSV没有被拼接成完整火灾。原180秒粗/中网格作为已有参考，未重新计为新火灾。

## 2. 实际完成的恢复对照

每布局先连续计算到100秒，正常保存周期断点；之后把求解器文件复制到新的恢复目录，以相同CHID从最近周期断点恢复到100秒。原参考文件不改，断点字节不编辑。仅恢复副本允许OVERWRITE=TRUE，FDS可以管理副本中的已有输出。

| 布局 | 被动断点/火灾秒 | 断点前原生记录逐值一致 | 气体/温度通道检查数 | 非近零点暴露项/全部项 | 恢复后点暴露最大相对差 |
|---|---:|---|---:|---:|---:|
{chr(10).join(control_rows)}

气体和温度按预先冻结的1%相对量加指定绝对容差检查；点暴露比较的是断点之后至100秒的短区间，非近零项要求相对差不超过1%，近零项另报绝对差。插值只用于离线配对诊断，原始输出保留实际时间戳。880个通道检查和220个点积分属于两场数值模型的内部检查，不能当作880个独立物理样本。

这些对照支持两个粗网格短时任务的恢复路径可用，不是所有网格、长时段或真实火灾都能无误恢复的保证。本轮细网格从头连续运行并周期保存，没有把粗网格断点移植给细网格。

另用缓存比较原0.5米180秒输出与本轮开启存档的0.5米100秒参考，在共同的前99秒时间戳完全一致；两布局最大气体差异为{max(p['max_gas_difference_mol_per_mol'] for p in parity):.3g} mol/mol，最大温度差异为{max(p['max_temperature_difference_C'] for p in parity):.3g}℃，接近打印精度。此诊断未新增求解器运行；它支持该范围内存档设置没有可观测扰动，不保证任何网格、任何长时段均如此。

细网格也与上一轮缓存的部分输出核对共同前缀，没有重跑旧任务或外推未覆盖时段：

{fine_parity_text or '本轮没有足够的共同细网格输出用于此项核对。'}

## 3. 两个长预算细网格的运行结果

| 布局 | 最后原生时刻/秒 | 结束状态 | 墙钟分钟 | 峰值常驻内存/MiB | 退出后通过完整性检查的被动断点 |
|---|---:|---|---:|---:|---|
{chr(10).join(case_rows)}

正常完成同时要求成功结束标志、无FDS错误、求解器末时间达到T_END及原生输出覆盖末端。容差仅用于打印精度。退出码为0或目录里有旧副本CSV，均不能独自证明恢复成功。

## 4. 网格必要条件及可复算证据

沿用上一轮预先固定的每布局40条路线：10起点、两个出口、60/120秒出发，速度0.8米/秒。只有在两套原生数据中均已走完的路线才积分。细网格FED低于1e-8时不报告百分比；其余路线要求0.25米与0.125米的相对差均不超过10%。

| 布局 | 已完整支持路线 | 非近零项 | 近零项 | 已支持非近零项最大相对差 |
|---|---:|---:|---:|---:|
{chr(10).join(route_rows)}

{witness_text or '本轮没有保存提前否决的完整路线证据；以已完成范围和统一检查状态判断。'}

只要一条既定完整路线违反必要条件，就能提前否决当前候选；不能提前宣布通过。提前结束的场景仍是部分火灾输出，不能称为完整180秒，更不能代替900秒。表中差异只描述已有支持范围，不能推广成全部路线的平均误差。

![原生观测曲线]({(OUT/'fine_mesh_native_curves.png').as_posix()})

0.125米是更细的数值参考，并非真实火灾真值。曲线差异可能包含空间分辨率、固定点采样与LES时变差异；本轮不能给出唯一因果解释。10%是原定工程初筛值，即便通过也不足以单独证明2%的控制收益稳健。

相对差的分母和绝对差必须同时解释。低暴露路线的百分比可以较大；这不是伤亡率或疏散风险变化的百分比，也不代表所有路线都有相同程度的差异。

## 4.1 已确认的观测定义问题

本轮XYZ设备未指定SPATIAL_STATISTIC。固定版本read.f90的默认值为null，dump.f90在该分支直接读取GET_IJK确定的单元状态；显式INTERPOLATION是另一种可选观测方式。因此两个网格虽然写了相同名义坐标，其默认观测支持体积并不相同。

例如名义坐标[21,4,1.5]米，在本轮均匀网格上，根据固定源码推算：0.25米单元中心为[21.125,4.125,1.625]米，0.125米单元中心为[21.0625,4.0625,1.5625]米。这里是单元支持几何的推算，不是新增测得的传感器坐标。

这项源代码证据确认了观测算子随网格改变，不能单独确定其对路线差异的贡献，也不能保证改成插值就能通过。下一轮应先分离数值场分辨率与采样定义的影响；本轮输入、路线、阈值和失败判定均未据此改动。源码定位、哈希及推算见[探头采样审计]({(OUT/'probe_sampling_audit.json').as_posix()})。

## 5. 本轮保留的问题与修正

1. L1连续参考已算完100秒后，首次驱动在打印numpy布尔值时失败。求解结果已保存，修复后显式复核并接着执行，未重复参考计算。
2. 首次L1恢复被OVERWRITE=FALSE拒绝，FDS返回码仍为0。结束标志与时间校验正确拒绝了该次结果；原失败目录保留，随后在全新恢复副本目录开启追加权限。
3. 一次独立代码审查提出4项重要问题，均先添加回归反例再修复：整记录尾部缺失、监控异常遗留子进程、损坏CSV阻断结果记录、时间覆盖判定过宽。20项针对性测试通过。未将这些实现检查称为新的疏散效果实验。

本轮共有{len(all_results)}次有记录的求解器尝试，其中{sum(r['normal_completion'] for r in all_results)}次正常完成；其中包含恢复诊断和被拒绝尝试，并非同样数量的独立火灾。没有改FDS可执行文件，未开展新训练、LLM请求或控制回合，未读取封存48场景。

## 6. 后续决定与产物

{decision}

复核{len(protected)}项父阶段保护哈希均未变化。完整状态、每条已支持路线、断点、运行日志、输入和实现修订均保留：

- [冻结计划]({(OUT/'execution_plan_frozen.json').as_posix()})
- [恢复核查]({(OUT/'recovery_validation.json').as_posix()})
- [路线与执行结果]({(OUT/'mesh_results.json').as_posix()})
- [本轮执行记录]({(OUT/'execution_progress.md').as_posix()})
- [上一轮参数与单位核对]({(ROOT/'reports/新建FDS场景_参数锁定与网格初筛结果_2026-10-19.md').as_posix()})

本模型的FED只计所选气体暴露，尚未把高温、热辐射与能见度转成通行约束。到达出口不能直接等同于安全疏散。
'''
    report = ROOT/'reports/FDS被动断点验证与细网格续项结果_2026-10-19.md'
    with report.open('x', encoding='utf-8') as file:
        file.write(text)
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    for p in [report, ROOT/'src/fire_evac/fds_checkpoint.py', ROOT/'tests/test_fds_checkpoint.py',
        ROOT/'docs/superpowers/plans/2026-10-19-fds-passive-checkpoint.md'] + list((ROOT/'scripts').glob('*post_safestep_checkpoint.py')):
        manifest[p.relative_to(ROOT).as_posix()] = sha(p)
    save(OUT/'completion_manifest.json', {'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage_status': status, 'artifact_sha256': manifest, 'protected_previous_files_verified': len(protected),
        'full_900s_inputs_accepted': False, 'formal_24_matrix_started': False})
    print(json.dumps({'report': str(report), 'status': status, 'protected_files': len(protected),
        'new_artifacts': len(manifest), 'witnesses_replayed': len(witnessed)}))


if __name__ == '__main__':
    main()
