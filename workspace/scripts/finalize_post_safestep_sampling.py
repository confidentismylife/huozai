"""Seal the cached diagnosis and the failed native interpolation checks."""
from datetime import datetime, timezone

from run_post_safestep_sampling import ROOT, OUT, read, save, sha


def main():
    protected = read(OUT/'protected_parent_hashes.json')
    for rel, value in protected.items():
        assert sha(ROOT/rel) == value, rel
    cached = read(OUT/'cached_route_diagnosis.json')['records']
    rows, segments = [], []
    for r in cached:
        for name, p in r['pair'].items():
            rows.append(f"| {r['layout']} | {p['dx_m']} | {p['FED']:.7g} | {p['components']['CO_base']:.7g} | "
                f"{p['components']['CO2_increment']:.7g} | {p['components']['O2_deficit']:.7g} | {100*p['gas_first_vs_frozen_rate_first_relative_difference']:.6f}% |")
        for s in r['segment_deltas']:
            segments.append(f"| {r['layout']} | {s['node']} | {s['xy_m']} | {s['candidate_minus_fine_FED']:.7g} |")
    fixtures = [read(OUT/'runs'/name/'run_result.json') for name in ('uniform_probe', 'temperature_step')]
    assert all(r['normal_completion'] for r in fixtures)
    save(OUT/'diagnosis_summary.json', {'status': 'NATIVE_INTERPOLATION_REJECTED_BY_FIXTURES',
        'cached_routes_reconstructed': 2, 'new_solver_runs': 2, 'new_fire_runs': 0,
        'uniform_fixture_pair_max_difference_mol_per_mol': .0977716807838,
        'temperature_t0_expected_default_C': 40., 'temperature_t0_expected_interpolation_C': 30.,
        'temperature_t0_observed_default_C': 40., 'temperature_t0_observed_interpolation_C': 40.,
        'planned_paired_fire_jobs_not_started': 4, 'original_mesh_failure_unchanged': True,
        'formal24_started': False, '900s_started': False, 'LLM_requests': 0, 'RL_updates': 0})
    # No fixture is relabelled as passing. The second planned fixture was run
    # separately after the first failed, solely to diagnose the same operator.
    text = f'''# 路线差异分解与原生插值探头检查

日期：2026-10-19。本轮是针对已见失败证据的事后数值诊断，没有开展控制效果实验。

已用原缓存精确重建两条失败路线；先前46.13%和37.69%的判定不变。两个很小的探头检查均正常算完，但原生空间插值的数值结果未通过预期检查，因此事先准备的四个70秒火灾任务均未启动。

## 缓存能解释什么

| 布局 | 网格/米 | 原FED | CO基础项 | CO₂放大增量 | 缺氧项 | 改为气体先插值后积分的剂量改变量 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

CO₂放大项包含与CO的交互，上表是加法记账，不是独立气体的因果贡献。缺氧项在这两条路线中为零。气体先插值采用0.025秒积分，并用0.05秒结果作步长减半核对；最大相对改变约0.143%，不足以在这项插值假设下解释原46.13%的差异。它不能恢复未记录的亚秒场，也不能证明原1秒采样充分。

| 布局 | 节点 | 坐标/米 | 0.25米减0.125米的路段FED |
|---|---:|---|---:|
{chr(10).join(segments)}

L1最大单段差异位于[23,5]米；L2位于[3,4]米。部分路段方向相反，不能仅用最高单点CO或整条路线平均浓度解释。

## 原生空间插值未通过小检查

1. 官方均匀混合气：默认探头CO约0.003242 mol/mol、CO₂约0.034313、O₂约0.097772，追加的原生插值探头三个通道均为零，三个时刻皆如此。最大差0.097772 mol/mol，未通过原1e-10容差。
2. 20/40℃分界、0.1米网格：t=0默认探头40℃，原生插值探头也为40℃，未得到已知八邻单元的30℃平均。未将观测到40℃改成新的通过标准。

固定6.11.1源码read.f90:13943附近构造插值范围，14317附近将范围经NINT转换成单元界限；dump.f90:6808附近按单元中心距离加权。这些范围与取整代码是后续追踪对象。本轮结果确认当前二进制、这些坐标和输入下的两种反例，尚未把具体内部错误唯一归因，也不声称所有坐标、平台或FDS版本都会失败。

官方[固定版本用户指南](https://github.com/firemodels/fds/blob/ff928dbf68a2623ced3a007be75391c6dc6c9f90/Manuals/FDS_User_Guide/FDS_User_Guide.tex)描述最近八个单元的线性插值；实际运行仍必须用已知值检查。原生插值不能未经核验就被当作更准确的观测。

## 执行决定

四个原生插值火灾任务均保留为未启动。先用显式八邻单元中心的默认读数重建名义坐标值，在均匀混合气及已知温度阶跃中分别验证。只有通过后，才做一条L1失败路线的两网格70秒有界诊断。它是新的探索性观测定义，不能翻转原网格否决，也不能验收180/900秒或正式24矩阵。

新增5项积分测试先失败后通过，与既有20项一起通过；后续八邻单元重建的4项测试另列，不能混作本轮原生探头已通过。复核{len(protected)}项父文件未变化。无RL训练、LLM调用或封存场景读取。

- [缓存逐段与气体分解]({(OUT/'cached_route_diagnosis.json').as_posix()})
- [原生探头检查状态]({(OUT/'fixture_validation.json').as_posix()})
- [温度反例原生CSV]({(OUT/'runs/temperature_step/temperature_step_devc.csv').as_posix()})
- [实验协议]({(ROOT/'docs/2026-10-19-fds-sampling-diagnostic.md').as_posix()})
'''
    report = ROOT/'reports/FDS路线差异分解与原生插值检查_2026-10-19.md'
    with report.open('x', encoding='utf-8') as file:
        file.write(text)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    files += [report, ROOT/'scripts/finalize_post_safestep_sampling.py']
    files += [ROOT/p for p in read(OUT/'protocol_frozen.json')['implementation_sha256']]
    save(OUT/'completion_manifest.json', {'created_UTC': datetime.now(timezone.utc).isoformat(),
        'stage_status': 'NATIVE_INTERPOLATION_REJECTED_BY_FIXTURES',
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))},
        'protected_previous_files_verified': len(protected)})
    print(report)


if __name__ == '__main__':
    main()
