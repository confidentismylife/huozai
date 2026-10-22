"""Revise only the sealed L1 diagnostic presentation; never launch simulations."""

from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from fire_evac.fds_cell_stencil import reconstruct
from fire_evac.fds_sampling_diagnostic import route_breakdown
from fire_evac.post_safestep_physics import read_device_csv

OUT = ROOT / 'outputs/post_safestep_stencil_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_manifest(path):
    entries = read(path)['artifact_sha256']
    failures = [rel for rel, expected in entries.items() if digest(ROOT / rel) != expected]
    if failures:
        raise ValueError(f'Sealed artifacts changed: {failures}')
    return len(entries)


def main():
    parent_manifest = ROOT / 'outputs/post_safestep_sampling_20261019/completion_manifest.json'
    stage_manifest = OUT / 'completion_manifest.json'
    verified = {
        str(path.relative_to(ROOT)): verify_manifest(path)
        for path in (parent_manifest, stage_manifest)
    }
    original_manifest_hashes = {str(p.relative_to(ROOT)): digest(p)
                                for p in (parent_manifest, stage_manifest)}
    result = read(OUT / 'sampling_comparison.json')
    assert result['status'] == 'COMPLETE_70S_STRICT_PREFIX_PARITY_FAILED'
    assert result['original_mesh_failure_unchanged']
    assert not result['formal24_started'] and not result['900s_started']
    assert not result['sealed_final48_accessed']
    assert result['LLM_requests'] == 0 and result['RL_updates'] == 0
    records = sorted(result['records'], key=lambda record: -record['dx_m'])
    assert len(records) == 2
    arrays = {}
    for record in records:
        assert record['run']['normal_completion']
        assert record['run']['native_end_s'] == 70
        case = record['case']
        raw, _ = read_device_csv(OUT / 'runs' / case / f'{case}_devc.csv')
        rebuilt = reconstruct(raw, record['route']['route'][:-1])
        for operator, data, prefix in [('default', raw, 'N'), ('eight_cell', rebuilt, 'S')]:
            value = route_breakdown(data, record['route'], prefix=prefix)['FED']
            np.testing.assert_allclose(value, record[operator]['FED'], rtol=1e-13, atol=0)
        arrays[record['dx_m']] = (raw, rebuilt)
    for operator in ('default', 'eight_cell'):
        medium, fine = (record[operator]['FED'] for record in records)
        comparison = result['four_dose_comparison'][operator]
        np.testing.assert_allclose(abs(medium - fine), comparison['absolute_gap_FED'], rtol=1e-13)
        np.testing.assert_allclose(abs(medium - fine) / fine, comparison['relative_gap'], rtol=1e-13)

    image_path = OUT / 'sampling_comparison_v2.png'
    report_path = ROOT / 'reports/FDS显式八单元采样诊断_L1_2026-10-20.md'
    revision_path = OUT / 'report_revision_manifest.json'
    for path in (image_path, report_path, revision_path):
        if path.exists():
            raise FileExistsError(f'Revision output already exists: {path}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(2)
    for offset, operator, label, color in [
        (-.18, 'default', '原默认单元读数', '#52657A'),
        (.18, 'eight_cell', '显式八单元重建', '#008578'),
    ]:
        values = [record[operator]['FED'] * 1e6 for record in records]
        bars = axes[0].bar(x + offset, values, width=.36, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
    axes[0].set_xticks(x, ['0.25米', '0.125米'])
    axes[0].set_ylim(0, .88)
    axes[0].set_ylabel(r'路线气体FED（单位：$10^{-6}$）')
    axes[0].set_title('L1固定路线：60秒出发，66.25秒到达')
    axes[0].legend(fontsize=8, loc='upper left')
    for dx, color in [(.25, '#155B80'), (.125, '#B04A29')]:
        raw, rebuilt = arrays[dx]
        for data, prefix, style, name in [(raw, 'N', '-', '默认'), (rebuilt, 'S', '--', '重建')]:
            mask = (data['Time'] >= 58) & (data['Time'] <= 69)
            axes[1].plot(data['Time'][mask], data[prefix + '077_CO'][mask] * 1e6,
                         linestyle=style, color=color, label=f'{dx}米 {name}')
    axes[1].set_title('节点77（[23,5]米）的局部CO读数')
    axes[1].set_xlabel('火灾时间/秒')
    axes[1].set_ylabel('CO / ppm')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.text(.5, .025, '仅一条事后选择路线；细网格严格时间戳复现未通过。',
             ha='center', fontsize=10)
    figure_bytes = io.BytesIO()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fig.tight_layout(rect=(0, .07, 1, 1))
        fig.savefig(figure_bytes, format='png', dpi=170)
    plt.close(fig)
    glyph_warnings = [str(item.message) for item in captured if 'Glyph' in str(item.message)]
    if glyph_warnings:
        raise ValueError(f'Figure has missing glyphs: {glyph_warnings}')

    original_report = ROOT / 'reports/FDS显式八单元采样诊断_L1_2026-10-19.md'
    report = original_report.read_text(encoding='utf-8')
    lead = '''
报告修订日期：2026-10-20。此次只修正图表单位字体、补充分母定义和解释，原始结果、冻结标准及2026-10-19版报告保持不变。

**本轮收获：在事后选中的L1路线中，网格间相对差从默认读数的46.13%变为八单元重建的8.95%。这说明本例对烟气采样定义敏感；尚不能认定网格已经收敛，或真实暴露误差已经降低。**

通俗地说，同一个名义位置在粗细网格里可能读到不同范围的单元。现在明确读取位置周围八个单元，先重建气体浓度，再计算这条路线的累计剂量。两个网格的结果更接近了，但平均也可能平滑峰值；没有独立真值，不能把“更接近”直接解释成“更准确”。

两套模拟都正常完成70秒。细网格新旧原生时间戳最大差约0.000000180秒，超过事先冻结的0.000000010秒门槛约17.96倍，因此严格复现仍判失败。这个差异很小，不能据此说模拟崩溃；也不能忽略它后宣布复现通过。

'''
    report = report.replace('# L1路线的显式八单元观测诊断\n',
                            '# L1路线的显式八单元观测诊断\n' + lead, 1)
    report = report.replace('有符号网格差的变化为',
        '以上两种相对差分别以各自观测定义下的0.125米细网格FED为分母。有符号网格差的变化为', 1)
    report = report.replace('相对于原绝对网格差', '相对于本轮默认观测的绝对网格差', 1)
    report = report.replace((OUT / 'sampling_comparison.png').as_posix(), image_path.as_posix())
    report += '''
## 本轮结束后的决定

本轮完成的是一项局部观测诊断，不是正式矩阵的通过凭证。原网格筛查否决、原生插值检查失败和本轮严格时间戳复现失败均保留；没有启动180/900秒或24场景矩阵。

若继续，下一项应先写清并冻结独立验证协议：固定观测算子，说明原生时间戳如何配对，论证应用相关的剂量尺度、绝对差及相对差要求，并预先选择本轮未用于诊断的路线和时间段。完成协议后再做小规模独立验证，不能利用本例结果反向调整门槛。L2及较晚出发路线尚待验证，不能提前宣告成功或失败。

本次修订从封存CSV重新计算四个剂量，核对两阶段清单中的文件哈希，并检查报告链接；没有新增火灾模拟、训练或模型请求。独立只读复核确认相对差、绝对差缩减、百分比换算和结论范围，提出的分母及对照表述建议已纳入本版。
'''
    local_links = re.findall(r'\]\(([^)]+)\)', report)
    missing = [link for link in local_links
               if re.match(r'^[A-Za-z]:/', link) and link != image_path.as_posix()
               and not Path(link).exists()]
    if missing:
        raise ValueError(f'Missing local report links: {missing}')

    # Recheck sealed files before writing only the new presentation artifacts.
    for path in (parent_manifest, stage_manifest):
        assert verify_manifest(path) == verified[str(path.relative_to(ROOT))]
        assert digest(path) == original_manifest_hashes[str(path.relative_to(ROOT))]
    with image_path.open('xb') as target:
        target.write(figure_bytes.getvalue())
    with report_path.open('x', encoding='utf-8') as target:
        target.write(report)
    artifacts = [Path(__file__).resolve(), image_path, report_path]
    revision = {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'revision_scope': 'presentation_and_explanation_only',
        'status_unchanged': result['status'],
        'sealed_manifest_sha256': original_manifest_hashes,
        'sealed_manifest_entries_verified': verified,
        'four_doses_recomputed_from_sealed_CSV': True,
        'local_report_links_checked': len(local_links),
        'missing_figure_glyph_warnings': glyph_warnings,
        'new_solver_runs': 0,
        'LLM_requests': 0,
        'RL_updates': 0,
        'sealed_final48_accessed': False,
        'artifact_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in artifacts},
    }
    with revision_path.open('x', encoding='utf-8') as target:
        json.dump(revision, target, ensure_ascii=False, indent=2)
        target.write('\n')
    print(json.dumps({'report': str(report_path), 'image': str(image_path),
                      'verified': verified, 'four_doses_recomputed': True,
                      'glyph_warnings': glyph_warnings}, ensure_ascii=False))


if __name__ == '__main__':
    main()
