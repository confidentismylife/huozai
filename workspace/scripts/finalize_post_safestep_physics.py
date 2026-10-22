"""Report finished physical-input checks, including failed and partial attempts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fire_evac.post_safestep_physics import read_device_csv
OUT = ROOT / 'outputs/post_safestep_physics_20261019'
REPORT = ROOT / 'reports/新建FDS场景_参数锁定与网格初筛结果_2026-10-19.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf-8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def plot_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, (lid, vent) in enumerate((('L1', 'V0'), ('L2', 'V1'))):
        for dx, color in ((.5, '#B66B25'), (.25, '#155B80'), (.125, '#008578')):
            name = f'{lid}_F2_G150_{vent}_dx{str(dx).replace(".", "p")}_adaptive'
            folder = OUT / 'runs' / name
            data, _ = read_device_csv(folder / (name + '_devc.csv'))
            partial = not read(folder / 'run_result.json')['normal_completion']
            label = f'{dx}米' + ('（未完成180秒）' if partial else '')
            co = np.max(np.array([v for k, v in data.items() if re.match(r'N\d+_CO$', k)]), axis=0) * 1e6
            temperature = np.max(np.array([v for k, v in data.items() if re.match(r'N\d+_T$', k)]), axis=0)
            axes[0, col].plot(data['Time'], co, color=color, label=label)
            axes[1, col].plot(data['Time'], temperature, color=color, label=label)
        axes[0, col].set_title(lid + '：相同固定观测位置')
        axes[0, col].set_ylabel('最大CO / ppm')
        axes[1, col].set_ylabel('最高温度 / ℃')
        axes[1, col].set_xlabel('火灾绝对时间 / 秒')
        for row in range(2):
            axes[row, col].set_xlim(0, 180)
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend(fontsize=9)
    fig.suptitle('物理数据初筛：曲线在实际原生数据末端停止，未向180秒外推')
    fig.tight_layout()
    fig.savefig(OUT / 'mesh_native_observation_curves.png', dpi=170)
    plt.close(fig)


def main():
    screen = read(OUT / 'mesh_screen_results.json')
    unit = read(OUT / 'dose_units_verification.json')
    chemistry = read(OUT / 'solver_chemistry_verification.json')
    assert chemistry['passed']
    for item in chemistry['checks']:
        assert sha(ROOT/item['source']) == item['source_sha256']
    params = read(OUT / 'physical_parameters_frozen.json')
    prior = read(ROOT / 'outputs/safestep_closure_plan_20261019/source_hashes.json')
    for relative, expected in prior.items():
        assert not any(word in relative.lower() for word in ('sealed', 'final48'))
        assert sha(ROOT / relative) == expected, relative
    results = [read(p) for p in (OUT / 'runs').glob('*/run_result.json')]
    started = list((OUT / 'runs').glob('*/process_start.json'))
    interrupted = list((OUT / 'runs').glob('*/interrupted_by_numeric_amendment.json'))
    sampling_checks = [item for case in screen['cases']
        for item in case['direct_solver_point_FED_checks']]
    sampling_relative = [item['relative_difference'] for item in sampling_checks
        if item['relative_difference'] is not None]
    termination_summary = [
        {'case': case['case'],
         'normal_completion': case['normal'],
         'native_end_s': case['end_s'],
         'termination_cause': read(OUT/'runs'/case['case']/'run_result.json')['termination_cause'],
         'early_route_rejection': (OUT/'runs'/case['case']/'mesh_failure_witness.json').exists(),
         'restart_files_present': [p.name for p in (OUT/'runs'/case['case']).glob('*restart*') if p.is_file()]}
        for case in screen['cases']]
    plot_curves()
    hoc = None
    header = (OUT / 'runs/L1_init/L1_init.out').read_text(errors='replace')
    match = re.search(r'Heat of Combustion \(kJ/kg\)\s*\n\s*N-HEPTANE\s+([0-9.]+)', header)
    if match:
        hoc = float(match[1])
    decision_text = (
        '本轮因细网格计算预算不足而未完成；没有完整非近零路线证据时，不能认定网格不收敛。下一轮先冻结可完成细网格的计算预算、周期检查点及运行实现，在新目录完成原定路线与180秒检查，之后检查900秒代表场景。'
        if screen['status'] == 'INCOMPLETE_RUNTIME_BUDGET' else
        ('已有预定完整路线违反10%必要条件，当前0.25米候选未通过；下一轮先检查空间采样与网格敏感性，不能直接接受正式矩阵。'
         if screen['early_failure_witnesses'] else
         '以本轮完整运行及路线检查状态决定下一步；180秒初筛即使通过，仍需完成900秒代表场景检查。'))
    status = {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'parent_structural_protocol': 'post_safestep_physical_design_20261019_v1_1',
        'P1_solver': 'PASS_VERSION_HASH_AND_STARTUP',
        'P2_parameters': 'LOCKED_TRANSFERRED_ASSUMPTIONS_NOT_NEW_EMPIRICAL_VALIDATION',
        'P3_geometry': 'PASS_TWO_REPRESENTATIVE_PARSERS_AND_GRAPH_CHECKS',
        'P4_gas_rate': 'PASS_FIXED_MIXTURE_SOLVER_CROSSCHECK',
        'P5_mesh': screen['status'],
        'P6_budget': 'MEASURED_PILOT_ONLY_900S_NOT_ACCEPTED',
        'solver_effective_heat_of_combustion_kJ_per_kg': hoc,
        'solver_chemistry_yield_checks_passed': len(chemistry['checks']),
        'solver_processes_started': len(started),
        'normal_completed_attempts': sum(r['normal_completion'] for r in results),
        'initial_dt_cap_interrupted_attempts': len(interrupted),
        'early_rejected_fine_mesh_attempts': len(screen['early_failure_witnesses']),
        'mesh_attempts': termination_summary,
        'point_sampling_check_count': len(sampling_checks),
        'point_sampling_nonzero_max_relative_difference': max(sampling_relative) if sampling_relative else None,
        'formal_24_matrix_started': False, 'full_900s_model_accepted': False,
        'policy_episodes': 0, 'new_RL_updates': 0, 'LLM_requests': 0,
        'sealed_final48_accessed': False, 'protected_previous_files_verified': len(prior),
        'next_step': decision_text,
    }
    save(OUT / 'physical_execution_status.json', status)
    table = []
    for record in screen['cases']:
        cause = read(OUT/'runs'/record['case']/'run_result.json')['termination_cause']
        reason = ('完成' if record['normal'] else
            ('完整路线触发提前否决' if record['status'].startswith('EARLY_REJECT') else str(cause or '见原始日志')))
        table.append(f"| {record['case'].split('_')[0]} | {read(OUT/'runs'/record['case']/'input_manifest.json')['mesh_spacing_m']} | "
            f"{record['end_s']:.3f} | {'正常完成' if record['normal'] else '未完成180秒'} | {reason} | {record['elapsed_wall_s']:.1f} | {record['max_HRR_kW']:.2f} |")
    comparison_table = []
    for item in screen['comparisons']:
        value = item['maximum_relative_difference']
        relative_text = f'{value*100:.2f}%' if value is not None else '无足够非零剂量项'
        comparison_table.append(f"| {item['layout']} | {item['dx_m']} 对 0.125 | {item['completed_routes']}/40 | "
            f"{item['finite_dose_routes']} | {item['near_zero_routes']} | {relative_text} | {item['routes_over_10pct']} |")
    witness_text = '\n'.join(
        f"- {w['layout']}：预先固定路线在{w['route_end_s']:.2f}秒结束，0.25米剂量{w['candidate_dx0p25_FED']:.9g}，0.125米剂量{w['reference_dx0p125_FED']:.9g}，相对差{w['relative_difference']*100:.2f}%。"
        for w in screen['early_failure_witnesses'])
    text = f'''# 新建FDS场景：参数锁定与网格初筛结果

_2026-10-19。运行范围为两个代表场景和实现核查，保留初始尝试、数值修订与未完成输出。当前结论是物理与数值输入检查，不是规划方法、LLM或RL的效果结果。_

---

## 📋 1 本轮结论

已完成燃料与产率的来源记录、CO/CO₂/O₂到暴露率的实现、两个代表布局的FDS输入和路径图，并实际运行求解器。已知浓度单位核对通过：离线累计FED为{unit['offline_accumulated_FED']:.12g}，FDS直接输出为{unit['solver_accumulated_FED']:.12g}，相对差{unit['relative_difference']:.3g}。

网格初筛状态为`{screen['status']}`。0.25米候选网格与0.125米参考网格的固定路线检查见下表。完整900秒模型尚未验收，24场景正式矩阵未启动，未训练RL或调用LLM。网格一致性、剂量代码一致性与真实火灾验证分别判断。

{decision_text}

## 📚 2 已锁定的参数与来源

| 项目 | 本轮固定值 | 来源与适用范围 |
|---|---|---|
| 求解器 | FDS-6.11.1-0-gff928db-release | 本机实际日志与可执行哈希 |
| 燃料 | 正庚烷，C₇H₁₆ | 官方用户指南WTC示例；用于新模型的迁移假设 |
| CO产率 | 0.008 kg/kg燃料 | 同一示例数值，不是本建筑实测 |
| 烟炱产率 | 0.015 kg/kg燃料 | 同一示例数值，不是通用材料值 |
| HCN | 不加入 | 当前无氮燃料的简化化学模型，不代表所有火灾没有HCN |
| 辐射份额 | 0.40 | 固定版指南正庚烷条目，仍为模型选择 |
| 壁面 | 导热率0.17 W/(m·K)，比热1.09 kJ/(kg·K)，密度650 kg/m³，厚0.013m | 官方ATF走廊例的石膏参数 |
| 壁面简化 | 全部固体表面统一衬里，背面绝热 | 本研究假设，未复制ATF整套实验 |
| 环境 | 20℃、相对湿度40% | 研究者指定 |

求解器报告的有效燃烧热为{hoc if hoc is not None else '见原始out文件'} kJ/kg，来自固定版本内置热化学数据与指定产率的计算。名义火源为1平方米、1000kW、150秒t²增长至峰值；几何与两种通风保持设计定义。参数来源、URL和源文件SHA256见[冻结参数](../outputs/post_safestep_physics_20261019/physical_parameters_frozen.json)。官方示例的验证结论不能随参数迁移到自建布局。[^1][^2]

原始启动日志保留了“建议用SPEC定义REAC燃料化学”的警告；当前采用指南示例的REAC元素写法，初始化与实际输出均已检查。该警告不等于本轮求解报错，后续若改写输入仍应单独记录版本并核查化学组分，不能悄悄替换已冻结输入。

另从两个初始化输出中的反应摩尔系数和分子量反算CO、烟炱的质量产率，4项核对均落在输出小数舍入所允许的范围内，与冻结的0.008和0.015 kg/kg一致。此项确认求解器实际采用的产率，仍不等于材料实测验证，记录见[求解器化学核对](../outputs/post_safestep_physics_20261019/solver_chemistry_verification.json)。

## 🧪 3 气体单位与暴露率

原始输入为湿基摩尔分数mol/mol。CO乘10⁶变ppm，CO₂和O₂乘100变体积百分数。采用固定版本源码中轻体力活动的CO系数2.7641667×10⁻⁵及指数1.036，CO₂作通气增强因子，O₂仅在摩尔分数小于0.20时加入低氧项，最后将每分钟率除以60得到每秒速率。HCN、NOx和刺激性组分未包括在本次所选化学模型中。[^3]

先在各原生时刻由浓度计算局部率，再对分段线性率沿实际路径驻留区间积分；不外推原生数据范围，也不把固定测点自起火累计的FED沿路线相加。该定义与SafeStep旧FER不是自动相同的绝对量。

本轮还输出温度和消光系数，但没有把它们转换成高温、热辐射或能见度的通行约束。因此这里的FED只表示所列气体的累计暴露；即使某条路线到达出口，也不能仅据此认定其在火灾中可生存或安全。

6项针对性实现检查通过。另运行一个由官方已知混合气体例派生的均匀浓度核查，比较FDS直接累计值和离线值，结果如第一节。它核对公式和秒/分钟，不校准实际材料或人体风险。完整记录见[单位核查](../outputs/post_safestep_physics_20261019/dose_units_verification.json)。[^4]

动态火灾另在每个网格的4个固定测点对照FDS直接累计FED，共{len(sampling_checks)}项，其中非近零项的最大相对差为{max(sampling_relative)*100 if sampling_relative else float('nan'):.3f}%。这是原生输出时间采样的近似误差；部分细网格只比较其已有时间范围，不能把它解释为完整180秒误差界。原始逐点值保留在[网格检查记录](../outputs/post_safestep_physics_20261019/mesh_screen_results.json)。

## 🧱 4 几何、开口与时间采样

两个代表场景分别为L1_F2_G150_V0和L2_F2_G150_V1。输入与导航图共用开放矩形；火源1平方米占地从导航图排除。每个布局的10个规定起点均可连接两个出口；吸收出口节点不当作室外浓度测点。FDS两次初始化均通过，未修改既有酒店案例。

![L1几何与路径图](../outputs/post_safestep_physics_20261019/L1_geometry_and_graph.png)

![L2几何与路径图](../outputs/post_safestep_physics_20261019/L2_geometry_and_graph.png)

目标DT_DEVC=1秒并不保证CSV严格落在整数秒。保留实际采集时间，在诊断中核对实际最大间隔；之后的在线延迟逻辑也必须以这些时刻计算送达，不重新标注成整数秒。离线曲线比较允许使用已生成的完整数据，不能据此给在线策略提供未来端点。

## 📊 5 三种网格的实际结果

| 布局 | 网格/米 | 最后原生时刻/秒 | 运行状态 | 完成或中止原因 | 实际耗时/秒 | 当前输出最大HRR/kW |
|---|---:|---:|---|---|---:|---:|
{chr(10).join(table)}

初始输入显式DT=0.01，且默认不允许时间步超过初值。一个粗网格已完整运行，另一个尝试在数值设置修订时中断并保留。修订后由FDS自动选取初始时间步并按CFL调整，燃料、产率、几何和剂量公式未变化。正式网格比较使用后缀adaptive的输入，初始尝试不混入比较。详见[数值修订](../outputs/post_safestep_physics_20261019/adaptive_timestep_amendment.json)。[^1]

每布局事先固定10起点到两出口的最短合法路线，分别60秒和120秒出发，共40个诊断积分。它们是数值检查，不是新策略训练或闭环控制回合。要求有足够剂量的这些路线在0.25与0.125米间均相差不超过10%，近零路线另列。

| 布局 | 比较网格/米 | 已有完整路径的比较数 | 非近零项 | 近零项 | 非近零项最大相对差 | 超10%的项数 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_table)}

若尚未运行至180秒，但某条预先固定路线已经在两套原生数据中完整结束，且违反上述必要条件，可提前判定该网格筛查不通过。此规则在任何0.125米结果生成前记录，只能提前否决，不能提前宣布通过。中止的细网格数据仍属于未完成输出，不能冒充完整180秒或900秒火灾。

{witness_text or '本轮没有触发提前否决的完整路线证据；结果以全时段记录为准。'}

![网格原生观测曲线](../outputs/post_safestep_physics_20261019/mesh_native_observation_curves.png)

图中曲线在各自最后原生时间结束，没有补齐未运行部分。0.125米仅为更细参考，未宣称它就是真值。差异可能包含空间分辨率、点采样、非线性剂量及LES时变差异，不能从一次网格比较给出唯一因果解释。数值规则的10%是工程筛查选择，不是物理或安全认证。

若仅因1200秒计算预算耗尽而中止，应判为未完成，不能据此说网格已不收敛；若已有完整预定路线违反规则，则该条证据足以否决当前候选，但其他尚未完成的路线仍不评分。即使日后通过10%初筛，也不能仅凭该结果解释2%的控制差异；还需检查方法比较对数值与采样选择是否稳健。

## ✍️ 6 后续决定与证据保留

当前尚未接受完整900秒输入。正式矩阵继续保持未启动，先处理代表场景的网格/采样一致性，再检查长时段与数据支撑范围；不能通过调整预测门槛来弥补物理输入不确定性。如果当前0.25米未通过，应报告这个结果，而不能称其已经足以比较2%的控制收益。

{decision_text}

本轮DT_RESTART设为100000秒，没有为20分钟预算配置有效的周期检查点。中止任务是否有可用重启文件逐项列在执行状态中；不能把部分CSV当作断点存档直接接成完整火灾。下一轮长任务应先验证检查点写出和恢复流程，保留本轮尝试并使用新目录。

本轮实际启动求解器{len(started)}次，正常完成{status['normal_completed_attempts']}次；初始DT设置下保留{len(interrupted)}次中断，细网格提前否决{len(screen['early_failure_witnesses'])}次。正常次数包含均匀浓度与初始化检查，不代表同样数量的完整火灾。无新训练、LLM请求或控制效果回合；48个既有最终人群场景未读取。

再次核对{len(prior)}项旧文件哈希均未变化。新输出、输入、来源、修订与运行状态见[本轮清单](../outputs/post_safestep_physics_20261019/completion_manifest.json)和[执行状态](../outputs/post_safestep_physics_20261019/physical_execution_status.json)。该报告可以支持下一轮数值决策，不能作为独立真人或真实建筑验证。

## 🔗 7 官方来源

[^1]: Firemodels. [FDS6.11.1固定用户指南](https://github.com/firemodels/fds/blob/FDS-6.11.1/Manuals/FDS_User_Guide/FDS_User_Guide.tex)，访问2026-10-19。核查WTC反应示例、正庚烷辐射条目、初始时间步和瞬时输出约定。
[^2]: Firemodels. [ATF_Corridors_240_kW固定输入](https://github.com/firemodels/fds/blob/FDS-6.11.1/Validation/ATF_Corridors/FDS_Input_Files/ATF_Corridors_240_kW.fds)，访问2026-10-19。仅迁移所列材料参数。
[^3]: Firemodels. [FDS6.11.1 func.f90](https://github.com/firemodels/fds/blob/FDS-6.11.1/Source/func.f90)，访问2026-10-19，FUNCTION FED。以本轮下载SHA256固定的原文为实现定位，避免网页缓存行号差异。
[^4]: Firemodels. [FED_FIC_SMIX固定核查输入](https://github.com/firemodels/fds/blob/FDS-6.11.1/Verification/Species/FED_FIC_SMIX.fds)，访问2026-10-19。仅采用其中CO/O₂/CO₂混合气作为实现核查，不采用其材料值作为火灾参数。
'''
    text = text.replace('../outputs/', ROOT.as_posix() + '/outputs/')
    with REPORT.open('x', encoding='utf-8') as file:
        file.write(text)
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    for path in (REPORT, ROOT/'src/fire_evac/post_safestep_physics.py', ROOT/'tests/test_post_safestep_physics.py'):
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in sorted((ROOT/'scripts').glob('*post_safestep*.py')):
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    save(OUT / 'completion_manifest.json', {'status': 'COMPLETE_PHYSICAL_PREFLIGHT_DECISION',
        'artifact_sha256': artifacts, 'stage_status': status,
        'verification_scope': 'Numerical input, units, outputs and grid screen only; no empirical validation or policy improvement claim'})
    print(json.dumps({'report': str(REPORT), 'status': screen['status'], 'artifact_files': len(artifacts),
        'old_files_unchanged': len(prior), 'FDS_started': len(started)}))


if __name__ == '__main__':
    main()
