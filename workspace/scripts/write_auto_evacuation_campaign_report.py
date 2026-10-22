"""Report executed and skipped stages without promoting development evidence."""
from pathlib import Path
import numpy as np
from auto_evacuation_campaign import ROOT,OUT,REPORT,PROTOCOL,read,sha,write


def link(label,path):return f'[{label}]({path.resolve().as_posix()})'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def write_report(reason):
    m=read(OUT/'manifest.json');mechanism=read(OUT/'mechanism_summary.json');budget=read(OUT/'budget.json');v=read(OUT/'verification.json');assert v['all_checks_passed']
    sections=['# 疏散自动实验阶段总报告\n\n日期：2026-10-05。用户已授权整套自动实验无需逐步确认。',
        '## 1. 运行结果与范围\n\n本报告是预先固定条件流程的开发研究结果，不是原门控资格认证或现实火灾效能验证。完成原因：`'+reason+'`。\n\n'
        f"累计启动完整模拟{budget['simulations_started']}条，完成{budget['simulations_completed']}条；其中机制科学模拟1536条，仪器一致性检查2条。矩阵学生拟合{budget['matrix_fits']}次、均值参考{budget['mean_reference_estimates']}项、PPO更新0、LLM请求{budget['llm_requests']}次。固定最大模拟预算44738条不是必须消耗的目标。\n\n"+
        link('固定总方案',PROTOCOL)+'；'+link('逐步预算',OUT/'budget.json')+'；'+link('独立验证',OUT/'verification.json')+'。',
        '## 2. 匹配人口的机制核查\n\n固定601和603，北/南两个火源世界，两个人口，每人口六未来。所有组合匹配人口及未来块，组合顺序随机化。先192条容量组×火势时序组对照，再补全五变量32组合；没有修改行为模型、危险过程或PPO参数。新增详细轨迹只作为仪器记录，不参与控制。\n\n'+
        table(['组组合','601平均等待请求','603平均等待请求','R收益(百分点)','M收益(百分点)','人口0 M','人口1 M'],[
            [r['cell'],f"{r['wait601']:.4f}",f"{r['wait603']:.4f}",f"{r['R_gain_pp']:+.6f}",f"{r['M_gain_pp']:+.6f}",f"{r['population0_M_pp']:+.6f}",f"{r['population1_M_pp']:+.6f}"] for r in mechanism['group192'].values()])+
        f"\n\n完整32组合中，两个人口均支持正M的组合{mechanism['positive_cells']}个，两个人口均支持负M的组合{mechanism['negative_cells']}个。学生阶段推进条件为同时存在这两类组合；本次判定{mechanism['proceed_to_students']}。两个固定人口与其六未来不是32个独立抽样场景；组合响应只适用于这两个参数端点组成的网格。\n\n"+
        table(['变量从训练端换到留出端','16个配置平均M变化','最小变化','最大变化','严格翻转数/16'],[
            [r['factor'],f"{r['mean_M_change_pp']:+.6f}",f"{r['min_M_change_pp']:+.6f}",f"{r['max_M_change_pp']:+.6f}",r['strict_sign_flips_of16']] for r in mechanism['factor_effects']])+
        '\n\n每个单变量的平均对比是在其他四变量16配置上完整计算；最小/最大展示效应随背景变化。它不是现实危险因素的贡献百分比。全部31项因子与交互都另存，避免只报告最大因素。相同种子不保证策略改变后每个行人使用同一抽样数，不能解释成逐人行为完全匹配。\n\n'+
        link('全部32组合',OUT/'mechanism_cells.csv')+'；'+link('所有主效应和交互',OUT/'mechanism_all_interactions.csv')+'；'+link('1536回合的过程摘要',OUT/'mechanism_trace_episodes.csv')+'。']
    plot_mechanism(mechanism)
    sections.append('![机制响应与因子变化]('+ (OUT/'mechanism_response.png').resolve().as_posix()+')')
    if (OUT/'student_summary.json').exists():
        s=read(OUT/'student_summary.json');g=read(OUT/'frozen_gates.json')
        sections.append('## 3. 全新场景的学生对照\n\n训练60、校准24、留出48家庭，全部场景签名与历史银行分离。四次拟合包括一个直接M核模型和三个家庭bootstrap种子的绝对世界损失模型，后者平均为主模型；它们不是新PPO训练种子。初始选择一次后，冻结PPO闭环执行9动作。\n\n'+
            table(['臂','失能差(百分点，负值更好)','按家庭95%区间','接管家庭','改善家庭','变差家庭'],[
                [r['arm'],f"{r['mean_delta_pp']:+.6f}",f"[{r['ci95_pp'][0]:+.6f}, {r['ci95_pp'][1]:+.6f}]",r['overrides'],r['improved_families'],r['harmed_families']] for r in s['all_arms']])+
            '\n\n主模型是absolute_margin相对601；三个单种子只作固定诊断，不从留出中择优。模型实际收益预测MAE：\n\n'+table(['模型','MAE(百分点)'],[[k,f'{v:.6f}'] for k,v in s['actual_gain_MAE_pp'].items()])+
            '\n\n校准余量(百分点)：'+json_margins(g)+f"。留出有益候选共{s['actual_beneficial_candidate_count']}个；候选机会召回另存摘要，无正例时为null、无接管时接管精度为null。\n\n"+
            f"固定推进判据：至少一次接管、主失能差区间上界<0、实际收益MAE低于零预测、变差家庭<=5%；结果为{s['proceed_to_robustness']}。这是开发价值检查，不表示95%/10%/5%/95%原资格标准通过。\n\n"+
            link('学生完整结果',OUT/'student_summary.json')+'；'+link('每家庭选择与结局',OUT/'student_family_results.csv')+'。')
    else:sections.append('## 3. 学生阶段未执行\n\n机制阶段没有满足预先固定的稳健收益转折条件，因而没有增加学生拟合或后续场景模拟。保留机制负结果，不改变条件重新寻找阳性。')
    if (OUT/'robustness_summary.json').exists():
        r=read(OUT/'robustness_summary.json');sections.append('## 4. 行为与观测变化\n\n冻结模型与余量，分别改变行为温度和light/medium观测噪声。所有选择先于该条件独立实际结局冻结；含噪两人口可出现不同初始选择，不借用真实火源判断。\n\n'+
            table(['条件','主失能差(百分点)','95%区间','变差家庭','接管人口数'],[[x['condition'],f"{x['mean_delta_pp']:+.6f}",str(x['ci95_pp']),x['harmed_families'],x['overrides']] for x in r['conditions']])+
            f"\n\n继续LLM阶段判定：{r['proceed_to_llm']}。该温度变化只是局部敏感性检查，不覆盖地图/FDS或真实行为分布。")
    else:sections.append('## 4. 行为与噪声阶段未执行\n\n前序固定推进条件未通过，本分支按方案停止，不把跳过说成泛化通过。')
    if (OUT/'llm_summary.json').exists():
        r=read(OUT/'llm_summary.json');sections.append('## 5. 本地真实LLM建议对照\n\n另48家庭，真实qwen3:4b建议在全部独立实际结局前冻结；五个PPO训练种子601—605、四臂、同9动作与物理信息。主归因对照固定raw_llm控制器，只更换LLM/规则建议；其他跨训练臂对照为次要。LLM推理成本单列，不能声称与规则计算成本相同。\n\n'+
            f"主失能差{r['primary_mean_delta_pp']:+.6f}个百分点，按家庭区间{r['primary_ci95_pp']}；各训练种子差{r['per_training_seed_delta_pp']}；固定改善信号判定{r['improvement_signal']}。这不等于现实火灾效能或原门控资格通过。\n\n"+link('LLM完整摘要',OUT/'llm_summary.json')+'。')
    else:sections.append('## 5. 真实LLM对照未执行\n\n前序证据条件未满足，因而未增加真实LLM调用或LLM控制比较；当前流程不能建立LLM独立贡献。Ollama可用不构成研究有效性的证据。')
    sections.append('## 6. 保全、局限和下一步\n\n所有已完成缓存单元均保存哈希，失败、中断标记与原始响应保留；旧模型、报告和旧状态字节继续保全。仪器一致性检查与原模拟帮助函数的11项结局完全一致，详细轨迹的人数守恒、失能/撤离、FED增量、放行容量和等待字段由独立验证检查。\n\n'
        '当前流程不追加PPO训练，不宣称跨地图或现实火灾效能，也不以初始选择冒充运行中动态切换/LLM纠正。已看留出只能继续作为开发诊断，不能作为修改后的最终评估。负结果应指导下一份独立方案，不能降低余量继续刷同一银行。\n\n'
        '停止原因与各阶段推进条件均已保存。若学生阶段失败，下一步首先查训练支持、绝对损失预测偏差与校准余量，而不是直接扩大PPO训练；若行为/噪声失败，则先解决相应观测/行为变化；若LLM无优势，则保留规则对照并报告没有独立贡献。\n\n'+link('运行状态',OUT/'campaign_status.json')+'；'+link('总清单',OUT/'manifest.json')+'。')
    REPORT.write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    write(OUT/'report_assets_manifest.json',{'report_sha256':sha(REPORT),'writer_sha256':sha(__file__),'verification_sha256':sha(OUT/'verification.json'),
        'figure_sha256':{f'mechanism_response.{ext}':sha(OUT/f'mechanism_response.{ext}') for ext in ('png','svg','pdf')},
        'figure_visual_inspection_pending':True})


def json_margins(g):return ', '.join(f'{k}={v*100:.6f}' for k,v in g['margins'].items())


def plot_mechanism(m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5));rows=m['all_cells'];x=np.arange(32)
    axes[0].bar(x,[r['M_gain_pp'] for r in rows],color=['#0072B2' if r['M_gain_pp']>0 else '#D55E00' for r in rows]);axes[0].axhline(0,color='black',lw=.7)
    axes[0].set_xlabel('Five-factor cell index (binary levels)');axes[0].set_ylabel('601 minus603 incapacitation (pp)');axes[0].set_title('Matched-population worst-world gains')
    names=['North capacity','South capacity','Growth rate','Onset step','Background hazard'];effects=m['factor_effects'];yy=np.arange(5)
    mean=np.array([r['mean_M_change_pp'] for r in effects]);low=np.array([r['min_M_change_pp'] for r in effects]);high=np.array([r['max_M_change_pp'] for r in effects])
    axes[1].errorbar(mean,yy,xerr=np.array([mean-low,high-mean]),fmt='o',color='#0072B2',capsize=3);axes[1].set_yticks(yy,names);axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.7)
    axes[1].set_xlabel('M change, training to heldout parameter (pp)');axes[1].set_title('Mean and range across16 other-factor settings')
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'mechanism_response.{ext}',dpi=180)
    plt.close(fig)
