"""Cached-data manuscript repair. No rollout, PPO training, or LLM calls.

Writes only to the v2 artifact directory; original records remain unchanged.
Sensitivity fits use the original audited numerical functions extracted by AST,
without importing the experiment runner (and its simulation/model dependencies).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/llm_ppo_paper_package_20261020'
OUT = ROOT / 'artifacts/llm_ppo_paper_v2_20261021'
DATA = OUT / 'data'
FIG = OUT / 'figures'
MECH = ROOT / 'outputs/final_mechanism_attribution_20261020'
LABELS = {'trained_llm_advice_ppo': 'LLM建议PPO', 'trained_pure_ppo': '独立纯PPO', 'trained_rule_advice_ppo': '规则建议PPO'}
ORDER = list(LABELS)
SOURCES = [MECH / 'replay_diagnostics.csv', MECH / 'prediction_dataset.csv',
           MECH / 'grouped_cv_predictions.csv', MECH / 'predictability_metrics.csv',
           MECH / 'mechanism_effects.csv',
           ROOT / 'outputs/final_pipeline_context_20261020/paired_effects.csv',
           ROOT / 'outputs/direction_capacity_decoupling_dev_20261020_v2/paired_effects.csv',
           OLD / '论文主表_统一结果.csv', OLD / '实验登记与证据等级表.csv',
           OLD / 'LLM_PPO火灾疏散控制_中文论文初稿_v1.md',
           ROOT / 'scripts/run_final_mechanism_attribution_20261020.py']
SOURCE_CODE = [ROOT / f for f in [
    'src/fire_evac/config.py', 'src/fire_evac/env.py', 'src/fire_evac/behavior.py',
    'src/fire_evac/hazard.py', 'src/fire_evac/dynamic_conflicts.py',
    'src/fire_evac/confirmatory.py', 'src/fire_evac/advice_protocol_v2.py',
    'src/fire_evac/dynamic_llm_rl.py', 'scripts/run_guarded_confirmatory_v2.py',
    'scripts/run_final_blind_evaluation_20261020.py',
    'outputs/behavior_pipeline/behavior_model.json',
    'outputs/guarded_confirmatory_v2/training_summary.csv',
    'outputs/guarded_confirmatory_v2/train_scenarios.json',
    'outputs/final_blind_evaluation_20261020/llm_cache/SCF0001.json',
    'artifacts/llm_ppo_paper_package_20261020/review/cached_evidence_audit.json']]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csv(frame, name):
    frame.to_csv(DATA / name, index=False, encoding='utf-8-sig')

def primary(frame):
    return frame[(frame.behavior_model == 'calibrated_logit') & (frame.noise_level == 'none')]

def md_table(frame):
    def cell(value):
        if pd.isna(value): return '—'
        if isinstance(value, (float, np.floating)): return f'{value:.4f}'
        return str(value)
    lines = ['|' + '|'.join(frame.columns) + '|', '|' + '|'.join(['---'] * len(frame.columns)) + '|']
    lines.extend('|' + '|'.join(cell(v) for v in row) + '|' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)

def repair_endpoints(frame):
    fixed = frame.copy()
    for percent in (50, 90):
        original = f'evacuated_{percent}_seconds'
        reached = frame.evacuation_rate >= percent / 100 - 1e-12
        fixed[f'legacy_{original}'] = frame[original]
        fixed[f'reached_{percent}'] = reached
        fixed[original] = frame[original].where(reached)
        # Every replaced value must be the known sentinel; never infer a time.
        assert np.allclose(frame.loc[~reached, original], frame.loc[~reached, 'evacuation_time_seconds'] + 5)
        assert fixed.loc[~reached, original].isna().all()
        assert (fixed.loc[reached, original] <= fixed.loc[reached, 'evacuation_time_seconds']).all()
    csv(fixed, 'replay_endpoints_corrected.csv')
    summaries = []
    # Pooled and each stratum are descriptive, not independent building samples.
    strata = [('all', fixed)] + [(f'{b}/{n}', g) for (b, n), g in fixed.groupby(['behavior_model', 'noise_level'])]
    for scope, subset in strata:
        for method, group in subset.groupby('method'):
            for percent in (50, 90):
                reached = group[f'reached_{percent}']
                summaries.append(dict(scope=scope, method=method, threshold_percent=percent,
                    episodes=len(group), reached=int(reached.sum()), not_reached=int((~reached).sum()),
                    attainment_percent=100 * reached.mean(),
                    conditional_mean_seconds=group[f'evacuated_{percent}_seconds'].mean(),
                    conditional_median_seconds=group[f'evacuated_{percent}_seconds'].median(),
                    interpretation='Descriptive only; time conditional on attainment; no censoring-model assumptions'))
    summary = pd.DataFrame(summaries)
    csv(summary, 'endpoint_attainment_summary.csv')
    pairs = []
    p = primary(fixed)
    for percent in (50, 90):
        wide = p.pivot(index=['scenario_id', 'training_seed'], columns='method', values=f'evacuated_{percent}_seconds')
        for baseline in ORDER[1:]:
            valid = wide[ORDER[0]].notna() & wide[baseline].notna()
            pairs.append(dict(candidate=ORDER[0], baseline=baseline, threshold_percent=percent,
                all_pairs=len(wide), both_attained=int(valid.sum()),
                conditional_mean_difference_seconds=(wide.loc[valid, ORDER[0]] - wide.loc[valid, baseline]).mean(),
                interpretation='Post-hoc selected successful pairs; not an unbiased full-population effect'))
    paired = pd.DataFrame(pairs)
    csv(paired, 'conditional_paired_times.csv')
    absolute = p.groupby('method', sort=False).agg(episodes=('run_id','size'),
        incapacitation_percent=('incapacitation_rate', lambda x: 100*x.mean()),
        mean_cumulative_fed=('mean_cumulative_fed','mean'),
        episode_end_seconds=('evacuation_time_seconds','mean'),
        cumulative_congestion=('cumulative_congestion','mean')).reset_index()
    csv(absolute, 'primary_absolute_outcomes.csv')
    seeds = p.groupby(['method','training_seed'])[['incapacitation_rate','mean_cumulative_fed','evacuation_time_seconds']].mean().reset_index()
    csv(seeds, 'primary_per_seed_means.csv')
    return summary, paired, absolute

def numerical_functions():
    source = SOURCES[-1]
    names = {'grouped_folds', 'fit_logistic', 'predict_logistic', 'binary_metrics', 'cluster_bootstrap_metrics'}
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == names
    namespace = dict(np=np, pd=pd, math=math, minimize=minimize, rankdata=rankdata,
                     CV_SEED=2026102023, BOOTSTRAP_SAMPLES=5000)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace

def sensitivity():
    f = numerical_functions()
    dataset = pd.read_csv(MECH / 'prediction_dataset.csv')
    old_predictions = pd.read_csv(MECH / 'grouped_cv_predictions.csv')
    observations = [c for c in dataset if c.startswith('initial_')]
    assert len(observations) == 34
    public = [c for c in observations if not c.startswith('initial_llm_')]
    assert len(public) == 29
    groups = dataset.scenario_id.to_numpy(str)
    folds = f['grouped_folds'](groups)
    results, predictions, reproduction = [], [], []
    features = {'public_observation': public, 'full_with_llm': observations}
    for baseline in ORDER[1:]:
        y = dataset[f'beneficial_vs_{baseline}'].to_numpy(int)
        for name, columns in features.items():
            for identities in (True, False):
                selected = columns + (['behavior_mixed_logit', 'noise_medium'] if identities else [])
                x = dataset[selected].to_numpy(float)
                prob, freq = np.empty(len(y)), np.empty(len(y))
                fold_ids = np.empty(len(y), dtype=int)
                for fold, (train, test) in enumerate(folds):
                    assert not set(groups[train]) & set(groups[test])
                    beta, mean, scale = f['fit_logistic'](x[train], y[train], penalty=1.0)
                    prob[test] = f['predict_logistic'](x[test], beta, mean, scale)
                    freq[test] = y[train].mean()
                    fold_ids[test] = fold
                if identities:
                    ref = old_predictions[(old_predictions.baseline == baseline) & (old_predictions.feature_set == name)]
                    aligned = dataset[['scenario_id','behavior_model','noise_level']].merge(ref,
                        on=['scenario_id','behavior_model','noise_level'], validate='one_to_one')
                    error = float(np.max(np.abs(aligned.predicted_probability - prob)))
                    assert error < 1e-8, error
                    assert np.array_equal(aligned.fold.to_numpy(), fold_ids)
                    assert np.array_equal(aligned.observed_beneficial.to_numpy(), y)
                    reproduction.append(dict(baseline=baseline, feature_set=name, maximum_probability_error=error))
                    continue
                index = len(results)
                measured = f['binary_metrics'](y, prob)
                base = f['binary_metrics'](y, freq)
                ci = f['cluster_bootstrap_metrics'](y, prob, groups, 2026102023 + index)
                improvement = 100 * (base['brier'] - measured['brier']) / base['brier']
                results.append(dict(baseline=baseline, feature_set=name, rows=len(y), scenarios=len(set(groups)),
                    feature_count=len(selected), **measured,
                    roc_auc_ci95_low=ci['roc_auc'][0], roc_auc_ci95_high=ci['roc_auc'][1],
                    prevalence_baseline_brier=base['brier'], brier_improvement_percent=improvement,
                    meets_original_reference_gate=int(ci['roc_auc'][0] > .60 and improvement >= 5),
                    status='post-hoc sensitivity; no experiment identity covariates; fixed OOF bootstrap'))
                for i in range(len(y)):
                    predictions.append(dict(scenario_id=groups[i], behavior_model=dataset.iloc[i].behavior_model,
                        noise_level=dataset.iloc[i].noise_level, baseline=baseline, feature_set=name,
                        fold=int(fold_ids[i]), observed_beneficial=int(y[i]), predicted_probability=prob[i],
                        fold_prevalence_probability=freq[i]))
    result = pd.DataFrame(results)
    csv(result, 'prediction_without_identity_metrics.csv')
    csv(pd.DataFrame(predictions), 'prediction_without_identity_oof.csv')
    csv(pd.DataFrame(reproduction), 'original_predictor_reproduction.csv')
    (DATA / 'sensitivity_feature_columns.json').write_text(json.dumps(features, indent=2), encoding='utf-8')
    return result

def update_main_table():
    table = pd.read_csv(OLD / '论文主表_统一结果.csv')
    # Withdraw contaminated time effects instead of substituting selected cases.
    table = table[~table['指标'].isin(['50%撤离时间','90%撤离时间'])].copy()
    table['比较'] = table['比较'].str.replace('LLM动态方向＋纯PPO容量', 'LLM建议PPO方向＋纯PPO容量', regex=False)
    table['指标'] = table['指标'].replace({'房间清空时间':'活动人员房间清空时间'})
    table['来源文件'] = ''
    table['筛选与分母'] = ''
    table['效应方向'] = '候选减基线；安全、暴露、时间指标负值较低；放行量不作优劣判定'
    for i, row in table.iterrows():
        level = row['证据等级']
        if '完整训练管线' in level:
            table.loc[i, ['预设门槛','是否通过','来源文件']] = ['无预设优越门槛；探索性区间解释','不适用','outputs/final_pipeline_context_20261020/paired_effects.csv']
        elif '共享控制器' in level:
            table.loc[i, '来源文件'] = 'outputs/final_blind_evaluation_20261020/paired_effects.csv'
        elif '动作级重放' in level:
            table.loc[i, '来源文件'] = 'outputs/final_mechanism_attribution_20261020/mechanism_effects.csv'
            table.loc[i, '回合或样本数'] = 240
        elif '交叉验证' in level:
            table.loc[i, '来源文件'] = 'outputs/final_mechanism_attribution_20261020/predictability_metrics.csv'
            table.loc[i, '比较'] = row['比较'].replace('公开观测','初始观测（含实验身份标签）')
            table.loc[i, '效应方向'] = 'AUC越大区分能力越强；区间重采样固定折外预测'
        elif '开发实验' in level:
            table.loc[i, '来源文件'] = 'outputs/direction_capacity_decoupling_dev_20261020_v2/paired_effects.csv'
            if '−原LLM联合控制' in row['比较']:
                rules = {'失能率': ('CI上限<+0.5百分点','未通过'), '人均累计FED':('CI上限<+0.01','未通过'),
                    '回合终止时间':('CI上限<0秒','通过'), '平均放行档位':('描述性指标','不适用')}
                table.loc[i, ['预设门槛','是否通过']] = list(rules[row['指标']])
            else:
                table.loc[i, ['预设门槛','是否通过']] = ['描述性补充比较','不适用']
        table.loc[i, '筛选与分母'] = ('60场景×5人口重复×5训练种子=1500配对单元' if '开发实验' in level
            else '48场景×2行为×2噪声=192单元；先平均5种子' if '交叉验证' in level
            else 'calibrated_logit / none；48场景×5种子=240配对单元')
    csv(table, '论文主表_修订.csv')
    registry = pd.read_csv(OLD / '实验登记与证据等级表.csv')
    for i, row in registry.iterrows():
        if row['实验'] == '收益可预测性':
            registry.loc[i, '回答的问题'] = '含实验身份标签的初始特征能否预测五种子平均的完整策略收益'
            registry.loc[i, '主要限制'] = '原模型含身份标签；本轮删除敏感性另报；不评价逐时刻覆盖'
        elif row['实验'] == '方向与容量解耦':
            registry.loc[i, '回答的问题'] = 'LLM建议PPO方向分量与纯PPO容量分量混合的闭环效果'
        elif row['实验'] == '动作机制归因':
            registry.loc[i, '主要限制'] = '原T50/T90含未达标替代值；v2已修正，不作单一因果归因'
    csv(registry, '实验登记与证据等级表_修订.csv')
    return table

def export_method_sources():
    cache = json.loads((ROOT/'outputs/final_blind_evaluation_20261020/llm_cache/SCF0001.json').read_text(encoding='utf-8'))
    (DATA/'prompt_example_SCF0001.txt').write_text(cache['prompt'],encoding='utf-8')
    extracts=[]
    for relative,names in [
        ('scripts/run_final_blind_evaluation_20261020.py', {'SCHEMA'}),
        ('src/fire_evac/dynamic_llm_rl.py', {'parse_dynamic_advice','ALLOWED_DYNAMIC_REASONS'})]:
        source=(ROOT/relative).read_text(encoding='utf-8-sig')
        for node in ast.parse(source).body:
            label=getattr(node,'name',None)
            if isinstance(node,ast.Assign):
                label=next((t.id for t in node.targets if isinstance(t,ast.Name)),None)
            if label in names: extracts.append(f'# Source: {relative}:{node.lineno}\n'+ast.get_source_segment(source,node))
    assert any('SCHEMA =' in e for e in extracts)
    (DATA/'advice_schema_and_parser.txt').write_text('\n\n'.join(extracts)+'\n',encoding='utf-8')

def setup_plotting():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 180,
                        'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)

def figures(summary, sensitivity_results):
    setup_plotting()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set(xlim=(0, 13), ylim=(0, 5)); ax.axis('off')
    boxes = [(0.3,3.15,2.3,1,'场景预测字段\n每场景生成一次'), (3.3,3.15,3,1,'冻结LLM／规则建议\n目标＋置信度'),
             (0.3,1.1,2.3,1.2,'当前环境观测\n动态矛盾特征'), (3.3,1.1,3,1.2,'34维输入 → PPO\n方向×4档放行＋等待'),
             (7,1.1,2.4,1.2,'指示保持与流量限制\n出口吞吐限制'), (10.1,1.1,2.4,1.2,'人群移动与危险更新\n下一步观测')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor='#e4eef8',edgecolor='#48647a'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center')
    for start,end in [((2.6,3.65),(3.3,3.65)),((4.8,3.15),(4.8,2.3)),((2.6,1.7),(3.3,1.7)),((6.3,1.7),(7,1.7)),((9.4,1.7),(10.1,1.7))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,color='#48647a'))
    ax.add_patch(FancyArrowPatch((2.6,3.15),(3.3,2.3),arrowstyle='-|>',mutation_scale=15,color='#48647a',linestyle='--'))
    ax.text(2.35,2.7,'原始预测',ha='center',fontsize=9)
    ax.annotate('',xy=(1.45,1.1),xytext=(11.3,1.1),arrowprops=dict(arrowstyle='->',connectionstyle='bar,fraction=-0.06',color='#48647a'))
    ax.text(6.5,4.65,'图1  最终实验实际使用的缓存建议闭环',ha='center',fontsize=15)
    ax.text(6.5,.05,'未执行实时LLM到达／过期拒绝或危险阈值硬屏蔽；PPO不具有默认安全保证。',ha='center',fontsize=10)
    save(fig,'图1_实际实验流程')

    pipe = pd.read_csv(SOURCES[5])
    fig, axes = plt.subplots(1,3,figsize=(12,3.6))
    for ax, (metric, label, scale) in zip(axes,[('incapacitation_rate','失能率差（百分点）',100),('mean_cumulative_fed','累计FED差',1),('evacuation_time_seconds','回合终止时间差（秒）',1)]):
        sub = primary(pipe)
        for j, baseline in enumerate(ORDER[1:]):
            row = sub[(sub.candidate == ORDER[0]) & (sub.baseline == baseline) & (sub.metric == metric)].iloc[0]
            m,l,h = [row[c]*scale for c in ['mean_difference','ci95_low','ci95_high']]
            ax.errorbar(m,j,xerr=[[m-l],[h-m]],fmt='o',capsize=4)
        ax.set_yticks([0,1],['对纯PPO','对规则PPO']); ax.set_xlabel(label); ax.axvline(0,ls='--',color='gray'); ax.grid(axis='x',alpha=.2)
    fig.suptitle('图2  LLM建议PPO减去各基线：主要条件差值与95%区间（探索性）')
    fig.tight_layout(); save(fig,'图2_完整管线效应')

    s = summary[(summary.scope == 'calibrated_logit/none') & (summary.threshold_percent == 90)].set_index('method').loc[ORDER]
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    axes[0].bar(range(3),s.attainment_percent,color=['#4e79a7','#59a14f','#f28e2b'])
    axes[0].set(ylim=(0,105),ylabel='达到90%撤离的回合比例（%）')
    for j, (_,row) in enumerate(s.iterrows()): axes[0].text(j,row.attainment_percent+2,f'{int(row.reached)}/{int(row.episodes)}',ha='center')
    axes[1].bar(range(3),s.conditional_mean_seconds,color=['#4e79a7','#59a14f','#f28e2b'])
    axes[1].set(ylabel='仅达标回合的T90均值（秒）')
    for ax in axes: ax.set_xticks(range(3),[LABELS[m] for m in ORDER])
    fig.suptitle('图3  分开报告达标比例与条件时间（主要条件；描述性）')
    fig.text(.5,.01,'各方法达标样本不同，右图不能作为无偏总体时间比较；未达标时间记为缺失。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96)); save(fig,'图3_撤离达标与条件时间')

    dec = pd.read_csv(SOURCES[6]); fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    specs=[('evacuation_time_seconds','时间差（秒）',1,0),('incapacitation_rate','失能率差（百分点）',100,.5),('mean_cumulative_fed','FED差',1,.01)]
    for ax,(metric,label,scale,limit) in zip(axes,specs):
        row=dec[(dec.candidate=='llm_policy_direction_pure_capacity') & (dec.baseline=='trained_llm_joint') & (dec.metric==metric)].iloc[0]
        m,l,h=[row[c]*scale for c in ['mean_difference','ci95_low','ci95_high']]
        ax.errorbar(m,0,xerr=[[m-l],[h-m]],fmt='o',capsize=5)
        ax.axvline(0,color='gray',ls='--'); ax.axvline(limit,color='#c44e52',ls=':',label=f'界值 {limit:g}')
        ax.set(yticks=[],xlabel=label,title='通过' if h<limit else '未通过'); ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle('图4  PPO方向／容量混合相对原LLM建议PPO：三个独立单位')
    fig.text(.5,.01,'旧开发场景；安全非劣未通过表示不能排除超界损失，不等于已证明危害。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94)); save(fig,'图4_解耦各指标门槛')

    old = pd.read_csv(MECH/'predictability_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,df,title in zip(axes,[old,sensitivity_results],['原模型：含实验身份标签','事后敏感性：移除身份标签']):
        m=df.roc_auc.to_numpy(); l=df.roc_auc_ci95_low.to_numpy(); h=df.roc_auc_ci95_high.to_numpy()
        ax.errorbar(m,np.arange(4),xerr=[m-l,h-m],fmt='o',capsize=4)
        ax.set_yticks(range(4),['纯PPO / 基础','纯PPO / +LLM','规则 / 基础','规则 / +LLM'])
        ax.set(xlim=(.35,.85),xlabel='ROC AUC及95%区间',title=title)
        ax.axvline(.5,color='gray',ls='--'); ax.axvline(.6,color='#c44e52',ls=':'); ax.invert_yaxis()
    fig.suptitle('图5  初始收益预测；同一场景分组8折与固定模型设定')
    fig.text(.5,.01,'区间按场景重采样固定折外预测，不含每次重拟合不确定性；仍须满足Brier条件。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94)); save(fig,'图5_预测身份标签敏感性')

def main():
    for folder in (OUT, DATA, FIG): folder.mkdir(parents=True,exist_ok=True)
    before = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES + SOURCE_CODE}
    frame=pd.read_csv(SOURCES[0]); assert len(frame)==2880
    assert not frame.duplicated(['scenario_id','training_seed','behavior_model','noise_level','method']).any()
    summary, pairs, absolute=repair_endpoints(frame)
    print('Endpoint repair complete.',flush=True)
    result=sensitivity(); print(result[['baseline','feature_set','roc_auc','brier_improvement_percent']].to_string(index=False),flush=True)
    update_main_table(); figures(summary,result); export_method_sources()
    for path in SOURCES + SOURCE_CODE: assert before[str(path.relative_to(ROOT))] == sha(path)
    report = ['# 缓存修正与低成本敏感性分析', '',
        '2026-10-21；事后修正。未新增疏散回合、PPO训练或LLM请求。只重拟合固定逻辑回归预测器。',
        '', '## 主要条件绝对结局', '',md_table(absolute), '', '## T90达标与条件时间', '',
        md_table(summary[(summary.scope=='calibrated_logit/none') & (summary.threshold_percent==90)]),
        '', '## 双方均达标的配对描述', '',md_table(pairs),
        '', '条件时间受到达标筛选影响，不作为总体无偏效应。未达标可能由失能或截止时仍未撤离造成，不能默认是独立右删失。未采用生存模型或新惩罚指标。',
        '', '## 去除实验身份标签', '',md_table(result),
        '', '使用原始192个单元、原分组8折、L2惩罚1.0和原有有益标签，去除behavior_mixed_logit及noise_medium。标准化仅拟合训练折；不搜索参数。4个原模型重拟合后逐条核对旧折外概率；结果见original_predictor_reproduction.csv。',
        '', 'AUC区间按场景重采样固定折外预测5000次，并未在每次bootstrap中重拟合。沿用旧门槛仅为参考，不赋予本轮确认性地位。移除身份字段也不意味着剩余所有预测字段已由真实传感器验证。']
    (OUT/'缓存修正结果.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    checks=dict(status='passed_scoped_checks', input_hashes_unchanged=True,source_sha256=before,
        corrected_rows=len(frame), t90_nonattainment=int((frame.evacuation_rate<.9-1e-12).sum()),
        original_ppo_training=0,new_rollouts=0,new_llm_requests=0,
        new_predictors=4,reproduced_reference_predictors=4,
        scope='endpoint and OOF consistency; not manuscript merit, citation verification, or independent validation',
        python=sys.version, pandas=pd.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__)
    (OUT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote {OUT}',flush=True)

if __name__=='__main__': main()
