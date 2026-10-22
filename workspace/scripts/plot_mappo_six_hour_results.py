"""General group-meeting figures with paired seeds and source tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'mappo_six_hour_extension_20261022'
ORDER=['mappo_34_raw','mappo_50_masked_control','mappo_50_rule_plan',
       'mappo_50_deepseek_retrieval','mappo_50_deepseek_expanded']
LABELS=['Raw 34D','Masked 50D','Rule plan','Old DS library','New DS library']
COLORS=['#333333','#777777','#0072B2','#D55E00','#009E73']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    source=OUT/('progress_analysis' if args.partial else 'analysis')
    target=source/'figures';target.mkdir(parents=True,exist_ok=True)
    analysis=json.loads((source/'analysis.json').read_text(encoding='utf-8'))
    records=[]
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.titlesize':11}):
        # Show every training-seed difference; CI uses crossed seed/scenario bootstrap.
        keys=list(analysis['primary_effects'])
        fig,ax=plt.subplots(figsize=(8.4,3.2),layout='constrained')
        for i,key in enumerate(keys):
            value=analysis['primary_effects'][key]['incapacitation_rate']
            points=np.array(list(value['by_training_seed'].values()))*100
            offset=np.linspace(-.1,.1,len(points))
            ax.scatter(points,i+offset,s=23,facecolors='none',edgecolors='#555555',label='Training seeds' if i==0 else None)
            lo,hi=np.array(value['crossed_ci95'])*100;mean=value['mean']*100
            ax.errorbar(mean,i+.23,xerr=[[mean-lo],[hi-mean]],fmt='D',color='#0072B2',capsize=4,
                        label='Mean / crossed 95% bootstrap CI' if i==0 else None)
        ax.axvline(0,color='black',linestyle='--',linewidth=.8)
        ax.set(yticks=range(len(keys)),yticklabels=['DS minus Raw','DS minus Rule'][:len(keys)],
               xlabel='Incapacitation difference (percentage points; negative favors DS)',
               title=f'Primary development confirmation: {len(analysis["primary_training_seeds"])} training seeds')
        ax.legend(fontsize=8,loc='best');ax.grid(axis='x',alpha=.2)
        records.append(save(fig,target/'primary_paired_effects',[source/'analysis.json']))
        mechanism_path=source/'mechanism_episode_results.csv'
        if mechanism_path.exists():
            data=pd.read_csv(mechanism_path)
            byseed=data.groupby(['config_id','training_seed','arm'],as_index=False).incapacitation_rate.mean()
            byseed.to_csv(target/'mechanism_seed_source_data.csv',index=False,encoding='utf-8-sig')
            configs=sorted(byseed.config_id.unique())
            fig,axes=plt.subplots(1,len(configs),figsize=(11,4),sharey=True,layout='constrained',squeeze=False)
            for ax,cid in zip(axes[0],configs):
                grouped=byseed[byseed.config_id==cid]
                for index,(arm,color) in enumerate(zip(ORDER,COLORS)):
                    vals=100*grouped[grouped.arm==arm].incapacitation_rate.to_numpy()
                    if not len(vals):continue
                    ax.scatter(index+np.linspace(-.1,.1,len(vals)),vals,color=color,s=30,marker='o',alpha=.8)
                    ax.scatter(index,vals.mean(),color='black',s=40,marker='D',zorder=4)
                title={'e003_i002':'Weak plan supervision (0.02)',
                       'e003_i005':'Stronger plan supervision (0.05)'}.get(cid,cid)
                ax.set(xticks=range(len(ORDER)),xticklabels=LABELS,title=title,ylim=(0,None))
                ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
            axes[0,0].set_ylabel('Incapacitation (%)')
            fig.suptitle('Exploratory mechanism: circles = training seeds; diamonds = means')
            records.append(save(fig,target/'mechanism_seed_results',[mechanism_path]))
        diagnostic_path=source/'actor_checkpoint_diagnostics.csv'
        if diagnostic_path.exists():
            history=pd.read_csv(diagnostic_path)
            history=history[history.stage=='new_mechanism']
            configs=sorted(c for c in history.config_id.unique() if c.startswith('e003_'))
            if configs:
                fig,axes=plt.subplots(2,len(configs),figsize=(11,6.8),sharex=True,layout='constrained',squeeze=False)
                for column,cid in enumerate(configs):
                    data=history[history.config_id.isin([cid,'shared_raw','shared_masked'])]
                    for index,(arm,color) in enumerate(zip(ORDER,COLORS)):
                        subset=data[data.arm==arm]
                        if subset.empty:continue
                        grouped=subset.groupby('timesteps').agg(
                            mean_rate=('validation_incapacitation_12','mean'),
                            min_rate=('validation_incapacitation_12','min'),
                            max_rate=('validation_incapacitation_12','max'),
                            mean_entropy=('flow_entropy','mean'))
                        x=grouped.index.to_numpy()/1000
                        rate=grouped.mean_rate.to_numpy()*100
                        marker=['o','s','^','D','P'][index]
                        axes[0,column].plot(x,rate,color=color,marker=marker,markersize=3,label=LABELS[index])
                        axes[0,column].fill_between(x,grouped.min_rate.to_numpy()*100,grouped.max_rate.to_numpy()*100,color=color,alpha=.07)
                        axes[0,column].scatter(subset.timesteps/1000,subset.validation_incapacitation_12*100,color=color,s=9,alpha=.18)
                        axes[1,column].scatter(subset.timesteps/1000,subset.flow_entropy,color=color,s=9,alpha=.18)
                        axes[1,column].plot(x,grouped.mean_entropy,color=color,marker=marker,markersize=3)
                    axes[0,column].set(title=cid,ylim=(0,None))
                    axes[0,column].grid(axis='y',alpha=.2)
                    axes[1,column].axhline(np.log(4),color='#555555',linestyle='--',linewidth=.8)
                    axes[1,column].set(xlabel='Training steps (thousands)',ylim=(0,np.log(4)*1.06))
                    axes[1,column].grid(axis='y',alpha=.2)
                axes[0,0].set_ylabel('12-episode validation incap. (%)')
                axes[1,0].set_ylabel('Flow actor entropy (training update)')
                common_top=max(ax.get_ylim()[1] for ax in axes[0])
                for ax in axes[0]:ax.set_ylim(0,common_top)
                axes[0,0].legend(fontsize=7,ncol=2,loc='upper right')
                fig.suptitle('Checkpoint diagnostics: means, training-seed ranges and individual seed points')
                records.append(save(fig,target/'actor_checkpoint_diagnostics',[diagnostic_path]))
        latency_path=source/'timing_return_check_results.csv'
        if latency_path.exists():
            data=pd.read_csv(latency_path)
            variants=['native','rule','shuffled','masked','instant_unchecked','instant_checked',
                      'delayed_unchecked','delayed_checked']
            labels=['Cached','Rule','Shuffled','Masked','API immediate','API + check',
                    'API delayed','Delay + check']
            fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
            for ax,metric,factor,ylabel in [(axes[0],'incapacitation_rate',100,'Incapacitation (%)'),
                (axes[1],'evacuation_time_seconds',1,'Evacuation time (s)')]:
                for scenario,g in data.groupby('scenario_id'):
                    values=g.set_index('provider_variant')[metric].reindex(variants)*factor
                    ax.plot(range(len(variants)),values,color='#999999',alpha=.45,linewidth=.7,marker='o',markersize=3)
                means=data.groupby('provider_variant')[metric].mean().reindex(variants)*factor
                ax.plot(range(len(variants)),means,color='black',linewidth=1.5,marker='D',markersize=5)
                ax.set(xticks=range(len(variants)),xticklabels=labels,ylabel=ylabel,ylim=(0,None))
                ax.tick_params(axis='x',rotation=38);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Timing / return check: one fixed policy, six paired development scenarios')
            records.append(save(fig,target/'timing_return_check',[latency_path]))
    (target/'figure_manifest.json').write_text(json.dumps({
        'purpose':'general internal group-meeting figures, not journal-specific validation',
        'partial':args.partial,'figures':records,'matplotlib_version':matplotlib.__version__,
        'encoding':'all seed/scenario points shown; no data-driven omissions or significance stars',
        'raw_baselines_reused_across_mechanism_panels':True},indent=2),encoding='utf-8')
    print(json.dumps({'figure_count':len(records),'directory':str(target)}))


def save(fig,path,sources):
    fig.savefig(path.with_suffix('.png'),dpi=220)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    return {'stem':str(path),'formats':['png','pdf'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


if __name__=='__main__':main()
