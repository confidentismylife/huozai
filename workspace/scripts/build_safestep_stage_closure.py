"""Build a source-checked meeting brief and portable local results bundle.

All numbers are summaries of existing caches. No solver, policy rollout,
training or LLM request is made. Original frozen files remain unchanged.
"""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'outputs/safestep_public_benchmark_20261017'
CASES=ROOT/'outputs/safestep_five_residual_cases_20261017'
OUT=ROOT/'outputs/safestep_stage_closure_20261017'
PACKAGE=ROOT/'artifacts/safestep_group_meeting_20261017'
ARCHIVE=ROOT/'artifacts/SafeStep阶段汇报与研究结果_2026-10-17.zip'
REPORT=ROOT/'reports/SafeStep阶段收敛汇报与后续研究决策_2026-10-17.md'
LINK=re.compile(r'\]\(([^)]+)\)')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def local_targets(text):
    for match in LINK.finditer(text):
        value=match.group(1).strip('<>')
        if re.match(r'^[A-Za-z]:[/\\]',value):yield Path(value)


def write_brief(headroom):
    table='\n'.join(f"| {r['departure_fire_s']} | {r['DQN_mean_FED']:.6f} | {r['reference_mean_FED']:.6f} | {r['absolute_mean_gap']:.6f} | {r['headroom_percent']:.2f}% | {r['FED_matching_starts']}/61 |" for r in headroom)
    content=f'''# SafeStep阶段汇报与后续研究决策

日期：2026-10-17。用途：组会汇报、研究范围确认与实验章节收尾。

**本阶段完成了作者公开危险率输入下的路径数值复现、基线比较和剩余差额审计。自建FDS的物理定量复现仍未完成；LLM新增控制收益仍未得到验证。**

## 一页摘要

研究对象为一个酒店楼层、一个公开火灾危险率背景、一个作者固定DQN权重。61个起点对应58个危险率区域，两个终点是楼梯入口格。DQN推理时读取当前位置；当前快照方法读取当前全图FER；完整未来参照拥有额外未来信息。四种方法的观察条件并不相同。

主要结果固定为300秒出发、61个起点等权，主计分使用出发格驻留区间的FER积分：

| 方法 | 平均FED | 平均行走时间/秒 |
|---|---:|---:|
| 最近出口 | 0.206062 | 53.22 |
| 作者DQN | 0.142790 | 65.90 |
| 当前危险率重规划 | 0.145939 | 68.31 |
| 完整未来最小FED参照 | 0.141517 | 63.72 |

DQN相对最近出口的平均FED下降30.71%，平均多走12.68秒；56/61个起点达到本轮45步时域内的最小FED，全体平均进一步下降空间约0.89%。5个剩余起点中，两个出口选择案例贡献84.01%的总差额，三个同出口局部路线案例贡献15.99%。

## 约5分钟汇报讲稿

这阶段我主要完成了SafeStep公开案例的路径规划复现。研究对象是一个酒店楼层，我们使用作者公开的地图、危险率数据和保存的DQN权重。之前已经重放出论文四条指定路线的FED，按作者计分方式与论文四位小数一致。这是公开输入下的路径数值复现。

接着，我比较了四种方法：最近出口、作者DQN、根据当前危险率逐步重规划，以及提前知道完整未来危险率的最小暴露参照。最后一种方法只用于衡量剩余空间。它掌握额外未来信息，因此不能作为公平在线竞争者。

本次统一对照包括61个起点、9个出发时刻，共2,196条轨迹。它们来自同一个火灾，不是2,196个独立火灾。重点看300秒出发时，最近出口的平均FED为0.2061，DQN为0.1428，平均下降约30.7%，但平均行走时间增加约12.7秒。当前危险率重规划的平均FED为0.1459，也比较接近DQN。

为了判断还有多少提升空间，我计算了相同动作与45步时域内的最小FED。DQN在61个起点中有56个达到这个参照，其余5个还有差距。全部达到参照时，平均FED相对DQN还能下降约0.89%。因此这个已知场景中的平均空间较小，但不能忽略少数位置。

我把这5个案例全部展开分析。两个案例可以通过改选出口减少暴露，占总差额约84%；另外三个案例的出口和步数相同，差别集中在电梯前区的早期走法。最大的单例下降空间约21%，但它不代表所有起点都有这么大收益。

其中R24有一个明确现象：如果把出发时烟气危险率保持不变，较长的出口B路线看起来更好；但考虑行走期间的危险率变化后，较短的出口A路线反而更好。这说明初始快照可能低估长路线中的后续暴露。不过，它没有解释DQN内部为什么选择B，也没有证明LLM就能预测正确。

我还补看了其余出发时刻。200秒和250秒时，平均FED距完整未来参照还有约6.03%和3.88%的下降空间；350至600秒时约为0.13%至0.58%。所以0.89%只对应300秒条件，不能概括全部时刻。

目前最扎实的成果是可追溯的复现、基线比较和差额解释。自建FDS与作者数据的物理量级差距仍存在，相关参数没有全部恢复。我建议先完成这一部分的实验章节与组会材料。如果继续LLM方向，需要另行明确新增信息、经典基线和独立验证数据，而不是直接在这些已知答案上增加一个LLM模块。

## 不同出发时刻的完整结果

下表由既有缓存重新汇总；没有新增轨迹或改动策略。每行均为61个起点等权。“下降空间”定义为`(DQN平均FED−完整未来参照平均FED)/DQN平均FED`，仅在共同动作、45步时域和同一公开危险率下成立。

| 出发火灾时刻/秒 | DQN平均FED | 未来参照平均FED | 平均绝对差额 | 相对下降空间 | FED匹配起点数 |
|---|---:|---:|---:|---:|---:|
{table}

早期出发的相对空间较大，但绝对差额也应同时看。例如200秒的平均绝对差额约0.001916，而600秒约0.008073。不能只用百分比决定某个条件是否更值得研究。

九个时刻的均值等权合并后，相对差额约0.43%；它受所选时刻及其绝对暴露量影响，不是跨火灾效应量，也不替代逐时刻结果。所有时刻共用同一火灾和同一权重，不据此进行独立样本显著性检验。

## 可以说什么、还不能说什么

| 证据 | 支持的表述 | 超出证据的表述 |
|---|---|---|
| 四条路线旧计分匹配 | 已复现公开输入下的指定路径数值 | 已完整恢复论文的物理火灾 |
| 300秒平均FED下降30.71% | 同一危险率下的路线暴露比较 | 真人伤亡下降30.71% |
| 56/61匹配45步最小FED | 该时刻、目标和时域下多数起点达到参照 | 原DQN复合奖励已全局最优 |
| 当前快照与时变计分反转 | 该对既有路线存在时间变化导致的计分偏好反转 | 已证明DQN训练失败原因或LLM预测有效 |
| 2,196条轨迹均到达 | 图上终点到达 | 所有人在真实建筑中安全疏散 |
| 9个出发时刻 | 同一火灾下的时刻变化诊断 | 跨建筑、跨火灾或跨人群泛化 |

## 接下来如何收敛

| 路线 | 当前是否具备证据 | 继续前需要什么 | 本阶段决定 |
|---|---|---|---|
| 公开危险率复现与基线审计 | 已具备 | 按报告/学位论文章节要求整理来源、方法和局限 | 完成写作与汇报 |
| 新控制算法或未来预测方法 | 当前只有一个已知火灾，信息条件也不同 | 独立环境变化、共同观测条件、普通规划/预测基线与冻结确认数据 | 作为新研究另行设计，不沿用5例充当确认集 |
| 老师提出的LLM模拟人群、RL负责引导 | 当前路径实验没有提供真人动态响应验证 | 决策前观测、引导内容与接收情况、保持/改道者和后续轨迹 | 数据不足时不作人群行为或因果主张 |

研究建议是先完成第一条路线。新研究必须提出可检验的新信息或约束；若独立数据暂不可得，可将当前工作定位为复现与方法审计，避免让参数试探或重复训练替代证据。

## 组会可能被问到的问题

**这轮到底用DQN还是PPO？** SafeStep复现使用作者DQN；前期LLM/PPO联合引导属于另一条实验线，不能合并成同一方法结果。

**有没有证明强化学习一定比普通规划好？** 没有。本轮信息和原训练投入不同，只能比较指定设置下的表现。当前快照规划已经较接近DQN，完整未来参照还拥有额外信息。

**为什么暂时没有增加LLM？** 当前已知场景的剩余空间有限且分布不均；部分偏差已有普通规划处理。需要先证明LLM提供可独立验证的新增信息，再检验控制收益。

**没有原始FDS文件，研究是否就结束了？** 公开危险率下的算法复现可以完成。物理火灾的精确复现范围需要收紧，自建候选继续标为探索性材料。

**这些材料现在能作为什么？** 可用于复现报告、学位论文实验章节或后续论文的基础证据。新方法贡献和独立环境验证仍需补充，不能仅靠改写文字取得。

## 阅读与复查入口

- [方法、结果与讨论初稿](D:/UserFolders/Documents/ChatGPT/LLM探索/reports/SafeStep公开危险率复现_方法结果与讨论初稿_2026-10-17.md)
- [完整统一计分对照](D:/UserFolders/Documents/ChatGPT/LLM探索/reports/SafeStep公开危险率统一计分对照_2026-10-17.md)
- [五个偏差案例分析](D:/UserFolders/Documents/ChatGPT/LLM探索/reports/SafeStep五个偏差案例分析与阶段结论_2026-10-17.md)
- [分时刻剩余空间数据](D:/UserFolders/Documents/ChatGPT/LLM探索/outputs/safestep_stage_closure_20261017/departure_headroom.csv)

数据来源：[SafeStep论文](https://doi.org/10.1016/j.jobe.2025.115132)及[固定版本公开仓库](https://github.com/hqfang3/SafeStep_v1/tree/573f613420e21e9a6731bc56fa36ea0b3f3a05f6)。本轮未新增原作者物理输入、独立火灾数据或LLM验证结果。
'''
    assert not REPORT.exists();REPORT.write_text(content,encoding='utf-8')


def build_portable_bundle():
    pending=[REPORT];mapping={};sources=[]
    while pending:
        p=pending.pop().resolve()
        if p in mapping:continue
        assert p.is_file() and p.is_relative_to(ROOT),p
        if p==REPORT.resolve():destination=Path('README.md')
        elif p.suffix.lower()=='.md':destination=Path('docs')/p.name
        elif p.suffix.lower() in ('.png','.svg','.jpg','.jpeg'):
            destination=Path('figures')/(p.parent.name+'_'+p.name)
        else:destination=Path('data')/(p.parent.name+'_'+p.name)
        assert destination not in mapping.values(),destination
        mapping[p]=destination;sources.append(p)
        if p.suffix.lower()=='.md':pending.extend(local_targets(p.read_text('utf-8')))
    PACKAGE.mkdir()
    for p in sources:
        destination=PACKAGE/mapping[p];destination.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix.lower()=='.md':
            def replace(match):
                target=match.group(1).strip('<>')
                if not re.match(r'^[A-Za-z]:[/\\]',target):return match.group(0)
                source=Path(target).resolve();assert source in mapping,source
                relative=posixpath.relpath(mapping[source].as_posix(),mapping[p].parent.as_posix())
                return ']('+('<'+relative+'>' if ' ' in relative else relative)+')'
            content=LINK.sub(replace,p.read_text('utf-8'))
            destination.write_text(content,encoding='utf-8')
        else:destination.write_bytes(p.read_bytes())
    (PACKAGE/'使用说明.txt').write_text(
        '从README.md开始阅读。文档、图表和派生结果使用相对链接，可整体移动或解压后阅读。\n'
        '这是汇报与数值结果归档包；独立复跑仍需要原项目代码环境及作者数据。\n'
        '原始输入文件未全部收录，公开来源与固定提交见文档；JSON中的原项目路径保留为来源记录。\n'
        '所有轨迹来自同一公开火灾，不能当作独立火灾重复。\n',encoding='utf-8')
    local_links=0
    for p in PACKAGE.rglob('*.md'):
        for m in LINK.finditer(p.read_text('utf-8')):
            target=m.group(1).strip('<>')
            if target.startswith(('https://','http://','#')):continue
            assert not re.match(r'^[A-Za-z]:',target),target
            resolved=(p.parent/target).resolve()
            assert resolved.is_relative_to(PACKAGE) and resolved.is_file(),(p,target)
            local_links+=1
    payload=[p for p in PACKAGE.rglob('*') if p.is_file()]
    manifest=dict(purpose='Portable meeting brief and source-linked derived results; not a standalone software distribution',
        original_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        bundle_sha256={p.relative_to(PACKAGE).as_posix():sha(p) for p in payload},
        local_relative_links_verified=local_links)
    save(PACKAGE/'MANIFEST.json',manifest)
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert not name.startswith(('/','\\')) and '..' not in Path(name).parts
            assert hashlib.sha256(z.read(name)).hexdigest()==sha(PACKAGE/name)
    return dict(files=len(payload)+1,local_links=local_links,
        archive_bytes=ARCHIVE.stat().st_size,source_to_bundle={p.relative_to(ROOT).as_posix():mapping[p].as_posix() for p in sources})


def main():
    assert not any(p.exists() for p in (OUT,PACKAGE,ARCHIVE,REPORT))
    protected={}
    plan=json.loads((CASES/'analysis_plan.json').read_text('utf-8'))
    for k in ('protected_prior_sha256','input_sha256'):protected.update(plan[k])
    for folder in (BENCH,CASES):
        for name in ('completion_manifest.json','review_manifest.json'):
            p=folder/name
            if not p.exists():continue
            protected.update(json.loads(p.read_text('utf-8'))['artifact_sha256'])
            protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    OUT.mkdir()
    save(OUT/'scope.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        stage='Stage closure, descriptive resummary and portable handoff',
        analysis_status='Post-hoc descriptive summary of already inspected departure results',
        new_rollouts=0,new_training=0,new_FDS=0,new_LLM_requests=0,final48_accessed=False,
        protected_prior_sha256=protected,script_sha256=sha(Path(__file__))))
    summary=rows(BENCH/'departure_summary.csv');pairs=rows(BENCH/'paired_comparisons.csv')
    headroom=[]
    for departure in sorted({int(r['departure_fire_s']) for r in summary}):
        methods={r['method']:r for r in summary if int(r['departure_fire_s'])==departure}
        d=float(methods['author_DQN_legal_mask']['mean_FED'])
        ref=float(methods['full_future_minFED_reference']['mean_FED'])
        comparison=next(r for r in pairs if int(r['departure_fire_s'])==departure
            and r['method_a']=='author_DQN_legal_mask' and r['method_b']=='full_future_minFED_reference')
        assert int(comparison['a_lower_FED'])==0
        headroom.append(dict(departure_fire_s=departure,DQN_mean_FED=d,reference_mean_FED=ref,
            absolute_mean_gap=d-ref,headroom_percent=100*(d-ref)/d,
            FED_matching_starts=int(comparison['ties']),starts=61))
    with (OUT/'departure_headroom.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(headroom[0]));w.writeheader();w.writerows(headroom)
    dtotal=sum(r['DQN_mean_FED'] for r in headroom);rtotal=sum(r['reference_mean_FED'] for r in headroom)
    save(OUT/'departure_headroom.json',dict(results=headroom,pooled_ratio_of_means_percent=100*(dtotal-rtotal)/dtotal,
        pooling='Equal start weights and equal weights of these nine dependent departure times; descriptive only',
        primary_departure_remains_300=True,independent_fire_count=1,independent_validation_claim=False))
    write_brief(headroom)
    bundle=build_portable_bundle()
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    save(OUT/'verification.json',dict(status='COMPLETE_STAGE_CLOSURE_AND_PORTABLE_MEETING_BUNDLE',
        all_nine_departures_reported=True,primary_300_scope_preserved=True,
        protected_prior_files_verified=len(protected),bundle=bundle,zip_integrity_passed=True,
        new_policy_rollouts=0,training_updates=0,FDS_runs=0,LLM_requests=0,final48_accessed=False))
    files=[p for p in OUT.iterdir() if p.is_file()]+[REPORT,Path(__file__),ARCHIVE]
    files.extend(p for p in PACKAGE.rglob('*') if p.is_file())
    save(OUT/'completion_manifest.json',dict(status='COMPLETE',
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        protected_prior_files_verified=len(protected)))
    print(json.dumps(dict(status='COMPLETE',bundle_files=bundle['files'],
        portable_links_verified=bundle['local_links'],archive_bytes=bundle['archive_bytes'],
        protected_prior_files_verified=len(protected),headroom=[(r['departure_fire_s'],r['headroom_percent']) for r in headroom]),ensure_ascii=True))


if __name__=='__main__':main()
