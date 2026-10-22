"""Readable Chinese synthesis plus immutable-source and budget audit."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
from run_timestamped_development_campaign import ROOT, read, save, sha, unique_output
from run_smoke_preference_development_confirmation import verify
from run_direction_counterfactual_audit import verify_audit


def run():
    verify()
    paths = {name: ROOT / ('outputs/' + leaf) for name, leaf in {
        'feasibility': 'direction_state_feasibility_20261007T031854810149Z',
        'counterfactual': 'direction_counterfactual_20261007T031906039073Z',
        'matched_release': 'direction_counterfactual_final_audit_20261007T032154325639Z',
        'mechanism': 'direction_mechanism_trace_20261007T032341078522Z',
        'confirmation': 'smoke_preference_development_confirmation_20261007T032614390080Z'}.items()}
    for p in paths.values():
        verify_audit(p)
    s = {name: read(p / 'summary.json') for name, p in paths.items()}
    assert s['feasibility']['exact_all_transition_and_terminal_checks']
    assert s['counterfactual']['terminal_rollouts'] == 7050
    assert s['mechanism']['terminal_rollouts'] == 60
    assert s['confirmation']['completed_full_episodes'] == 1056
    assert all(not any(s[k]['violations'].values()) for k in ('counterfactual', 'confirmation'))
    a = {r['metric']: r for r in s['matched_release']['same_release_contrasts']
         if r['contrast'] == 'lower_direction_same_release_all'}
    c = {r['controller']: r for r in s['confirmation']['constraint_effects']
         if r['behavior'] == 'all' and r['metric'] == 'mean_cumulative_fed'}
    means = {(r['controller'], r['constraints']): r for r in s['confirmation']['means']}
    inc = {r['controller']: r for r in s['confirmation']['constraint_effects']
           if r['behavior'] == 'all' and r['metric'] == 'incapacitation_rate'}
    out = unique_output('direction_research_round_final_audit')
    report = ROOT / 'reports/论文开发实验_方向覆盖与等待反事实审计_2026-10-07.md'
    assert not report.exists(), 'Keep report append-only.'
    lines = ['# 火灾疏散论文开发实验：方向覆盖、等待与统一约束的审计', '',
        '日期：2026年10月7日。用户停止27B提速后，本轮继续论文开发实验。实际完成7,050条终止反事实轨迹、60条机制追踪轨迹、1,056个完整疏散回合。三类实验用途不同，不能合并称为8,166个独立测试场景。', '',
        '## 本轮最重要的结论', '',
        '**不能把“门控拒绝较高烟气建议”直接理解为“PPO必须换到低烟气出口”。本轮中，简单强制换向通常使最终暴露更大；在另一个开发子集的全程控制中，这种规则也使PPO表现变差。**', '',
        '缓存LLM＋同权重PPO仍没有稳定优势。这是开发阶段的负结果和机制证据，没有证明27B有效，也没有证明原PPO已经最优或安全。', '',
        '## 1. 从上一轮的问题出发', '',
        '上一轮已完成1,488个开发回合，以及480个同权重/错误建议对照回合。错误建议生成器在实时烟气差大于0.01时提名烟气较高方向；这只是人为构造的可观察烟气偏好冲突，不一定是综合收益上的错误建议。', '',
        '上一轮观察到8,723次提案因实时烟气矛盾被拒绝，其中596次PPO仍选择相同较高烟气方向。此前这596次没有进行可执行性和后果评估，不能称为596次已证实纠错失败。', '',
        '本轮重新按历史动作重放240个错误建议回合，核对10,483个控制步及每个回合终点。应用动作、实际指示牌、放行人数、容量限制和终点指标全部一致。596个目标状态来自11个场景、129个回合。', '',
        '|目标状态类型|数量|含义|', '|---|---:|---|',
        '|允许立即换到低烟气方向|486|动作mask确实允许比较换向|',
        '|指示牌最短保持时间限制|110|出口虽开放，当前不能立即切换方向|', '',
        '这110次要从“能够立即覆盖却没有覆盖”的口径中排除。等待动作只停止房间放行，路口已有人群仍然继续选路。', '',
        '## 2. 反事实如何设计', '',
        '保持历史时刻的个人属性、当前位置、累计FED、既有PPO权重和物理限制相同。分别比较原PPO动作、mask允许的低烟气方向25%/50%/75%/100%放行档位、等待一步后恢复PPO、等待1—3步并在合法时换向后恢复PPO。每条轨迹跑到终止。', '',
        '校准概率行为使用4个预先冻结的未来种子，并按同一人、同一绝对时刻配对随机数；改变路线后不会把不同人的随机数误配。确定性规则行为只使用1个未来。干预最多4步，然后恢复同一PPO和相同类型的现场错误建议生成逻辑。', '',
        '总计7,050条后续终止轨迹。即时换向只运行合法动作，没有绕过指示牌保持或容量限制；封闭/超烟气阈值走廊进入、超量放行、非法实际方向记录均为0。这个0只说明执行约束符合本仿真规则，不代表疏散伤害为0。', '',
        '本轮专门从候选库中提取486个“保持原放行档位，仅改变方向”的对照，避免把少放人误算为方向收益。统计先对未来种子平均，再对同一回合中的状态平均，再对场景平均。', '',
        '## 3. 保持放行档位后，强制换向是否更好', '',
        '|指标|场景等权平均差（换向−原PPO）|描述性场景bootstrap 95%区间|改善/恶化/不变状态|',
        '|---|---:|---|---|']
    labels = {'mean_cumulative_fed': '人均累积FED', 'cumulative_congestion': '累计拥堵',
              'evacuation_rate': '疏散率', 'incapacitation_rate': '失能率'}
    for metric in ('mean_cumulative_fed', 'cumulative_congestion', 'evacuation_rate', 'incapacitation_rate'):
        r = a[metric]
        lo, hi = r['descriptive_scene_bootstrap_95_interval']
        factor = 100 if metric.endswith('_rate') else 1
        unit = '个百分点' if factor == 100 else ''
        lines.append(f"|{labels[metric]}|{r['equal_scene_episode_state_mean_difference'] * factor:+.6f}{unit}|{lo * factor:.6f}, {hi * factor:.6f}{unit}|{r['improved_states']}/{r['worsened_states']}/{r['unchanged_states']}|")
    lines += ['', '428/486个状态的最终FED增加，58个减少；134个状态的平均失能率增加，15个减少。两个行为模型分开看，FED平均差也都是正值。该结果反对在当前策略与仿真设置下直接采用“烟气更低就强制换向”。', '',
        '等待也不是无代价。对全部596个状态，等待一步后恢复PPO的场景等权平均FED增加0.052916，平均失能率增加4.197个百分点。对110个受保持时间限制的状态，等待到能够换向后再切换，平均失能率增加2.331个百分点。', '',
        '把完整候选库事后逐个查看，73/596个状态至少有一个候选降低FED，72个存在降低FED且不增加失能的候选；其余523个在这个有限候选库和冻结未来种子下都没有FED改善。它不是全局最优证明。事后知道真实后果再选方案属于乐观上限，不能作为实时控制器。', '',
        '## 4. 为什么低烟气出口可能导致更高总暴露', '',
        '额外冻结了8个探索案例：按行为模型、低烟气出口容量较大/较小分层，分别选择已观察到的FED差最小和最大的状态，共60条机制追踪。这些是根据结果挑选的案例，只能解释可能的路径，不能估计所有场景的机制发生率。', '',
        '对每一步累计各区域的人数×真实模拟烟气增量，分解后续FED，并记录房间/路口总停留人秒、实际走向和放行。60条重跑的终点全部与原反事实一致，区域FED分解与总FED严格一致。', '',
        '例如AULL0005的规则行为案例（人口种子1407100040、PPO种子605、step 4）：保持放行档位换向后，人均FED增加0.154419。房间和路口贡献分别增加约0.070977和0.089825；两条走廊的合计贡献反而减少约0.006383。房间总停留增加4,030人秒，路口增加1,870人秒。也就是说，走廊稍干净带来的收益，被上游等待的暴露抵消。', '',
        '另一个规则案例（AULL0005、人口种子1407100041、PPO种子602、step 4）FED降低约0.052054，但失能率增加0.417个百分点。因此只看平均FED也可能把有伤害的方案判成改善。', '',
        '这不是说低烟气路线本身不好。方向、容量、人群响应、指示牌保持和后续PPO动作会共同改变人员停留位置与时长。单步控制干预可能改变多个后续步，不能把局部对照直接累加成全程效果。', '',
        '## 5. 用另一个开发子集验证全程影响', '',
        '从既有AULL开发库另行冻结AULL0013—AULL0024共12个场景，六类冲突各2个。它们未参与此次596状态分析，但曾存在于旧开发工作，不能称为首次使用的最终测试集。真正的保留最终场景没有打开。', '',
        '对纯规则、相同既有权重但无建议的PPO、缓存LLM＋相同权重PPO，比较原操作约束与烟气优先实验约束。实验约束仅在低烟气方向有合法动作时屏蔽高烟气方向，保留保持时间和物理容量限制。', '',
        '两种行为模型、每场景2个人口种子、PPO组5套既有权重，总计1,056个完整回合。两套约束和两套PPO建议模式使用同一批人和按人/时间配对的未来随机数。实验约束只作为比较探针，没有采用为实际修正。', '',
        '|控制器|FED相对原约束变化|失能率变化|完整配对回合|', '|---|---:|---:|---:|']
    for ctrl in ('pure_rule', 'same_weight_ppo', 'cached_llm_same_weight_ppo'):
        relative = c[ctrl]['mean_difference'] / means[(ctrl, 'current_operational_constraints')]['mean_cumulative_fed'] * 100
        name = {'pure_rule': '纯规则', 'same_weight_ppo': '同权重无建议PPO', 'cached_llm_same_weight_ppo': '缓存LLM＋同权重PPO'}[ctrl]
        lines.append(f"|{name}|{relative:+.2f}%|{inc[ctrl]['mean_difference'] * 100:+.3f}个百分点|{c[ctrl]['paired_episodes']}|")
    lines += ['', '两个PPO组在烟气优先约束下都变差；纯规则在这批场景中结果不变。结论是当前这条简单硬规则不应作为已验证的纠错方法。它不排除未来经过正确训练和验证的策略能更好地利用烟气信息。', '',
        '## 6. 本轮也重新检查了LLM建议的独立收益', '',
        '原操作约束下，缓存LLM＋同权重PPO相对无建议PPO的FED平均差为+0.004111，开发场景描述性95%区间为[-0.000499, +0.008645]；失能率差为+0.231个百分点，区间跨零。此轮没有稳定收益证据，均差方向还略差。', '',
        '这些建议来自2026年10月5日保存的Qwen3 4B响应，初始时刻仅发一次，15秒后过期。没有新的27B或8B调用，不能把这组结果表述为27B在线智能体效果。无建议PPO使用同一raw_llm_ppo权重，这是去掉建议的消融，不等于重新训练了新的纯PPO模型。', '',
        '## 7. 对研究方案的调整', '',
        '1. 将596次重新命名为“烟气偏好冲突状态”，停止把它们当作已证实的错误动作或纠错失败。',
        '2. 不采用本轮烟气优先硬屏蔽规则；不因拒绝建议的数量高，就宣称门控有效。',
        '3. 继续保留所有控制器共同的出口封闭、烟气绝对阈值、保持时间与放行容量约束。这些是操作限制，尚不是通过资格审查的预测安全盾。',
        '4. 下一轮应分开验证“建议能否合法执行”和“覆盖建议是否降低综合损失”，以失能不增加为前提检查FED、拥堵和等待损失。模型输入必须来自现场观测；反事实真实未来只作离线标签或审计，不能成为实时控制器输入。',
        '5. 暂不扩大PPO训练。先在冻结开发数据上查明提案器和覆盖收益预测是否能区分真正有益/有害的方案，再谈训练增量和最终测试。', '',
        '## 8. 证据边界与可复现材料', '',
        '区间按场景聚类，条件于已有5套权重；没有把重复控制步和人口种子当成独立场景，也没有把复用开发集当成最终推断。机制案例为结果挑选；不同即时档位只在各自合法子集内比较，不能直接排名其均值。', '',
        'FED是简化图模型中的合成每步增量，没有真实建筑轨迹或FDS验证；当前证据不能直接外推为真实火场剂量或安全性。仿真结束包含失能终止，只有全员成功疏散回合才有all_evacuated_seconds。', '',
        '新功能与相关回归测试共29项通过。全部7,050条反事实及1,056个完整验证回合，封闭/超阈值走廊进入、放行超限和非法实际方向记录均为0。正式第三轮训练、预测盾资格状态和默认LLM选择保持原样。', '',
        '原文件、权重和台账均核验哈希，未改写；新增实验代码与结果使用独立文件和目录。一次报告生成因传入相对路径失败，之后改用绝对路径完成后处理，没有重复原7,050条仿真。60条机制重跑是另行冻结、用于新增区域暴露观测的实验。', '',
        '详细文件：', '']
    for name, path in paths.items():
        target = path / ('report.md' if (path / 'report.md').exists() else 'summary.json')
        lines.append(f"- [{name}]({target.as_posix()})")
    lines += ['', '本报告对应的最终审计目录：' + out.as_posix(), '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    save(out / 'summary.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        report=str(report), new_terminal_counterfactuals=7050, new_instrumented_terminal_traces=60,
        new_full_development_episodes=1056, fresh_llm_requests=0, new_training=0,
        tests_passed=29, reserved_final_scenarios_opened=0,
        decisions=['relabel_higher_smoke_steps_as_preference_conflicts', 'do_not_adopt_smoke_only_hard_preference',
                   'no_claim_of_stable_cached_llm_benefit', 'no_qualification_or_training_status_change'],
        protected_selection_sha256=sha(ROOT / 'outputs/local_llm_selection.json'),
        protected_research_status_sha256=sha(ROOT / 'outputs/third_round_development_status.json'),
        sources={p.relative_to(ROOT).as_posix(): sha(p) for p in [Path(__file__), report] +
                 [p / 'completion_manifest.json' for p in paths.values()]}))
    verify()
    save(out / 'completion_manifest.json', {'state': 'complete', 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(out), 'report': str(report), 'sources_unchanged': True}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
