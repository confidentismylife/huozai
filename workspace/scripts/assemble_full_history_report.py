"""Assemble verified retrospective sections, without touching experiment outputs."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = ROOT / "artifacts/full_history_report_assets"
NAME = "LLM_RL火灾疏散全流程实验研究总报告_2026-10-02"


def read(name):
    return (REPORTS / name).read_text(encoding="utf-8")


def sections(text):
    pieces = re.split(r"(?m)^## (.+)\n", text)
    return pieces[0], list(zip(pieces[1::2], pieces[2::2]))


def link(label, path):
    target = ROOT / path
    if not target.exists():
        raise FileNotFoundError(target)
    return f"[{label}]({target.as_posix()})"


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    framework = read("总报告素材_总体框架.md").replace("图1 五分区容量流环境及控制位置", "图2：五分区容量流环境及控制位置")
    framework = framework.replace("**整理日期：2026年10月2日**", "**实验内容截至：2026年10月2日；报告整理完成：2026年10月3日**")
    overview = """
### 1.5 全部实验如何串起来

下表是阅读路线。每一阶段都是为了解决前一阶段暴露的问题；各行的场景、观测和训练预算不同，表中不做跨轮成绩排名。

| 阶段 | 想回答的问题 | 最主要的发现与下一转折 |
|---|---|---|
| 原型与多种子PPO | 控制标志能否学习；是否依赖训练随机性 | 能学习，但强危险规则仍有优势；单次好成绩不代表稳定方法 |
| 行为交叉、历史与LSTM | 人群行为变化和记忆是否影响策略 | 行为错配明显；增加模型复杂度未自动改善；最坏组与平均存在取舍 |
| 公开行为数据预测 | 当前LLM是否足以充当经过验证的人员模型 | Qwen弱于同任务频率，外部数据概率表现未过关；合成校准仍不能代替真人证据 |
| 单次与动态LLM建议 | 建议是否帮助控制；PPO是否动态纠正 | 简单场景成绩好，但逐回合与静态建议完全相同；动态纠正尚未证明 |
| 信任门控、监督与联合容量 | 等待、覆盖、限流能否处理冲突 | 监督准确率高仍可能部署变差；联合动作改善只属开发证据 |
| 动作审计与两轮公平确认 | 相同信息、动作、预算下LLM是否有增益 | 首轮LLM明确退化；第二轮主比较未建立稳定改善 |
| 护栏与完整回合反事实 | 学到的安全筛选是否够可靠 | 空集合回退影响有效召回；完整回合标签与历史仍未达到资格 |
| 教师、可观测学生与2×2 | 教师是否值得学；扩大训练和排序是否有效 | 旧开发教师优势未稳定复现；最新四组区间均跨零，正式第三轮未启动 |

![图1：研究路线及三条尚待连接的证据链]({tracks})

""".format(tracks=(ASSETS / "research_tracks.png").as_posix())
    framework = framework.replace("## 2. 仿真与代码实际怎样工作", overview + "## 2. 仿真与代码实际怎样工作")
    _, behavior = sections(read("总报告素材_行为数据与研究边界.md"))
    chapter3 = ["## 3. 老师方向、文献与真实行为数据：哪些做到了，哪些尚未做到\n\n",
                "当前控制实验的合成校准与公开真人选择预测是两条尚未连接的证据链。本章先说明研究来源，再说明已经下载、实际使用及尚未取得的资料。\n\n"]
    for i, (title, body) in enumerate(behavior, 1):
        title = re.sub(r"^\d+\.\s*", "", title)
        title = title.replace("本轮确实读到了截图", "原始聊天截图的可核对要求")
        title = title.replace("总报告应保留的结论边界", "文献与数据可以支撑的结论边界")
        body = re.sub(r"(?m)^### \d+\.(\d+) (.+)$", lambda m: f"#### 3.{i}.{m[1]} {m[2]}", body)
        chapter3.extend([f"### 3.{i} {title}\n", body])
    _, early = sections(read("总报告素材_早期实验.md"))
    chapter4 = ["## 4. 从原型到行为稳健性，再到本地LLM接入\n\n",
                "早期训练与评估首先用于发现机制和完善代码。旧材料中的‘锁定测试’后来多次参与研发；尤其反复查看的20个门控场景，本报告统一称为开发评估集。\n\n"]
    chapter5 = ["## 5. 从静态建议到动态纠正、信任门控与联合放行\n\n",
                "这一阶段逐步把‘选择安全方向’扩展为‘根据时间、拥堵与预测错误决定何时等待、何时覆盖、放行多少人’。下面完整保留改善与失败。\n\n"]
    for i, (title, body) in enumerate(early):
        if i < 9:
            chapter4.extend([f"### 4.{i+1} {title}\n", body])
        else:
            title = title.replace("这一段历史可以支持的结论", "动态门控阶段可以支持的结论")
            chapter5.extend([f"### 5.{i-8} {title}\n", body])
    _, late = sections(read("总报告素材_后期审计.md"))
    late_chapters = []
    for raw_title, body in late:
        old_no = int(re.match(r"(\d+)\.", raw_title)[1])
        if not 1 <= old_no <= 11:
            continue
        chapter = old_no + 5
        title = re.sub(r"^\d+\.\s*", "", raw_title)
        subcounter = [0]
        def renumber(match):
            subcounter[0] += 1
            return f"### {chapter}.{subcounter[0]} {match[1]}"
        body = re.sub(r"(?m)^### (.+)$", renumber, body)
        late_chapters.extend([f"## {chapter}. {title}\n", body])
        if chapter == 16:
            late_chapters.append(f"\n![图3：最新2×2开发实验在同一审计场景上的比较]({(ROOT/'outputs/baseline_relative_plan_distillation_development_v1/initial_factorial_comparison.png').as_posix()})\n\n")
    evidence = [
        ("E01", "初始研究方案与原始报告", [("初始报告", "reports/LLM_RL火灾疏散研究与实验实施报告.md")]),
        ("E02", "环境、危险、人群与动作执行", [("配置", "src/fire_evac/config.py"), ("环境", "src/fire_evac/env.py"), ("危险场", "src/fire_evac/hazard.py"), ("行为模型", "src/fire_evac/behavior.py")]),
        ("E03", "合成行为校准的来源与结果", [("数据来源和指标", "outputs/behavior_pipeline/metrics.json"), ("模型", "outputs/behavior_pipeline/behavior_model.json")]),
        ("E04", "五种子基础PPO", [("元数据", "outputs/ppo_multiseed/experiment_metadata.json"), ("聚合", "outputs/ppo_multiseed/multiseed_aggregate.csv")]),
        ("E05", "行为交叉与四步历史", [("元数据", "outputs/behavior_cross/experiment_metadata.json"), ("矩阵", "outputs/behavior_cross/cross_matrix.csv")]),
        ("E06", "LSTM公平对照", [("比较", "outputs/recurrent_comparison/architecture_comparison.csv"), ("元数据", "outputs/recurrent_comparison/experiment_metadata.json")]),
        ("E07", "自适应最坏行为采样", [("排名", "outputs/robust_ensemble/robust_methods_ranking.csv"), ("课程日志", "outputs/robust_ensemble/curriculum_log.csv")]),
        ("E08", "公开人类选择验证", [("Snopková", "outputs/local_llm_validation/test_metrics.csv"), ("镜像", "outputs/mirror_validation/validation_metrics.csv"), ("Haghani", "outputs/haghani_external/external_metrics.csv")]),
        ("E09", "已发表行为公式的复现", [("Lu2025", "outputs/lu_warden_validation/metrics.json"), ("Lu2026", "outputs/li_group_validation/metrics.json"), ("全文状态", "artifacts/papers/README.md")]),
        ("E10", "单次LLM与PPO冲突选择", [("单次", "outputs/conflict_llm_rl/summary.csv"), ("多种子", "outputs/conflict_multiseed/aggregate_summary.csv")]),
        ("E11", "动态建议提示失败与修正", [("v1失败", "outputs/dynamic_llm_rl_prompt_v1_failure/summary.csv"), ("v2", "outputs/dynamic_llm_rl_v2/summary.csv")]),
        ("E12", "动态跨行为与静态等价审计", [("聚合", "outputs/dynamic_multiseed_cross/aggregate_cross_summary.csv"), ("逐回合等价", "outputs/dynamic_multiseed_cross/static_equivalence_audit.csv")]),
        ("E13", "冲突场景中的动态纠正", [("汇总", "outputs/dynamic_conflict_correction/aggregate_summary.csv")]),
        ("E14", "恢复时间与信任门控", [("元数据", "outputs/llm_trust_gate/metadata.json"), ("汇总", "outputs/llm_trust_gate/aggregate_summary.csv")]),
        ("E15", "三固定反事实计划与监督", [("监督", "outputs/supervised_trust_gate/supervised_training_summary.csv"), ("部署", "outputs/supervised_trust_gate/aggregate_summary.csv")]),
        ("E16", "动态计划与持续辅助损失", [("训练", "outputs/dynamic_auxiliary_gate/dynamic_training_summary.csv"), ("部署", "outputs/dynamic_auxiliary_gate/aggregate_summary.csv")]),
        ("E17", "域随机化、矛盾特征与容量动作", [("域随机化", "outputs/domain_randomized_contradiction/aggregate_summary.csv"), ("周期放行", "outputs/capacity_throttle_contradiction/aggregate_summary.csv"), ("联合动作", "outputs/joint_route_release/aggregate_summary.csv")]),
        ("E18", "统一动作与公平实验冻结", [("第一轮协议", "reports/confirmatory_joint_control_protocol.md"), ("冻结清单", "outputs/confirmatory_protocol/manifest.json")]),
        ("E19", "第一次确认性实验", [("完整报告", "reports/确认性联合控制最终实验报告.md"), ("种子效应", "outputs/confirmatory_joint_control/paired_training_seed_effects.csv")]),
        ("E20", "第二次确认性实验", [("协议", "reports/guarded_confirmatory_v2_protocol.md"), ("结果", "reports/安全护栏LLM确认性实验2报告.md"), ("种子效应", "outputs/guarded_confirmatory_v2/paired_training_seed_effects.csv")]),
        ("E21", "条件护栏运行审计与605干预", [("运行审计", "outputs/conditional_shield_audit_v1/audit_summary.json"), ("容量干预", "outputs/guarded_seed605_diagnosis_v2/paired_intervention_summary.csv")]),
        ("E22", "完整回合三分类护栏", [("资格结果", "outputs/terminal_shield_development_v1/validation_summary.json"), ("诊断", "outputs/terminal_shield_diagnosis_v1/diagnosis_summary.json")]),
        ("E23", "两可能火源世界与候选教师", [("12计划", "outputs/source_world_planning_development_v1/summary.json"), ("17候选", "outputs/ppo_extended_world_planning_development_v1/summary.json")]),
        ("E24", "首次可观测计划蒸馏", [("结果", "outputs/observable_plan_distillation_development_v1/summary.json"), ("诊断", "outputs/observable_plan_distillation_development_v1/distillation_diagnostic_summary.csv")]),
        ("E25", "最新扩充训练×相对排序2×2", [("预设开发清单", "outputs/baseline_relative_plan_distillation_development_v1/development_manifest.json"), ("最终分析", "outputs/baseline_relative_plan_distillation_development_v1/summary.json"), ("运行规模", "outputs/baseline_relative_plan_distillation_development_v1/run_summary.json")]),
    ]
    evidence_rows = ["| 编号 | 证据内容 | 主要文件 |", "|---|---|---|"]
    for number, description, paths in evidence:
        evidence_rows.append(f"| {number} | {description} | {'；'.join(link(label, path) for label,path in paths)} |")
    closing = read("总报告素材_解释与后续.md").replace("<!-- EVIDENCE_INDEX -->", "\n".join(evidence_rows))
    closing = closing.replace("第一轮原型中，PPO比中性标志好", "从原型进入PPO实验后，PPO比中性标志好")
    closing = closing.replace("容量控制显著改变了动作能力", "容量控制改变了动作能力")
    audit_note = "\n本次文件保全审计记录119个文件快照，并对其中93条既有哈希记录逐一比对，全部一致。它覆盖所列核心协议、结果、模型与代码文件，不代表对所有缓存和回合文件做了穷尽审计；缺少历史哈希的文件仅保留当前快照。[保全审计记录](" + (ASSETS/'experiment_preservation_audit.json').as_posix() + ")。\n"
    closing += audit_note
    assembled = framework.rstrip() + "\n\n" + "".join(chapter3 + chapter4 + chapter5 + late_chapters) + closing
    assembled = assembled.replace("第一轮确认性实验中，错误LLM建议造成了明确退化", "第一轮确认性实验中，该版本LLM建议组出现了明确退化")
    assembled = assembled.replace("先检查老师是否提供了", "先检查反事实教师是否提供了")
    assembled = assembled.replace("三个未来最小值会产生选择乐观偏差，已经另存规划−检验差诊断", "每个候选只用三个规划未来估计均值，再从有限候选中选取低代价计划，可能产生选择乐观偏差；已经另存规划−检验差诊断")
    assembled = assembled.replace("四项区间都包含零", "四项主要对照及交互的区间都包含零")
    assembled = assembled.replace("二元标签以当前九候选", "早期二元护栏所用的反事实标注以当前九候选")
    assembled = assembled.replace("明显劣于最佳的动作被标为危险，近似平局为不确定。某动作被标为", "同时处于这三个安全容差内的动作标为安全；达到预设失能或FED劣化阈值的动作标为危险，其余为不确定。某动作被标为")
    assembled = assembled.replace("初版还强制保留词典序赢家，后期共识教师才不强行制造安全赢家", "初版还强制保留词典序赢家；后期共识标签要求各教师一致，因此不再强制制造共同安全赢家")
    assembled = assembled.replace("后期固定候选的计划选择比较主要按场景家庭重采样。", "后期固定候选的计划选择比较主要按场景家庭重采样。这些家庭Bootstrap区间条件于已保存的控制器与学生，没有同时量化重新训练初始化的变异。")
    # Archive temporary source screenshots referenced in the final report.
    for target in set(re.findall(r"\]\((C:/Users/[^)]+/Temp/codex-clipboard-[^)]+\.png)\)", assembled)):
        src = Path(target)
        if not src.exists():
            raise FileNotFoundError(src)
        destination = ASSETS / "teacher_screenshots" / src.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
        assembled = assembled.replace(target, destination.as_posix())
    # Archive all seven human-provided files, not just the three linked above.
    temp = Path("C:/Users/摩天轮~1/AppData/Local/Temp")
    names = ["1fd26494-6208-412d-b4e0-2df6b346fb6f", "868f1511-40cb-4054-acdd-215e8c7cb154", "eecbb761-2a88-4493-8965-a6fc160e8ed3", "c325bf1f-9dff-4cb7-a775-e49c094a590a", "cd983fe6-0816-4b3e-844e-116dfe0f7226", "0d297fe7-ef3b-4caf-bb37-c8916fbd001c", "0a039c40-1827-4849-8a00-795b2d3d9ca6"]
    for name in names:
        src = temp / f"codex-clipboard-{name}.png"
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, ASSETS / "teacher_screenshots" / src.name)
    assembled = assembled.replace("总报告应", "本报告应").replace("供总报告整合", "用于本报告")
    destination = REPORTS / f"{NAME}.md"
    destination.write_text(assembled.strip() + "\n", encoding="utf-8")
    print({"report": str(destination), "characters": len(assembled), "chapters": len(re.findall(r"(?m)^## \d+\.", assembled)), "evidence_entries": len(evidence)})


if __name__ == "__main__":
    build()
