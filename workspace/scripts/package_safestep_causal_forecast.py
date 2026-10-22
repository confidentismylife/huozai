"""Make a portable, read-only results/review bundle; not a standalone simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import posixpath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/safestep_causal_forecast_results_20261018"
ARCHIVE = ROOT / "artifacts/SafeStep历史预测规划实验与独立核查_2026-10-18.zip"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not DEST.exists() and not ARCHIVE.exists()
    result_roots = [ROOT / "outputs/safestep_causal_forecast_20261018",
                    ROOT / "outputs/safestep_causal_forecast_review_20261018"]
    originals = []
    protected = {}
    for out in result_roots:
        manifest = json.loads((out / "completion_manifest.json").read_text("utf-8"))
        protected.update(manifest["artifact_sha256"])
        originals.extend(p for p in out.rglob("*") if p.is_file())
    for rel, digest in protected.items():
        assert sha(ROOT / rel) == digest, rel
    originals += [ROOT / "reports/SafeStep历史观测预测与路径规划开发实验_2026-10-18.md",
                  ROOT / "reports/SafeStep预测规划实验_结果解读与独立核查_2026-10-18.md",
                  ROOT / "src/fire_evac/safestep_causal_forecast.py",
                  ROOT / "tests/test_safestep_causal_forecast.py",
                  ROOT / "schemas/safestep_causal_forecast_20261018.json",
                  ROOT / "scripts/run_safestep_causal_forecast.py",
                  ROOT / "scripts/review_safestep_causal_forecast.py", Path(__file__)]
    originals = sorted(set(originals))
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in originals}
    DEST.mkdir()
    converted_links = 0
    for p in originals:
        rel = p.relative_to(ROOT)
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".md":
            def replace_link(match):
                nonlocal converted_links
                linked = Path(match.group(1))
                if not linked.is_relative_to(ROOT):
                    return match.group(0)
                link_rel = linked.relative_to(ROOT)
                assert (ROOT / link_rel) in originals, link_rel
                converted_links += 1
                relative = posixpath.relpath(link_rel.as_posix(), rel.parent.as_posix())
                return "](" + relative + ")"
            content = re.sub(r"\]\((D:/[^)]+)\)", replace_link, p.read_text("utf-8"))
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(p.read_bytes())
    readme = '''# SafeStep历史预测规划实验与独立核查

日期：2026-10-18。先读[结果解读与独立核查](reports/SafeStep预测规划实验_结果解读与独立核查_2026-10-18.md)，再查[完整实验报告](reports/SafeStep历史观测预测与路径规划开发实验_2026-10-18.md)。

本轮实现了只用历史观测的趋势预测与动态路径规划，并完成固定参数敏感性检查。

- 4,392条方法轨迹、43,532个动作，全部到达图上出口；对应同一火灾中的549个起点—出发时刻组合。
- 300秒出发：主趋势比快照平均FED低1.75%，改善全部来自R22一个起点，其余60个不变。
- 跨本轮549个组合：主趋势11个改善、4个有害、534个不变。半斜率9个改善、0个有害、540个不变，且漏掉了部分主趋势收益。
- 原先提出问题的R24仍未被主方案解决。300秒主趋势平均FED仍高于既有DQN；相关比较的信息条件不同。
- 13项实现检查、全轨迹独立积分及51项预测／动作核查通过。未训练RL、调用LLM或运行FDS，48个最终人群场景未读取。

## 怎么查数据

- outputs/safestep_causal_forecast_20261018：冻结计划、每条轨迹、逐步动作、全部参数、预测误差、计算时间。
- outputs/safestep_causal_forecast_review_20261018：独立剂量复算、收益与伤害计数、四个案例、固定计划排序。
- src、tests、scripts、schemas：本轮新增实现、检查和运行方案。

这是结果与代码审阅包。独立运行还需要原项目的公共评分模块、作者数据适配模块、固定版本地图／危险率工作表及Python依赖；包内不含这些全部输入。运行入口还会检查项目内此前冻结文件，不能直接在此包内启动主实验。

Markdown已转换为包内相对链接。JSON中的历史路径和来源哈希保留，可能指向原项目。MANIFEST.json另记录本包字节哈希和原文件哈希；Markdown链接转换导致字节不同属于明确记录的打包转换。

本轮是开发可行性证据，只有一个已知火灾。不能将其表述为跨火灾验证、真人安全保证或LLM收益证据。
'''
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    verified = 0
    for p in DEST.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text("utf-8")):
            assert (p.parent / link).resolve().is_file(), (p, link)
            verified += 1
    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    manifest = dict(status="COMPLETE_RESULTS_AND_CODE_REVIEW_BUNDLE",
        standalone_runnable=False, source_files_unchanged=True,
        converted_markdown_links=converted_links, relative_links_verified=verified,
        source_sha256=source_hashes,
        bundle_sha256={p.relative_to(DEST).as_posix(): sha(p) for p in files})
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for rel, digest in manifest["bundle_sha256"].items():
            assert hashlib.sha256(z.read(rel)).hexdigest() == digest
        count = len(z.namelist())
    for rel, digest in source_hashes.items():
        assert sha(ROOT / rel) == digest, rel
    print(json.dumps(dict(status="COMPLETE", archive=str(ARCHIVE), files=count,
        bytes=ARCHIVE.stat().st_size, relative_links_verified=verified,
        zip_sha256=sha(ARCHIVE)), ensure_ascii=True))


if __name__ == "__main__":
    main()
