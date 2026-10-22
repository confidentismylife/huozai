"""Verify v5 manuscript, frozen-source diagnostics, DOCX structure, and PDF."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import csv
import hashlib
import json
import math
import re

from lxml import etree
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/llm_ppo_paper_v5_20261021"
MD = OUT / "固定LLM建议与PPO疏散控制_期刊式改写稿_v5.md"
DOCX = MD.with_suffix(".docx")
PDF = ROOT / "output/pdf" / MD.with_suffix(".pdf").name
CSV = ROOT / "outputs/final_pipeline_context_20261020/combined_episode_results.csv"
QA = ROOT / "tmp/pdfs/v5_qa"
source = MD.read_text(encoding="utf-8")
checks: dict[str, bool] = {}


def check(name: str, condition: bool) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


check("seven_numbered_sections", len(re.findall(r"^## [1-7] ", source, re.M)) == 7)
check("six_body_tables", sum(1 for line in source.splitlines() if re.fullmatch(r"表\d+ .+", line)) == 6)
check("three_numbered_equations", len(re.findall(r"\\tag\{[123]\}", source)) == 3)
images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
check("five_images_resolve", len(images) == 5 and all((OUT / path).is_file() for path in images))
check("precision_scope_present", "事后精度审计" in source and "仅作为后续研究的启发式量" in source)
check("mde_not_exclusion_bound", "不能把1.9个百分点误写为本研究已经排除的效应界限" in source)
check("future_mid_not_prespecified", "当前研究没有预先设定或验证这一界值" in source)
check("no_general_llm_invalidity_claim", "不能据此推断LLM在疏散控制中普遍无效" in source)
check("no_proven_presence_only_claim", "只学到建议存在" not in source)
check("action_mismatch_caveat", "现有聚合记录没有分解所有来源" in source)

with CSV.open(encoding="utf-8-sig", newline="") as stream:
    all_rows = list(csv.DictReader(stream))
rows = [row for row in all_rows if row["behavior_model"] == "calibrated_logit" and row["noise_level"] == "none"]
check("main_rows_720", len(rows) == 720)
groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)
expected = {
    "trained_llm_advice_ppo": (0.0, 0, 7460, 0.9237480100052965),
    "trained_pure_ppo": (0.02311009594786769, 19, 6380, 0.933489102421004),
    "trained_rule_advice_ppo": (0.07199702577105169, 55, 6895, 0.9401389861212616),
}
for method, (wait_mean, wait_episodes, steps, match_mean) in expected.items():
    group = groups[method]
    waits = [float(row["wait_fraction"]) for row in group]
    matches = [float(row["requested_actual_joint_match_rate"]) for row in group]
    check(method + "_episodes", len(group) == 240)
    check(method + "_wait_mean", math.isclose(sum(waits) / len(waits), wait_mean, abs_tol=1e-14))
    check(method + "_wait_episodes", sum(value > 0 for value in waits) == wait_episodes)
    check(method + "_steps", sum(int(row["steps"]) for row in group) == steps)
    check(method + "_match", math.isclose(sum(matches) / len(matches), match_mean, abs_tol=1e-14))

audit = json.loads((OUT / "新增精度与动作诊断.json").read_text(encoding="utf-8"))
for method, (wait_mean, wait_episodes, steps, match_mean) in expected.items():
    item = audit["action_diagnostics"][method]
    check("audit_" + method + "_wait", math.isclose(item["mean_wait_fraction"], wait_mean, abs_tol=1e-14))
    check("audit_" + method + "_count", item["episodes_with_wait"] == wait_episodes)
    check("audit_" + method + "_steps", item["total_control_steps"] == steps)
    check("audit_" + method + "_match", math.isclose(item["mean_requested_actual_joint_match_rate"], match_mean, abs_tol=1e-14))
precision = audit["precision_audit"]
check("precision_is_post_hoc", precision["status"] == "post_hoc_planning_sensitivity_not_confirmatory_evidence")
check("precision_design", precision["scenarios"] == 48 and precision["training_seeds"] == 5 and precision["episodes"] == 240)
check("precision_estimate", round(precision["mean_difference_fraction"] * 100, 4) == -0.6632)
check("precision_se", 0.68 < precision["bootstrap_derived_se_percentage_points"] < 0.70)
check("precision_mde", 1.90 < precision["normal_approx_mde_80pct_percentage_points"] < 1.95)
check("precision_power", 0.15 < precision["normal_approx_power_at_observed_effect"] < 0.17)
check("precision_effective_n", precision["iid_equivalent_effective_n_heuristic"] == 72)

for token in [
    "−0.6632个百分点［−2.3160，0.4132］",
    "0.000669［−0.012287，0.013746］",
    "22.50 s［−0.65，40.52］",
    "−1.1215个百分点［−2.8455，0.2500］",
    "0.1374［0.0386，0.2188］",
    "25.52 s［−1.15，47.27］",
    "0.896",
]:
    check("main_text_" + hashlib.sha1(token.encode()).hexdigest()[:8], token in source)

revision = json.loads((OUT / "revision_sources.json").read_text(encoding="utf-8"))
for path, digest in revision["parent_and_sources"].items():
    check("unchanged_" + path, hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest)
evidence = json.loads((OUT / "evidence_and_layout_record.json").read_text(encoding="utf-8"))
check("manuscript_hash_current", hashlib.sha256(MD.read_bytes()).hexdigest() == evidence["manuscript_sha256"])

ns = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}
with ZipFile(DOCX) as archive:
    xml = etree.fromstring(archive.read("word/document.xml"))
    check("docx_seven_tables_including_abstract", len(xml.findall(".//w:tbl", ns)) == 7)
    check("docx_three_native_math_equations", len(xml.findall(".//m:oMath", ns)) == 3)
    check("docx_five_embedded_figures", len(xml.findall(".//wp:inline", ns)) == 5)
    check("docx_no_raw_latex", b"\\tag{" not in archive.read("word/document.xml"))

pdf = PdfReader(PDF)
check("pdf_twelve_pages", len(pdf.pages) == 12)
pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
check("pdf_no_replacement_glyph", "\ufffd" not in pdf_text)
check("pdf_contains_precision", "1.9" in pdf_text and "16%" in pdf_text)
check("pdf_contains_action_diagnostics", "7460" in pdf_text and "0.9237" in pdf_text)
check("all_final_page_images_exist", all((QA / f"final-{number:02d}.png").is_file() for number in range(1, 13)))

record = {
    "status": "passed_assistant_checks_requires_author_review",
    "automated_checks": checks,
    "passed_count": len(checks),
    "final_pdf_pages": len(pdf.pages),
    "visual_inspection": {
        "by": "AI assistant",
        "pages": list(range(1, 13)),
        "render": "Microsoft Word PDF export then Poppler PNG rendering",
        "result": "No observed clipping, overlap, missing glyphs, or broken tables.",
        "human_verified": False,
    },
    "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [MD, DOCX, PDF]},
    "new_training": 0,
    "new_simulation_episodes": 0,
    "new_llm_calls": 0,
}
(OUT / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": record["status"], "checks": len(checks), "pdf_pages": len(pdf.pages)}, ensure_ascii=True))
