"""Validate and render the report; never run evacuation simulations."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
NAME = "LLM_RL火灾疏散全流程实验研究总报告_2026-10-02"
ASSETS = ROOT / "artifacts/full_history_report_assets"
WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def plain(text):
    code = []
    def protect(match):
        code.append(match[1])
        return f"\uf000{len(code)-1}\uf001"
    text = re.sub(chr(96) + r"([^" + chr(96) + r"]+)" + chr(96), protect, text)
    text = re.sub(r"!?\[([^\]]*)\]\(([^)]+)\)", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    for i, value in enumerate(code):
        text = text.replace(f"\uf000{i}\uf001", value)
    return text


def build(only_source=False):
    md = ROOT / "reports" / f"{NAME}.md"
    text = md.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text)
    local = sorted({target for target in targets if re.match(r"^[A-Za-z]:[/\\]", target)})
    missing = [target for target in local if not Path(target).exists()]
    chapters = re.findall(r"(?m)^## (\d+)\.", text)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(md), "source_characters": len(text),
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "chapters": chapters, "chapters_contiguous_1_to_19": chapters == [str(i) for i in range(1, 20)],
        "local_link_count": len(local), "missing_local_links": missing,
        "placeholder_present": "<!--" in text or "EVIDENCE_INDEX" in text,
        "referenced_temporary_clipboard_files": "/Temp/codex-clipboard" in text,
    }
    assert not missing and report["chapters_contiguous_1_to_19"]
    assert not report["placeholder_present"] and not report["referenced_temporary_clipboard_files"]
    if only_source:
        print(json.dumps(report, ensure_ascii=False))
        return report
    docx = ROOT / "artifacts" / f"{NAME}.docx"
    pdf = ROOT / "artifacts" / f"{NAME}.pdf"
    with zipfile.ZipFile(docx) as archive:
        xml = ET.fromstring(archive.read("word/document.xml"))
        word_text = "".join(item.text or "" for item in xml.iter(WNS + "t"))
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        hyperlinks = [rel.attrib.get("Target") for rel in rels if rel.attrib.get("Type", "").endswith("/hyperlink")]
    expected = []
    in_code = False
    for line in text.splitlines():
        if line.startswith(chr(96) * 3):
            in_code = not in_code
            continue
        if not line.strip() or (line.startswith("# ") and not in_code):
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                expected.extend(plain(cell) for cell in cells)
        elif in_code:
            expected.append(line)
        else:
            expected.append(plain(re.sub(r"^(?:#{2,4} |[-] |\d+\. )", "", line)))
    normalized_word = normalize(word_text)
    word_missing = [item[:150] for item in expected if normalize(item) not in normalized_word]
    report.update({
        "docx_tables": len(list(xml.iter(WNS + "tbl"))),
        "docx_inline_images": len(list(xml.iter(WNS + "drawing"))),
        "docx_unique_hyperlinks": len(hyperlinks),
        "expected_text_units": len(expected),
        "docx_missing_text_units": word_missing,
    })
    assert not word_missing, word_missing
    pdf_doc = fitz.open(pdf)
    pdf_text = "\n".join(
        "".join(block[4] for block in page.get_text("blocks")
                if block[1] >= 49 and block[3] <= page.rect.height-50)
        for page in pdf_doc
    )
    (ASSETS / "report_pdf_extracted_text.txt").write_text(pdf_text, encoding="utf-8")
    normalized_pdf = normalize(pdf_text)
    pdf_missing = [item[:150] for item in expected if normalize(item) not in normalized_pdf]
    empty_pages = [i + 1 for i, page in enumerate(pdf_doc) if not page.get_text().strip() and not page.get_images()]
    outside = []
    for i, page in enumerate(pdf_doc):
        for block in page.get_text("blocks"):
            if block[0] < -1 or block[1] < -1 or block[2] > page.rect.width + 1 or block[3] > page.rect.height + 1:
                outside.append({"page": i + 1, "bbox": list(block[:4])})
    report.update({
        "pdf_pages": len(pdf_doc), "pdf_text_characters": len(pdf_text),
        "pdf_missing_text_units": pdf_missing, "pdf_empty_pages": empty_pages,
        "pdf_text_blocks_outside_page": outside,
        "pdf_replacement_character_count": pdf_text.count("\ufffd"),
    })
    thumb_paths = []
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    for start in range(0, len(pdf_doc), 9):
        sheet = Image.new("RGB", (720, 1080), "white")
        draw = ImageDraw.Draw(sheet)
        for index in range(start, min(start + 9, len(pdf_doc))):
            pixmap = pdf_doc[index].get_pixmap(matrix=fitz.Matrix(.37, .37), alpha=False)
            thumbnail = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            thumbnail.thumbnail((225, 327))
            slot = index - start
            x, y = 8 + (slot % 3) * 240, 8 + (slot // 3) * 360
            draw.text((x, y), f"第{index+1}页", font=font, fill="#25384a")
            sheet.paste(thumbnail, (x, y + 24))
        target = ASSETS / f"contact_sheet_{start//9+1:02d}.png"
        sheet.save(target)
        thumb_paths.append(str(target))
    detailed = []
    markers = ("图1：研究路线", "五分区容量流", "合成校准Logit失能率", "第一次确认性实验", "最新2×2开发实验", "相同九动作的第一次", "小MSE五初始化期望")
    detail_pages = {0, 1, len(pdf_doc) - 1}
    for i, page in enumerate(pdf_doc):
        if any(normalize(marker) in normalize(page.get_text()) for marker in markers):
            detail_pages.add(i)
    for i in sorted(detail_pages):
        target = ASSETS / f"review_page_{i+1:03d}.png"
        pdf_doc[i].get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(target)
        detailed.append(str(target))
    report.update({"contact_sheets": thumb_paths, "detailed_review_pages": detailed})
    audit = json.loads((ASSETS / "experiment_preservation_audit.json").read_text(encoding="utf-8"))
    changed = [record["path"] for record in audit["records"] if digest(Path(record["path"])) != record["current_sha256"]]
    report["original_snapshot_records_rechecked"] = len(audit["records"])
    report["original_unique_snapshot_files_rechecked"] = len({record["path"] for record in audit["records"]})
    report["original_experiment_files_changed"] = changed
    report["artifacts"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in (md, docx, pdf)]
    assert not pdf_missing and not empty_pages and not outside and not changed
    target = ASSETS / "report_qa_manifest.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"artifacts", "contact_sheets", "detailed_review_pages", "chapters"}}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    build(parser.parse_args().source_only)
