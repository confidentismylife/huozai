"""Build v5 DOCX by reusing the visually verified v4 typesetter."""
from pathlib import Path
import hashlib
import importlib.util
import json


ROOT = Path(__file__).resolve().parents[1]
V4_SCRIPT = ROOT / "scripts/build_llm_ppo_journal_v4_20261021.py"
OUT = ROOT / "artifacts/llm_ppo_paper_v5_20261021"
MD = OUT / "固定LLM建议与PPO疏散控制_期刊式改写稿_v5.md"
DOCX = MD.with_suffix(".docx")


def main() -> None:
    spec = importlib.util.spec_from_file_location("llm_ppo_v4_typesetter", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load v4 typesetter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.MD = MD
    module.DOCX = DOCX
    module.build()

    record_path = OUT / "evidence_and_layout_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["version"] = "v5"
    record["parent_docx_sha256"] = hashlib.sha256(
        (ROOT / "artifacts/llm_ppo_paper_v4_20261021/固定LLM建议与PPO疏散控制_期刊式改写稿_v4.docx").read_bytes()
    ).hexdigest()
    record["v5_revision_scope"] = "post-hoc precision audit, action diagnostics, and tighter task-design interpretation"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(DOCX).encode("ascii", "backslashreplace").decode())


if __name__ == "__main__":
    main()
