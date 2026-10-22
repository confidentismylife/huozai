"""Re-run the frozen read-only audit with NumPy console JSON serialization.

The frozen audit already wrote its complete output before console printing failed.
This wrapper preserves that version and only supplies the missing JSON default.
No simulation, model fit, or LLM call is introduced.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/observable_teacher_label_audit_development_v1"
SOURCE = ROOT / "scripts/audit_observable_teacher_labels.py"


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    manifest = json.loads((OUT / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest["code_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_observable_audit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_dumps = json.dumps

    def scalar_dumps(*args, **kwargs):
        kwargs.setdefault("default", numpy_scalar)
        return original_dumps(*args, **kwargs)

    try:
        json.dumps = scalar_dumps
        module.main()
    finally:
        json.dumps = original_dumps
    print(original_dumps({"cached_analysis_console_repair_completed": True,
                          "frozen_source_unchanged": True,
                          "new_simulations": 0, "new_training_runs": 0, "new_llm_calls": 0}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
