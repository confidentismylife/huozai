from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/qa_tools/python_deps"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_haghani_external import file_hash
from run_haghani_external import route_prompt, score_case


def test_locked_cases_have_no_labels_or_repeated_ids():
    path = ROOT / "outputs/haghani_external/locked_input_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    audit = json.loads((path.parent / "data_audit.json").read_text(encoding="utf-8"))
    assert file_hash(path) == audit["locked_input_sha256"]
    assert len(cases) == 90
    assert len({case["record_id"] for case in cases}) == 90
    assert {size: sum(c["choice_set_size"] == size for c in cases) for size in (2, 3, 4)} == {2: 30, 3: 30, 4: 30}
    assert all("selected_exit" not in case for case in cases)


def test_prompt_hides_unseen_exit_distance_and_congestion():
    case = {
        "alternatives": [
            {"exit": 1, "distance_m": 11.2, "congestion_people": 3, "visible": True, "flow_to_exit_people": 2},
            {"exit": 2, "distance_m": 987.6, "congestion_people": 999, "visible": False, "flow_to_exit_people": 1},
        ]
    }
    prompt = route_prompt(case)
    assert "11.2米" in prompt
    assert "987.6" not in prompt
    assert "999" not in prompt
    assert "拥堵人数未知" in prompt


def test_multiclass_ties_receive_fractional_credit():
    scored = score_case([0.4, 0.4, 0.2], 2)
    assert scored["top1_ties_fractional"] == 0.5
    assert abs(scored["brier_multiclass"] - 0.56) < 1e-12
