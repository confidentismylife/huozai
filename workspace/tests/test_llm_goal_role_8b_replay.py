"""Protect the frozen question set and request budget during a model switch."""
from pathlib import Path
import copy
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_llm_goal_role_8b import replay_requests

BASE = Path(__file__).resolve().parents[1] / 'outputs/llm_goal_role_20261006/manifest.json'


def test_replay_changes_only_model_and_preserves_private_reference_and_order():
    source = json.loads(BASE.read_text(encoding='utf-8'))
    preserved = copy.deepcopy(source)
    changed = replay_requests(source)
    assert source == preserved
    assert len(changed) == 32
    for before, after in zip(source['request_order'], changed):
        assert {k for k in before if before[k] != after[k]} == {'payload', 'payload_sha256'}
        assert {k for k in before['payload'] if before['payload'][k] != after['payload'][k]} == {'model'}
        assert after['payload']['model'] == 'qwen3:8b'
        assert after['private_reference'] == before['private_reference']
    changed[0]['payload']['options']['num_predict'] = 1
    assert source == preserved


def test_replay_rejects_unexpected_baseline_model():
    source = json.loads(BASE.read_text(encoding='utf-8'))
    source['request_order'][0]['payload']['model'] = 'unverified_model'
    with pytest.raises(AssertionError):
        replay_requests(source)
