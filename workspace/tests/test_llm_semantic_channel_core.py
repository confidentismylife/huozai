"""Meaningful interface privacy and direction/confidence isolation checks."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llm_semantic_channel_core import prompt, parse, probes, fixed_advice, feature_advice


def test_repair_changes_semantic_names_without_leaking_outcomes():
    scene = probes()[0]
    original, repaired = prompt(scene, 'original'), prompt(scene, 'repaired')
    changed = dict(scene, source_branch=1, conflict_type='PRIVATE', scenario_id='PRIVATE')
    assert prompt(changed, 'repaired') == repaired
    old_data = json.loads(original.split('Data: ')[1])
    new_data = json.loads(repaired.split('Data: ')[1])
    assert new_data.pop('predicted_fire_location') == old_data.pop('forecast_source')
    assert new_data == old_data
    assert 'computed_safe_target' not in repaired and 'MUST be south' not in repaired


def test_parser_has_no_implicit_direction_flip_or_rule_fallback():
    assert parse('{"recommended_exit":"north","confidence":0.8}', 'repaired')['target_branch'] == 0
    assert parse('{"target":"south","confidence":0.8}', 'original')['target_branch'] == 1
    for raw in ('{"recommended_exit":"bad","confidence":0.8}',
                '{"recommended_exit":"north","confidence":true}',
                '{"recommended_exit":"north","confidence":NaN}',
                '{"recommended_exit":"north","confidence":1.1}'):
        with pytest.raises(ValueError):
            parse(raw, 'repaired')


def test_factorial_only_changes_direction_confidence_and_derived_contradiction():
    observation = np.arange(49, dtype=np.float32) / 100
    untouched = np.r_[np.arange(28), np.arange(33, 49)]
    vectors = {arm: feature_advice(observation, fixed_advice(arm))
               for arm in ('north_c50', 'north_c95', 'south_c50', 'south_c95')}
    for x in vectors.values():
        np.testing.assert_array_equal(x[untouched], observation[untouched])
    np.testing.assert_array_equal(vectors['north_c50'][[28, 29, 31, 32]], vectors['north_c95'][[28, 29, 31, 32]])
    assert vectors['north_c50'][30] != vectors['north_c95'][30]
    assert vectors['north_c95'][30] == vectors['south_c95'][30]
    assert vectors['north_c95'][28] == vectors['south_c95'][29] == 1


def test_probe_mirrors_have_identical_physics_except_fire_side():
    records = probes()
    assert len(records) == 12
    for a, b in zip(records[::2], records[1::2]):
        assert a['forecast_source_branch'] == 0 and b['forecast_source_branch'] == 1
        assert a['north_exit_capacity'] == a['south_exit_capacity']
        for key in a.keys() - {'scenario_id', 'source_branch', 'forecast_source_branch'}:
            assert a[key] == b[key]
