from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_relative_plan_distillation import (
    ARMS, METRICS, VIRTUAL_GROUPS, append_expectations, paired_family_deltas,
    bootstrap_effect, summarize,
)


def episode(arm, family="family_a", future=1, value=.2):
    row = {"arm": arm, "state_id": family + "_0", "source_id": family,
           "step": 0, "forecast_wrong": False, "supported_world_count": 2,
           "future_seed": future, "conflict_type": "blocked", "selected_plan": arm}
    row.update({metric: value for metric in METRICS})
    return row


def test_five_initialization_expectation_averages_outcomes_and_never_votes_on_actions():
    rows = [episode(arm, value=index / 10) for members in VIRTUAL_GROUPS.values()
            for index, arm in enumerate(members)]
    result = append_expectations(rows)
    virtual = [r for r in result if r["arm"] in VIRTUAL_GROUPS]
    assert len(virtual) == len(VIRTUAL_GROUPS)
    assert all(r["selected_plan"] == "expected_uniform_member_outcome" for r in virtual)
    np.testing.assert_allclose([r["final_incapacitation"] for r in virtual], .2)


def test_five_initialization_expectation_rejects_missing_or_duplicate_seed_records():
    rows = [episode(arm) for members in VIRTUAL_GROUPS.values() for arm in members]
    with pytest.raises(ValueError, match="five aligned"):
        append_expectations(rows[1:])
    with pytest.raises(ValueError, match="five aligned"):
        append_expectations(rows + [rows[0]])


def test_bootstrap_unit_is_family_after_within_family_repeat_average():
    control, treatment = [], []
    for family, values in (("family_a", [.1, .3, .5]), ("family_b", [.9])):
        for future, value in enumerate(values):
            control.append(episode("control", family, future, 0))
            treatment.append(episode("treatment", family, future, value))
    deltas = paired_family_deltas(treatment, control)
    assert deltas == {"family_a": pytest.approx(.3), "family_b": .9}
    effect = bootstrap_effect(deltas, np.random.default_rng(20261703))
    assert effect["families"] == 2
    assert effect["delta_pp"] == pytest.approx(60)  # Equal families, not .45 pooled rows.


def test_paired_comparison_rejects_different_futures_or_duplicate_rows():
    with pytest.raises(ValueError, match="identical unique"):
        paired_family_deltas([episode("treatment", future=1)], [episode("control", future=2)])
    row = episode("control")
    with pytest.raises(ValueError, match="duplicate control"):
        paired_family_deltas([episode("treatment")], [row, row])


def test_factorial_interaction_is_direct_family_difference_of_differences(tmp_path):
    values = {"small_mse": .2, "small_rank": .18, "large_mse": .15, "large_rank": .10}
    episodes = []
    for family in ("a", "b"):
        for arm in ARMS:
            group = arm.rsplit("_", 1)[0]
            episodes.append(episode(arm, family, value=values.get(group, .2)))
    summaries, effects, interactions = summarize(episodes, tmp_path)
    interaction = next(r for r in interactions if r["subset"] == "initial")
    assert interaction["families"] == 2
    assert interaction["delta_pp"] == pytest.approx(-3)
    assert interaction["exploratory_ci_low_pp"] == pytest.approx(-3)
    assert interaction["exploratory_ci_high_pp"] == pytest.approx(-3)
    mean = next(r for r in summaries if r["subset"] == "initial" and r["arm"] == "five_large_rank_expectation")
    assert mean["records"] == 2 and mean["families"] == 2
    assert mean["final_incapacitation"] == pytest.approx(.1)
    assert (tmp_path / "family_paired_deltas.csv").is_file()
