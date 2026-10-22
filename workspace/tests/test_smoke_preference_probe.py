from dataclasses import replace
from fire_evac.timestamped_pipeline import Snapshot, CommonActionConstraints
from fire_evac.smoke_preference_probe import SmokePreferenceProbeConstraints


def snapshot():
    return Snapshot('unit', 3, 15., 'unit', (0.5, 0.5, 0., 0., 0.),
                    (.001, .001, .002, .01, .08), (8, 8), (False, False),
                    0, .01, 0, 0., 1, 0, 20, 240)


def test_only_admissible_direction_preference_changes():
    s = snapshot()
    probe = SmokePreferenceProbeConstraints()
    mask = probe.action_mask(s)
    assert mask[0] and mask[1] and mask[2]
    assert not mask[4:8].any() and mask[8]
    assert probe.branch_allowed(s, 1)
    assert probe.room_limit(s) == CommonActionConstraints().room_limit(s)


def test_hold_is_not_overridden_and_wait_is_not_forced():
    s = replace(snapshot(), sign_hold_remaining=2)
    probe = SmokePreferenceProbeConstraints()
    assert (probe.action_mask(s) == CommonActionConstraints().action_mask(s)).all()
    assert probe.action_mask(s)[4]


def test_closed_lower_exit_preserves_open_direction():
    s = replace(snapshot(), closed=(True, False), capacities=(0, 8))
    assert (SmokePreferenceProbeConstraints().action_mask(s) == CommonActionConstraints().action_mask(s)).all()
