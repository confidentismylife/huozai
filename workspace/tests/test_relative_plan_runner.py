from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_relative_plan_distillation import (
    ARMS, COLLECTION_STEPS, training_seeds, audit_seeds, initial_global_plan,
)


def test_new_random_future_streams_are_unique_and_disjoint_including_sensors():
    training = {s for j in range(60) for r in range(2) for t in range(5) for s in training_seeds(j, r, t)[1]}
    planning = {s for j in range(36) for r in range(2) for t in range(5) for s in audit_seeds(j, r, t)[1]}
    evaluation = {s for j in range(36) for r in range(2) for t in range(5) for s in audit_seeds(j, r, t)[2]}
    streams = [training, planning, evaluation]
    streams += [{s + 9_000_001 for s in stream} for stream in streams.copy()]
    assert [len(stream) for stream in streams] == [1800, 1080, 1080, 1800, 1080, 1080]
    for i, stream in enumerate(streams):
        assert all(not stream & other for other in streams[:i])
    assert COLLECTION_STEPS == (0, 8, 16, 24, 40)
    assert len(ARMS) == len(set(ARMS)) == 38


def test_global_fixed_plan_uses_only_own_train_initial_states_with_equal_family_mass():
    targets = np.ones((6, 17, 6)) * 10
    # Family a has two initial rows, b one. Equal-family means pick plan12.
    targets[:2, 12] = 0
    targets[2, 12] = 2
    targets[:2, 13] = 2
    targets[2, 13] = 0
    targets[2, 13, 0] = .5
    targets[3:, 16] = -1000  # Later-train/calibration/excluded rows cannot steer it.
    families = np.asarray(["a", "a", "b", "b", "cal", "excluded"])
    train = np.asarray([True, True, True, True, False, False])
    steps = np.asarray([0, 0, 0, 8, 0, 0])
    assert initial_global_plan(targets, families, train, steps) == 12
