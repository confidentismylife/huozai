import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fire_evac.safestep_baseline import SmallMap
from fire_evac.safestep_exact_supervision import snapshot_teacher, legal_mse, quality

SOURCE = json.loads((Path(__file__).resolve().parents[1]/'schemas/safestep_shared_observation_20261014.json').read_text(encoding='utf-8'))


def test_exact_snapshot_labels_are_negative_and_include_the_deadline():
    grid = SmallMap.paper_case_map()
    snapshot = np.full(24, .00005/.012, dtype=np.float32)
    snapshot[list(grid.exits)] = 0
    q = snapshot_teacher(grid, snapshot, SOURCE, 2/.3)
    assert q[0, 23, 2] == pytest.approx(-20*8*(.00005*2/.3+.00125), abs=1e-7)
    assert q[127, 23, 2] == pytest.approx(-40-20*(.00005*2/.3+.00125), abs=1e-7)
    assert np.all(q[np.isfinite(q)] < 0)


def test_supervision_ignores_illegal_targets_and_their_gradients():
    predictions = torch.tensor([[0., 0., 100., -200.]], requires_grad=True)
    target = torch.tensor([[-1., -3., 0., 0.]])
    legal = torch.tensor([[True, True, False, False]])
    loss = legal_mse(predictions, target, legal)
    assert loss.item() == 5
    loss.backward()
    assert predictions.grad.tolist() == [[1., 3., 0., 0.]]


def test_negative_predictions_do_not_guarantee_correct_ranking():
    truth = np.array([[-1., -2., 0., 0.]])
    prediction = np.array([[-3., -1., 100., 100.]])
    legal = np.array([[True, True, False, False]])
    scores = quality(prediction, truth, legal, 1e-6)
    assert scores['positive_selected_Q_fraction'] == 0
    assert scores['optimal_action_accuracy'] == 0
    assert scores['mean_action_regret'] == 1
