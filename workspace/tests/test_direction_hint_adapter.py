import numpy as np
import torch
from types import SimpleNamespace
from fire_evac.direction_hint_adapter import direction_hint_action
from fire_evac.timestamped_pipeline import ppo_action


class ToyPolicy:
    def __init__(self, probabilities):
        self.probabilities = torch.tensor([probabilities], dtype=torch.float64)

    def obs_to_tensor(self, observation):
        return observation, False

    def get_distribution(self, observation):
        return SimpleNamespace(distribution=SimpleNamespace(probs=self.probabilities))


def test_target_hint_cannot_change_release_level_when_direction_already_matches():
    model = SimpleNamespace(policy=ToyPolicy([.1, .18, .20, .19, .08, .08, .07, .06, .04]))
    proposal = SimpleNamespace(preferred_branch=0, confidence=.95, north_share=1., wait_steps=0)
    gate = SimpleNamespace(accepted=True, valid_candidate_actions=(0, 1))
    mask = np.ones(9, bool)
    old, _ = ppo_action(model, np.zeros(34), mask, proposal, gate)
    new, _ = direction_hint_action(model, np.zeros(34), mask, proposal, gate)
    assert old == 1  # Old partial candidate bonus changes the release ranking.
    assert new == 2


def test_route_change_preserves_target_branch_release_ranking_and_legality():
    model = SimpleNamespace(policy=ToyPolicy([.1, .22, .1, .08, .24, .09, .07, .06, .04]))
    proposal = SimpleNamespace(preferred_branch=0, confidence=.95)
    gate = SimpleNamespace(accepted=True)
    mask = np.ones(9, bool)
    new, _ = direction_hint_action(model, np.zeros(34), mask, proposal, gate)
    assert new == 1
    mask[:4] = False
    new, _ = direction_hint_action(model, np.zeros(34), mask, proposal, gate)
    assert new == 4


def test_rejected_advice_does_not_change_action():
    model = SimpleNamespace(policy=ToyPolicy([.1, .22, .1, .08, .24, .09, .07, .06, .04]))
    proposal = SimpleNamespace(preferred_branch=0, confidence=.95)
    new, _ = direction_hint_action(model, np.zeros(34), np.ones(9, bool), proposal, SimpleNamespace(accepted=False))
    assert new == 4
