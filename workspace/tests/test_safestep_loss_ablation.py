"""Protect the anchoring/scale contrasts and their interpretation under Adam."""
from copy import deepcopy
import unittest

import torch

from fire_evac.safestep_frequency_ablation import public_vector_loss
from fire_evac.safestep_loss_ablation import variant_loss


class LossVariantTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.policy = torch.nn.Linear(3, 4, bias=False)
        self.target = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.policy.weight.fill_(.7)
            self.target.weight.copy_(torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]))
        self.data = (torch.eye(3), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]),
                     torch.tensor([0, 1, 2]), torch.tensor([1., 2., 3.]), torch.tensor([True, False, False]), .95)

    def test_original_loss_and_shared_targets_exact(self):
        old = public_vector_loss(self.policy, self.target, *self.data)
        original = variant_loss("full_anchor", self.policy, self.target, *self.data)
        for a, b in zip(old, original):
            self.assertTrue(torch.equal(a, b))
        for variant in ("selected", "selected_quarter"):
            result = variant_loss(variant, self.policy, self.target, *self.data)
            for a, b in zip(original[1:], result[1:]):
                self.assertTrue(torch.equal(a, b))

    def test_gradient_components_and_factor_four(self):
        gradients = {}
        for name in ("full_anchor", "selected", "selected_quarter"):
            policy = deepcopy(self.policy)
            loss = variant_loss(name, policy, self.target, *self.data)[0]
            loss.backward()
            gradients[name] = policy.weight.grad.clone()
        for state in range(3):
            chosen = state
            torch.testing.assert_close(gradients["full_anchor"][chosen, state],
                                       gradients["selected_quarter"][chosen, state])
            for action in range(4):
                if action != chosen:
                    self.assertEqual(float(gradients["selected_quarter"][action, state]), 0.)
                    self.assertNotEqual(float(gradients["full_anchor"][action, state]), 0.)
        torch.testing.assert_close(gradients["selected"], 4 * gradients["selected_quarter"])
        self.assertTrue(all(p.grad is None for p in self.target.parameters()))

    def test_adam_step_is_not_four_times_larger(self):
        results = {}
        for name in ("selected", "selected_quarter"):
            policy = deepcopy(self.policy)
            opt = torch.optim.Adam(policy.parameters(), lr=.008)
            loss = variant_loss(name, policy, self.target, *self.data)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            results[name] = policy.weight.detach().clone()
        torch.testing.assert_close(results["selected"], results["selected_quarter"], atol=2e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
