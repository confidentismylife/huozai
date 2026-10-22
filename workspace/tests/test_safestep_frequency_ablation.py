"""Checks protecting the single-factor interpretation of the diagnostic."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fire_evac.safestep_author_numerics import AuthorDQN, load_author_inputs
from fire_evac.safestep_author_training import load_original_modules
from fire_evac.safestep_frequency_ablation import public_vector_loss, schedule

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/safestep_paper_numerics_20261015/author_repository"


class FrequencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_vector_loss_and_adam_match_original_scalar_optimizer(self):
        maze, alg = load_original_modules(SOURCE)
        inputs = load_author_inputs(SOURCE)
        env = SimpleNamespace()
        env.unwrapped = SimpleNamespace(maze_environment=SimpleNamespace(empty=inputs["empty"], actions=maze.Maze.actions))
        original = alg.DQL_NetworkModel(env)
        torch.manual_seed(705)
        policy, target = AuthorDQN(61), AuthorDQN(61)
        scalar_policy, scalar_target = deepcopy(policy), deepcopy(target)
        original.optimizer = torch.optim.Adam(scalar_policy.parameters(), lr=.008)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.008)
        rng = np.random.default_rng(92)
        identity = torch.eye(61)
        for update in range(8):
            states = rng.integers(0, 61, size=64)
            actions = rng.integers(0, 4, size=64)
            nxt = rng.integers(0, 61, size=64)
            rewards = rng.normal(size=64)
            status = rng.integers(0, 3, size=64)
            batch = [(inputs["empty"][s], maze.Action(int(a)), inputs["empty"][n], float(r), maze.Status(int(t)))
                     for s, a, n, r, t in zip(states, actions, nxt, rewards, status)]
            old = original.optimize(.95, batch, scalar_policy, scalar_target)
            loss, _, _, _ = public_vector_loss(policy, target, identity, torch.tensor(states), torch.tensor(actions),
                torch.tensor(nxt), torch.tensor(rewards, dtype=torch.float32), torch.tensor(status == 0), .95)
            optimizer.zero_grad()
            loss.backward()
            for a, b in zip(policy.parameters(), scalar_policy.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
            optimizer.step()
            torch.testing.assert_close(loss, old, rtol=2e-5, atol=2e-6)
            for a, b in zip(policy.parameters(), scalar_policy.parameters()):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_timeout_keeps_bootstrap_and_unchosen_anchors(self):
        policy, target = AuthorDQN(61), AuthorDQN(61)
        with torch.no_grad():
            for model in (policy, target):
                for p in model.parameters():
                    p.zero_()
            target.out.bias.copy_(torch.tensor([1., 2., 3., 4.]))
        loss, q, labels, anchor = public_vector_loss(policy, target, torch.eye(61),
            torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.tensor([61, 4, 5]),
            torch.ones(3), torch.tensor([True, False, False]), .95)
        torch.testing.assert_close(labels, torch.tensor([1., 4.8, 4.8]))
        expected = torch.tensor([[1., 2., 3., 4.], [1., 4.8, 3., 4.], [1., 2., 4.8, 4.]])
        torch.testing.assert_close(loss, (expected ** 2).mean())
        self.assertGreater(float(anchor), 0.)

    def test_schedule_changes_only_update_frequency(self):
        decays, syncs, updates = {}, {}, {}
        for arm in ("episode_end", "every_step"):
            last = 0
            decays[arm], syncs[arm], updates[arm] = [], [], []
            for step in range(1, 302):
                update, decay, sync = schedule(arm, step, step in (100, 200, 301), min(step, 1000), last)
                if update:
                    updates[arm].append(step)
                if decay:
                    decays[arm].append(step)
                if sync:
                    syncs[arm].append(step)
                    last = step
        self.assertEqual(updates["episode_end"], [100, 200, 301])
        self.assertEqual(updates["every_step"], list(range(65, 302)))
        self.assertEqual(syncs["episode_end"], syncs["every_step"])
        self.assertEqual(decays["episode_end"], decays["every_step"])
        self.assertEqual(schedule("episode_end", 70, False, 70, 0), (False, False, False))
        self.assertEqual(schedule("every_step", 70, False, 70, 0), (True, False, False))
        self.assertEqual(schedule("episode_end", 150, True, 150, 100), (True, True, False))


if __name__ == "__main__":
    unittest.main()
