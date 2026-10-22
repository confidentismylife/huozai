from __future__ import annotations

from pathlib import Path
import sys
import csv
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import EpisodeBehaviorEnsemble, MixedLogitBehavior, RuleBehavior
from fire_evac.controllers import HazardAwareController
from fire_evac.behavior_data import (
    classification_metrics,
    fit_hierarchical_logit,
    generate_synthetic_behavior_data,
    rows_for_participants,
    split_participants,
)
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import run_episode
from fire_evac.llm_interface import LLMDecisionError, canonical_state_hash, parse_llm_decision
from fire_evac.qlearning import train_q_learning
from fire_evac.gym_env import GymFireEvacuationEnv, HistoryStackWrapper
from fire_evac.comparison import hierarchical_bootstrap_multiseed, paired_bootstrap_comparisons
from fire_evac.hazard import FDSCsvHazardProvider, ZONE_NAMES
from fire_evac.ppo import RecurrentPPOController
from fire_evac.robust_training import worst_case_sampling_weights, worst_case_score


class EnvironmentTests(unittest.TestCase):
    def test_reset_is_reproducible(self) -> None:
        env = FireEvacuationEnv(MixedLogitBehavior())
        obs1, info1 = env.reset(seed=123)
        obs2, info2 = env.reset(seed=123)
        np.testing.assert_allclose(obs1, obs2)
        self.assertEqual(info1["scenario"], info2["scenario"])

    def test_episode_conserves_population(self) -> None:
        env = FireEvacuationEnv(RuleBehavior())
        result = run_episode(env, HazardAwareController(), seed=456)
        total = result["evacuation_rate"] + result["incapacitation_rate"] + result["remaining_rate"]
        self.assertAlmostEqual(total, 1.0, places=9)
        self.assertGreaterEqual(result["evacuation_rate"], 0.0)

    def test_q_learning_produces_policy(self) -> None:
        env = FireEvacuationEnv(MixedLogitBehavior())
        controller = train_q_learning(env, episodes=25, seed=7)
        obs, _ = env.reset(seed=8)
        self.assertIn(controller.act(obs), (0, 1, 2))
        self.assertGreater(len(controller.q), 0)

    def test_llm_decision_schema(self) -> None:
        payload = {
            "start_evacuation": True,
            "target_exit": "exit_B",
            "follow_sign": False,
            "follow_crowd": True,
            "decision_delay_s": 6.8,
            "confidence": 0.62,
            "reason_codes": ["familiar_route", "follow_majority"],
        }
        decision = parse_llm_decision(payload)
        self.assertEqual(decision.target_exit, "exit_B")
        self.assertEqual(canonical_state_hash(payload), canonical_state_hash(dict(reversed(list(payload.items())))))
        with self.assertRaises(LLMDecisionError):
            parse_llm_decision({**payload, "confidence": 1.5})

    def test_participant_split_has_no_leakage(self) -> None:
        rows = generate_synthetic_behavior_data(participants=20, trials_per_participant=8, seed=91)
        split = split_participants(rows, seed=12)
        split.assert_disjoint()
        self.assertEqual(set(split.train) | set(split.validation) | set(split.test), {r.participant_id for r in rows})

    def test_behavior_model_fits_and_generalizes(self) -> None:
        rows = generate_synthetic_behavior_data(participants=40, trials_per_participant=15, seed=92)
        split = split_participants(rows, seed=13)
        train = rows_for_participants(rows, split.train)
        test = rows_for_participants(rows, split.test)
        model = fit_hierarchical_logit(train)
        metrics = classification_metrics(test, model.predict_probability(test))
        self.assertGreater(metrics["accuracy"], 0.60)
        self.assertLess(metrics["log_loss"], 0.68)

    def test_gymnasium_adapter(self) -> None:
        env = GymFireEvacuationEnv(MixedLogitBehavior())
        observation, info = env.reset(seed=77)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["step"], 0)
        observation, reward, terminated, truncated, info = env.step(2)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated and truncated)

    def test_paired_bootstrap_keeps_episode_pairing(self) -> None:
        rows = []
        for seed in range(10):
            rows.append({"behavior_model": "demo", "controller": "ppo", "seed": seed, "loss": float(seed)})
            rows.append({"behavior_model": "demo", "controller": "neutral", "seed": seed, "loss": float(seed + 1)})
        result = paired_bootstrap_comparisons(
            rows, candidate="ppo", baselines=("neutral",), metrics=("loss",), bootstrap_samples=200, seed=4
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result[0]["mean_difference"]), -1.0)
        self.assertAlmostEqual(float(result[0]["ci_2_5"]), -1.0)

    def test_fds_provider_is_drop_in_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fds.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scenario_id", "step", "zone", "fed_increment"])
                writer.writeheader()
                for step in (0, 1):
                    for zone_index, zone in enumerate(ZONE_NAMES):
                        writer.writerow(
                            {"scenario_id": "fds-demo", "step": step, "zone": zone, "fed_increment": 0.001 * (zone_index + 1)}
                        )
            provider = FDSCsvHazardProvider(path)
            scenario = provider.sample_scenario(np.random.default_rng(2))
            self.assertEqual(scenario.scenario_id, "fds-demo")
            np.testing.assert_allclose(provider.values(scenario, 1), np.arange(1, 6) * 0.001)

            incomplete = Path(directory) / "incomplete.csv"
            incomplete.write_text("scenario_id,step,zone,fed_increment\na,0,room_a,0.01\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                FDSCsvHazardProvider(incomplete)

    def test_hierarchical_bootstrap_uses_training_seeds(self) -> None:
        rows = []
        for training_seed in (1, 2):
            for episode_seed in range(5):
                rows.append(
                    {
                        "behavior_model": "demo",
                        "controller": "ppo",
                        "training_seed": training_seed,
                        "seed": episode_seed,
                        "loss": float(episode_seed - training_seed),
                    }
                )
        for episode_seed in range(5):
            rows.append(
                {
                    "behavior_model": "demo",
                    "controller": "neutral",
                    "training_seed": "baseline",
                    "seed": episode_seed,
                    "loss": float(episode_seed),
                }
            )
        result = hierarchical_bootstrap_multiseed(
            rows, candidate="ppo", baselines=("neutral",), metrics=("loss",), bootstrap_samples=200, seed=9
        )
        self.assertEqual(int(result[0]["training_seeds"]), 2)
        self.assertAlmostEqual(float(result[0]["mean_difference"]), -1.5)

    def test_history_stack_shape_and_shift(self) -> None:
        env = HistoryStackWrapper(GymFireEvacuationEnv(MixedLogitBehavior()), history_length=4)
        observation, _ = env.reset(seed=17)
        self.assertEqual(observation.shape, (56,))
        np.testing.assert_allclose(observation[:14], observation[-14:])
        next_observation, _, _, _, _ = env.step(2)
        np.testing.assert_allclose(next_observation[:42], observation[14:])

    def test_behavior_ensemble_selects_once_per_episode(self) -> None:
        ensemble = EpisodeBehaviorEnsemble((RuleBehavior, MixedLogitBehavior))
        env = FireEvacuationEnv(ensemble)
        _, info = env.reset(seed=23)
        selected = info["active_behavior_model"]
        self.assertIn(selected, {"rule", "mixed_logit"})
        _, _, _, _, info_after_step = env.step(2)
        self.assertEqual(info_after_step["active_behavior_model"], selected)

    def test_recurrent_controller_resets_hidden_state(self) -> None:
        class FakeRecurrentModel:
            def __init__(self) -> None:
                self.episode_starts: list[bool] = []

            def predict(self, observation, state, episode_start, deterministic):
                del observation, deterministic
                self.episode_starts.append(bool(episode_start[0]))
                next_state = 1 if state is None else int(state) + 1
                return np.array([2]), next_state

        model = FakeRecurrentModel()
        controller = RecurrentPPOController(model)
        observation = np.zeros(14, dtype=np.float32)
        self.assertEqual(controller.act(observation), 2)
        self.assertEqual(controller.act(observation), 2)
        controller.reset()
        self.assertEqual(controller.act(observation), 2)
        self.assertEqual(model.episode_starts, [True, False, True])

    def test_worst_case_weights_focus_on_low_reward(self) -> None:
        rewards = {"a": -0.1, "b": -0.8, "c": -0.2}
        weights = worst_case_sampling_weights(rewards, ("a", "b", "c"), temperature=0.1, exploration_mass=0.3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(int(np.argmax(weights)), 1)
        self.assertTrue(np.all(weights >= 0.1 - 1e-12))
        self.assertAlmostEqual(worst_case_score(rewards), -0.8)


if __name__ == "__main__":
    unittest.main()

