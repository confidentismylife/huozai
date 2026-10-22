from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior import CalibratedLogitBehavior, LLMSurrogateBehavior, MixedLogitBehavior, RuleBehavior
from fire_evac.controllers import HazardAwareController, NeutralController, StaticNorthController
from fire_evac.env import FireEvacuationEnv
from fire_evac.experiment import evaluate, summarize, write_csv
from fire_evac.gym_env import GymFireEvacuationEnv
from fire_evac.ppo import PPOController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and cross-evaluate PPO evacuation guidance.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ppo")
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=ROOT / "outputs" / "behavior_pipeline" / "behavior_model.json",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit("PPO dependencies are missing. Install the project's optional 'rl' dependencies.") from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = CalibratedLogitBehavior(args.calibration_artifact)
    check_env(GymFireEvacuationEnv(calibration), warn=True)
    training_env = Monitor(GymFireEvacuationEnv(calibration))
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        seed=args.seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model_path = args.output_dir / "ppo_fire_evac"
    model.save(model_path)

    controllers = [NeutralController(), StaticNorthController(), HazardAwareController(), PPOController(model)]
    behavior_models = [RuleBehavior(), MixedLogitBehavior(), calibration, LLMSurrogateBehavior()]
    all_rows: list[dict[str, float | int | str]] = []
    for behavior in behavior_models:
        all_rows.extend(
            evaluate(FireEvacuationEnv(behavior), controllers, episodes=args.eval_episodes, seed=20_000)
        )
    summary_rows = summarize(all_rows)
    write_csv(all_rows, args.output_dir / "ppo_episode_results.csv")
    with (args.output_dir / "ppo_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(
            f"{row['behavior_model']:<28} {row['controller']:<18} "
            f"evac={row['evacuation_rate_mean']:.4f} incap={row['incapacitation_rate_mean']:.4f} "
            f"fed={row['mean_cumulative_fed']:.4f} reward={row['reward_mean']:.4f}"
        )
    print(f"Model:   {model_path}.zip")
    print(f"Summary: {args.output_dir / 'ppo_summary.csv'}")


if __name__ == "__main__":
    main()
