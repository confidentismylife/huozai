from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .behavior import BehaviorModel, CalibratedLogitBehavior
from .gym_env import GymFireEvacuationEnv, HistoryStackWrapper


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 50_000
    n_envs: int = 4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    validation_interval_steps: int = 5_000
    validation_episodes: int = 50
    early_stop_min_evals: int = 4
    early_stop_patience: int = 5


def train_ppo_seed(
    calibration_artifact: str | Path,
    output_dir: str | Path,
    seed: int,
    config: PPOTrainingConfig,
    behavior_factory: Callable[[], BehaviorModel] | None = None,
    experiment_label: str = "calibrated_logit",
    history_length: int = 1,
) -> tuple[Path, dict[str, object]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    artifact_path = Path(calibration_artifact).resolve()
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    if behavior_factory is None:
        behavior_factory = lambda: CalibratedLogitBehavior(artifact_path)

    def make_training_env():
        base_env = GymFireEvacuationEnv(behavior_factory())
        return HistoryStackWrapper(base_env, history_length) if history_length > 1 else base_env

    train_env = make_vec_env(make_training_env, n_envs=config.n_envs, seed=seed)
    validation_env = Monitor(make_training_env())
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config.early_stop_patience,
        min_evals=config.early_stop_min_evals,
        verbose=0,
    )
    callback_frequency = max(1, config.validation_interval_steps // config.n_envs)
    eval_callback = EvalCallback(
        validation_env,
        best_model_save_path=str(seed_dir),
        log_path=str(seed_dir),
        eval_freq=callback_frequency,
        n_eval_episodes=config.validation_episodes,
        deterministic=True,
        callback_after_eval=stop_callback,
        warn=False,
        verbose=0,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback, progress_bar=False)
    final_model_path = seed_dir / "final_model"
    model.save(final_model_path)
    best_model_path = seed_dir / "best_model.zip"
    selected_model_path = best_model_path if best_model_path.exists() else final_model_path.with_suffix(".zip")
    evaluations_path = seed_dir / "evaluations.npz"
    best_validation_reward: float | None = None
    if evaluations_path.exists():
        import numpy as np

        evaluations = np.load(evaluations_path)
        best_validation_reward = float(np.max(np.mean(evaluations["results"], axis=1)))
    metadata: dict[str, object] = {
        "training_seed": seed,
        "requested_timesteps": config.total_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "selected_model": str(selected_model_path.resolve()),
        "best_validation_reward": best_validation_reward,
        "calibration_artifact": str(artifact_path),
        "experiment_label": experiment_label,
        "history_length": history_length,
        "config": asdict(config),
    }
    (seed_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_env.close()
    validation_env.close()
    return selected_model_path, metadata


def train_recurrent_ppo_seed(
    calibration_artifact: str | Path,
    output_dir: str | Path,
    seed: int,
    config: PPOTrainingConfig,
    behavior_factory: Callable[[], BehaviorModel] | None = None,
    experiment_label: str = "recurrent_calibrated_logit",
) -> tuple[Path, dict[str, object]]:
    import numpy as np
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    artifact_path = Path(calibration_artifact).resolve()
    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if behavior_factory is None:
        behavior_factory = lambda: CalibratedLogitBehavior(artifact_path)

    def make_training_env() -> GymFireEvacuationEnv:
        return GymFireEvacuationEnv(behavior_factory())

    train_env = make_vec_env(make_training_env, n_envs=config.n_envs, seed=seed)
    validation_env = Monitor(make_training_env())
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config.early_stop_patience,
        min_evals=config.early_stop_min_evals,
        verbose=0,
    )
    eval_callback = EvalCallback(
        validation_env,
        best_model_save_path=str(seed_dir),
        log_path=str(seed_dir),
        eval_freq=max(1, config.validation_interval_steps // config.n_envs),
        n_eval_episodes=config.validation_episodes,
        deterministic=True,
        callback_after_eval=stop_callback,
        warn=False,
        verbose=0,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={
            "net_arch": [64],
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
    )
    model.learn(total_timesteps=config.total_timesteps, callback=eval_callback, progress_bar=False)
    final_model_path = seed_dir / "final_model"
    model.save(final_model_path)
    best_model_path = seed_dir / "best_model.zip"
    selected_model_path = best_model_path if best_model_path.exists() else final_model_path.with_suffix(".zip")
    evaluations_path = seed_dir / "evaluations.npz"
    best_validation_reward: float | None = None
    if evaluations_path.exists():
        evaluations = np.load(evaluations_path)
        best_validation_reward = float(np.max(np.mean(evaluations["results"], axis=1)))
    metadata: dict[str, object] = {
        "algorithm": "RecurrentPPO",
        "training_seed": seed,
        "requested_timesteps": config.total_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "selected_model": str(selected_model_path.resolve()),
        "best_validation_reward": best_validation_reward,
        "calibration_artifact": str(artifact_path),
        "experiment_label": experiment_label,
        "config": asdict(config),
        "policy_kwargs": {
            "net_arch": [64],
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
    }
    (seed_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_env.close()
    validation_env.close()
    return selected_model_path, metadata
