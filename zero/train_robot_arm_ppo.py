#!/usr/bin/env python3
"""
使用Stable-Baselines3的PPO算法训练机械臂进行位置跟踪
"""

import argparse
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from robot_arm_env import RobotArmEnv

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("logs") / "ppo"
MODEL_DIR = Path("models") / "ppo"
INTERRUPTED_DIR = MODEL_DIR / "interrupted"
BEST_MODEL_DIR = MODEL_DIR / "best_model"
BEST_MODEL_PATH = BEST_MODEL_DIR / "best_model"
BEST_VEC_NORMALIZE_PATH = BEST_MODEL_DIR / "vec_normalize.pkl"
FINAL_MODEL_PATH = MODEL_DIR / "ppo_robot_arm_final"
VEC_NORMALIZE_PATH = MODEL_DIR / "vec_normalize.pkl"


def resolve_path(path_like) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


class ManualInterruptCallback(BaseCallback):
    """
    允许手动中断训练并保存模型的回调函数
    """

    def __init__(self, verbose=0):
        super(ManualInterruptCallback, self).__init__(verbose)
        self.interrupted = False
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        del sig, frame
        print("\nInterrupt received, saving model...")
        self.interrupted = True
        self.save_model()
        print("Model saved, exiting.")
        sys.exit(0)

    def save_model(self):
        if self.model is not None:
            interrupted_dir = resolve_path(INTERRUPTED_DIR)
            interrupted_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(interrupted_dir / "ppo_robot_arm_interrupted"))
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(str(interrupted_dir / "vec_normalize.pkl"))
            print(f"Saved interrupted model and normalize stats to {interrupted_dir}")

    def _on_step(self) -> bool:
        if self.interrupted:
            return False
        return True


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """
    当 EvalCallback 发现新 best 时，同步保存归一化参数。
    """

    def __init__(self, normalize_save_path: str):
        super().__init__(verbose=0)
        self.normalize_save_path = normalize_save_path

    def _on_step(self) -> bool:
        if self.model is None:
            return True
        env = self.model.get_vec_normalize_env()
        if env is not None:
            normalize_path = resolve_path(self.normalize_save_path)
            normalize_path.parent.mkdir(parents=True, exist_ok=True)
            env.save(str(normalize_path))
            print(f"[Best Model] Synced normalize stats: {normalize_path}", flush=True)
        return True


class EpisodeSummaryCallback(BaseCallback):
    """仅按回合窗口打印聚合日志，避免刷屏影响进度条。"""

    def __init__(self, episode_log_interval=64):
        super().__init__(verbose=0)
        self.episode_log_interval = max(1, int(episode_log_interval))
        self.total_successes = 0
        self._reset_window()

    def _reset_window(self):
        self.window_episodes = 0
        self.window_successes = 0
        self.window_collisions = 0
        self.window_return_sum = 0.0
        self.window_final_distance_sum = 0.0
        self.window_min_distance_sum = 0.0
        self.window_avg_speed_sum = 0.0
        self.window_reasons = Counter()

    def _flush_window(self):
        if self.window_episodes <= 0:
            return
        episodes = float(self.window_episodes)
        print(
            "[Train Summary] "
            f"eps_window={self.window_episodes} "
            f"succ_window={self.window_successes} "
            f"succ_total={self.total_successes} "
            f"succ_rate={self.window_successes / episodes:.2%} "
            f"coll_rate={self.window_collisions / episodes:.2%} "
            f"ret_mean={self.window_return_sum / episodes:.2f} "
            f"dist_final_mean={self.window_final_distance_sum / episodes:.4f} "
            f"dist_min_mean={self.window_min_distance_sum / episodes:.4f} "
            f"speed_avg_mean={self.window_avg_speed_sum / episodes:.4f} "
            f"reasons={','.join(f'{k}:{v}' for k, v in self.window_reasons.most_common(3))}"
        , flush=True)
        self._reset_window()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            summary = info.get("episode_summary")
            if not isinstance(summary, dict):
                continue
            self.window_episodes += 1
            success_flag = int(summary.get("episode_success_count", 0) > 0)
            self.window_successes += success_flag
            self.total_successes += success_flag
            self.window_collisions += int(summary.get("episode_collision_count", 0) > 0)
            self.window_return_sum += float(summary.get("episode_return", 0.0))
            self.window_final_distance_sum += float(summary.get("final_distance", 0.0))
            self.window_min_distance_sum += float(summary.get("min_distance", 0.0))
            self.window_avg_speed_sum += float(summary.get("avg_speed", 0.0))
            self.window_reasons[str(summary.get("done_reason", "unknown"))] += 1
        if self.window_episodes >= self.episode_log_interval:
            self._flush_window()
        return True

    def _on_training_end(self) -> None:
        self._flush_window()


def resolve_ppo_hyperparams(_n_envs):
    """PPO 单独训练推荐参数。"""
    return {
        "learning_rate": 3e-4,
        "batch_size": 4096,
        "n_steps": 1024,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.003,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


def evaluate_final_policy(model, train_env, final_eval_episodes=20):
    """
    训练结束后统一评估一次模型，减少训练中额外计算负担。
    """
    eval_env = make_vec_env(lambda: RobotArmEnv(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=max(1, int(final_eval_episodes)),
        deterministic=True,
    )
    eval_env.close()
    return float(mean_reward), float(std_reward)


def train_robot_arm(
    total_timesteps=5000000,
    n_envs=20,
    episode_log_interval=64,
    final_eval_episodes=20,
    eval_freq=200000,
    eval_episodes=10,
):
    """
    训练机械臂进行位置跟踪
    """
    print("Creating robot arm environments...")

    vec_env_cls = SubprocVecEnv if int(n_envs) > 1 else DummyVecEnv
    env = make_vec_env(lambda: RobotArmEnv(), n_envs=n_envs, vec_env_cls=vec_env_cls)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],
            vf=[512, 512, 256],
        ),
        activation_fn=nn.ReLU,
    )

    hparams = resolve_ppo_hyperparams(n_envs)
    print(
        "[Train Config] "
        f"n_envs={n_envs} lr={hparams['learning_rate']} batch={hparams['batch_size']} "
        f"n_steps={hparams['n_steps']} n_epochs={hparams['n_epochs']} "
        f"ent_coef={hparams['ent_coef']} eval_freq={eval_freq} eval_episodes={eval_episodes} "
        f"final_eval_episodes={final_eval_episodes}",
        flush=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="auto",
        learning_rate=float(hparams["learning_rate"]),
        batch_size=int(hparams["batch_size"]),
        n_steps=int(hparams["n_steps"]),
        n_epochs=int(hparams["n_epochs"]),
        gamma=float(hparams["gamma"]),
        gae_lambda=float(hparams["gae_lambda"]),
        clip_range=float(hparams["clip_range"]),
        ent_coef=float(hparams["ent_coef"]),
        vf_coef=float(hparams["vf_coef"]),
        max_grad_norm=float(hparams["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
    )

    manual_interrupt_callback = ManualInterruptCallback(verbose=1)
    resolve_path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    resolve_path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    resolve_path(BEST_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # 评估环境仅用于周期性评估和 best_model 选择。
    eval_env = make_vec_env(lambda: RobotArmEnv(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    save_vec_on_best_callback = SaveVecNormalizeOnBestCallback(BEST_VEC_NORMALIZE_PATH)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(resolve_path(BEST_MODEL_DIR)),
        log_path=str(resolve_path(LOG_DIR)),
        eval_freq=max(1, int(eval_freq // max(1, int(n_envs)))),
        n_eval_episodes=max(1, int(eval_episodes)),
        deterministic=True,
        render=False,
        callback_on_new_best=save_vec_on_best_callback,
    )

    print(f"Start training... (n_envs={n_envs})", flush=True)
    print("Tip: press Ctrl+C to stop early and save interrupted model.", flush=True)
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[manual_interrupt_callback, eval_callback],
        log_interval=1000,
        progress_bar=True,
    )

    mean_reward, std_reward = evaluate_final_policy(model, env, final_eval_episodes=final_eval_episodes)
    print(f"[Final Eval] episodes={int(final_eval_episodes)} mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}", flush=True)

    env.save(str(resolve_path(VEC_NORMALIZE_PATH)))
    model.save(str(resolve_path(FINAL_MODEL_PATH)))
    if not resolve_path(BEST_MODEL_PATH).with_suffix(".zip").exists():
        model.save(str(resolve_path(BEST_MODEL_PATH)))
        env.save(str(resolve_path(BEST_VEC_NORMALIZE_PATH)))
        print("[Best Model] No periodic-best found, fallback to final model.", flush=True)
    eval_env.close()

    end_time = time.time()
    print(f"Training finished, elapsed: {end_time - start_time:.2f}s", flush=True)
    return model, env


def test_robot_arm(model_path=FINAL_MODEL_PATH, normalize_path=VEC_NORMALIZE_PATH, num_episodes=10):
    """
    测试训练好的模型
    """
    print("Loading model for evaluation...")
    env = make_vec_env(lambda: RobotArmEnv(render_mode="human"), n_envs=1)

    resolved_normalize_path = resolve_path(normalize_path)
    resolved_model_path = resolve_path(model_path)
    if resolved_normalize_path.exists():
        env = VecNormalize.load(str(resolved_normalize_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(resolved_model_path), env=env)
    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        target_pos = env.venv.envs[0].env.unwrapped.target_pos
        print(f"Episode {episode + 1} target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        for i in range(5000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            del info
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            time.sleep(0.01)
            env.render()
            if done:
                print(f"Episode {episode + 1} finished after {i + 1} timesteps")
                print(f"Episode reward: {total_reward}")
                episode_rewards.append(total_reward)
                break

    env.close()
    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test robot arm with PPO")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--best", action="store_true", help="Test the best saved model")
    parser.add_argument("--final", action="store_true", help="Test the final saved model")
    parser.add_argument("--model-path", type=str, default=FINAL_MODEL_PATH, help="Path to the model for testing")
    parser.add_argument("--normalize-path", type=str, default=VEC_NORMALIZE_PATH, help="Path to the normalization parameters")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to test")
    parser.add_argument("--total-timesteps", type=int, default=5000000, help="Total timesteps for training")
    parser.add_argument("--n-envs", type=int, default=20, help="Number of parallel environments for training")
    parser.add_argument("--episode-log-interval", type=int, default=64, help="Aggregate training logs every N episodes")
    parser.add_argument("--eval-freq", type=int, default=200000, help="Env steps between periodic evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per periodic evaluation")
    parser.add_argument("--final-eval-episodes", type=int, default=20, help="Episodes for final evaluation after training")
    args = parser.parse_args()

    if args.test:
        if args.best:
            if resolve_path(BEST_MODEL_PATH).with_suffix(".zip").exists():
                model_path = BEST_MODEL_PATH
                normalize_path = BEST_VEC_NORMALIZE_PATH
            else:
                print("Best model not found, fallback to final model.")
                model_path = FINAL_MODEL_PATH
                normalize_path = VEC_NORMALIZE_PATH
        elif args.final:
            model_path = FINAL_MODEL_PATH
            normalize_path = VEC_NORMALIZE_PATH
        else:
            model_path = args.model_path
            normalize_path = args.normalize_path
        test_robot_arm(model_path, normalize_path, args.episodes)
    else:
        train_robot_arm(
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            episode_log_interval=args.episode_log_interval,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
        )






