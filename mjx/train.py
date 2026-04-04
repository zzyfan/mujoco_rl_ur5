#!/usr/bin/env python3
"""独立 MJX 训练脚本（与原 train.py 并行共存）。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mjx.backend import resolve_physics_backend
    from mjx.env import MJXEnvConfig, MJXReachEnv
else:
    from .backend import resolve_physics_backend
    from .env import MJXEnvConfig, MJXReachEnv


@dataclass
class MJXTrainArgs:
    test: bool = False
    algo: str = "sac"
    timesteps: int = 1_000_000
    episodes: int = 5
    seed: int = 42
    n_envs: int = 1
    device: str = "cuda"
    render: bool = False
    render_mode: str = "human"
    render_freq: int = 1
    model_dir: str = "models/mjx"
    log_dir: str = "logs/mjx"
    run_name: str = "mjx_reach"
    robot: str = "zero_robotiq"
    physics_backend: str = "auto"
    frame_skip: int = 1
    max_steps: int = 3000
    success_threshold: float = 0.01
    batch_size: int = 256
    gradient_steps: int = 1
    learning_starts: int = 10000
    action_noise_sigma: float = 2.5
    curriculum_stage1_fixed_episodes: int = 200
    curriculum_stage2_random_episodes: int = 800
    curriculum_stage2_range_scale: float = 0.35
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None
    ur5_target_x_min: float = -0.95
    ur5_target_x_max: float = -0.60
    ur5_target_y_min: float = 0.15
    ur5_target_y_max: float = 0.50
    ur5_target_z_min: float = 0.12
    ur5_target_z_max: float = 0.30
    zero_target_x_min: float = -1.00
    zero_target_x_max: float = -0.62
    zero_target_y_min: float = 0.08
    zero_target_y_max: float = 0.48
    zero_target_z_min: float = 0.10
    zero_target_z_max: float = 0.35
    safe_disable_constraints: bool = True
    resume: bool = False
    resume_model_path: str = ""
    resume_normalize_path: str = ""
    model_path: str = ""
    normalize_path: str = ""


class RenderDuringTrainingCallback(BaseCallback):
    """训练期间按频率调用渲染。"""

    def __init__(self, render_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = max(int(render_freq), 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq != 0:
            return True
        try:
            if hasattr(self.training_env, "envs") and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
                while hasattr(env, "env"):
                    env = env.env
                if hasattr(env, "render"):
                    env.render()
        except Exception as e:
            if self.verbose > 0:
                print(f"MJX 渲染报错: {e}")
        return True


def _build_run_paths(args: MJXTrainArgs) -> dict[str, str]:
    base_dir = Path(__file__).resolve().parents[1]
    model_root = Path(args.model_dir)
    log_root = Path(args.log_dir)
    if not model_root.is_absolute():
        model_root = base_dir / model_root
    if not log_root.is_absolute():
        log_root = base_dir / log_root
    run_dir = str(model_root / args.algo / args.robot / args.run_name)
    log_run_dir = str(log_root / args.algo / args.robot / args.run_name)
    return {
        "run_dir": run_dir,
        "log_run_dir": log_run_dir,
        "final_model": os.path.join(run_dir, "final", "model"),
        "final_norm": os.path.join(run_dir, "final", "vec_normalize.pkl"),
        "interrupted_model": os.path.join(run_dir, "interrupted", "model"),
        "interrupted_norm": os.path.join(run_dir, "interrupted", "vec_normalize.pkl"),
    }


def _build_env_config(args: MJXTrainArgs) -> MJXEnvConfig:
    cfg = MJXEnvConfig(
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        curriculum_stage1_fixed_episodes=args.curriculum_stage1_fixed_episodes,
        curriculum_stage2_random_episodes=args.curriculum_stage2_random_episodes,
        curriculum_stage2_range_scale=args.curriculum_stage2_range_scale,
        fixed_target_x=args.fixed_target_x,
        fixed_target_y=args.fixed_target_y,
        fixed_target_z=args.fixed_target_z,
        physics_backend=args.physics_backend,
        safe_disable_constraints=args.safe_disable_constraints,
        render_mode=args.render_mode if args.render else None,
    )
    if args.robot == "zero_robotiq":
        cfg.model_xml = "assets/zero_arm/zero_with_robotiq_reach.xml"
        cfg.home_pose_mode = "direct6"
        cfg.home_joint1 = 0.0
        cfg.home_joint2 = -0.85
        cfg.home_joint3 = 1.35
        cfg.home_joint4 = -0.5
        cfg.home_joint5 = 0.0
        cfg.home_joint6 = 0.0
        cfg.target_x_min = float(args.zero_target_x_min)
        cfg.target_x_max = float(args.zero_target_x_max)
        cfg.target_y_min = float(args.zero_target_y_min)
        cfg.target_y_max = float(args.zero_target_y_max)
        cfg.target_z_min = float(args.zero_target_z_min)
        cfg.target_z_max = float(args.zero_target_z_max)
        if args.fixed_target_x is None:
            cfg.fixed_target_x = float(np.clip(cfg.target_x_max - 0.06, cfg.target_x_min, cfg.target_x_max))
        if args.fixed_target_y is None:
            cfg.fixed_target_y = float(
                np.clip(cfg.target_y_min + 0.65 * (cfg.target_y_max - cfg.target_y_min), cfg.target_y_min, cfg.target_y_max)
            )
        if args.fixed_target_z is None:
            cfg.fixed_target_z = float(
                np.clip(cfg.target_z_min + 0.55 * (cfg.target_z_max - cfg.target_z_min), cfg.target_z_min, cfg.target_z_max)
            )
    else:
        cfg.model_xml = "assets/robotiq_cxy/lab_env.xml"
        cfg.home_pose_mode = "ur5_coupled"
        cfg.home_joint1 = np.deg2rad(29.7)
        cfg.home_joint2 = np.deg2rad(-85.0)
        cfg.home_joint3 = np.deg2rad(115.0)
        cfg.home_joint4 = 0.0
        cfg.home_joint5 = 0.0
        cfg.home_joint6 = 0.0
        cfg.target_x_min = float(args.ur5_target_x_min)
        cfg.target_x_max = float(args.ur5_target_x_max)
        cfg.target_y_min = float(args.ur5_target_y_min)
        cfg.target_y_max = float(args.ur5_target_y_max)
        cfg.target_z_min = float(args.ur5_target_z_min)
        cfg.target_z_max = float(args.ur5_target_z_max)
    return cfg


def _make_env_fn(args: MJXTrainArgs):
    base_cfg = _build_env_config(args)

    def _init():
        return MJXReachEnv(config=replace(base_cfg))

    return _init


def _build_model(args: MJXTrainArgs, env, device: str):
    if args.algo == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=float(args.action_noise_sigma) * np.ones(n_actions, dtype=np.float32),
        )
        return TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=1,
            seed=args.seed,
            device=device,
            learning_rate=3e-4,
            buffer_size=2_000_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=0.99,
            gradient_steps=args.gradient_steps,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]), activation_fn=nn.ReLU),
        )
    if args.algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            device=device,
            learning_rate=3e-4,
            buffer_size=2_000_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=0.99,
            gradient_steps=args.gradient_steps,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]), activation_fn=nn.ReLU),
        )
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), activation_fn=nn.ReLU),
    )


def _load_model(args: MJXTrainArgs, model_path: str, env, device: str):
    if args.algo == "td3":
        return TD3.load(model_path, env=env, device=device)
    if args.algo == "sac":
        return SAC.load(model_path, env=env, device=device)
    return PPO.load(model_path, env=env, device=device)


def train(args: MJXTrainArgs):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    paths = _build_run_paths(args)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["log_run_dir"], exist_ok=True)

    device = args.device
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("请求 cuda，但当前不可用，回退到 cpu")
        device = "cpu"
    print(f"当前训练设备: {device}")
    print(f"当前物理后端: {resolve_physics_backend(args.physics_backend)}")
    if args.render and args.n_envs > 1:
        print("MJX 渲染模式建议 n_envs=1，否则只显示一个环境窗口且显著降速。")

    base_env = make_vec_env(_make_env_fn(args), n_envs=args.n_envs, seed=args.seed)
    if args.resume:
        resume_model_path = args.resume_model_path or paths["interrupted_model"]
        resume_norm_path = args.resume_normalize_path or paths["interrupted_norm"]
        if not os.path.exists(resume_model_path + ".zip") and not os.path.exists(resume_model_path):
            raise FileNotFoundError(f"未找到继续训练模型: {resume_model_path}")
        if os.path.exists(resume_norm_path):
            env = VecNormalize.load(resume_norm_path, base_env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
        model = _load_model(args, resume_model_path, env, device)
        print(f"继续训练模型: {resume_model_path}")
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
        model = _build_model(args, env, device)

    print("开始 MJX 训练...")
    start_t = time.time()
    try:
        callback = RenderDuringTrainingCallback(render_freq=args.render_freq) if args.render else None
        model.learn(
            total_timesteps=max(args.timesteps, 1),
            progress_bar=True,
            reset_num_timesteps=not args.resume,
            callback=callback,
        )
    except KeyboardInterrupt:
        os.makedirs(os.path.dirname(paths["interrupted_model"]), exist_ok=True)
        model.save(paths["interrupted_model"])
        env.save(paths["interrupted_norm"])
        env.close()
        print("检测到中断，已保存 interrupted 模型。")
        print(f"中断模型路径: {paths['interrupted_model']}.zip")
        print(f"中断归一化路径: {paths['interrupted_norm']}")
        return

    elapsed = time.time() - start_t
    os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)
    model.save(paths["final_model"])
    env.save(paths["final_norm"])
    env.close()
    print(f"MJX 训练完成，总耗时 {elapsed:.2f}s")
    print(f"最终模型: {paths['final_model']}.zip")
    print(f"最终归一化: {paths['final_norm']}")


def test(args: MJXTrainArgs):
    paths = _build_run_paths(args)
    model_path = args.model_path or paths["final_model"]
    norm_path = args.normalize_path or paths["final_norm"]
    model_exists = os.path.exists(model_path) or os.path.exists(model_path + ".zip")
    # 若没有 final 模型，则自动回退到 interrupted，方便 Ctrl+C 后直接测试。
    if not model_exists and not args.model_path:
        fallback_model = paths["interrupted_model"]
        if os.path.exists(fallback_model) or os.path.exists(fallback_model + ".zip"):
            print("未找到 final 模型，自动回退到 interrupted 模型进行测试。")
            model_path = fallback_model
            norm_fallback = paths["interrupted_norm"]
            if not args.normalize_path and os.path.exists(norm_fallback):
                norm_path = norm_fallback
        else:
            raise FileNotFoundError(f"未找到可用模型（final/interrupted）: {model_path}")
    device = args.device
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("请求 cuda，但当前不可用，回退到 cpu")
        device = "cpu"
    print(f"当前物理后端: {resolve_physics_backend(args.physics_backend)}")

    env = make_vec_env(_make_env_fn(args), n_envs=1, seed=args.seed + 1)
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    model = _load_model(args, model_path, env, device)
    rewards = []
    for ep in range(max(args.episodes, 1)):
        obs = env.reset()
        done = np.array([False], dtype=bool)
        total = 0.0
        steps = 0
        for _ in range(max(args.max_steps, 1)):
            if bool(done[0]):
                break
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            if args.render:
                env.render()
            total += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            steps += 1
        print(f"第 {ep + 1} 回合: 步数={steps}, 奖励={total:.3f}")
        rewards.append(total)
    env.close()
    if rewards:
        print(f"平均奖励: {float(np.mean(rewards)):.3f}")


def parse_args() -> MJXTrainArgs:
    p = argparse.ArgumentParser(description="独立 MJX 训练脚本")
    p.add_argument("--test", action="store_true")
    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", action="store_false", dest="render")
    p.set_defaults(render=False)
    p.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    p.add_argument("--render-freq", type=int, default=1)
    p.add_argument("--model-dir", type=str, default="models/mjx")
    p.add_argument("--log-dir", type=str, default="logs/mjx")
    p.add_argument("--run-name", type=str, default="mjx_reach")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="zero_robotiq")
    p.add_argument("--physics-backend", choices=["auto", "mjx", "warp"], default="auto")
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--success-threshold", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=10000)
    p.add_argument("--action-noise-sigma", type=float, default=2.5)
    p.add_argument("--curriculum-stage1-fixed-episodes", type=int, default=200)
    p.add_argument("--curriculum-stage2-random-episodes", type=int, default=800)
    p.add_argument("--curriculum-stage2-range-scale", type=float, default=0.35)
    p.add_argument("--fixed-target-x", type=float, default=None)
    p.add_argument("--fixed-target-y", type=float, default=None)
    p.add_argument("--fixed-target-z", type=float, default=None)
    p.add_argument("--ur5-target-x-min", type=float, default=-0.95)
    p.add_argument("--ur5-target-x-max", type=float, default=-0.60)
    p.add_argument("--ur5-target-y-min", type=float, default=0.15)
    p.add_argument("--ur5-target-y-max", type=float, default=0.50)
    p.add_argument("--ur5-target-z-min", type=float, default=0.12)
    p.add_argument("--ur5-target-z-max", type=float, default=0.30)
    p.add_argument("--zero-target-x-min", type=float, default=-1.00)
    p.add_argument("--zero-target-x-max", type=float, default=-0.62)
    p.add_argument("--zero-target-y-min", type=float, default=0.08)
    p.add_argument("--zero-target-y-max", type=float, default=0.48)
    p.add_argument("--zero-target-z-min", type=float, default=0.10)
    p.add_argument("--zero-target-z-max", type=float, default=0.35)
    p.add_argument("--safe-disable-constraints", action="store_true")
    p.add_argument("--strict-constraints", action="store_false", dest="safe_disable_constraints")
    p.set_defaults(safe_disable_constraints=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-model-path", type=str, default="")
    p.add_argument("--resume-normalize-path", type=str, default="")
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--normalize-path", type=str, default="")
    ns = p.parse_args()
    return MJXTrainArgs(
        test=ns.test,
        algo=ns.algo,
        timesteps=ns.timesteps,
        episodes=ns.episodes,
        seed=ns.seed,
        n_envs=ns.n_envs,
        device=ns.device,
        render=ns.render,
        render_mode=ns.render_mode,
        render_freq=ns.render_freq,
        model_dir=ns.model_dir,
        log_dir=ns.log_dir,
        run_name=ns.run_name,
        robot=ns.robot,
        physics_backend=ns.physics_backend,
        frame_skip=ns.frame_skip,
        max_steps=ns.max_steps,
        success_threshold=ns.success_threshold,
        batch_size=ns.batch_size,
        gradient_steps=ns.gradient_steps,
        learning_starts=ns.learning_starts,
        action_noise_sigma=ns.action_noise_sigma,
        curriculum_stage1_fixed_episodes=ns.curriculum_stage1_fixed_episodes,
        curriculum_stage2_random_episodes=ns.curriculum_stage2_random_episodes,
        curriculum_stage2_range_scale=ns.curriculum_stage2_range_scale,
        fixed_target_x=ns.fixed_target_x,
        fixed_target_y=ns.fixed_target_y,
        fixed_target_z=ns.fixed_target_z,
        ur5_target_x_min=ns.ur5_target_x_min,
        ur5_target_x_max=ns.ur5_target_x_max,
        ur5_target_y_min=ns.ur5_target_y_min,
        ur5_target_y_max=ns.ur5_target_y_max,
        ur5_target_z_min=ns.ur5_target_z_min,
        ur5_target_z_max=ns.ur5_target_z_max,
        zero_target_x_min=ns.zero_target_x_min,
        zero_target_x_max=ns.zero_target_x_max,
        zero_target_y_min=ns.zero_target_y_min,
        zero_target_y_max=ns.zero_target_y_max,
        zero_target_z_min=ns.zero_target_z_min,
        zero_target_z_max=ns.zero_target_z_max,
        safe_disable_constraints=ns.safe_disable_constraints,
        resume=ns.resume,
        resume_model_path=ns.resume_model_path,
        resume_normalize_path=ns.resume_normalize_path,
        model_path=ns.model_path,
        normalize_path=ns.normalize_path,
    )


def main():
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
