#!/usr/bin/env python3
"""SBX experiment line for the UR5 reach task.

这条实验线的定位不是替代 `classic/` 或 `warp_gpu/`：
- `classic/` 继续承担 `SB3 + HER` 成功率主线
- `warp_gpu/` 继续承担 GPU 高吞吐验证线
- `sbx_runner/` 用来验证 “JAX 算法层 + 现有 MuJoCo/Gym 环境” 是否能把两条路的实现方法进一步收敛

当前实现刻意保持保守：
- 沿用 `classic` 的环境和日志体系
- 默认启用 sparse reward + joint_position_delta + curriculum
- 暂时不启用 HER
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

try:
    from sbx import PPO, SAC, TD3, TQC

    SBX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - 依赖本机安装情况
    PPO = SAC = TD3 = TQC = None  # type: ignore[assignment]
    SBX_IMPORT_ERROR = exc

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from classic.train import (
        ENV_ID,
        ManualInterruptCallback,
        SaveVecNormalizeCallback,
        TrainArgs as ClassicTrainArgs,
        TrainingLogCallback,
        _apply_classic_throughput_preset,
        _apply_zero_original_preset,
        _build_run_paths,
        _make_env,
        _make_train_vec_env,
        _normalize_device_arg,
        _sync_legacy_run_artifacts,
        register_env,
    )
else:
    from classic.train import (
        ENV_ID,
        ManualInterruptCallback,
        SaveVecNormalizeCallback,
        TrainArgs as ClassicTrainArgs,
        TrainingLogCallback,
        _apply_classic_throughput_preset,
        _apply_zero_original_preset,
        _build_run_paths,
        _make_env,
        _make_train_vec_env,
        _normalize_device_arg,
        _sync_legacy_run_artifacts,
        register_env,
    )


@dataclass
class TrainArgs:
    """SBX 实验线参数。

    这里保留和 `classic` 尽量一致的任务参数，但把算法限制在 SBX 当前更容易维护的一小组：
    - `sac`
    - `td3`
    - `tqc`
    - `ppo`
    """

    algo: str = "sac"
    timesteps: int = 5_000_000
    seed: int = 42
    n_envs: int = 128
    device: str = "cuda"
    render: bool = False
    render_mode: str = "human"
    model_dir: str = "models/sbx"
    log_dir: str = "logs/sbx"
    run_name: str = "ur5_sbx_experiment"
    eval_freq: int = 10_000
    n_eval_episodes: int = 1
    save_best_model: bool = True
    log_interval: int = 1000
    batch_size: int = 1024
    buffer_size: int = 3_000_000
    gradient_steps: int = 4
    learning_starts: int = 20_000
    max_steps: int = 3000
    success_threshold: float = 0.01
    stage1_success_threshold: float = 0.05
    stage2_success_threshold: float = 0.03
    frame_skip: int = 2
    action_target_scale: float = 0.6
    action_smoothing_alpha: float = 0.75
    controller_mode: str = "joint_position_delta"
    joint_position_delta_scale: float = 0.08
    position_control_kp: float = 45.0
    position_control_kd: float = 3.0
    goal_conditioned: bool = True
    reward_mode: str = "sparse"
    physics_backend: str = "mujoco"
    legacy_zero_ee_velocity: bool = False
    robot: str = "ur5_cxy"
    lock_camera: bool = False
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
    curriculum_stage1_fixed_episodes: int = 200
    curriculum_stage2_random_episodes: int = 800
    curriculum_stage2_range_scale: float = 0.35
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="SBX experiment line for UR5 reach")
    p.add_argument("--algo", choices=["sac", "td3", "tqc", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-envs", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", action="store_false", dest="render")
    p.set_defaults(render=False)
    p.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")
    p.add_argument("--model-dir", type=str, default="models/sbx")
    p.add_argument("--log-dir", type=str, default="logs/sbx")
    p.add_argument("--run-name", type=str, default="ur5_sbx_experiment")
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=1)
    p.add_argument("--save-best-model", action="store_true", dest="save_best_model")
    p.add_argument("--no-save-best-model", action="store_false", dest="save_best_model")
    p.set_defaults(save_best_model=True)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--buffer-size", type=int, default=3_000_000)
    p.add_argument("--gradient-steps", type=int, default=4)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--success-threshold", type=float, default=0.01)
    p.add_argument("--stage1-success-threshold", type=float, default=0.05)
    p.add_argument("--stage2-success-threshold", type=float, default=0.03)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--action-target-scale", type=float, default=0.6)
    p.add_argument("--action-smoothing-alpha", type=float, default=0.75)
    p.add_argument("--controller-mode", choices=["torque", "joint_position_delta"], default="joint_position_delta")
    p.add_argument("--joint-position-delta-scale", type=float, default=0.08)
    p.add_argument("--position-control-kp", type=float, default=45.0)
    p.add_argument("--position-control-kd", type=float, default=3.0)
    p.add_argument("--goal-conditioned", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reward-mode", choices=["dense", "sparse"], default="sparse")
    p.add_argument("--physics-backend", choices=["auto", "mujoco", "warp"], default="mujoco")
    p.add_argument("--legacy-zero-ee-velocity", action="store_true")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")
    p.add_argument("--lock-camera", action="store_true", dest="lock_camera")
    p.add_argument("--free-camera", action="store_false", dest="lock_camera")
    p.set_defaults(lock_camera=False)
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
    p.add_argument("--curriculum-stage1-fixed-episodes", type=int, default=200)
    p.add_argument("--curriculum-stage2-random-episodes", type=int, default=800)
    p.add_argument("--curriculum-stage2-range-scale", type=float, default=0.35)
    p.add_argument("--fixed-target-x", type=float, default=None)
    p.add_argument("--fixed-target-y", type=float, default=None)
    p.add_argument("--fixed-target-z", type=float, default=None)
    ns = p.parse_args()
    return TrainArgs(**vars(ns))


def _to_classic_args(args: TrainArgs) -> ClassicTrainArgs:
    """把 SBX 参数映射成 classic 的环境/目录参数。

    这样可以直接复用：
    - 课程学习
    - 日志回调
    - VecNormalize
    - 目录结构
    """

    return ClassicTrainArgs(
        algo=args.algo,
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        device=args.device,
        render=args.render,
        render_mode=args.render_mode,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_best_model=args.save_best_model,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        stage1_success_threshold=args.stage1_success_threshold,
        stage2_success_threshold=args.stage2_success_threshold,
        frame_skip=args.frame_skip,
        action_target_scale=args.action_target_scale,
        action_smoothing_alpha=args.action_smoothing_alpha,
        controller_mode=args.controller_mode,
        joint_position_delta_scale=args.joint_position_delta_scale,
        position_control_kp=args.position_control_kp,
        position_control_kd=args.position_control_kd,
        goal_conditioned=args.goal_conditioned,
        reward_mode=args.reward_mode,
        physics_backend=args.physics_backend,
        legacy_zero_ee_velocity=args.legacy_zero_ee_velocity,
        robot=args.robot,
        lock_camera=args.lock_camera,
        ur5_target_x_min=args.ur5_target_x_min,
        ur5_target_x_max=args.ur5_target_x_max,
        ur5_target_y_min=args.ur5_target_y_min,
        ur5_target_y_max=args.ur5_target_y_max,
        ur5_target_z_min=args.ur5_target_z_min,
        ur5_target_z_max=args.ur5_target_z_max,
        zero_target_x_min=args.zero_target_x_min,
        zero_target_x_max=args.zero_target_x_max,
        zero_target_y_min=args.zero_target_y_min,
        zero_target_y_max=args.zero_target_y_max,
        zero_target_z_min=args.zero_target_z_min,
        zero_target_z_max=args.zero_target_z_max,
        curriculum_stage1_fixed_episodes=args.curriculum_stage1_fixed_episodes,
        curriculum_stage2_random_episodes=args.curriculum_stage2_random_episodes,
        curriculum_stage2_range_scale=args.curriculum_stage2_range_scale,
        fixed_target_x=args.fixed_target_x,
        fixed_target_y=args.fixed_target_y,
        fixed_target_z=args.fixed_target_z,
    )


def _build_model(args: TrainArgs, env, device: str):
    if SBX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "未检测到 `sbx-rl`，无法运行 SBX 实验线。"
            "请先执行 `pip install sbx-rl` 或安装 requirements.txt。"
        ) from SBX_IMPORT_ERROR

    policy_name = "MultiInputPolicy" if args.goal_conditioned else "MlpPolicy"
    common_kwargs = dict(
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        verbose=1,
    )
    if args.algo == "sac":
        return SAC(
            policy_name,
            env,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
            train_freq=1,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]), activation_fn=nn.ReLU),
            **common_kwargs,
        )
    if args.algo == "td3":
        return TD3(
            policy_name,
            env,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
            train_freq=1,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]), activation_fn=nn.ReLU),
            **common_kwargs,
        )
    if args.algo == "tqc":
        return TQC(
            policy_name,
            env,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
            train_freq=1,
            policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]), activation_fn=nn.ReLU),
            **common_kwargs,
        )
    return PPO(
        policy_name,
        env,
        n_steps=512,
        batch_size=args.batch_size,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]), activation_fn=nn.ReLU),
        **common_kwargs,
    )


def train(args: TrainArgs) -> None:
    """运行 SBX 实验线训练。

    这条线默认：
    - 启用课程学习
    - 使用 `joint_position_delta`
    - 使用 sparse reward
    - 可选 goal-conditioned
    - 暂不启用 HER，先验证 JAX 算法层本身的收益
    """

    base_args = _to_classic_args(args)
    _apply_zero_original_preset(base_args)
    _apply_classic_throughput_preset(base_args)
    register_env()
    os.makedirs(base_args.model_dir, exist_ok=True)
    os.makedirs(base_args.log_dir, exist_ok=True)
    paths = _build_run_paths(base_args)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["log_run_dir"], exist_ok=True)
    _sync_legacy_run_artifacts(base_args, paths)

    device = _normalize_device_arg(base_args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("请求使用 cuda，但当前不可用，自动回退到 cpu")
        device = "cpu"

    print(f"SBX 实验线算法: {args.algo}")
    print(f"当前训练设备: {device}")
    print(f"goal_conditioned={base_args.goal_conditioned} reward_mode={base_args.reward_mode}")
    print(f"controller_mode={base_args.controller_mode} n_envs={base_args.n_envs}")

    base_env = _make_train_vec_env(base_args, base_args.render_mode if base_args.render else None)
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
    model = _build_model(args, env, device)

    interrupt_callback = ManualInterruptCallback(
        interrupted_model_path=paths["interrupted_model"],
        interrupted_norm_path=paths["interrupted_norm"],
    )
    callbacks: list[BaseCallback] = [interrupt_callback]
    eval_env = None
    eval_callback = None
    if base_args.eval_freq > 0:
        from stable_baselines3.common.env_util import make_vec_env

        eval_env = make_vec_env(
            ENV_ID,
            n_envs=1,
            seed=base_args.seed + 1,
            env_kwargs=_make_env(base_args, None),
        )
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = env.obs_rms
        best_model_save_path = None
        if base_args.save_best_model:
            os.makedirs(paths["best_model_dir"], exist_ok=True)
            best_model_save_path = paths["best_model_dir"]
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=paths["log_run_dir"],
            eval_freq=max(base_args.eval_freq, 1),
            n_eval_episodes=max(base_args.n_eval_episodes, 1),
            deterministic=True,
            render=False,
        )
        callbacks = [eval_callback, interrupt_callback]
        if base_args.save_best_model:
            callbacks = [
                eval_callback,
                SaveVecNormalizeCallback(eval_callback=eval_callback, save_path=paths["best_norm"], verbose=1),
                interrupt_callback,
            ]
    train_log_callback = TrainingLogCallback(
        log_freq_timesteps=max(base_args.log_interval, 1) * max(base_args.n_envs, 1),
        eval_callback=eval_callback,
    )
    callbacks.append(train_log_callback)

    start_t = time.time()
    model.learn(
        total_timesteps=max(base_args.timesteps, 1),
        callback=callbacks,
        log_interval=max(base_args.log_interval, 1),
        progress_bar=True,
    )
    elapsed = time.time() - start_t
    os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)
    env.save(paths["final_norm"])
    model.save(paths["final_model"])
    env.close()
    if eval_env is not None:
        eval_env.close()

    print(f"训练完成，总耗时 {elapsed:.2f}s")
    print(f"最终模型路径: {paths['final_model']}.zip")
    print(f"最终归一化路径: {paths['final_norm']}")
    if base_args.save_best_model:
        print(f"最优模型目录: {paths['best_model_dir']}")


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
