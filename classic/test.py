#!/usr/bin/env python3
"""单独验证 classic reach 模型的推理路径。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from classic.train import (  # 推理时沿用训练脚本的目录规则，避免“训练能跑、测试找不到模型”。
        ENV_ID,
        TrainArgs,
        _build_run_paths,
        _make_env,
        _resolve_test_artifact_paths,
        _sync_legacy_run_artifacts,
        register_env,
    )
else:
    from .train import (  # 推理时沿用训练脚本的目录规则，避免“训练能跑、测试找不到模型”。
        ENV_ID,
        TrainArgs,
        _build_run_paths,
        _make_env,
        _resolve_test_artifact_paths,
        _sync_legacy_run_artifacts,
        register_env,
    )


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="测试 MuJoCo 到点模型")
    parser.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")  # 必须和训练算法一致，否则权重结构对不上。
    parser.add_argument("--run-name", type=str, default="ur5_mujoco")  # 不手填路径时，用 run-name 回溯训练产物。
    parser.add_argument("--model-path", type=str, default="")  # 手工指定模型时优先于 run-name。
    parser.add_argument("--norm-path", type=str, default="")  # VecNormalize 统计必须和训练时匹配，否则距离观测会失真。
    parser.add_argument("--episodes", type=int, default=3)  # 看稳定性时建议至少跑几个回合，不只看单次成功。
    parser.add_argument("--max-steps", type=int, default=3000)  # 应和训练时 max_steps 基本一致。
    parser.add_argument("--device", type=str, default="cuda")  # 只影响策略推理，不改变环境物理。
    parser.add_argument("--physics-backend", choices=["auto", "mujoco", "warp"], default="mujoco")  # 用来复现训练时的后端选择。
    parser.add_argument("--legacy-zero-ee-velocity", action="store_true", help="启用旧版 `cvel[:3]` 速度读取")
    parser.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 机器人和模型文件、目标范围绑定。
    parser.add_argument("--render", action="store_true", help="启用渲染（默认关闭，避免无头环境卡住）")
    parser.add_argument("--no-render", action="store_false", dest="render", help="关闭渲染")
    parser.set_defaults(render=False)
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")  # `human` 适合看动作，`rgb_array` 适合录制。
    parser.add_argument("--random-policy", action="store_true", help="不加载模型，使用随机动作测试环境")  # 用来确认环境本身没坏。
    parser.add_argument("--print-step-reward", action="store_true", help="打印每一步奖励与关键信息")
    parser.add_argument("--print-reward-info", action="store_true", help="打印环境 info 里的奖励分项（若存在）")
    return parser.parse_args()


def main():
    """测试入口函数。"""
    args = parse_args()
    register_env()
    path_cfg = TrainArgs(  # 测试脚本故意借用训练参数类，保证目录布局和命名完全一致。
        algo=args.algo,
        robot=args.robot,
        run_name=args.run_name,
        model_dir="models/classic",
        log_dir="logs/classic",
        model_path=args.model_path,
        normalize_path=args.norm_path,
    )
    paths = _build_run_paths(path_cfg)
    _sync_legacy_run_artifacts(path_cfg, paths)
    model_path, norm_path = _resolve_test_artifact_paths(path_cfg, paths)

    device = args.device
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("请求 cuda 但不可用，自动切换到 cpu")
        device = "cpu"

    env_cfg = TrainArgs(  # 推理环境也走 `_make_env`，这样 reach 任务参数不会在测试脚本里再维护一份。
        algo=args.algo,
        robot=args.robot,
        physics_backend=args.physics_backend,
        legacy_zero_ee_velocity=bool(args.legacy_zero_ee_velocity),
        render=bool(args.render),
        render_mode=args.render_mode,
        max_steps=args.max_steps,
    )

    env = None
    try:
        env = make_vec_env(
            ENV_ID,
            n_envs=1,  # 测试阶段只保留 1 个环境。
            seed=123,
            env_kwargs=_make_env(env_cfg, args.render_mode if args.render else None),
        )

        if not args.random_policy:
            try:
                env = VecNormalize.load(norm_path, env)
                env.training = False  # 测试时必须冻结统计，否则观测分布会被当前轨迹污染。
                env.norm_reward = False  # 看真实回报时关闭奖励归一化更直观。
                print(f"已加载归一化参数: {norm_path}")
            except Exception as e:
                print(f"未加载归一化参数（继续测试）: {e}")

        model = None
        if not args.random_policy:
            if args.algo == "td3":
                model = TD3.load(model_path, env=env, device=device)
            elif args.algo == "sac":
                model = SAC.load(model_path, env=env, device=device)
            else:
                model = PPO.load(model_path, env=env, device=device)
            print(f"已加载模型: {model_path}")

        for ep in range(args.episodes):
            obs = env.reset()
            done = np.array([False], dtype=bool)
            total_reward = 0.0
            steps = 0
            while not bool(done[0]) and steps < args.max_steps:
                if model is None:
                    action = np.array([env.action_space.sample()])  # 随机策略主要用来排查环境接口，不看效果。
                else:
                    action, _ = model.predict(obs, deterministic=True)  # 评估 reach 模型时默认看确定性策略。
                obs, reward, done, info = env.step(action)
                if args.render:
                    env.render()
                step_reward = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
                total_reward += step_reward
                steps += 1
                if args.print_step_reward:
                    info0 = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else {}
                    distance = info0.get("distance")
                    success = info0.get("success")
                    if distance is None:
                        print(f"[step {steps}] reward={step_reward:.6f}")
                    else:
                        print(f"[step {steps}] reward={step_reward:.6f}, distance={float(distance):.6f}, success={bool(success)}")
                    if args.print_reward_info:
                        reward_info = info0.get("reward_info")  # 环境若返回奖励分项，则在这里打印。
                        if isinstance(reward_info, dict):
                            print(f"  reward_info={reward_info}")
                if args.render and args.render_mode == "human":
                    time.sleep(0.01)  # 给 viewer 一点刷新时间，否则机械臂动作会快到看不清。
            print(f"Episode {ep + 1}: steps={steps}, reward={total_reward:.3f}")  # 先看回合总回报，再结合 step 日志看失败原因。
    finally:
        if env is not None:
            env.close()
    print("测试完成。")
    if args.render and args.render_mode == "human":
        # 某些 GLX/X11 组合在解释器回收 viewer 对象时会触发 `GLXBadDrawable`。
        # 测试脚本到这里已经完成推理输出，直接退出进程可以避免卡在窗口销毁阶段。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
