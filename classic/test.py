#!/usr/bin/env python3
"""独立测试脚本：用于验证 Gym 环境与 SB3 模型是否正常运行。"""

from __future__ import annotations

import argparse  # 标准库：命令行参数解析。
import sys
import time  # 标准库：控制渲染节奏。
from pathlib import Path

import numpy as np  # 数值计算与数组处理。
import torch  # 设备可用性检查。
from stable_baselines3 import PPO, SAC, TD3  # 三种可选算法模型。
from stable_baselines3.common.env_util import make_vec_env  # 创建向量化环境。
from stable_baselines3.common.vec_env import VecNormalize  # 归一化参数加载。

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from classic.train import (  # 复用训练脚本里的环境构造与路径解析逻辑。
        ENV_ID,
        TrainArgs,
        _build_run_paths,
        _make_env,
        _resolve_test_artifact_paths,
        _sync_legacy_run_artifacts,
        register_env,
    )
else:
    from .train import (  # 复用训练脚本里的环境构造与路径解析逻辑。
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
    parser = argparse.ArgumentParser(description="测试 MuJoCo 到点模型")  # 创建参数解析器。
    parser.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")  # 算法选择。
    parser.add_argument("--run-name", type=str, default="ur5_mujoco")  # 运行名，用于自动解析默认模型路径。
    parser.add_argument("--model-path", type=str, default="")  # 模型路径；为空时自动按 run-name 解析。
    parser.add_argument("--norm-path", type=str, default="")  # 归一化路径；为空时自动按 run-name 解析。
    parser.add_argument("--episodes", type=int, default=3)  # 测试回合数。
    parser.add_argument("--max-steps", type=int, default=3000)  # 每回合最大步数。
    parser.add_argument("--device", type=str, default="cuda")  # 推理设备。
    parser.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 机械臂模型选择。
    parser.add_argument("--render", action="store_true", help="启用渲染（默认关闭，避免无头环境卡住）")
    parser.add_argument("--no-render", action="store_false", dest="render", help="关闭渲染")
    parser.set_defaults(render=False)
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")  # 渲染模式。
    parser.add_argument("--random-policy", action="store_true", help="不加载模型，使用随机动作测试环境")  # 随机动作模式。
    return parser.parse_args()  # 返回解析结果。


def main():
    """测试入口函数。"""
    args = parse_args()  # 读取命令行参数。
    register_env()  # 注册 Gym 环境 id，确保 make_vec_env 可用。
    path_cfg = TrainArgs(  # 用训练脚本同一套分层目录规则解析默认测试路径。
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

    device = args.device  # 读取目标设备。
    if device.lower() == "cuda" and not torch.cuda.is_available():  # CUDA 不可用时自动回退。
        print("请求 cuda 但不可用，自动切换到 cpu")  # 打印回退提示。
        device = "cpu"  # 设置 CPU 设备。

    env_cfg = TrainArgs(  # 复用 TrainArgs 与 _make_env，保证环境参数风格一致。
        algo=args.algo,  # 算法名（仅用于后续一致性）。
        robot=args.robot,  # 机械臂模型（与训练时保持一致）。
        render=bool(args.render),  # 测试是否渲染由命令行控制。
        render_mode=args.render_mode,  # 渲染模式。
        max_steps=args.max_steps,  # 单回合最大步数。
    )

    env = None
    try:
        env = make_vec_env(  # 创建向量化测试环境。
            ENV_ID,  # 环境 id。
            n_envs=1,  # 测试阶段单环境即可。
            seed=123,  # 固定种子便于复现。
            env_kwargs=_make_env(env_cfg, args.render_mode if args.render else None),  # 把配置传给环境构造函数。
        )

        if not args.random_policy:  # 非随机策略模式才尝试加载归一化参数。
            try:
                env = VecNormalize.load(norm_path, env)  # 加载训练期观测归一化统计。
                env.training = False  # 推理阶段不更新统计。
                env.norm_reward = False  # 推理阶段关闭奖励归一化。
                print(f"已加载归一化参数: {norm_path}")  # 打印加载成功信息。
            except Exception as e:  # 归一化文件缺失时继续测试，不中断。
                print(f"未加载归一化参数（继续测试）: {e}")  # 打印异常。

        model = None  # 默认无模型（随机策略模式）。
        if not args.random_policy:  # 仅在模型策略模式下加载模型。
            if args.algo == "td3":  # TD3 分支。
                model = TD3.load(model_path, env=env, device=device)  # 加载 TD3 模型。
            elif args.algo == "sac":  # SAC 分支。
                model = SAC.load(model_path, env=env, device=device)  # 加载 SAC 模型。
            else:  # PPO 分支。
                model = PPO.load(model_path, env=env, device=device)  # 加载 PPO 模型。
            print(f"已加载模型: {model_path}")  # 打印模型路径。

        for ep in range(args.episodes):  # 逐回合测试。
            obs = env.reset()  # VecEnv reset 返回 batched obs。
            done = np.array([False], dtype=bool)  # done 使用批量布尔数组。
            total_reward = 0.0  # 当前回合累计奖励。
            steps = 0  # 当前回合步数。
            while not bool(done[0]) and steps < args.max_steps:  # 回合终止或达到步数上限时退出。
                if model is None:  # 随机策略模式。
                    action = np.array([env.action_space.sample()])  # 采样一个随机动作并补 batch 维。
                else:  # 模型策略模式。
                    action, _ = model.predict(obs, deterministic=True)  # 输出确定性动作。
                obs, reward, done, info = env.step(action)  # VecEnv step 返回 4 元组。
                if args.render:
                    env.render()  # 按需刷新渲染窗口。
                total_reward += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)  # 兼容数组/标量奖励。
                steps += 1  # 步数累加。
                if args.render and args.render_mode == "human":  # human 模式下减速，便于观察。
                    time.sleep(0.01)  # 每步暂停 10ms。
            print(f"Episode {ep + 1}: steps={steps}, reward={total_reward:.3f}")  # 输出回合结果。
    finally:
        if env is not None:
            env.close()  # 关闭环境并释放渲染资源。
    print("测试完成。")  # 测试结束提示。


if __name__ == "__main__":
    main()  # 直接运行脚本时执行入口函数。
