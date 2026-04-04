#!/usr/bin/env python3
"""独立测试脚本：用于验证 Gym 环境与 SB3 模型是否正常运行。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import argparse  # 标准库：命令行参数解析。
import sys  # 依赖导入：先认清这个文件需要哪些外部能力
import time  # 标准库：控制渲染节奏。
from pathlib import Path  # 依赖导入：先认清这个文件需要哪些外部能力

import numpy as np  # 数值计算与数组处理。
import torch  # 设备可用性检查。
from stable_baselines3 import PPO, SAC, TD3  # 三种可选算法模型。
from stable_baselines3.common.env_util import make_vec_env  # 创建向量化环境。
from stable_baselines3.common.vec_env import VecNormalize  # 归一化参数加载。

if __package__ in (None, ""):  # 条件分支：学习时先看触发条件，再看两边行为差异
    ROOT = Path(__file__).resolve().parents[1]  # 状态或中间变量：调试时多观察它的值如何流动
    if str(ROOT) not in sys.path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        sys.path.insert(0, str(ROOT))  # 代码执行语句：结合上下文理解它对后续流程的影响
    from classic.train import (  # 复用训练脚本里的环境构造与路径解析逻辑。
        ENV_ID,  # 代码执行语句：结合上下文理解它对后续流程的影响
        TrainArgs,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _build_run_paths,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _make_env,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _resolve_test_artifact_paths,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _sync_legacy_run_artifacts,  # 代码执行语句：结合上下文理解它对后续流程的影响
        register_env,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素
else:  # 兜底分支：当前面条件都不满足时走这里
    from .train import (  # 复用训练脚本里的环境构造与路径解析逻辑。
        ENV_ID,  # 代码执行语句：结合上下文理解它对后续流程的影响
        TrainArgs,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _build_run_paths,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _make_env,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _resolve_test_artifact_paths,  # 代码执行语句：结合上下文理解它对后续流程的影响
        _sync_legacy_run_artifacts,  # 代码执行语句：结合上下文理解它对后续流程的影响
        register_env,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def parse_args():  # 命令行解析入口：外部调参首先会影响这里
    """解析命令行参数。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    parser = argparse.ArgumentParser(description="测试 MuJoCo 到点模型")  # 创建参数解析器。
    parser.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")  # 算法选择。
    parser.add_argument("--run-name", type=str, default="ur5_mujoco")  # 运行名，用于自动解析默认模型路径。
    parser.add_argument("--model-path", type=str, default="")  # 模型路径；为空时自动按 run-name 解析。
    parser.add_argument("--norm-path", type=str, default="")  # 归一化路径；为空时自动按 run-name 解析。
    parser.add_argument("--episodes", type=int, default=3)  # 测试回合数。
    parser.add_argument("--max-steps", type=int, default=3000)  # 每回合最大步数。
    parser.add_argument("--device", type=str, default="cuda")  # 推理设备。
    parser.add_argument("--physics-backend", choices=["auto", "mujoco", "warp"], default="mujoco")  # 物理后端。
    parser.add_argument("--legacy-zero-ee-velocity", action="store_true", help="兼容 zero 原始 `cvel[:3]` 末端速度读取")  # 命令行参数：这里就是直接的调参入口
    parser.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 机械臂模型选择。
    parser.add_argument("--render", action="store_true", help="启用渲染（默认关闭，避免无头环境卡住）")  # 命令行参数：这里就是直接的调参入口
    parser.add_argument("--no-render", action="store_false", dest="render", help="关闭渲染")  # 命令行参数：这里就是直接的调参入口
    parser.set_defaults(render=False)  # 状态或中间变量：调试时多观察它的值如何流动
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")  # 渲染模式。
    parser.add_argument("--random-policy", action="store_true", help="不加载模型，使用随机动作测试环境")  # 随机动作模式。
    parser.add_argument("--print-step-reward", action="store_true", help="打印每一步奖励与关键信息")  # 命令行参数：这里就是直接的调参入口
    parser.add_argument("--print-reward-info", action="store_true", help="打印环境 info 里的奖励分项（若存在）")  # 命令行参数：这里就是直接的调参入口
    return parser.parse_args()  # 返回解析结果。


def main():  # 脚本入口：先看它如何把各个步骤串起来
    """测试入口函数。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    args = parse_args()  # 读取命令行参数。
    register_env()  # 注册 Gym 环境 id，确保 make_vec_env 可用。
    path_cfg = TrainArgs(  # 用训练脚本同一套分层目录规则解析默认测试路径。
        algo=args.algo,  # 状态或中间变量：调试时多观察它的值如何流动
        robot=args.robot,  # 状态或中间变量：调试时多观察它的值如何流动
        run_name=args.run_name,  # 状态或中间变量：调试时多观察它的值如何流动
        model_dir="models/classic",  # 状态或中间变量：调试时多观察它的值如何流动
        log_dir="logs/classic",  # 状态或中间变量：调试时多观察它的值如何流动
        model_path=args.model_path,  # 状态或中间变量：调试时多观察它的值如何流动
        normalize_path=args.norm_path,  # 状态或中间变量：调试时多观察它的值如何流动
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    paths = _build_run_paths(path_cfg)  # 状态或中间变量：调试时多观察它的值如何流动
    _sync_legacy_run_artifacts(path_cfg, paths)  # 代码执行语句：结合上下文理解它对后续流程的影响
    model_path, norm_path = _resolve_test_artifact_paths(path_cfg, paths)  # 状态或中间变量：调试时多观察它的值如何流动

    device = args.device  # 读取目标设备。
    if device.lower() == "cuda" and not torch.cuda.is_available():  # CUDA 不可用时自动回退。
        print("请求 cuda 但不可用，自动切换到 cpu")  # 打印回退提示。
        device = "cpu"  # 设置 CPU 设备。

    env_cfg = TrainArgs(  # 复用 TrainArgs 与 _make_env，保证环境参数风格一致。
        algo=args.algo,  # 算法名（仅用于后续一致性）。
        robot=args.robot,  # 机械臂模型（与训练时保持一致）。
        physics_backend=args.physics_backend,  # 物理后端选择（默认 auto）。
        legacy_zero_ee_velocity=bool(args.legacy_zero_ee_velocity),  # 是否兼容 zero 原始末端速度读取。
        render=bool(args.render),  # 测试是否渲染由命令行控制。
        render_mode=args.render_mode,  # 渲染模式。
        max_steps=args.max_steps,  # 单回合最大步数。
    )  # 收束上一段结构，阅读时回看上面的参数或元素

    env = None  # 状态或中间变量：调试时多观察它的值如何流动
    try:  # 异常保护：把高风险调用包起来，避免主流程中断
        env = make_vec_env(  # 创建向量化测试环境。
            ENV_ID,  # 环境 id。
            n_envs=1,  # 测试阶段单环境即可。
            seed=123,  # 固定种子便于复现。
            env_kwargs=_make_env(env_cfg, args.render_mode if args.render else None),  # 把配置传给环境构造函数。
        )  # 收束上一段结构，阅读时回看上面的参数或元素

        if not args.random_policy:  # 非随机策略模式才尝试加载归一化参数。
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
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
                if args.render:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    env.render()  # 按需刷新渲染窗口。
                step_reward = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)  # 兼容数组/标量奖励。
                total_reward += step_reward  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                steps += 1  # 步数累加。
                if args.print_step_reward:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    info0 = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else {}  # 状态或中间变量：调试时多观察它的值如何流动
                    distance = info0.get("distance")  # 状态或中间变量：调试时多观察它的值如何流动
                    success = info0.get("success")  # 状态或中间变量：调试时多观察它的值如何流动
                    if distance is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
                        print(f"[step {steps}] reward={step_reward:.6f}")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    else:  # 兜底分支：当前面条件都不满足时走这里
                        print(f"[step {steps}] reward={step_reward:.6f}, distance={float(distance):.6f}, success={bool(success)}")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    if args.print_reward_info:  # 条件分支：学习时先看触发条件，再看两边行为差异
                        reward_info = info0.get("reward_info")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                        if isinstance(reward_info, dict):  # 条件分支：学习时先看触发条件，再看两边行为差异
                            print(f"  reward_info={reward_info}")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                if args.render and args.render_mode == "human":  # human 模式下减速，便于观察。
                    time.sleep(0.01)  # 每步暂停 10ms。
            print(f"Episode {ep + 1}: steps={steps}, reward={total_reward:.3f}")  # 输出回合结果。
    finally:  # 代码执行语句：结合上下文理解它对后续流程的影响
        if env is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            env.close()  # 关闭环境并释放渲染资源。
    print("测试完成。")  # 测试结束提示。


if __name__ == "__main__":  # 条件分支：学习时先看触发条件，再看两边行为差异
    main()  # 直接运行脚本时执行入口函数。
