#!/usr/bin/env python3
# 主线训练与测试入口（zero-arm 风格重写）。
#
# 本模块负责 UR5 到点任务的命令行解析、环境构建、算法初始化、
# 回调注册、模型保存和测试流程。
#
# 涉及的主要外部库：
# - `Stable-Baselines3`：提供 TD3 / SAC / PPO 和回调系统。
# - `Gymnasium`：提供环境接口规范。
# - `MuJoCo`：提供机器人动力学和渲染。
# - `VecNormalize`：负责观测和奖励标准化。

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ur5_reach_env import UR5ReachEnv


def _model_class(algo: str):
    mapping = {"td3": TD3, "sac": SAC, "ppo": PPO}
    if algo not in mapping:
        raise ValueError(f"不支持的算法: {algo}")
    return mapping[algo]


class SaveVecNormalizeCallback(BaseCallback):
    # 在保存最佳模型时同时保存 VecNormalize 参数。
    def __init__(self, eval_callback: EvalCallback, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 检查是否有新的最佳模型。
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            if self.verbose > 0:
                print("保存与最佳模型对应的 VecNormalize 参数")
            os.makedirs("./logs/best_model", exist_ok=True)
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save("./logs/best_model/vec_normalize.pkl")
        return True


class ManualInterruptCallback(BaseCallback):
    # 允许手动中断训练并保存模型的回调函数。
    def __init__(self, algo: str, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.algo = algo
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame) -> None:
        del sig, frame
        print("\n接收到中断信号，正在保存模型...")
        self.interrupted = True
        self._save_model()
        print("模型已保存，退出程序")
        raise SystemExit(0)

    def _save_model(self) -> None:
        if self.model is None:
            return
        os.makedirs("./models/interrupted", exist_ok=True)
        self.model.save(f"./models/interrupted/{self.algo}_ur5_interrupted")
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save("./models/interrupted/vec_normalize.pkl")
        print("已保存中断时的模型和参数到 ./models/interrupted/")

    def _on_step(self) -> bool:
        return not self.interrupted


class TrainRenderCallback(BaseCallback):
    # 训练时按频率触发环境渲染。
    def __init__(self, render_every: int = 1, render_index: int = 0, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.render_every = max(1, int(render_every))
        self.render_index = int(render_index)
        self.render_failed = False

    def _on_step(self) -> bool:
        if self.render_failed or self.n_calls % self.render_every != 0:
            return True
        try:
            self.training_env.env_method("render", indices=self.render_index)
        except Exception as exc:
            self.render_failed = True
            print(f"训练渲染失败，后续将关闭训练渲染: {exc}")
        return True


def make_training_envs(n_envs: int, render: bool) -> VecNormalize:
    # 创建训练环境。开启渲染时所有环境声明同一 render_mode，
    # 但训练过程中仅主动渲染第一个环境，避免弹出多个窗口。
    render_mode = "human" if render else None

    def _make_env(render_mode=render_mode):
        return UR5ReachEnv(render_mode=render_mode)

    env_fns = [_make_env for _ in range(max(int(n_envs), 1))]
    env = DummyVecEnv(env_fns)
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_eval_env(train_env: VecNormalize) -> VecNormalize:
    # 创建评估环境，并同步训练环境的 obs 统计量。
    eval_env = DummyVecEnv([lambda: UR5ReachEnv(render_mode=None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    if hasattr(train_env, "obs_rms"):
        eval_env.obs_rms = train_env.obs_rms
    return eval_env


def build_policy_kwargs() -> dict:
    # 自定义网络结构：Actor / Critic 均使用 [512, 512, 256]。
    return {"net_arch": dict(pi=[512, 512, 256], qf=[512, 512, 256]), "activation_fn": nn.ReLU}  # ReLU 激活


def train_robot_arm(
    algo: str,
    total_timesteps: int,
    n_envs: int,
    render: bool,
    render_every: int,
    device: str,
) -> None:
    print("创建机械臂环境...")  # 训练入口日志
    if render and n_envs > 1:
        print("已开启并行训练渲染：仅显示第 1 个环境，其余环境保持无头模式")

    env = make_training_envs(n_envs=n_envs, render=render)  # 构造训练 VecEnv

    # 设置动作噪声
    # - TD3 使用 action noise 来提升探索
    # - SAC 自带熵正则，不强依赖外部噪声
    # - PPO 为 on-policy，不使用动作噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.5 * np.ones(n_actions))

    policy_kwargs = build_policy_kwargs()

    # 创建模型
    # 这里把各算法的关键超参写死在代码里，便于和 zero-arm 对照。
    # 如果需要更细粒度调参，可把这些参数改为 CLI 选项。
    model_cls = _model_class(algo)
    if algo == "ppo":
        model = model_cls(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,  # Adam 学习率
            batch_size=256,  # 每次更新的样本数
            gamma=0.99,  # 折扣因子
            n_steps=2048,  # rollout 长度（每次采样的步数）
            n_epochs=10,  # 每批数据的优化轮数
            gae_lambda=0.95,  # GAE 优势估计参数
            ent_coef=0.0,  # 熵正则系数（探索强度）
            vf_coef=0.5,  # 价值函数损失系数
            clip_range=0.2,  # PPO clip 范围
            policy_kwargs=policy_kwargs,
        )
    else:
        model = model_cls(
            "MlpPolicy",
            env,
            action_noise=action_noise if algo == "td3" else None,
            verbose=1,
            device=device,
            learning_rate=3e-4,  # Adam 学习率
            buffer_size=3_000_000,  # 回放池容量（off-policy 必需）
            learning_starts=10_000,  # 预热步数
            batch_size=256,  # 每次更新的样本数
            tau=0.005,  # 目标网络软更新系数
            gamma=0.99,  # 折扣因子
            train_freq=1,  # 每步更新频率
            gradient_steps=1,  # 每次采样后更新次数
            policy_delay=4 if algo == "td3" else 1,  # TD3 延迟更新策略网络
            target_policy_noise=0.2 if algo == "td3" else 0.0,  # TD3 目标噪声
            target_noise_clip=0.5 if algo == "td3" else 0.0,  # TD3 目标噪声裁剪
            policy_kwargs=policy_kwargs,
        )

    # 创建评估环境和回调函数
    eval_env = make_eval_env(env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    save_vec_normalize_callback = SaveVecNormalizeCallback(eval_callback, verbose=1)
    manual_interrupt_callback = ManualInterruptCallback(algo=algo, verbose=1)

    callbacks = [eval_callback, save_vec_normalize_callback, manual_interrupt_callback]  # 训练回调集合
    if render:
        callbacks.append(TrainRenderCallback(render_every=render_every, render_index=0, verbose=1))

    os.makedirs("./logs", exist_ok=True)  # 训练日志目录
    os.makedirs("./models", exist_ok=True)  # 模型保存目录

    print("开始训练...")  # 真正进入 learn 阶段
    print("提示: 按 Ctrl+C 可以中途停止训练并保存最后一次模型数据")
    print(f"训练配置: algo={algo}, total_timesteps={total_timesteps}, n_envs={n_envs}, render={render}, render_every={render_every}")
    start_time = time.time()

    model.learn(
        total_timesteps=int(total_timesteps),
        callback=callbacks,
        log_interval=1000,
        progress_bar=True,
    )

    env.save("./models/vec_normalize.pkl")  # 保存归一化统计量
    model.save(f"./models/{algo}_ur5_final")  # 保存最终模型

    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")


def test_robot_arm(
    algo: str,
    model_path: str,
    normalize_path: str,
    num_episodes: int,
) -> None:
    print("加载模型并测试...")  # 测试入口日志

    env = DummyVecEnv([lambda: UR5ReachEnv(render_mode="human")])
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False

    model = _model_class(algo).load(model_path, env=env)
    episode_rewards: list[float] = []

    for episode in range(max(int(num_episodes), 1)):
        obs = env.reset()
        total_reward = 0.0

        for step in range(int(UR5ReachEnv.EPISODE_LENGTH)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            total_reward += reward_value
            time.sleep(0.01)  # 控制渲染频率，防止窗口卡顿
            env.render()  # human 渲染窗口刷新
            if bool(done):
                print(f"Episode {episode + 1} finished after {step + 1} timesteps")
                print(f"Episode reward: {total_reward:.3f}")
                episode_rewards.append(total_reward)
                break

    env.close()
    if episode_rewards:
        print(f"Average reward over {len(episode_rewards)} episodes: {float(np.mean(episode_rewards)):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or test UR5 with TD3/SAC/PPO (zero-arm style)")
    parser.add_argument("--algo", choices=["td3", "sac", "ppo"], default="td3", help="训练算法")  # algo 选择
    parser.add_argument("--test", action="store_true", help="测试已训练模型")  # 只跑测试
    parser.add_argument("--model-path", type=str, default="./models/td3_ur5_final", help="测试模型路径")  # 模型路径
    parser.add_argument("--normalize-path", type=str, default="./models/vec_normalize.pkl", help="归一化参数路径")  # VecNormalize
    parser.add_argument("--episodes", type=int, default=10, help="测试回合数")  # 测试回合数
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="训练步数")  # 训练总步数
    parser.add_argument("--n-envs", type=int, default=1, help="并行环境数")  # 并行环境数
    parser.add_argument("--render", action="store_true", help="训练时打开人类可视化窗口")  # 训练渲染开关
    parser.add_argument("--render-every", type=int, default=1, help="训练渲染刷新间隔")  # 训练渲染频率
    parser.add_argument("--device", type=str, default="auto", help="训练设备，例如 auto/cpu/cuda")  # 训练设备

    args = parser.parse_args()
    if args.test:
        test_robot_arm(
            algo=args.algo,
            model_path=args.model_path,
            normalize_path=args.normalize_path,
            num_episodes=args.episodes,
        )
    else:
        train_robot_arm(
            algo=args.algo,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            render=args.render,
            render_every=args.render_every,
            device=args.device,
        )


if __name__ == "__main__":
    main()
