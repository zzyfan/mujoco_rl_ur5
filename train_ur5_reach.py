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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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


class EpisodeLogCallback(BaseCallback):
    # 每回合输出：碰撞次数、成功次数、平均距离、最小距离、平均速度
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._collision_counts = None  # 每回合碰撞次数统计
        self._success_counts = None  # 每回合成功次数统计
        self._sum_distance = None  # 累积距离（用于平均距离）
        self._min_distance = None  # 最小距离
        self._sum_speed = None  # 累积速度（用于平均速度）
        self._step_counts = None  # 步数计数

    def _init_arrays(self, n_envs: int) -> None:
        self._collision_counts = np.zeros(n_envs, dtype=np.int32)  # 初始化碰撞计数
        self._success_counts = np.zeros(n_envs, dtype=np.int32)  # 初始化成功计数
        self._sum_distance = np.zeros(n_envs, dtype=np.float32)  # 初始化距离累积
        self._min_distance = np.full(n_envs, np.inf, dtype=np.float32)  # 初始化最小距离
        self._sum_speed = np.zeros(n_envs, dtype=np.float32)  # 初始化速度累积
        self._step_counts = np.zeros(n_envs, dtype=np.int32)  # 初始化步数计数

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        n_envs = len(infos)
        if self._collision_counts is None:
            self._init_arrays(n_envs)
        for idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            if "collision" in info and bool(info["collision"]):
                self._collision_counts[idx] += 1  # 统计碰撞
            if "success" in info and bool(info["success"]):
                self._success_counts[idx] += 1  # 统计成功
            if "distance" in info:
                try:
                    distance = float(info["distance"])  # 当前距离
                    self._sum_distance[idx] += distance  # 累加距离
                    self._min_distance[idx] = min(self._min_distance[idx], distance)  # 更新最小距离
                except (TypeError, ValueError):
                    pass
            if "ee_speed" in info:
                try:
                    self._sum_speed[idx] += float(info["ee_speed"])  # 累加速度
                except (TypeError, ValueError):
                    pass
            self._step_counts[idx] += 1  # 步数累加
            if idx < len(dones) and bool(dones[idx]):
                steps = max(int(self._step_counts[idx]), 1)  # 防止除零
                avg_distance = float(self._sum_distance[idx]) / steps  # 平均距离
                avg_speed = float(self._sum_speed[idx]) / steps  # 平均速度
                print(
                    f"[episode] env={idx} collisions={int(self._collision_counts[idx])} "
                    f"successes={int(self._success_counts[idx])} "
                    f"avg_dist={avg_distance:.4f} min_dist={float(self._min_distance[idx]):.4f} "
                    f"avg_speed={avg_speed:.4f}"
                )
                self._collision_counts[idx] = 0  # 清空碰撞计数
                self._success_counts[idx] = 0  # 清空成功计数
                self._sum_distance[idx] = 0.0  # 清空距离累积
                self._min_distance[idx] = np.inf  # 清空最小距离
                self._sum_speed[idx] = 0.0  # 清空速度累积
                self._step_counts[idx] = 0  # 清空步数计数
        return True


def make_training_envs(n_envs: int, render_mode: str | None) -> VecNormalize:
    # 创建训练环境。render_mode 使用 Gymnasium 官方命名（None / "human"）。
    # 训练过程中仅主动渲染第一个环境，避免弹出多个窗口。

    def _make_env(render_mode=render_mode):
        return UR5ReachEnv(render_mode=render_mode)

    env_fns = [_make_env for _ in range(max(int(n_envs), 1))]  # 生成环境工厂列表
    # 并行环境优先用 SubprocVecEnv（多进程），单环境时用 DummyVecEnv
    env = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)  # 多进程/单进程切换
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def _safe_float(value) -> float | None:
    # 尽量把 NumPy 标量、普通数字或布尔值转成 Python `float`，失败时返回 `None`。
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    # 和 `_safe_float(...)` 类似，但这里希望得到整数日志字段。
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class DetailedTrainLogCallback(BaseCallback):
    # 训练过程诊断日志：
    # 1. 开始训练时打印观测向量每一段的真实含义。
    # 2. 每个向量化 step 输出一行聚合诊断。
    # 3. 每个回合结束时输出完整摘要。

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._episode_log_order = 0

    def _on_training_start(self) -> None:
        print("observation_schema:")
        for obs_slice, name, meaning in UR5ReachEnv.observation_schema():
            print(f"  {obs_slice} {name}: {meaning}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float32).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        if not infos:
            return True

        distances: list[float] = []
        speeds: list[float] = []
        episode_returns: list[float] = []
        active_collision_counts: list[int] = []
        lifetime_success_total = 0
        stage_counts: dict[str, int] = {}
        representative_info = None

        for info in infos:
            if not isinstance(info, dict):
                continue
            distance = _safe_float(info.get("relative_distance", info.get("distance")))
            speed = _safe_float(info.get("relative_speed", info.get("ee_speed")))
            episode_return = _safe_float(info.get("episode_return"))
            collision_count = _safe_int(info.get("episode_collision_count"))
            success_total = _safe_int(info.get("lifetime_success_count"))
            stage = info.get("curriculum_stage")
            if distance is not None:
                distances.append(distance)
            if speed is not None:
                speeds.append(speed)
            if episode_return is not None:
                episode_returns.append(episode_return)
            if collision_count is not None:
                active_collision_counts.append(collision_count)
            if success_total is not None:
                lifetime_success_total += success_total
            if isinstance(stage, str):
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if representative_info is None:
                representative_info = info

        reward_mean = float(np.mean(rewards)) if rewards.size else 0.0
        mean_distance = float(np.mean(distances)) if distances else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_episode_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        active_collisions = int(np.sum(active_collision_counts)) if active_collision_counts else 0
        representative_step = _safe_int((representative_info or {}).get("step_in_episode")) or 0
        representative_episode = _safe_int((representative_info or {}).get("episode_index")) or 0
        stage_summary = ",".join(f"{name}:{count}" for name, count in sorted(stage_counts.items())) if stage_counts else "unknown"
        print(
            f"[train_step] env_steps={self.num_timesteps} episode={representative_episode} "
            f"step={representative_step} rel_dist_mean={mean_distance:.4f} rel_speed_mean={mean_speed:.4f} "
            f"success_total={lifetime_success_total} episode_return_mean={mean_episode_return:.3f} "
            f"collision_count_active={active_collisions} reward_mean={reward_mean:.3f} stage_mix={stage_summary}"
        )

        finished_episodes: list[tuple[int, dict]] = []
        for idx, done in enumerate(dones):
            if not bool(done) or idx >= len(infos) or not isinstance(infos[idx], dict):
                continue
            summary = infos[idx].get("episode_summary") or {}
            finished_episodes.append((idx, summary))

        for idx, summary in sorted(finished_episodes, key=lambda item: item[0]):
            self._episode_log_order += 1
            print(
                f"[episode_end] order={self._episode_log_order} env={idx} "
                f"episode={_safe_int(summary.get('episode_index')) or 0} "
                f"done_reason={summary.get('done_reason', 'unknown')} total_reward={_safe_float(summary.get('episode_return')) or 0.0:.3f} "
                f"steps={_safe_int(summary.get('episode_steps')) or 0} final_distance={_safe_float(summary.get('final_distance')) or 0.0:.4f} "
                f"min_distance={_safe_float(summary.get('min_distance')) or 0.0:.4f} "
                f"final_speed={_safe_float(summary.get('final_speed')) or 0.0:.4f} "
                f"collisions={_safe_int(summary.get('episode_collision_count')) or 0} "
                f"episode_successes={_safe_int(summary.get('episode_success_count')) or 0} "
                f"lifetime_successes={_safe_int(summary.get('lifetime_success_count')) or 0} "
                f"stage={summary.get('curriculum_stage', 'unknown')}"
            )
        return True


def build_parser() -> argparse.ArgumentParser:
    # 构造主线 CLI。
    #
    # 参数分成三组：
    # - 基础参数：训练/测试共用。
    # - Training：算法训练流程参数。
    # - Environment：任务环境参数。
    train_defaults = RLTrainConfig()
    env_defaults = UR5ReachEnvConfig()
    # 先取一份 dataclass 默认值，后面 CLI 的默认参数直接从这里读，
    # 保证“代码默认值”和“命令行默认值”始终一致。
    parser = argparse.ArgumentParser(description="UR5 reach 单线复现入口，仅保留 UR5，支持 td3 / sac / ppo。")
    
    # 这组参数控制脚本整体行为，例如训练还是测试、产物目录名字以及是否手动指定模型。
    parser.add_argument("--algo", choices=["td3", "sac", "ppo"], default=train_defaults.algo, help="算法名称。")
    parser.add_argument("--test", action="store_true", help="切换到测试模式。")
    parser.add_argument("--run-name", type=str, default=train_defaults.run_name, help="实验名字，也是产物目录名字。")
    parser.add_argument("--seed", type=int, default=train_defaults.seed, help="随机种子。")
    parser.add_argument("--episodes", type=int, default=5, help="测试模式运行多少回合。")
    parser.add_argument("--max-steps", type=int, default=None, help="测试模式覆盖环境回合长度。")
    parser.add_argument("--render", action="store_true", help="测试模式是否打开人类可视化窗口。")
    parser.add_argument("--print-reward-terms", action="store_true", help="测试模式打印奖励分解。")
    parser.add_argument("--model-path", type=str, default="", help="手动指定模型路径。")
    parser.add_argument("--normalize-path", type=str, default="", help="手动指定 VecNormalize 路径。")

    # Training 组对应 `RLTrainConfig`，主要控制 SB3 算法更新和训练过程。
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--total-timesteps", type=int, default=train_defaults.total_timesteps, help="总训练步数。")
    train_group.add_argument("--n-envs", type=int, default=train_defaults.n_envs, help="并行环境数量。")
    train_group.add_argument("--eval-freq", type=int, default=train_defaults.eval_freq, help="评估间隔。")
    train_group.add_argument("--eval-episodes", type=int, default=train_defaults.eval_episodes, help="每次评估回合数。")
    train_group.add_argument("--device", type=str, default=train_defaults.device, help="训练设备，例如 auto、cpu、cuda。")
    train_group.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate, help="学习率。")
    train_group.add_argument("--buffer-size", type=int, default=train_defaults.buffer_size, help="回放池大小，PPO 下忽略。")
    train_group.add_argument("--learning-starts", type=int, default=train_defaults.learning_starts, help="开始学习前的预热步数，PPO 下忽略。")
    train_group.add_argument("--batch-size", type=int, default=train_defaults.batch_size, help="训练 batch 大小。")
    train_group.add_argument("--tau", type=float, default=train_defaults.tau, help="目标网络软更新系数，PPO 下忽略。")
    train_group.add_argument("--gamma", type=float, default=train_defaults.gamma, help="折扣因子。")
    train_group.add_argument("--train-freq", type=int, default=train_defaults.train_freq, help="离策略算法的训练触发频率，PPO 下忽略。")
    train_group.add_argument("--gradient-steps", type=int, default=train_defaults.gradient_steps, help="离策略算法每次更新的梯度步数，PPO 下忽略。")
    train_group.add_argument("--policy-delay", type=int, default=train_defaults.policy_delay, help="TD3 actor 更新延迟。")
    train_group.add_argument("--target-policy-noise", type=float, default=train_defaults.target_policy_noise, help="TD3 目标策略噪声。")
    train_group.add_argument("--target-noise-clip", type=float, default=train_defaults.target_noise_clip, help="TD3 目标噪声裁剪。")
    train_group.add_argument("--action-noise-sigma", type=float, default=train_defaults.action_noise_sigma, help="TD3 探索动作噪声。")
    train_group.add_argument("--ppo-n-steps", type=int, default=train_defaults.ppo_n_steps, help="PPO rollout 长度。")
    train_group.add_argument("--ppo-n-epochs", type=int, default=train_defaults.ppo_n_epochs, help="PPO 每轮 rollout 的优化轮数。")
    train_group.add_argument("--ppo-gae-lambda", type=float, default=train_defaults.ppo_gae_lambda, help="PPO 的 GAE lambda。")
    train_group.add_argument("--ppo-ent-coef", type=float, default=train_defaults.ppo_ent_coef, help="PPO 的熵正则系数。")
    train_group.add_argument("--ppo-vf-coef", type=float, default=train_defaults.ppo_vf_coef, help="PPO value loss 权重。")
    train_group.add_argument("--ppo-clip-range", type=float, default=train_defaults.ppo_clip_range, help="PPO 裁剪范围。")
    train_group.add_argument("--render-training", action="store_true", help="训练时打开窗口。")
    train_group.add_argument("--render-every", type=int, default=train_defaults.render_every, help="训练渲染刷新间隔。")
    train_group.add_argument("--spectator-render", action="store_true", help="训练无头多并行时，在主进程单独打开一个旁观窗口。")
    train_group.add_argument("--spectator-render-every", type=int, default=train_defaults.spectator_render_every, help="旁观窗口每隔多少个训练 step 更新一次。")
    train_group.add_argument("--no-spectator-deterministic", action="store_false", dest="spectator_deterministic", help="让旁观模式使用随机动作采样，而不是确定性动作。")
    parser.set_defaults(spectator_deterministic=train_defaults.spectator_deterministic)

    # Environment 组对应 `UR5ReachEnvConfig`，主要控制任务定义和课程学习。
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--frame-skip", type=int, default=env_defaults.frame_skip, help="每个 RL step 对应多少个物理步。")
    env_group.add_argument("--episode-length", type=int, default=env_defaults.episode_length, help="训练模式的回合长度。")
    env_group.add_argument("--control-mode", choices=["joint_delta", "torque"], default=env_defaults.control_mode, help="控制模式。")
    env_group.add_argument("--joint-delta-scale", type=float, default=env_defaults.joint_delta_scale, help="关节增量控制时每步最大变化量。")
    env_group.add_argument("--position-kp", type=float, default=env_defaults.position_kp, help="PD 比例增益。")
    env_group.add_argument("--position-kd", type=float, default=env_defaults.position_kd, help="PD 阻尼增益。")
    env_group.add_argument("--action-smoothing-alpha", type=float, default=env_defaults.action_smoothing_alpha, help="动作平滑系数。")
    env_group.add_argument("--curriculum-fixed-episodes", type=int, default=env_defaults.curriculum_fixed_episodes, help="固定目标阶段回合数。")
    env_group.add_argument("--curriculum-local-random-episodes", type=int, default=env_defaults.curriculum_local_random_episodes, help="局部随机阶段回合数。")
    env_group.add_argument("--curriculum-local-scale", type=float, default=env_defaults.curriculum_local_scale, help="局部随机范围比例。")
    env_group.add_argument("--fixed-target-x", type=float, default=env_defaults.fixed_target_x, help="固定目标 x。")
    env_group.add_argument("--fixed-target-y", type=float, default=env_defaults.fixed_target_y, help="固定目标 y。")
    env_group.add_argument("--fixed-target-z", type=float, default=env_defaults.fixed_target_z, help="固定目标 z。")
    env_group.add_argument("--target-x-min", type=float, default=env_defaults.target_x_min, help="目标采样 x 最小值。")
    env_group.add_argument("--target-x-max", type=float, default=env_defaults.target_x_max, help="目标采样 x 最大值。")
    env_group.add_argument("--target-y-min", type=float, default=env_defaults.target_y_min, help="目标采样 y 最小值。")
    env_group.add_argument("--target-y-max", type=float, default=env_defaults.target_y_max, help="目标采样 y 最大值。")
    env_group.add_argument("--target-z-min", type=float, default=env_defaults.target_z_min, help="目标采样 z 最小值。")
    env_group.add_argument("--target-z-max", type=float, default=env_defaults.target_z_max, help="目标采样 z 最大值。")
    return parser


def build_policy_kwargs(algo: str) -> dict:
    # 自定义网络结构：Actor / Critic 使用 [512, 512, 256]。
    # PPO 需要 pi/vf，SAC/TD3 需要 pi/qf。
    if algo == "ppo":
        return {"net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]), "activation_fn": nn.ReLU}  # ReLU 激活
    return {"net_arch": dict(pi=[512, 512, 256], qf=[512, 512, 256]), "activation_fn": nn.ReLU}  # ReLU 激活


def train_robot_arm(
    algo: str,
    total_timesteps: int,
    n_envs: int,
    render_mode: str | None,
    render_every: int,
    device: str,
) -> None:
    print("创建机械臂环境...")  # 训练入口日志
    if render_mode == "human" and n_envs > 1:
        print("已开启并行训练渲染：仅显示第 1 个环境，其余环境保持无头模式")

    env = make_training_envs(n_envs=n_envs, render_mode=render_mode)  # 构造训练 VecEnv

    # 设置动作噪声
    # - TD3 使用 action noise 来提升探索
    # - SAC 自带熵正则，不强依赖外部噪声
    # - PPO 为 on-policy，不使用动作噪声
    n_actions = env.action_space.shape[-1]  # 动作维度
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.5 * np.ones(n_actions))

    policy_kwargs = build_policy_kwargs(algo)  # 按算法生成网络结构

    # 创建模型
    # 这里把各算法的关键超参写死在代码里，便于和 zero-arm 对照。
    # 如果需要更细粒度调参，可把这些参数改为 CLI 选项。
    model_cls = _model_class(algo)  # 选择算法类
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
        # SAC 与 TD3 参数并不完全一致，这里用条件参数避免传入无效字段。
        td3_only_kwargs = {}  # TD3 专用参数
        if algo == "td3":
            td3_only_kwargs = {
                "policy_delay": 4,  # TD3 延迟更新策略网络
                "target_policy_noise": 0.2,  # TD3 目标噪声
                "target_noise_clip": 0.5,  # TD3 目标噪声裁剪
            }
        action_noise_kw = {"action_noise": action_noise} if algo == "td3" else {}  # TD3 动作噪声
        model = model_cls(
            "MlpPolicy",
            env,
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
            policy_kwargs=policy_kwargs,
            **action_noise_kw,
            **td3_only_kwargs,
        )

    # 创建评估环境和回调函数
    eval_env = make_eval_env(env)  # 评估环境
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    callbacks.append(SaveVecNormalizeCallback(eval_callback, paths.best_normalize_path, verbose=1))
    callbacks.append(ManualInterruptCallback(paths.interrupted_model_path, paths.interrupted_normalize_path, verbose=1))
    callbacks.append(DetailedTrainLogCallback(verbose=1))
    # 最后再按模式决定是否追加训练渲染或旁观渲染回调。
    if train_config.render_training:
        callbacks.append(TrainRenderCallback(train_config.render_every, verbose=1))
    elif train_config.spectator_render:
        callbacks.append(
            SpectatorRenderCallback(
                env_config=env_config,
                render_every=train_config.spectator_render_every,
                deterministic=train_config.spectator_deterministic,
                verbose=1,
            )
        )

    callbacks = [
        eval_callback,
        save_vec_normalize_callback,
        manual_interrupt_callback,
        EpisodeLogCallback(verbose=1),
    ]  # 训练回调集合
    if render_mode == "human":
        callbacks.append(TrainRenderCallback(render_every=render_every, render_index=0, verbose=1))

    os.makedirs("./logs", exist_ok=True)  # 训练日志目录
    os.makedirs("./models", exist_ok=True)  # 模型保存目录

    print("开始训练...")  # 真正进入 learn 阶段
    print("提示: 按 Ctrl+C 可以中途停止训练并保存最后一次模型数据")
    print(
        f"训练配置: algo={algo}, total_timesteps={total_timesteps}, n_envs={n_envs}, render_mode={render_mode}, render_every={render_every}"
    )
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
    parser.add_argument(
        "--render-mode",
        choices=["none", "human"],
        default="none",
        help="训练渲染模式（符合 Gymnasium 官方命名）",
    )  # 训练渲染模式
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
            render_mode=None if args.render_mode == "none" else args.render_mode,
            render_every=args.render_every,
            device=args.device,
        )


if __name__ == "__main__":
    main()
