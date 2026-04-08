#!/usr/bin/env python3
# 主线训练与测试入口。
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
import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ur5_reach_config import (
    RLTrainConfig,
    UR5ReachEnvConfig,
    build_run_paths,
    ensure_run_directories,
    save_run_configuration,
)
from ur5_reach_env import UR5ReachEnv


def _model_class(algo: str):
    # 根据算法名字返回对应的 SB3 模型类。
    #
    # 这一步把字符串参数和真实 Python 类解耦，后面训练和测试都复用这个映射。
    mapping = {"td3": TD3, "sac": SAC, "ppo": PPO}
    if algo not in mapping:
        raise ValueError(f"不支持的算法: {algo}")
    return mapping[algo]


def _artifact_run_name(run_name: str, env_config: UR5ReachEnvConfig) -> str:
    # 为不同末端模型分开保存产物，避免同名实验互相覆盖。
    # 规则：
    # - 带夹爪：沿用原始 run_name
    # - 不带夹爪：自动追加 `__no_gripper`
    return f"{run_name}__no_gripper" if bool(env_config.disable_gripper_end_effector) else run_name


class SaveVecNormalizeCallback(BaseCallback):
    # 在 best model 更新时同步保存 VecNormalize 参数。
    #
    # 训练时如果启用了观测归一化或奖励归一化，模型文件本身并不包含这些统计量，
    # 所以这里需要和最佳模型一起额外保存一份。

    def __init__(self, eval_callback: EvalCallback, normalize_path: Path, verbose: int = 0) -> None:
        # `eval_callback` 负责告诉我们“当前 best reward 是多少”，
        # `normalize_path` 负责告诉我们“最佳统计量保存到哪里”。
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.normalize_path = normalize_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 每一步都检查一次最佳评估回报是否刷新。
        # 一旦刷新，就把当前 VecNormalize 统计量落盘。
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(str(self.normalize_path))
                if self.verbose > 0:
                    print(f"已同步保存最佳 VecNormalize 参数: {self.normalize_path}")
        return True


class ManualInterruptCallback(BaseCallback):
    # 在用户按下 Ctrl+C 时保存中断模型。
    #
    # 这个回调的作用是让长时间训练在手动停止时也能保留当前进度。

    def __init__(self, model_path: Path, normalize_path: Path, verbose: int = 0) -> None:
        # 这里把输出路径保存下来，并且注册 SIGINT 处理函数。
        super().__init__(verbose=verbose)
        self.model_path = model_path
        self.normalize_path = normalize_path
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame) -> None:
        del sig, frame
        # 用户按 Ctrl+C 时，不直接粗暴退出，而是先保存模型和归一化参数。
        print("\n收到 Ctrl+C，正在保存中断模型...")
        self.interrupted = True
        self._save_checkpoint()
        raise SystemExit(0)

    def _save_checkpoint(self) -> None:
        # 这里保存两类东西：
        # 1. 模型参数
        # 2. 观测/奖励归一化统计量
        if self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_path))
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(self.normalize_path))
        print(f"中断模型已保存到: {self.model_path}")

    def _on_step(self) -> bool:
        return not self.interrupted


class TrainRenderCallback(BaseCallback):
    # 训练时定期渲染一个向量化环境实例。

    def __init__(self, render_every: int, verbose: int = 0) -> None:
        # `render_every` 控制渲染频率，避免每一步都渲染导致训练极慢。
        super().__init__(verbose=verbose)
        self.render_every = max(int(render_every), 1)
        self.render_failed = False

    def _on_step(self) -> bool:
        # 训练渲染只尝试渲染一个环境实例，而且一旦失败就永久关闭渲染，
        # 避免图形问题让整个训练中断。
        if self.render_failed or self.n_calls % self.render_every != 0:
            return True
        try:
            self.training_env.env_method("render", indices=0)
        except Exception as exc:
            self.render_failed = True
            print(f"训练渲染失败，后续关闭渲染: {exc}")
        return True


class SpectatorRenderCallback(BaseCallback):
    # 在训练保持无头并行时，主进程额外运行一个旁观环境。
    #
    # 这和 `--render-training` 的区别是：
    # - 训练环境本体仍然可以使用多并行。
    # - 可视化环境是主进程中的独立环境，只负责展示当前策略效果。

    def __init__(
        self,
        env_config: UR5ReachEnvConfig,
        render_every: int,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        # 旁观环境和训练环境不是同一个实例。
        # 这样训练环境仍然可以保持无头多并行，而旁观环境单独显示当前策略行为。
        super().__init__(verbose=verbose)
        self.env_config = env_config
        self.render_every = max(int(render_every), 1)
        self.deterministic = bool(deterministic)
        self.spectator_env: UR5ReachEnv | None = None
        self.spectator_obs = None
        self.render_failed = False

    def _normalize_for_policy(self, observation: np.ndarray) -> np.ndarray:
        # 旁观环境拿到的是原始观测，但模型训练时可能看到的是归一化后的观测，
        # 所以这里要先把旁观观测按当前 VecNormalize 统计量做一次同样的预处理。
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is None:
            return observation
        normalized = vec_env.normalize_obs(observation.copy())
        return np.asarray(normalized, dtype=np.float32)

    def _on_training_start(self) -> None:
        # 训练开始时，单独创建一个 human 模式的环境实例用于旁观。
        try:
            self.spectator_env = UR5ReachEnv(config=self.env_config, render_mode="human")
            self.spectator_obs, _info = self.spectator_env.reset(seed=12345)
            self.spectator_env.render()
        except Exception as exc:
            self.render_failed = True
            self.spectator_env = None
            self.spectator_obs = None
            print(f"旁观窗口启动失败，已自动关闭 spectator 模式: {exc}")

    def _on_step(self) -> bool:
        # 旁观模式的单步流程：
        # 1. 把旁观环境观测归一化
        # 2. 用当前模型预测动作
        # 3. 推进一步旁观环境
        # 4. 若回合结束则自动 reset
        if self.render_failed or self.spectator_env is None or self.n_calls % self.render_every != 0:
            return True
        try:
            policy_obs = self._normalize_for_policy(np.asarray(self.spectator_obs, dtype=np.float32))
            action, _state = self.model.predict(policy_obs, deterministic=self.deterministic)
            next_obs, _reward, terminated, truncated, _info = self.spectator_env.step(action)
            self.spectator_env.render()
            if bool(terminated) or bool(truncated):
                self.spectator_obs, _reset_info = self.spectator_env.reset()
                self.spectator_env.render()
            else:
                self.spectator_obs = next_obs
        except Exception as exc:
            self.render_failed = True
            print(f"旁观窗口更新失败，已自动关闭 spectator 模式: {exc}")
            if self.spectator_env is not None:
                self.spectator_env.close()
                self.spectator_env = None
                self.spectator_obs = None
        return True

    def _on_training_end(self) -> None:
        # 训练结束时主动关闭旁观环境，避免 viewer 句柄泄露。
        if self.spectator_env is not None:
            self.spectator_env.close()
            self.spectator_env = None
            self.spectator_obs = None


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
    # 2. 多环境训练时只在回合结束时输出摘要，避免刷掉进度条。
    # 3. 只有单环境训练时才保留逐步聚合日志。

    def __init__(self, verbose: int = 0, train_step_log_every: int = 100) -> None:
        super().__init__(verbose=verbose)
        self._episode_log_order = 0
        self.max_lifetime_success_total = 0
        # SB3 的 progress bar 本身会持续刷新单行状态。
        # 如果我们每个向量化 step 都 `print(...)`，终端会被训练日志刷满，
        # 进度条看起来就像“消失了”。因此这里把聚合训练日志做节流，
        # 只保留固定间隔输出；而回合结束摘要仍然实时打印。
        self._train_step_log_every = max(int(train_step_log_every), 1)

    def _on_training_start(self) -> None:
        print("[observation_schema]")
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
        self.max_lifetime_success_total = max(self.max_lifetime_success_total, int(lifetime_success_total))
        should_print_step_log = int(getattr(self.training_env, "num_envs", 1)) == 1
        if should_print_step_log and self.n_calls % self._train_step_log_every == 0:
            print(
                f"[train_step] env_steps={self.num_timesteps} episode={representative_episode} "
                f"step={representative_step} stage_mix={stage_summary}\n"
                f"  rel_dist_mean={mean_distance:.4f} rel_speed_mean={mean_speed:.4f} reward_mean={reward_mean:.3f}\n"
                f"  success_total={lifetime_success_total} episode_return_mean={mean_episode_return:.3f} "
                f"collision_count_active={active_collisions}",
                flush=True,
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
                f"done_reason={summary.get('done_reason', 'unknown')} "
                f"stage={summary.get('curriculum_stage', 'unknown')}\n"
                f"  total_reward={_safe_float(summary.get('episode_return')) or 0.0:.3f} "
                f"steps={_safe_int(summary.get('episode_steps')) or 0} "
                f"final_distance={_safe_float(summary.get('final_distance')) or 0.0:.4f} "
                f"min_distance={_safe_float(summary.get('min_distance')) or 0.0:.4f} "
                f"final_speed={_safe_float(summary.get('final_speed')) or 0.0:.4f}\n"
                f"  contact_success={int(bool(summary.get('target_contact_achieved')))} "
                f"hold_success={int(bool(summary.get('hold_success')))} "
                f"target_contact_steps={_safe_int(summary.get('target_contact_steps')) or 0}\n"
                f"  collisions={_safe_int(summary.get('episode_collision_count')) or 0} "
                f"episode_successes={_safe_int(summary.get('episode_success_count')) or 0} "
                f"lifetime_successes={_safe_int(summary.get('lifetime_success_count')) or 0}",
                flush=True,
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
    
    # 这组参数控制脚本整体行为，例如训练还是测试、产物目录名字以及测试时选择哪一份模型。
    parser.add_argument("--algo", choices=["td3", "sac", "ppo"], default=train_defaults.algo, help="算法名称。")
    parser.add_argument("--test", action="store_true", help="切换到测试模式。")
    parser.add_argument("--run-name", type=str, default=train_defaults.run_name, help="实验名字，也是产物目录名字。")
    parser.add_argument("--seed", type=int, default=train_defaults.seed, help="随机种子。")
    parser.add_argument("--episodes", type=int, default=5, help="测试模式运行多少回合。")
    parser.add_argument(
        "--success-target",
        type=int,
        default=None,
        help="测试模式下累计成功多少次后提前结束。训练模式下不会使用这个参数。",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="测试模式覆盖环境回合长度。")
    parser.add_argument(
        "--model",
        "--model-mode",
        dest="model_variant",
        choices=["best", "final"],
        default="best",
        help="测试模式加载哪一份模型目录。推荐直接传 best 或 final。",
    )
    parser.add_argument("--render-mode", choices=["none", "human"], default="none", help="测试模式渲染模式。")
    parser.add_argument("--print-reward-terms", action="store_true", help="测试模式打印奖励分解。")

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
    env_group.add_argument(
        "--disable-gripper-end-effector",
        action="store_true",
        default=env_defaults.disable_gripper_end_effector,
        help="切换到不带夹爪的简化末端模型。",
    )
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


def build_env_config(args: argparse.Namespace) -> UR5ReachEnvConfig:
    # 把 CLI 参数显式映射到环境配置 dataclass。
    #
    # 这里使用显式赋值而不是 `**vars(args)`，目的是让参数来源更直观，
    # 也方便代码学习时逐项追踪“命令行参数 -> dataclass -> 环境实现”的链路。
    # 这里挑出和环境定义真正相关的参数，构造出一份干净的环境配置对象。
    return UR5ReachEnvConfig(
        frame_skip=args.frame_skip,
        episode_length=args.episode_length,
        disable_gripper_end_effector=bool(args.disable_gripper_end_effector),
        control_mode=args.control_mode,
        joint_delta_scale=args.joint_delta_scale,
        position_kp=args.position_kp,
        position_kd=args.position_kd,
        action_smoothing_alpha=args.action_smoothing_alpha,
        curriculum_fixed_episodes=args.curriculum_fixed_episodes,
        curriculum_local_random_episodes=args.curriculum_local_random_episodes,
        curriculum_local_scale=args.curriculum_local_scale,
        fixed_target_x=args.fixed_target_x,
        fixed_target_y=args.fixed_target_y,
        fixed_target_z=args.fixed_target_z,
        target_x_min=args.target_x_min,
        target_x_max=args.target_x_max,
        target_y_min=args.target_y_min,
        target_y_max=args.target_y_max,
        target_z_min=args.target_z_min,
        target_z_max=args.target_z_max,
    )


def build_eval_env_config(env_config: UR5ReachEnvConfig) -> UR5ReachEnvConfig:
    # 构造评估环境配置。
    #
    # 评估阶段关闭课程学习，让测试始终在最终完整任务分布上进行。
    # 通过 dataclass `replace(...)` 复制一份配置，再只覆盖课程学习相关字段。
    return replace(env_config, curriculum_fixed_episodes=0, curriculum_local_random_episodes=0)


def build_train_config(args: argparse.Namespace) -> RLTrainConfig:
    # 把 CLI 参数映射到训练配置 dataclass。
    # 这里挑出和训练流程相关的参数，和环境参数明确分开。
    return RLTrainConfig(
        algo=args.algo,
        run_name=args.run_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        action_noise_sigma=args.action_noise_sigma,
        ppo_n_steps=args.ppo_n_steps,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_clip_range=args.ppo_clip_range,
        render_training=bool(args.render_training),
        render_every=args.render_every,
        spectator_render=bool(args.spectator_render),
        spectator_render_every=args.spectator_render_every,
        spectator_deterministic=bool(args.spectator_deterministic),
    )


def make_env_factory(env_config: UR5ReachEnvConfig, render_mode: str | None = None):
    # 构造环境工厂函数。
    #
    # SB3 的向量化环境接口要求传入一个可调用对象，而不是环境实例本身，
    # 所以这里返回一个闭包供 `make_vec_env` 延迟创建环境。
    def _factory():
        # 真正的环境实例会在这里被创建。
        return UR5ReachEnv(config=env_config, render_mode=render_mode)
    return _factory


def make_training_env(train_config: RLTrainConfig, env_config: UR5ReachEnvConfig) -> VecNormalize:
    # 创建训练环境并包一层 `VecNormalize`。
    #
    # 实现方式：
    # - 有头训练时使用 `DummyVecEnv`，避免多进程窗口渲染不稳定。
    # - 无头多并行训练时使用 `SubprocVecEnv` 提高采样吞吐。
    # - 最外层统一包 `VecNormalize`，让模型读取的是标准化后的观测。
    render_mode = "human" if train_config.render_training else None
    # 如果训练过程中要直接开窗口，就退回单进程环境；
    # 否则在多环境场景下优先用多进程向量环境。
    vec_cls = DummyVecEnv if train_config.n_envs == 1 or train_config.render_training else SubprocVecEnv
    env = make_vec_env(
        make_env_factory(env_config=env_config, render_mode=render_mode),
        n_envs=max(int(train_config.n_envs), 1),
        seed=int(train_config.seed),
        vec_env_cls=vec_cls,
    )
    # 最外层统一包一层 VecNormalize，让模型训练时总是看到处理后的观测。
    return VecNormalize(env, norm_obs=bool(train_config.normalize_observation), norm_reward=bool(train_config.normalize_reward))


def make_eval_env(train_config: RLTrainConfig, env_config: UR5ReachEnvConfig) -> VecNormalize:
    # 创建评估环境。
    #
    # 评估环境固定为单环境，并关闭奖励归一化，避免评估数值被训练期统计量扭曲。
    # 评估环境只需要单环境，目标是稳定可比，不追求采样吞吐。
    env = make_vec_env(
        make_env_factory(env_config=build_eval_env_config(env_config), render_mode=None),
        n_envs=1,
        seed=int(train_config.seed) + 10_000,
        vec_env_cls=DummyVecEnv,
    )
    # 评估时仍然沿用观测归一化，但关闭 reward 归一化和 training 标记。
    return VecNormalize(env, norm_obs=bool(train_config.normalize_observation), norm_reward=False, training=False)


def build_policy_kwargs(train_config: RLTrainConfig) -> dict:
    # 构造 SB3 策略网络结构描述。
    #
    # SB3 中：
    # - PPO 使用 `pi` / `vf`
    # - SAC / TD3 使用 `pi` / `qf`
    # 因此这里根据算法族切换键名，但隐藏层结构都来自同一份训练配置。
    if train_config.algo == "ppo":
        # PPO 的 value 网络键名是 `vf`。
        net_arch = dict(pi=list(train_config.actor_layers), vf=list(train_config.critic_layers))
    else:
        # SAC / TD3 的 critic 网络键名是 `qf`。
        net_arch = dict(pi=list(train_config.actor_layers), qf=list(train_config.critic_layers))
    return {"net_arch": net_arch, "activation_fn": nn.ReLU}


def build_model(train_config: RLTrainConfig, env: VecNormalize, resolved_run_name: str):
    # 实例化选定算法。
    #
    # 这里把三类算法分开写，是为了让“每个参数最终传给了哪个库函数”一目了然。
    # 先准备网络结构描述和当前实验的输出目录。
    policy_kwargs = build_policy_kwargs(train_config)
    paths = build_run_paths(train_config.algo, resolved_run_name)

    if train_config.algo == "td3":
        # TD3 需要额外的高斯动作噪声，用于连续控制下的探索。
        action_dim = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=float(train_config.action_noise_sigma) * np.ones(action_dim, dtype=np.float32),
        )
        # 这里把 TD3 需要的超参数逐项显式传入，便于对照 CLI 和 dataclass。
        return TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=1,
            device=train_config.device,
            tensorboard_log=str(paths.tensorboard_dir),
            learning_rate=float(train_config.learning_rate),
            buffer_size=int(train_config.buffer_size),
            learning_starts=int(train_config.learning_starts),
            batch_size=int(train_config.batch_size),
            tau=float(train_config.tau),
            gamma=float(train_config.gamma),
            train_freq=int(train_config.train_freq),
            gradient_steps=int(train_config.gradient_steps),
            policy_delay=int(train_config.policy_delay),
            target_policy_noise=float(train_config.target_policy_noise),
            target_noise_clip=float(train_config.target_noise_clip),
            policy_kwargs=policy_kwargs,
            seed=int(train_config.seed),
        )

    if train_config.algo == "sac":
        # SAC 走离策略训练，但不需要像 TD3 那样手动构造动作噪声对象。
        # SAC 仍然需要 replay buffer、soft update 和批量更新等参数。
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device=train_config.device,
            tensorboard_log=str(paths.tensorboard_dir),
            learning_rate=float(train_config.learning_rate),
            buffer_size=int(train_config.buffer_size),
            learning_starts=int(train_config.learning_starts),
            batch_size=int(train_config.batch_size),
            tau=float(train_config.tau),
            gamma=float(train_config.gamma),
            train_freq=int(train_config.train_freq),
            gradient_steps=int(train_config.gradient_steps),
            policy_kwargs=policy_kwargs,
            seed=int(train_config.seed),
        )

    # PPO 是在轨策略算法，因此不使用 replay buffer、target network 等参数。
    # PPO 走在轨训练，因此主要依赖 rollout 长度、epoch 次数和 GAE 参数。
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=train_config.device,
        tensorboard_log=str(paths.tensorboard_dir),
        learning_rate=float(train_config.learning_rate),
        batch_size=int(train_config.batch_size),
        gamma=float(train_config.gamma),
        n_steps=int(train_config.ppo_n_steps),
        n_epochs=int(train_config.ppo_n_epochs),
        gae_lambda=float(train_config.ppo_gae_lambda),
        ent_coef=float(train_config.ppo_ent_coef),
        vf_coef=float(train_config.ppo_vf_coef),
        clip_range=float(train_config.ppo_clip_range),
        policy_kwargs=policy_kwargs,
        seed=int(train_config.seed),
    )


def train(train_config: RLTrainConfig, env_config: UR5ReachEnvConfig) -> None:
    # 执行完整训练流程并保存产物。
    #
    # 主要步骤：
    # 1. 生成产物目录并保存配置。
    # 2. 创建训练环境和评估环境。
    # 3. 初始化算法模型。
    # 4. 注册评估、归一化保存、中断保存和渲染回调。
    # 5. 调用 SB3 `learn()` 开始训练。
    # 第一步：构造路径并创建目录。
    resolved_run_name = _artifact_run_name(train_config.run_name, env_config)
    paths = build_run_paths(train_config.algo, resolved_run_name)
    ensure_run_directories(paths)
    # 第二步：先把这次实验的参数落盘，方便中途或事后复现。
    save_run_configuration(paths, env_config, train_config)

    # 第三步：创建训练环境、评估环境和模型。
    env = make_training_env(train_config, env_config)
    eval_env = make_eval_env(train_config, env_config)
    model = build_model(train_config, env, resolved_run_name)

    # 回调列表集中处理“评估、保存、手动中断、渲染”这几类辅助流程。
    callbacks: list[BaseCallback] = []
    stop_callback = None
    if train_config.save_best_reward_threshold is not None:
        # 如果设置了奖励阈值，最佳模型达到阈值后可以提前停训。
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=float(train_config.save_best_reward_threshold),
            verbose=1,
        )
    # `EvalCallback` 负责周期性评估，并把最佳模型保存到 `best_dir`。
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths.best_dir),
        log_path=str(paths.run_dir),
        eval_freq=max(int(train_config.eval_freq // max(int(train_config.n_envs), 1)), 1),
        n_eval_episodes=int(train_config.eval_episodes),
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )
    callbacks.append(eval_callback)
    callbacks.append(SaveVecNormalizeCallback(eval_callback, paths.best_normalize_path, verbose=1))
    callbacks.append(ManualInterruptCallback(paths.interrupted_model_path, paths.interrupted_normalize_path, verbose=1))
    detailed_log_callback = DetailedTrainLogCallback(verbose=1)
    callbacks.append(detailed_log_callback)
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

    print(
        "[train_start]\n"
        f"  algo={train_config.algo}\n"
        f"  run_dir={paths.run_dir}\n"
        f"  artifact_run_name={resolved_run_name}\n"
        f"  disable_gripper_end_effector={bool(env_config.disable_gripper_end_effector)}\n"
        f"  total_timesteps={train_config.total_timesteps}\n"
        f"  n_envs={train_config.n_envs}\n"
        f"  device={train_config.device}\n"
        f"  control_mode={env_config.control_mode}\n"
        f"  episode_length={env_config.episode_length}",
        flush=True,
    )
    if train_config.spectator_render and not train_config.render_training:
        print(
            "[spectator]\n"
            f"  enabled=True\n"
            f"  every={train_config.spectator_render_every}\n"
            f"  deterministic={train_config.spectator_deterministic}",
            flush=True,
        )
    start_time = time.time()
    # 真正开始训练。`learn(...)` 内部会循环调用环境、采样数据并更新模型。
    model.learn(total_timesteps=int(train_config.total_timesteps), callback=callbacks, log_interval=1000, progress_bar=True)

    # 训练结束后再额外保存一份最终模型和最终归一化参数。
    env.save(str(paths.final_normalize_path))
    model.save(str(paths.final_model_path))
    # 训练结束后再单独跑一轮最终评估，统一输出成功率、最小距离和最大回报。
    # 这里固定测试 `final` 产物，而不是训练中途的最佳模型，回答的是“这次实验最终交付物表现如何”。
    final_eval = evaluate_final_success_rate(
        algo=train_config.algo,
        run_name=resolved_run_name,
        env_config=env_config,
        episodes=max(int(train_config.eval_episodes), 1),
        seed=int(train_config.seed),
    )
    final_eval["training_total_successes"] = int(detailed_log_callback.max_lifetime_success_total)
    final_eval_path = paths.run_dir / "final_eval.json"
    final_eval_path.write_text(json.dumps(final_eval, indent=2, ensure_ascii=False), encoding="utf-8")
    elapsed = time.time() - start_time
    print(
        "[train_complete]\n"
        f"  elapsed_seconds={elapsed:.2f}\n"
        f"  training_total_successes={int(detailed_log_callback.max_lifetime_success_total)}\n"
        f"  final_model={paths.final_model_path}\n"
        f"  vec_normalize={paths.final_normalize_path}",
        flush=True,
    )
    print(
        "[final_eval]\n"
        f"  algo={train_config.algo}\n"
        f"  training_total_successes={int(detailed_log_callback.max_lifetime_success_total)}\n"
        f"  success_rate={final_eval['success_rate']:.2%}\n"
        f"  successes={final_eval['successes']}/{final_eval['episodes']}\n"
        f"  min_distance={final_eval['min_distance'] if final_eval['min_distance'] is not None else float('nan'):.4f}\n"
        f"  max_return={final_eval['max_return']:.3f}\n"
        f"  avg_return={final_eval['average_return']:.3f}\n"
        f"  avg_steps={final_eval['average_steps']:.1f}\n"
        f"  saved_to={final_eval_path}",
        flush=True,
    )
    env.close()
    eval_env.close()


def resolve_test_paths(algo: str, run_name: str, model_variant: str, env_config: UR5ReachEnvConfig | None = None) -> tuple[Path, Path]:
    # 解析测试时要加载的模型路径和归一化路径。
    #
    # 当前测试 CLI 不再要求用户自己写完整路径，而是只传 `best` 或 `final`。
    # 解析顺序：
    # 1. 先根据末端模型开关，把 run_name 解析成实际保存目录名。
    # 2. 先查当前作用域，再自动补查 `local` / `server` 两套新目录结构。
    # 3. 最后回退到旧目录结构兼容路径。
    resolved_run_name = _artifact_run_name(run_name, env_config or UR5ReachEnvConfig())
    paths = build_run_paths(algo, resolved_run_name)
    project_root = paths.project_root

    # 测试时优先查当前分区，但也自动补查 `local` / `server` 两套新目录结构，
    # 这样把服务器模型拉回本地后，不需要手动设置 `UR5_ARTIFACT_SCOPE` 也能直接测。
    scoped_run_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for run_dir in (
        paths.run_dir,
        project_root / "runs" / "local" / "main" / algo / resolved_run_name,
        project_root / "runs" / "server" / "main" / algo / resolved_run_name,
    ):
        if run_dir not in seen_dirs:
            scoped_run_dirs.append(run_dir)
            seen_dirs.add(run_dir)

    model_candidates: list[Path] = []
    norm_candidates: list[Path] = []
    if model_variant == "best":
        for run_dir in scoped_run_dirs:
            model_candidates.append(run_dir / "best_model" / "best_model.zip")
            norm_candidates.append(run_dir / "best_model" / "vec_normalize.pkl")
    else:
        for run_dir in scoped_run_dirs:
            model_candidates.append(run_dir / "final_model" / "final_model.zip")
            norm_candidates.append(run_dir / "final_model" / "vec_normalize.pkl")
            model_candidates.append(run_dir / "final" / "final_model.zip")
            norm_candidates.append(run_dir / "final" / "vec_normalize.pkl")

    legacy_run_dir = project_root / "runs" / algo / resolved_run_name
    legacy_best_model = legacy_run_dir / "best_model" / "best_model.zip"
    legacy_best_norm = legacy_run_dir / "best_model" / "vec_normalize.pkl"
    legacy_final_model = legacy_run_dir / "final" / "final_model.zip"
    legacy_final_norm = legacy_run_dir / "final" / "vec_normalize.pkl"
    legacy_final_model_dir_model = legacy_run_dir / "final_model" / "final_model.zip"
    legacy_final_model_dir_norm = legacy_run_dir / "final_model" / "vec_normalize.pkl"
    root_models_final = project_root / "models" / f"{algo}_ur5_final.zip"
    root_models_norm = project_root / "models" / "vec_normalize.pkl"

    if model_variant == "best":
        model_candidates.append(legacy_best_model)
        norm_candidates.append(legacy_best_norm)
    else:
        model_candidates.extend((legacy_final_model, legacy_final_model_dir_model, root_models_final))
        norm_candidates.extend((legacy_final_norm, legacy_final_model_dir_norm, root_models_norm))

    resolved_model = next((candidate for candidate in model_candidates if candidate.exists()), model_candidates[0])
    resolved_norm = next((candidate for candidate in norm_candidates if candidate.exists()), norm_candidates[0])
    return resolved_model, resolved_norm


def evaluate_final_success_rate(
    algo: str,
    run_name: str,
    env_config: UR5ReachEnvConfig,
    episodes: int,
    seed: int,
) -> dict[str, float | int | None]:
    # 训练结束后单独跑一轮确定性评估，统一输出最终成功率统计。
    #
    # 这里固定加载 `final_model/` 下的最终产物，避免把“最佳中间模型”和“最终交付模型”
    # 混在一起。输出口径和 Warp 线保持一致：最小距离、最大回报、成功次数、成功率。
    resolved_model, resolved_norm = resolve_test_paths(algo, run_name, "final", env_config)
    eval_config = build_eval_env_config(env_config)
    env = make_vec_env(
        make_env_factory(eval_config, render_mode=None),
        n_envs=1,
        seed=int(seed),
        vec_env_cls=DummyVecEnv,
    )
    if resolved_norm.exists():
        env = VecNormalize.load(str(resolved_norm), env)
        env.training = False
        env.norm_reward = False

    model = _model_class(algo).load(str(resolved_model), env=env)
    successes = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    min_distance = float("inf")

    for _ in range(max(int(episodes), 1)):
        observation = env.reset()
        done = np.array([False], dtype=bool)
        total_reward = 0.0
        step = 0
        episode_success = False
        while not bool(done[0]) and step < int(eval_config.episode_length):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            total_reward += reward_value
            step += 1
            info0 = info[0] if isinstance(info, (list, tuple)) and info else {}
            distance = float(info0.get("relative_distance", info0.get("distance", np.inf)))
            min_distance = min(min_distance, distance)
            if bool(done[0]):
                episode_success = bool(info0.get("success"))
        successes += int(episode_success)
        episode_returns.append(total_reward)
        episode_lengths.append(step)

    env.close()
    total_episodes = max(int(episodes), 1)
    return {
        "episodes": total_episodes,
        "successes": int(successes),
        "success_rate": float(successes) / float(total_episodes),
        "average_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "max_return": float(np.max(episode_returns)) if episode_returns else 0.0,
        "average_steps": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_distance": float(min_distance) if np.isfinite(min_distance) else None,
    }


def test(
    algo: str,
    run_name: str,
    env_config: UR5ReachEnvConfig,
    model_variant: str,
    episodes: int,
    seed: int,
    success_target: int | None,
    max_steps: int | None,
    render_mode: str,
    print_reward_terms: bool,
) -> None:
    # 加载训练好的主线模型并执行测试回合。
    #
    # 这里会自动处理两件事：
    # 1. 加载 VecNormalize 统计量，让测试时的观测分布和训练保持一致。
    # 2. 复用环境 `info` 中的调试信息，按需打印奖励分解。
    # 第一步：解析要加载的模型文件和归一化文件。
    resolved_model, resolved_norm = resolve_test_paths(algo, run_name, model_variant, env_config)
    if not resolved_model.exists():
        raise FileNotFoundError(f"模型不存在: {resolved_model}")

    # 第二步：构造测试环境配置。
    eval_config = build_eval_env_config(env_config)
    if max_steps is not None:
        # 如果 CLI 显式指定了最大步数，就只覆盖 episode_length，不改其他环境参数。
        eval_config = replace(eval_config, episode_length=max(int(max_steps), 1))

    # 测试环境固定为单环境，避免日志输出和渲染被多并行干扰。
    env = make_vec_env(
        make_env_factory(eval_config, render_mode=None if render_mode == "none" else render_mode),
        n_envs=1,
        seed=int(seed),
        vec_env_cls=DummyVecEnv,
    )
    if resolved_norm.exists():
        # 如果训练时用过 VecNormalize，这里必须一起加载，否则策略看到的观测分布会变。
        env = VecNormalize.load(str(resolved_norm), env)
        env.training = False
        env.norm_reward = False
        print(
            f"[test_load]\n"
            f"  disable_gripper_end_effector={bool(env_config.disable_gripper_end_effector)}\n"
            f"  vec_normalize={resolved_norm}",
            flush=True,
        )

    # 第三步：按算法类型加载模型对象。
    model = _model_class(algo).load(str(resolved_model), env=env)
    print(
        f"[test_load]\n"
        f"  disable_gripper_end_effector={bool(env_config.disable_gripper_end_effector)}\n"
        f"  model={resolved_model}",
        flush=True,
    )

    # 第四步：循环执行若干测试回合，并按需打印 reward 分解。
    total_successes = 0
    requested_episodes = max(int(episodes), 1)
    effective_success_target = None if success_target is None else max(int(success_target), 1)
    for episode_index in range(requested_episodes):
        observation = env.reset()
        done = np.array([False], dtype=bool)
        total_reward = 0.0
        step = 0
        episode_success = False
        while not bool(done[0]) and step < int(eval_config.episode_length):
            # 测试阶段固定使用确定性动作，方便比较不同实验。
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            if render_mode == "human":
                env.render()
            step += 1
            reward_value = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            total_reward += reward_value
            info0 = info[0] if isinstance(info, (list, tuple)) and info else {}
            # 推理测试固定是单环境，这里按你的要求逐步输出。
            print(
                f"[test_step] episode={episode_index + 1} step={step}\n"
                f"  reward={reward_value:.4f} "
                f"distance={float(info0.get('distance', 0.0)):.4f} "
                f"rel_speed={float(info0.get('relative_speed', info0.get('ee_speed', 0.0))):.4f} "
                f"done_reason={info0.get('done_reason', 'running')}",
                flush=True,
            )
            if print_reward_terms and isinstance(info0.get("reward_terms"), dict):
                # `reward_terms` 来自环境 `info`，可以直接帮助分析策略当前主要吃的是哪一项奖励。
                print(f"  reward_terms={info0['reward_terms']}", flush=True)
            if bool(done[0]):
                episode_success = bool(info0.get("success"))
                total_successes += int(episode_success)
        info0 = info[0] if isinstance(info, (list, tuple)) and info else {}
        print(
            f"[test_episode_end] episode={episode_index + 1}\n"
            f"  steps={step}\n"
            f"  total_reward={total_reward:.3f}\n"
            f"  distance={float(info0.get('distance', 0.0)):.4f}\n"
            f"  done_reason={info0.get('done_reason', 'unknown')}\n"
            f"  episode_success={int(episode_success)} cumulative_successes={total_successes}",
            flush=True,
        )
        if effective_success_target is not None and total_successes >= effective_success_target:
            print(
                f"[test_stop]\n"
                f"  reason=success_target_reached\n"
                f"  cumulative_successes={total_successes}\n"
                f"  success_target={effective_success_target}\n"
                f"  episodes_executed={episode_index + 1}",
                flush=True,
            )
            break
    env.close()
    if render_mode == "human":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main() -> None:
    # 主线脚本入口。
    #
    # 入口职责很简单：
    # - 先解析参数
    # - 再构造环境配置
    # - 最后根据 `--test` 在训练和测试之间分流
    # 先解析 CLI，再构造环境配置。
    parser = build_parser()
    args = parser.parse_args()
    env_config = build_env_config(args)
    if args.test:
        # 带 `--test` 时走测试分支，不进入训练流程。
        test(
            algo=args.algo,
            run_name=args.run_name,
            env_config=env_config,
            model_variant=args.model_variant,
            episodes=args.episodes,
            seed=args.seed,
            success_target=args.success_target,
            max_steps=args.max_steps,
            render_mode=args.render_mode,
            print_reward_terms=bool(args.print_reward_terms),
        )
        return
    # 默认走训练分支。
    train(build_train_config(args), env_config)


if __name__ == "__main__":
    main()
