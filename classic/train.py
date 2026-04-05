#!/usr/bin/env python3
"""MuJoCo 独立训练脚本（SAC/PPO/TD3），不依赖 ROS/Gazebo。"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from classic.env import ENV_ID, MujocoEnvConfig, register_env  # 训练、测试和 smoke 都要走同一个环境注册入口。
else:
    from .env import ENV_ID, MujocoEnvConfig, register_env  # 训练、测试和 smoke 都要走同一个环境注册入口。


@dataclass  # 这一组参数同时决定任务定义、SB3 超参数和产物保存路径。
class TrainArgs:
    """训练/测试运行参数。"""

    test: bool = False  # `True` 时运行测试流程。
    algo: str = "sac"  # 算法：`sac` / `ppo` / `td3`。
    timesteps: int = 5_000_000  # 总训练步数。
    episodes: int = 5000  # 测试回合数。
    seed: int = 42  # 随机种子。
    n_envs: int = 8  # 对 SAC/TD3 来说它直接决定采样吞吐，也会影响样本利用率。
    device: str = "cuda"  # 训练设备：`cuda` 或 `cpu`。
    render: bool = False  # 训练时是否显示窗口。
    render_mode: str = "human"  # 渲染模式：`human` 或 `rgb_array`。
    model_dir: str = "models/classic"  # 模型输出目录。
    log_dir: str = "logs/classic"  # 日志输出目录。
    run_name: str = "ur5_mujoco"  # 运行名（拼接到模型文件名）。
    eval_freq: int = 5000  # 评估太频繁会拖训练，太稀又看不出奖励什么时候拐头。
    n_eval_episodes: int = 1  # 每次评估回合数（越小越快）。
    save_best_model: bool = True  # 评估时是否额外保存 best_model 与 VecNormalize。
    render_freq: int = 1  # 渲染频率（每多少个回调 step 渲染一次）。
    log_interval: int = 1000  # SB3 日志间隔。
    batch_size: int = 512  # 单次梯度更新的小批量大小。
    buffer_size: int = 200_000  # 回放池太小会更快覆盖旧轨迹，太大则恢复训练更占内存。
    gradient_steps: int = 4  # 配合 n_envs 一起看；并行越大，同样 gradient_steps 的样本利用率越低。
    learning_starts: int = 20000  # 经验池至少积累多少步后开始学习。
    action_noise_sigma: float = 2.5  # TD3 动作噪声标准差。
    policy_delay: int = 4  # TD3 actor 延迟更新步数。
    target_policy_noise: float = 0.2  # TD3 目标策略噪声。
    target_noise_clip: float = 0.5  # TD3 目标噪声裁剪。
    max_steps: int = 3000  # 单回合最大步数。
    success_threshold: float = 0.01  # 到点成功阈值（米）。
    frame_skip: int = 1  # 每个 RL step 对应 MuJoCo 积分步数。
    physics_backend: str = "mujoco"  # 物理后端：mujoco（默认）/warp/auto（优先 warp）。
    legacy_zero_ee_velocity: bool = False  # 是否启用旧版 `cvel[:3]` 速度读取。
    robot: str = "ur5_cxy"  # 机械臂模型：`ur5_cxy` 或 `zero_robotiq`。
    lock_camera: bool = False  # 是否锁定到 XML 固定相机（默认 False，可鼠标拖动）。
    ur5_target_x_min: float = -0.95  # UR5 目标采样范围 x 最小值。
    ur5_target_x_max: float = -0.60  # UR5 目标采样范围 x 最大值。
    ur5_target_y_min: float = 0.15  # UR5 目标采样范围 y 最小值。
    ur5_target_y_max: float = 0.50  # UR5 目标采样范围 y 最大值。
    ur5_target_z_min: float = 0.12  # UR5 目标采样范围 z 最小值。
    ur5_target_z_max: float = 0.30  # UR5 目标采样范围 z 最大值。
    zero_target_x_min: float = -1.00  # ZERO 目标采样范围 x 最小值。
    zero_target_x_max: float = -0.62  # ZERO 目标采样范围 x 最大值。
    zero_target_y_min: float = 0.08  # ZERO 目标采样范围 y 最小值。
    zero_target_y_max: float = 0.48  # ZERO 目标采样范围 y 最大值。
    zero_target_z_min: float = 0.10  # ZERO 目标采样范围 z 最小值。
    zero_target_z_max: float = 0.35  # ZERO 目标采样范围 z 最大值。
    curriculum_stage1_fixed_episodes: int = 200  # 课程阶段 1 回合数（固定点）。
    curriculum_stage2_random_episodes: int = 800  # 课程阶段 2 回合数（小范围随机）。
    curriculum_stage2_range_scale: float = 0.35  # 阶段 2 随机范围缩放。
    fixed_target_x: float | None = None  # 阶段 1 固定目标 x。
    fixed_target_y: float | None = None  # 阶段 1 固定目标 y。
    fixed_target_z: float | None = None  # 阶段 1 固定目标 z。
    model_path: str = ""  # 测试时可手动指定模型路径。
    normalize_path: str = ""  # 测试时可手动指定归一化参数路径。
    resume: bool = False  # 是否从已有模型继续训练。
    resume_model_path: str = ""  # 继续训练的模型路径（为空时走默认 interrupted 路径）。
    resume_normalize_path: str = ""  # 继续训练的归一化路径（为空时走默认 interrupted 路径）。
    resume_replay_path: str = ""  # 继续训练的回放缓存路径（SAC/TD3 可用）。
    skip_replay_buffer: bool = False  # 继续训练时是否跳过旧回放缓存恢复。


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
                    env = env.env  # 一直解包到最内层 MuJoCo 环境，render 才能真正生效。
                if hasattr(env, "render"):
                    env.render()
        except Exception as e:
            if self.verbose > 0:
                print(f"渲染报错: {e}")
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """当评估最优回报更新时，同步保存 VecNormalize 参数。"""

    def __init__(self, eval_callback: EvalCallback, save_path: str, verbose: int = 0):
        super().__init__(verbose)  # 初始化父类。
        self.eval_callback = eval_callback  # EvalCallback 实例引用。
        self.save_path = save_path  # 归一化参数输出路径。
        self.best_mean_reward = -np.inf  # 记录历史最佳评估回报。

    def _on_step(self) -> bool:
        if self.eval_callback.best_mean_reward > self.best_mean_reward:  # 评估最好成绩刷新时才保存。
            self.best_mean_reward = float(self.eval_callback.best_mean_reward)  # 更新最佳值缓存。
            vec_env = self.model.get_vec_normalize_env()  # 获取当前模型绑定的 VecNormalize。
            if vec_env is not None:  # 仅在使用 VecNormalize 时保存。
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)  # 自动创建目标目录。
                vec_env.save(self.save_path)  # 写入归一化统计文件。
                if self.verbose > 0:  # 控制台可选提示。
                    print("已为最佳模型保存 VecNormalize 参数")  # 打印成功信息。
        return True  # 不中断训练。


class ManualInterruptCallback(BaseCallback):
    """支持 Ctrl+C 中断保存（模型 + 归一化参数）。"""

    def __init__(self, interrupted_model_path: str, interrupted_norm_path: str, verbose: int = 0):
        super().__init__(verbose)  # 初始化父类。
        self.interrupted = False  # 中断标志位。
        self._saved_once = False  # 防止重复保存中断模型。
        self.interrupted_model_path = interrupted_model_path  # 中断模型保存路径。
        self.interrupted_norm_path = interrupted_norm_path  # 中断归一化参数保存路径。
        signal.signal(signal.SIGINT, self.signal_handler)  # 注册 Ctrl+C 信号处理函数。

    def signal_handler(self, _sig, _frame):
        if self.interrupted:  # 第二次 Ctrl+C 直接强退。
            print("\n再次收到中断信号，立即强制退出。")  # 提示用户进入强退。
            os._exit(130)  # 不再等待任何清理逻辑。
        print("\n收到中断信号：将在当前 step 后保存并退出。")  # 第一次 Ctrl+C 走优雅停止。
        self.interrupted = True  # 标记为需要停止。

    def save_model(self):
        if self.model is not None:  # 仅在模型已初始化时保存。
            os.makedirs(os.path.dirname(self.interrupted_model_path), exist_ok=True)  # 自动创建保存目录。
            self.model.save(self.interrupted_model_path)  # 保存 SB3 模型权重。
            vec_env = self.model.get_vec_normalize_env()  # 取归一化封装。
            if vec_env is not None:  # 存在归一化封装时同步保存。
                vec_env.save(self.interrupted_norm_path)  # 保存 VecNormalize 统计。

    def _on_step(self) -> bool:
        if self.interrupted and not self._saved_once:  # 首次进入中断分支时执行保存。
            self.save_model()  # 保存中断模型与归一化统计。
            self._saved_once = True  # 标记已保存。
            print("已保存中断模型，正在退出训练...")  # 给用户反馈。
            return False  # 停止训练循环。
        return not self.interrupted  # 未中断时继续训练。


class TrainingLogCallback(BaseCallback):
    """按固定时间步输出最近训练回合的统计信息。"""

    def __init__(self, log_freq_timesteps: int, eval_callback: EvalCallback | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq_timesteps = max(int(log_freq_timesteps), 1)  # 控制台日志按环境总步数触发。
        self.eval_callback = eval_callback  # 若开启评估，则顺带打印最近一次评估回报。
        self.next_log_timestep = self.log_freq_timesteps
        self.recent_rewards: deque[float] = deque(maxlen=20)
        self.recent_lengths: deque[float] = deque(maxlen=20)
        self.recent_distances: deque[float] = deque(maxlen=20)
        self.recent_successes: deque[float] = deque(maxlen=20)
        self.recent_collisions: deque[float] = deque(maxlen=20)

    def _maybe_append_episode_stats(self) -> None:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return
        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            episode_info = info.get("episode", {})
            if "r" in episode_info:
                self.recent_rewards.append(float(episode_info["r"]))
            if "l" in episode_info:
                self.recent_lengths.append(float(episode_info["l"]))
            if "distance" in info:
                self.recent_distances.append(float(info["distance"]))
            if "success" in info:
                self.recent_successes.append(float(info["success"]))
            if "collision" in info:
                self.recent_collisions.append(float(info["collision"]))

    @staticmethod
    def _mean_or_none(values: deque[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(values))

    def _on_step(self) -> bool:
        self._maybe_append_episode_stats()
        if self.num_timesteps < self.next_log_timestep:
            return True

        parts = [f"[训练日志] timesteps={self.num_timesteps}"]
        recent_reward = self._mean_or_none(self.recent_rewards)
        recent_length = self._mean_or_none(self.recent_lengths)
        recent_distance = self._mean_or_none(self.recent_distances)
        success_rate = self._mean_or_none(self.recent_successes)
        collision_rate = self._mean_or_none(self.recent_collisions)

        if recent_reward is not None:
            parts.append(f"recent_reward={recent_reward:.3f}")
        if recent_length is not None:
            parts.append(f"recent_ep_len={recent_length:.1f}")
        if recent_distance is not None:
            parts.append(f"recent_distance={recent_distance:.4f}")
        if success_rate is not None:
            parts.append(f"success_rate={success_rate:.2%}")
        if collision_rate is not None:
            parts.append(f"collision_rate={collision_rate:.2%}")
        if self.eval_callback is not None:
            last_mean_reward = getattr(self.eval_callback, "last_mean_reward", None)
            if last_mean_reward is not None and np.isfinite(last_mean_reward):
                parts.append(f"eval_reward={float(last_mean_reward):.3f}")

        print(" ".join(parts))
        self.next_log_timestep += self.log_freq_timesteps
        return True


def _normalize_device_arg(device_raw: str) -> str:
    """规范化并校验设备参数，尽早给出可读错误信息。"""
    device = str(device_raw).strip()
    lowered = device.lower()
    if lowered in {"cpu", "cuda", "mps"}:
        return lowered
    if lowered.startswith("cuda:") and lowered.split(":", 1)[1].isdigit():
        return lowered
    if lowered.startswith("cud"):
        raise ValueError(
            f"无效的 --device 参数: {device_raw!r}。"
            "你可能想写的是 'cuda'。若你命令里还有 '--eval-freq 5000'，请确认参数之间有空格："
            "--device cuda --eval-freq 5000"
        )
    raise ValueError(
        f"无效的 --device 参数: {device_raw!r}。可用值示例: cpu / cuda / cuda:0 / mps"
    )


def _apply_zero_original_preset(args: TrainArgs) -> None:
    """在 `zero_robotiq` 机器人下补齐默认参数组。"""
    if args.robot != "zero_robotiq":
        return

    applied: list[str] = []
    if args.n_envs == 8:
        args.n_envs = 1
        applied.append("n_envs=1")
    if args.batch_size == 512:
        args.batch_size = 256
        applied.append("batch_size=256")
    if args.buffer_size == 200_000:
        args.buffer_size = 1_000_000
        applied.append("buffer_size=1000000")
    if args.gradient_steps == 4:
        args.gradient_steps = 1
        applied.append("gradient_steps=1")
    if args.learning_starts == 20000:
        args.learning_starts = 50000
        applied.append("learning_starts=50000")
    if args.max_steps == 3000:
        args.max_steps = 5000
        applied.append("max_steps=5000")
    if abs(float(args.success_threshold) - 0.01) < 1e-12:
        args.success_threshold = 0.001
        applied.append("success_threshold=0.001")
    if args.algo == "td3" and abs(float(args.action_noise_sigma) - 2.5) < 1e-12:
        args.action_noise_sigma = 0.12
        applied.append("action_noise_sigma=0.12")

    if applied:
        print(f"已应用 zero_robotiq 默认参数: {', '.join(applied)}")


def _make_env(args: TrainArgs, render_mode: str | None = None):
    """构造 `make_vec_env` 需要的 `env_kwargs`。"""
    cfg = MujocoEnvConfig(  # 把训练参数映射成环境配置对象。
        frame_skip=args.frame_skip,  # 仿真 frame-skip。
        physics_backend=args.physics_backend,  # 物理后端选择。
        legacy_zero_ee_velocity=args.legacy_zero_ee_velocity,  # 是否启用旧版末端速度读取。
        max_steps=args.max_steps,  # 单回合最大步数。
        success_threshold=args.success_threshold,  # 成功判定阈值。
        viewer_lock_camera=args.lock_camera,  # 是否锁定固定相机。
        curriculum_stage1_fixed_episodes=args.curriculum_stage1_fixed_episodes,  # 阶段 1 回合数。
        curriculum_stage2_random_episodes=args.curriculum_stage2_random_episodes,  # 阶段 2 回合数。
        curriculum_stage2_range_scale=args.curriculum_stage2_range_scale,  # 阶段 2 范围缩放。
        fixed_target_x=args.fixed_target_x,  # 固定目标 x。
        fixed_target_y=args.fixed_target_y,  # 固定目标 y。
        fixed_target_z=args.fixed_target_z,  # 固定目标 z。
    )
    # 三种算法共用同一套奖励结构，但奖励系数按算法做适配：
    # PPO 更怕学成“站着不动”，SAC 更容易乱甩，TD3 介于两者之间。
    if args.algo == "ppo":
        cfg.improvement_gain = 160.0  # PPO 仍需要较强稠密正反馈，但收小后不再把“直冲目标”奖励得过激。
        cfg.regress_gain = 80.0  # 保留明显退步惩罚，让 PPO 更快放弃擦边碰撞轨迹。
        cfg.direction_reward_gain = 4.0  # 方向奖励保留为主信号，但不再主导策略一味加速前冲。
        cfg.speed_penalty_value = 0.35  # 提高超速约束，帮助 PPO 在接近目标前学会减速。
        cfg.action_magnitude_penalty_gain = 0.003  # 保持探索空间，同时显式约束过大的扭矩输出。
        cfg.action_change_penalty_gain = 0.002  # 放松但不取消动作切换约束，避免高噪声碰撞策略。
        cfg.idle_penalty_value = 0.35  # 仍然惩罚发呆，但不再把 PPO 推成“必须大动作”的发散探索。
    elif args.algo == "sac":
        cfg.improvement_gain = 130.0  # SAC 仍以逼近为主，但收小后减少 critic 把“猛冲”视为高价值。
        cfg.regress_gain = 75.0  # 稍提高退步惩罚，帮助 SAC 更快淘汰回合末大偏移轨迹。
        cfg.direction_reward_gain = 3.0  # 降低方向奖励主导性，避免只要朝着目标就不顾接触风险。
        cfg.speed_penalty_value = 0.7  # 加强超速惩罚，抑制接近目标阶段的过冲碰撞。
        cfg.action_magnitude_penalty_gain = 0.01  # 提高动作幅值惩罚，减少大扭矩直线撞击。
        cfg.action_change_penalty_gain = 0.007  # 提高动作切换惩罚，减少临近目标时的急修正撞击。
        cfg.idle_penalty_value = 0.12  # 保留轻度静止惩罚，避免为了防撞又退回到发呆策略。
    else:
        cfg.improvement_gain = 140.0  # TD3 保持较强逼近驱动，但比 PPO 更克制。
        cfg.regress_gain = 80.0  # 退步惩罚维持中高，帮助 value 估计更快远离发散轨迹。
        cfg.direction_reward_gain = 2.8  # 收小方向奖励，减少 TD3 走成“目标方向硬冲”的局部最优。
        cfg.speed_penalty_value = 0.55  # 加强速度约束，让 TD3 更容易学会靠近时减速。
        cfg.action_magnitude_penalty_gain = 0.008  # 提高扭矩约束，降低大动作碰撞。
        cfg.action_change_penalty_gain = 0.006  # 提高动作切换惩罚，减少目标附近抖动式碰撞。
        cfg.idle_penalty_value = 0.15  # 保留轻中度静止惩罚，兼顾推进和稳定。
    if args.robot == "zero_robotiq":  # zero 机械臂 + Robotiq 夹爪配置分支。
        cfg.model_xml = "assets/zero_arm/zero_with_robotiq_reach.xml"  # 切换到 zero+夹爪模型。
        cfg.home_pose_mode = "direct6"  # zero 机械臂采用直接六关节初始角。
        cfg.home_joint1 = 0.0  # 初始肩关节角。
        cfg.home_joint2 = -0.85  # 初始大臂关节角。
        cfg.home_joint3 = 1.35  # 初始小臂关节角。
        cfg.home_joint4 = -0.5  # 初始腕关节 1 角。
        cfg.home_joint5 = 0.0  # 初始腕关节 2 角。
        cfg.home_joint6 = 0.0  # 初始腕关节 3 角。
        cfg.render_camera_name = "workbench_camera"  # zero 模型里也提供同名固定相机。
        cfg.target_x_min = float(args.zero_target_x_min)  # ZERO 目标范围 x 最小值。
        cfg.target_x_max = float(args.zero_target_x_max)  # ZERO 目标范围 x 最大值。
        cfg.target_y_min = float(args.zero_target_y_min)  # ZERO 目标范围 y 最小值。
        cfg.target_y_max = float(args.zero_target_y_max)  # ZERO 目标范围 y 最大值。
        cfg.target_z_min = float(args.zero_target_z_min)  # ZERO 目标范围 z 最小值。
        cfg.target_z_max = float(args.zero_target_z_max)  # ZERO 目标范围 z 最大值。
        # 该机器人分支使用更宽的扭矩范围。
        cfg.torque_low = -20.0
        cfg.torque_high = 20.0
        # 切换到旧版目标采样方式。
        cfg.zero_original_mode = True
        # 如果用户没有手动指定阶段 1 固定目标点，则给 zero 一个“离机身更远”的默认点，
        # 避免课程初期目标与机身重合导致学习不稳定。
        if args.fixed_target_x is None:
            cfg.fixed_target_x = float(np.clip(cfg.target_x_max - 0.06, cfg.target_x_min, cfg.target_x_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
        if args.fixed_target_y is None:
            cfg.fixed_target_y = float(np.clip(cfg.target_y_min + 0.65 * (cfg.target_y_max - cfg.target_y_min), cfg.target_y_min, cfg.target_y_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
        if args.fixed_target_z is None:
            cfg.fixed_target_z = float(np.clip(cfg.target_z_min + 0.55 * (cfg.target_z_max - cfg.target_z_min), cfg.target_z_min, cfg.target_z_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
    else:
        cfg.target_x_min = float(args.ur5_target_x_min)  # UR5 目标范围 x 最小值。
        cfg.target_x_max = float(args.ur5_target_x_max)  # UR5 目标范围 x 最大值。
        cfg.target_y_min = float(args.ur5_target_y_min)  # UR5 目标范围 y 最小值。
        cfg.target_y_max = float(args.ur5_target_y_max)  # UR5 目标范围 y 最大值。
        cfg.target_z_min = float(args.ur5_target_z_min)  # UR5 目标范围 z 最小值。
        cfg.target_z_max = float(args.ur5_target_z_max)  # UR5 目标范围 z 最大值。
    return {"render_mode": render_mode, "config": cfg}  # make_vec_env 会把这个字典传给环境构造函数。


def _build_run_paths(args: TrainArgs) -> dict[str, str]:
    """按算法/运行名构造分层保存路径。"""
    run_dir = os.path.join(args.model_dir, args.algo, args.robot, args.run_name)  # 每个算法+机械臂独立目录。
    log_run_dir = os.path.join(args.log_dir, args.algo, args.robot, args.run_name)  # 每个算法+机械臂独立日志目录。
    stem = f"{args.algo}_{args.robot}_{args.run_name}"  # 旧目录命名使用的前缀。
    return {
        "run_dir": run_dir,
        "log_run_dir": log_run_dir,
        "final_model": os.path.join(run_dir, "final", "model"),
        "final_norm": os.path.join(run_dir, "final", "vec_normalize.pkl"),
        "final_replay": os.path.join(run_dir, "final", "replay_buffer.pkl"),
        "interrupted_model": os.path.join(run_dir, "interrupted", "model"),
        "interrupted_norm": os.path.join(run_dir, "interrupted", "vec_normalize.pkl"),
        "interrupted_replay": os.path.join(run_dir, "interrupted", "replay_buffer.pkl"),
        "best_model_dir": os.path.join(log_run_dir, "best_model"),
        "best_norm": os.path.join(log_run_dir, "best_model", "vec_normalize.pkl"),
        "legacy_final_model": os.path.join(args.model_dir, f"{stem}_final"),
        "legacy_final_norm": os.path.join(args.model_dir, f"{stem}_vec_normalize.pkl"),
    }


def _path_exists_with_optional_zip(path: str) -> bool:
    """同时接受 `model` 和 `model.zip` 两种路径写法。"""
    return os.path.exists(path) or os.path.exists(path + ".zip")


def _copy_tree_missing(src_dir: str, dst_dir: str) -> int:
    """递归复制目录中的缺失文件，不覆盖当前分层目录里已有文件。"""
    if not os.path.isdir(src_dir):
        return 0
    copied = 0
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        current_dst_dir = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(current_dst_dir, exist_ok=True)
        for name in files:
            src_file = os.path.join(root, name)
            dst_file = os.path.join(current_dst_dir, name)
            if os.path.exists(dst_file):
                continue
            shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


def _sync_legacy_run_artifacts(args: TrainArgs, paths: dict[str, str]) -> None:
    """把旧输出目录里的产物同步到当前分层目录。"""
    model_dir_parent = os.path.dirname(os.path.normpath(args.model_dir))
    log_dir_parent = os.path.dirname(os.path.normpath(args.log_dir))
    legacy_model_run_dir = os.path.join(model_dir_parent, args.algo, args.robot, args.run_name)
    legacy_log_run_dir = os.path.join(log_dir_parent, args.algo, args.robot, args.run_name)

    copied = 0
    if os.path.normpath(legacy_model_run_dir) != os.path.normpath(paths["run_dir"]):
        copied += _copy_tree_missing(os.path.join(legacy_model_run_dir, "final"), os.path.join(paths["run_dir"], "final"))
        copied += _copy_tree_missing(
            os.path.join(legacy_model_run_dir, "interrupted"),
            os.path.join(paths["run_dir"], "interrupted"),
        )
    if os.path.normpath(legacy_log_run_dir) != os.path.normpath(paths["log_run_dir"]):
        copied += _copy_tree_missing(legacy_log_run_dir, paths["log_run_dir"])

    if not _path_exists_with_optional_zip(paths["final_model"]) and _path_exists_with_optional_zip(paths["legacy_final_model"]):
        os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)
        legacy_model_file = paths["legacy_final_model"] if os.path.exists(paths["legacy_final_model"]) else paths["legacy_final_model"] + ".zip"
        final_model_file = paths["final_model"] if not legacy_model_file.endswith(".zip") else paths["final_model"] + ".zip"
        shutil.copy2(legacy_model_file, final_model_file)
        copied += 1
    if not os.path.exists(paths["final_norm"]) and os.path.exists(paths["legacy_final_norm"]):
        os.makedirs(os.path.dirname(paths["final_norm"]), exist_ok=True)
        shutil.copy2(paths["legacy_final_norm"], paths["final_norm"])
        copied += 1

    if copied > 0:
        print(f"已将 {copied} 个旧版产物同步到当前分层目录: {paths['run_dir']}")


def _resolve_test_artifact_paths(args: TrainArgs, paths: dict[str, str]) -> tuple[str, str]:
    """为测试模式解析模型与归一化路径，优先使用规范分层目录。"""
    model_candidates = [paths["final_model"], paths["interrupted_model"], paths["legacy_final_model"]]
    norm_candidates = [paths["final_norm"], paths["interrupted_norm"], paths["legacy_final_norm"]]

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = next((path for path in model_candidates if _path_exists_with_optional_zip(path)), model_candidates[0])
    if args.normalize_path:
        norm_path = args.normalize_path
    else:
        norm_path = next((path for path in norm_candidates if os.path.exists(path)), norm_candidates[0])
    return model_path, norm_path


def _make_train_vec_env(args: TrainArgs, render_mode: str | None):
    """按并行数自动选择训练用 VecEnv 实现。"""
    vec_env_cls = DummyVecEnv
    vec_env_kwargs: dict[str, object] = {}
    selected_start_method: str | None = None
    if int(args.n_envs) > 1 and not args.render:
        start_methods = ("fork", "forkserver", "spawn")
        last_error: Exception | None = None
        for start_method in start_methods:
            try:
                env = make_vec_env(
                    ENV_ID,
                    n_envs=args.n_envs,
                    seed=args.seed,
                    env_kwargs=_make_env(args, render_mode),
                    vec_env_cls=SubprocVecEnv,
                    vec_env_kwargs={"start_method": start_method},
                )
                vec_env_cls = SubprocVecEnv
                vec_env_kwargs = {"start_method": start_method}
                selected_start_method = start_method
                print(f"训练环境包装器: SubprocVecEnv (n_envs={args.n_envs}, start_method={start_method})")
                return env
            except Exception as e:
                last_error = e
                print(f"SubprocVecEnv 启动失败（start_method={start_method}）: {e}")
        print(f"SubprocVecEnv 初始化失败，自动回退到 DummyVecEnv: {last_error}")
    elif int(args.n_envs) > 1 and args.render:
        print("render 模式下为避免多进程窗口问题，训练环境保持 DummyVecEnv。")

    env = make_vec_env(
        ENV_ID,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=_make_env(args, render_mode),
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    if selected_start_method is None:
        print(f"训练环境包装器: {vec_env_cls.__name__} (n_envs={args.n_envs})")
    return env


def _load_model(args: TrainArgs, model_path: str, env, device: str):
    """按算法类型加载模型（训练恢复/测试共用）。"""
    if args.algo == "td3":  # TD3 模型加载分支。
        return TD3.load(
            model_path,
            env=env,
            device=device,
            custom_objects={
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "gradient_steps": args.gradient_steps,
                "learning_starts": args.learning_starts,
                "policy_delay": args.policy_delay,
                "target_policy_noise": args.target_policy_noise,
                "target_noise_clip": args.target_noise_clip,
            },
        )
    if args.algo == "sac":  # SAC 模型加载分支。
        return SAC.load(
            model_path,
            env=env,
            device=device,
            custom_objects={
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "gradient_steps": args.gradient_steps,
                "learning_starts": args.learning_starts,
            },
        )
    return PPO.load(  # PPO 模型加载分支。
        model_path,
        env=env,
        device=device,
        custom_objects={"batch_size": args.batch_size},
    )


def _rebuild_replay_buffer_if_needed(model, env) -> None:
    """当回放缓存与当前并行环境数不一致时，重建空回放缓存。"""
    if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
        return
    rb = model.replay_buffer
    rb_n_envs = getattr(rb, "n_envs", None)
    env_n_envs = getattr(env, "num_envs", None)
    if rb_n_envs is None or env_n_envs is None:
        return
    if int(rb_n_envs) == int(env_n_envs):
        return
    print(
        f"检测到回放缓存并行数不匹配（replay n_envs={rb_n_envs}, current n_envs={env_n_envs}），"
        "将重建空回放缓存以继续训练。"
    )
    rb_class = getattr(model, "replay_buffer_class", None)
    if rb_class is None:
        model.replay_buffer = None
        return
    rb_kwargs = dict(getattr(model, "replay_buffer_kwargs", {}) or {})
    rb_kwargs.pop("n_envs", None)
    model.replay_buffer = rb_class(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=int(env_n_envs),
        optimize_memory_usage=getattr(model, "optimize_memory_usage", False),
        handle_timeout_termination=getattr(model, "handle_timeout_termination", True),
        **rb_kwargs,
    )


def _build_model(args: TrainArgs, env, device: str):
    """按参数创建 TD3/SAC/PPO 模型。"""
    if args.algo == "td3":  # TD3 分支。
        n_actions = env.action_space.shape[-1]  # 动作维度（6）。
        action_noise = NormalActionNoise(  # TD3 训练时的高斯动作噪声。
            mean=np.zeros(n_actions, dtype=np.float32),  # 均值向量全 0。
            sigma=float(args.action_noise_sigma) * np.ones(n_actions, dtype=np.float32),  # 每维同样噪声标准差。
        )
        policy_kwargs = dict(  # 网络结构配置。
            net_arch=dict(  # 分开 actor 和 critic 网络结构。
                pi=[512, 512, 256],  # 策略网络层宽。
                qf=[512, 512, 256],  # Q 网络层宽。
            ),
            activation_fn=nn.ReLU,  # 激活函数。
        )
        return TD3(  # 返回 TD3 模型实例。
            "MlpPolicy",  # 多层感知机策略。
            env,  # 训练环境。
            action_noise=action_noise,  # 动作噪声。
            verbose=1,  # 打印训练日志。
            seed=args.seed,  # 随机种子。
            device=device,  # 训练设备。
            learning_rate=3e-4,  # 学习率。
            buffer_size=args.buffer_size,  # 回放池大小。
            learning_starts=args.learning_starts,  # 多少步后开始学习。
            batch_size=args.batch_size,  # batch 大小。
            tau=0.005,  # 软更新系数。
            gamma=0.99,  # 折扣因子。
            train_freq=1,  # 每步训练一次。
            gradient_steps=args.gradient_steps,  # 每次训练做几步梯度更新。
            policy_delay=args.policy_delay,  # actor 延迟更新。
            target_policy_noise=args.target_policy_noise,  # 目标动作噪声。
            target_noise_clip=args.target_noise_clip,  # 目标噪声裁剪。
            policy_kwargs=policy_kwargs,  # 网络配置。
        )

    if args.algo == "sac":  # SAC 分支。
        policy_kwargs = dict(  # 网络结构配置。
            net_arch=dict(  # SAC 中 pi/qf 分开定义。
                pi=[512, 512, 256],  # 策略网络层宽。
                qf=[512, 512, 256],  # Q 网络层宽。
            ),
            activation_fn=nn.ReLU,  # 激活函数。
        )
        return SAC(  # 返回 SAC 模型实例。
            "MlpPolicy",  # MLP 策略。
            env,  # 训练环境。
            verbose=1,  # 输出日志。
            seed=args.seed,  # 随机种子。
            device=device,  # 训练设备。
            learning_rate=3e-4,  # 学习率。
            buffer_size=args.buffer_size,  # 回放池大小。
            learning_starts=args.learning_starts,  # 开始学习步数。
            batch_size=args.batch_size,  # batch 大小。
            tau=0.005,  # 目标网络软更新。
            gamma=0.99,  # 折扣因子。
            train_freq=1,  # 训练频率。
            gradient_steps=args.gradient_steps,  # 梯度步数。
            policy_kwargs=policy_kwargs,  # 网络配置。
        )

    policy_kwargs = dict(  # PPO 分支网络配置。
        net_arch=dict(  # PPO 用 `vf` 表示 value function 分支。
            pi=[512, 512, 256],  # 策略网络层宽。
            vf=[512, 512, 256],  # 价值网络层宽。
        ),
        activation_fn=nn.ReLU,  # 激活函数。
        log_std_init=0.0,  # 把初始动作方差收回到中等水平，避免 PPO 早期直接发散成高噪声碰撞策略。
    )
    return PPO(  # 返回 PPO 模型实例。
        "MlpPolicy",  # MLP 策略。
        env,  # 训练环境。
        verbose=1,  # 输出日志。
        seed=args.seed,  # 随机种子。
        device=device,  # 训练设备。
        learning_rate=3e-4,  # 学习率回到更稳的量级，减少策略在前期因为更新过猛而发散。
        n_steps=512,  # rollout 长度；收短一点能让 PPO 更快更新，减少前期长时间小幅试探。
        batch_size=args.batch_size,  # batch 大小。
        gamma=0.99,  # 折扣因子。
        gae_lambda=0.95,  # GAE 参数。
        ent_coef=0.005,  # 保留探索，但避免熵过高把策略长期维持在高噪声撞击状态。
        policy_kwargs=policy_kwargs,  # 网络配置。
    )


def train(args: TrainArgs):  # 训练主流程：排查训练问题优先顺着这里读
    """训练主流程。"""
    _apply_zero_original_preset(args)  # `zero_robotiq` 会自动套用一组默认训练参数。
    register_env()  # 注册 Gym 环境 id。
    os.makedirs(args.model_dir, exist_ok=True)  # 创建模型根目录。
    os.makedirs(args.log_dir, exist_ok=True)  # 创建日志根目录。
    paths = _build_run_paths(args)  # 生成按算法/运行名分层的保存路径。
    os.makedirs(paths["run_dir"], exist_ok=True)  # 确保运行目录存在。
    os.makedirs(paths["log_run_dir"], exist_ok=True)  # 确保运行日志目录存在。
    _sync_legacy_run_artifacts(args, paths)  # 先把旧目录补齐到当前分层目录。

    if args.render and args.n_envs > 1:  # 有头渲染时多环境会显著拖慢且显示混乱。
        print("render 时建议 n_envs=1，否则只会显示一个环境画面。")  # 给出提示。
    train_render_mode = args.render_mode if args.render else None  # 训练时是否启用渲染模式。

    device = _normalize_device_arg(args.device)  # 先读取并规范化设备参数。
    if device.startswith("cuda") and not torch.cuda.is_available():  # CUDA 不可用时回退 CPU。
        print("请求使用 cuda，但当前不可用，自动回退到 cpu")  # 打印设备回退提示。
        device = "cpu"  # 设置为 CPU。
    print(f"当前训练设备: {device}")  # 打印最终设备。
    print(f"classic 物理后端: {args.physics_backend}")  # 打印物理后端设置。

    base_env = _make_train_vec_env(args, train_render_mode)  # 创建训练用向量化环境。

    if args.resume:  # 继续训练模式。
        resume_model_path = args.resume_model_path or paths["interrupted_model"]  # 默认从 interrupted 模型恢复。
        resume_norm_path = args.resume_normalize_path or paths["interrupted_norm"]  # 默认从 interrupted 归一化恢复。
        resume_replay_path = args.resume_replay_path or paths["interrupted_replay"]  # 默认从 interrupted 回放缓存恢复。
        # 若当前默认路径不存在，则尝试从上一级 models 根目录查找旧路径。
        if not _path_exists_with_optional_zip(resume_model_path):
            model_dir_parent = os.path.dirname(os.path.normpath(args.model_dir))
            legacy_interrupted_dir = os.path.join(model_dir_parent, args.algo, args.robot, args.run_name, "interrupted")
            legacy_model_path = os.path.join(legacy_interrupted_dir, "model")
            if not args.resume_model_path and (
                _path_exists_with_optional_zip(legacy_model_path)
            ):
                print(f"当前分层路径不存在，已回退到旧路径继续训练: {legacy_model_path}")
                resume_model_path = legacy_model_path
                if not args.resume_normalize_path:
                    resume_norm_path = os.path.join(legacy_interrupted_dir, "vec_normalize.pkl")
                if not args.resume_replay_path:
                    resume_replay_path = os.path.join(legacy_interrupted_dir, "replay_buffer.pkl")
        if not _path_exists_with_optional_zip(resume_model_path):
            raise FileNotFoundError(f"未找到继续训练模型: {resume_model_path}")
        if os.path.exists(resume_norm_path):  # 若存在归一化统计，先恢复统计。
            env = VecNormalize.load(resume_norm_path, base_env)
            env.training = True
            env.norm_reward = True
        else:
            print(f"未找到归一化文件，使用新统计开始继续训练: {resume_norm_path}")
            env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
        model = _load_model(args, resume_model_path, env, device)  # 加载模型权重并绑定当前环境。
        if args.skip_replay_buffer:
            print("已按参数跳过旧回放缓存恢复，将以空回放池继续训练。")
        elif hasattr(model, "load_replay_buffer") and os.path.exists(resume_replay_path):  # SAC/TD3 尝试恢复回放缓存。
            try:
                model.load_replay_buffer(resume_replay_path)
                print(f"已恢复回放缓存: {resume_replay_path}")
            except Exception as e:
                print(f"回放缓存恢复失败（继续训练）: {e}")
        _rebuild_replay_buffer_if_needed(model, env)  # n_envs 改变时避免回放缓存形状不匹配。
        if hasattr(model, "buffer_size"):
            print(f"当前回放池容量上限: {int(model.buffer_size)}")
        print(f"继续训练模型: {resume_model_path}")
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True)  # 新训练模式初始化归一化封装。
        model = _build_model(args, env, device)  # 创建算法模型。
        if hasattr(model, "buffer_size"):
            print(f"当前回放池容量上限: {int(model.buffer_size)}")

    try:
        probe = torch.zeros(1, device=device)  # 创建设备探针张量验证设备可写。
        print(f"设备探针张量所在设备: {probe.device}")  # 打印探针设备。
        del probe  # 释放探针变量。
    except Exception as e:  # 探针失败不阻塞训练。
        print(f"设备探针失败: {e}")  # 打印异常信息。

    interrupt_callback = ManualInterruptCallback(  # Ctrl+C 时保存中断模型。
        interrupted_model_path=paths["interrupted_model"],  # 中断模型路径（按算法独立）。
        interrupted_norm_path=paths["interrupted_norm"],  # 中断归一化路径（按算法独立）。
    )
    callbacks: list[BaseCallback] = [interrupt_callback]  # 默认先启用中断回调（保证可快停）。
    eval_env = None  # 评估关闭时保持 None，避免额外开销。
    eval_callback = None  # 日志回调会读取这里的最近评估结果。
    if args.eval_freq > 0:  # 评估频率大于 0 时才创建评估环境与回调。
        eval_env = make_vec_env(  # 创建评估环境（与训练环境分离）。
            ENV_ID,  # 环境 id。
            n_envs=1,  # 评估只需 1 个环境。
            seed=args.seed + 1,  # 用不同种子。
            env_kwargs=_make_env(args, None),  # 评估阶段不启用渲染。
        )
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # 评估只归一化观测。
        eval_env.obs_rms = env.obs_rms  # 共享训练环境的观测统计量。

        best_model_save_path = None
        if args.save_best_model:
            os.makedirs(paths["best_model_dir"], exist_ok=True)  # 创建最优模型目录。
            best_model_save_path = paths["best_model_dir"]
        eval_callback = EvalCallback(  # SB3 评估回调。
            eval_env,  # 评估环境。
            best_model_save_path=best_model_save_path,  # 可选：最优模型保存目录。
            log_path=paths["log_run_dir"],  # 评估日志目录（按算法独立）。
            eval_freq=max(args.eval_freq, 1),  # 防止评估频率为 0。
            n_eval_episodes=max(args.n_eval_episodes, 1),  # 每次评估回合数。
            deterministic=True,  # 评估用确定性策略。
            render=False,  # 评估时不渲染。
        )
        callbacks = [eval_callback, interrupt_callback]  # 启用评估与中断回调。
        if args.save_best_model:
            save_norm_callback = SaveVecNormalizeCallback(  # 最优模型刷新时同步保存 VecNormalize。
                eval_callback=eval_callback,  # 关联评估回调。
                save_path=paths["best_norm"],  # 归一化统计文件路径。
                verbose=1,  # 打印保存日志。
            )
            callbacks = [eval_callback, save_norm_callback, interrupt_callback]
        else:
            print("评估已开启，但已禁用 best_model 保存（--no-save-best-model）。")
    else:
        print("评估回调已关闭（--eval-freq 0），训练结束会更快退出。")  # 提示当前是快速退出模式。
    train_log_callback = TrainingLogCallback(
        log_freq_timesteps=max(args.log_interval, 1) * max(args.n_envs, 1),  # 向量环境每次 step 会累计 n_envs 个时间步。
        eval_callback=eval_callback,
    )
    callbacks.append(train_log_callback)
    if args.render:  # 开启训练渲染时追加渲染回调。
        callbacks.append(RenderDuringTrainingCallback(render_freq=args.render_freq))  # 按频率刷新窗口。

    print("开始训练...")  # 训练启动提示。
    start_t = time.time()  # 记录开始时间。
    model.learn(  # 进入 SB3 训练主循环。
        total_timesteps=max(args.timesteps, 1),  # 防止传入 0 步。
        callback=callbacks,  # 回调列表。
        log_interval=max(args.log_interval, 1),  # 防止日志间隔为 0。
        progress_bar=True,  # 显示进度条。
        reset_num_timesteps=not args.resume,  # 继续训练时保留时间轴。
    )
    elapsed = time.time() - start_t  # 计算训练耗时。

    final_reward_label = ""
    final_reward_value: float | None = None
    if eval_callback is not None:
        last_mean_reward = getattr(eval_callback, "last_mean_reward", None)
        if last_mean_reward is not None and np.isfinite(last_mean_reward):
            final_reward_label = "最终评估回报"
            final_reward_value = float(last_mean_reward)
    if final_reward_value is None:
        ep_info_buffer = getattr(model, "ep_info_buffer", None)
        if ep_info_buffer:
            recent_reward = ep_info_buffer[-1].get("r")
            if recent_reward is not None:
                final_reward_label = "最近训练回合回报"
                final_reward_value = float(recent_reward)

    if interrupt_callback.interrupted:  # 若是中断退出，不写 final，避免与完整训练混淆。
        if hasattr(model, "save_replay_buffer"):  # SAC/TD3 中断时额外保存回放缓存。
            try:
                os.makedirs(os.path.dirname(paths["interrupted_replay"]), exist_ok=True)
                model.save_replay_buffer(paths["interrupted_replay"])
            except Exception as e:
                print(f"中断回放缓存保存失败（可忽略）: {e}")
        env.close()
        if eval_env is not None:
            eval_env.close()
        print(f"训练已中断，总耗时 {elapsed:.2f}s")
        if final_reward_value is not None:
            print(f"{final_reward_label}: {final_reward_value:.3f}")
        print(f"中断模型路径: {paths['interrupted_model']}.zip")
        print(f"中断归一化路径: {paths['interrupted_norm']}")
        return

    os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)  # 创建 final 目录。
    env.save(paths["final_norm"])  # 保存训练后归一化统计。
    model.save(paths["final_model"])  # 保存最终模型。
    if hasattr(model, "save_replay_buffer"):  # SAC/TD3 会额外保存 replay buffer。
        try:
            model.save_replay_buffer(paths["final_replay"])
        except Exception as e:
            print(f"最终回放缓存保存失败（可忽略）: {e}")
    env.close()  # 关闭训练环境。
    if eval_env is not None:  # 仅在创建了评估环境时关闭。
        eval_env.close()  # 关闭评估环境。

    print(f"训练完成，总耗时 {elapsed:.2f}s")  # 输出总耗时。
    if final_reward_value is not None:
        print(f"{final_reward_label}: {final_reward_value:.3f}")  # 输出最后一次可用回报。
    print(f"最终模型路径: {paths['final_model']}.zip")  # 输出模型路径。
    print(f"最终归一化路径: {paths['final_norm']}")  # 输出归一化路径。
    if args.save_best_model:
        print(f"最优模型目录: {paths['best_model_dir']}")  # 输出最优模型目录。
    else:
        print("最优模型保存: 已关闭（仅保留评估日志）。")


def test(args: TrainArgs):  # 测试主流程：验收模型表现时会走这里
    """测试主流程（确定性策略推理）。"""
    _apply_zero_original_preset(args)  # zero 测试也保持与训练同一套默认语义。
    register_env()  # 注册环境。
    paths = _build_run_paths(args)  # 按算法/运行名生成标准路径。
    _sync_legacy_run_artifacts(args, paths)  # 测试前也先把旧目录补齐到当前分层目录。
    model_path, norm_path = _resolve_test_artifact_paths(args, paths)  # 自动解析 final/interrupted 路径。

    device = _normalize_device_arg(args.device)  # 读取并规范化目标设备。
    if device.startswith("cuda") and not torch.cuda.is_available():  # CUDA 不可用时回退。
        print("请求使用 cuda，但当前不可用，自动回退到 cpu")  # 提示回退。
        device = "cpu"  # 使用 CPU。
    print(f"classic 物理后端: {args.physics_backend}")  # 打印物理后端设置。

    env = make_vec_env(  # 创建测试环境。
        ENV_ID,  # 环境 id。
        n_envs=1,  # 测试阶段 1 环境即可。
        seed=args.seed + 2,  # 与训练/评估使用不同种子。
        env_kwargs=_make_env(args, args.render_mode if args.render else None),  # 测试时按 render 开关决定是否创建渲染上下文。
    )
    if os.path.exists(norm_path):  # 有归一化文件时加载。
        env = VecNormalize.load(norm_path, env)  # 载入归一化统计。
        env.training = False  # 推理时不再更新统计。
        env.norm_reward = False  # 推理时关闭奖励归一化。

    model = _load_model(args, model_path, env, device)  # 按算法类型加载模型。

    rewards = []  # 存放每回合总奖励。
    for ep in range(max(args.episodes, 1)):  # 按回合循环推理。
        obs = env.reset()  # VecEnv reset 返回 batched obs。
        done = np.array([False], dtype=bool)  # VecEnv done 也是批量布尔数组。
        total = 0.0  # 当前回合累计奖励。
        steps = 0  # 当前回合步数计数。
        for _ in range(max(args.max_steps, 1)):  # 每回合最多执行 max_steps。
            if bool(done[0]):  # 第 0 个环境 done 则结束本回合。
                break  # 跳出步循环。
            action, _states = model.predict(obs, deterministic=True)  # 用确定性策略输出动作。
            obs, reward, done, _info = env.step(action)  # VecEnv step 返回 4 元组。
            total += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)  # 同时接受标量和数组奖励。
            if args.render:
                env.render()  # 刷新可视化窗口。
            if args.render and args.render_mode == "human":  # human 模式下稍微 sleep，避免画面过快。
                time.sleep(0.01)  # 10ms 间隔。
            steps += 1  # 步数 +1。
        print(f"第 {ep + 1} 回合: 步数={steps}, 奖励={total:.3f}")  # 打印单回合结果。
        rewards.append(total)  # 保存单回合总奖励。
    env.close()  # 关闭测试环境。
    if rewards:  # 非空时计算平均奖励。
        print(f"平均奖励: {float(np.mean(rewards)):.3f}")  # 打印平均奖励。
    if args.render and args.render_mode == "human":
        # 与 `classic/test.py` 一样，human viewer 在部分 GLX/X11 环境里会在解释器收尾时抛 `GLXBadDrawable`。
        # 这里在结果已经打印完成后直接退出，避免卡在窗口销毁阶段。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def parse_args() -> TrainArgs:  # 命令行解析入口：外部调参首先会影响这里
    """解析命令行参数并组装成 TrainArgs。"""
    p = argparse.ArgumentParser(description="纯 MuJoCo UR5 训练脚本（Gym + SB3）")  # 创建参数解析器。
    p.add_argument("--test", action="store_true")  # 进入测试模式。
    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")  # 选择算法。
    p.add_argument("--timesteps", type=int, default=5_000_000)  # 训练步数。
    p.add_argument("--episodes", type=int, default=5000)  # 测试回合数。
    p.add_argument("--seed", type=int, default=42)  # 随机种子。
    p.add_argument("--n-envs", type=int, default=8)  # 并行环境数（UR5 默认优化值）。
    p.add_argument("--device", type=str, default="cuda")  # 设备。
    p.add_argument("--render", action="store_true")  # 显式打开渲染。
    p.add_argument("--no-render", action="store_false", dest="render")  # 显式关闭渲染。
    p.set_defaults(render=False)  # 默认训练不渲染。
    p.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")  # 渲染模式。
    p.add_argument("--model-dir", type=str, default="models/classic")  # 模型目录。
    p.add_argument("--log-dir", type=str, default="logs/classic")  # 日志目录。
    p.add_argument("--run-name", type=str, default="ur5_mujoco")  # 运行名称。
    p.add_argument("--eval-freq", type=int, default=5000)  # 评估频率。
    p.add_argument("--n-eval-episodes", type=int, default=1)  # 每次评估回合数。
    p.add_argument("--save-best-model", action="store_true", dest="save_best_model")  # 开启最优模型保存。
    p.add_argument("--no-save-best-model", action="store_false", dest="save_best_model")  # 关闭最优模型保存。
    p.set_defaults(save_best_model=True)
    p.add_argument("--render-freq", type=int, default=1)  # 渲染频率。
    p.add_argument("--log-interval", type=int, default=1000)  # 日志间隔。
    p.add_argument("--batch-size", type=int, default=512)  # batch 大小（UR5 默认优化值）。
    p.add_argument("--buffer-size", type=int, default=200_000)  # SAC/TD3 回放池容量（16G 内存友好）。
    p.add_argument("--gradient-steps", type=int, default=4)  # 梯度步数（UR5 默认优化值）。
    p.add_argument("--learning-starts", type=int, default=20000)  # 开始学习步数。
    p.add_argument("--action-noise-sigma", type=float, default=2.5)  # TD3 动作噪声。
    p.add_argument("--policy-delay", type=int, default=4)  # TD3 actor 延迟更新。
    p.add_argument("--target-policy-noise", type=float, default=0.2)  # TD3 目标策略噪声。
    p.add_argument("--target-noise-clip", type=float, default=0.5)  # TD3 目标噪声裁剪。
    p.add_argument("--max-steps", type=int, default=3000)  # 单回合最大步数。
    p.add_argument("--success-threshold", type=float, default=0.01)  # 成功阈值。
    p.add_argument("--frame-skip", type=int, default=1)  # frame-skip。
    p.add_argument("--physics-backend", choices=["auto", "mujoco", "warp"], default="mujoco")  # 物理后端。
    p.add_argument("--legacy-zero-ee-velocity", action="store_true")  # 启用旧版 `cvel[:3]` 速度读取。
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 机械臂模型选择。
    p.add_argument("--lock-camera", action="store_true", dest="lock_camera")  # 锁定到固定相机。
    p.add_argument("--free-camera", action="store_false", dest="lock_camera")  # 使用自由相机。
    p.set_defaults(lock_camera=False)  # 默认使用自由相机。
    p.add_argument("--ur5-target-x-min", type=float, default=-0.95)  # UR5 目标范围 x 最小值。
    p.add_argument("--ur5-target-x-max", type=float, default=-0.60)  # UR5 目标范围 x 最大值。
    p.add_argument("--ur5-target-y-min", type=float, default=0.15)  # UR5 目标范围 y 最小值。
    p.add_argument("--ur5-target-y-max", type=float, default=0.50)  # UR5 目标范围 y 最大值。
    p.add_argument("--ur5-target-z-min", type=float, default=0.12)  # UR5 目标范围 z 最小值。
    p.add_argument("--ur5-target-z-max", type=float, default=0.30)  # UR5 目标范围 z 最大值。
    p.add_argument("--zero-target-x-min", type=float, default=-1.00)  # ZERO 目标范围 x 最小值。
    p.add_argument("--zero-target-x-max", type=float, default=-0.62)  # ZERO 目标范围 x 最大值。
    p.add_argument("--zero-target-y-min", type=float, default=0.08)  # ZERO 目标范围 y 最小值。
    p.add_argument("--zero-target-y-max", type=float, default=0.48)  # ZERO 目标范围 y 最大值。
    p.add_argument("--zero-target-z-min", type=float, default=0.10)  # ZERO 目标范围 z 最小值。
    p.add_argument("--zero-target-z-max", type=float, default=0.35)  # ZERO 目标范围 z 最大值。
    p.add_argument("--curriculum-stage1-fixed-episodes", type=int, default=200)  # 阶段 1 回合数。
    p.add_argument("--curriculum-stage2-random-episodes", type=int, default=800)  # 阶段 2 回合数。
    p.add_argument("--curriculum-stage2-range-scale", type=float, default=0.35)  # 阶段 2 范围缩放。
    p.add_argument("--fixed-target-x", type=float, default=None)  # 固定目标 x。
    p.add_argument("--fixed-target-y", type=float, default=None)  # 固定目标 y。
    p.add_argument("--fixed-target-z", type=float, default=None)  # 固定目标 z。
    p.add_argument("--model-path", type=str, default="")  # 测试模型路径覆盖。
    p.add_argument("--normalize-path", type=str, default="")  # 测试归一化路径覆盖。
    p.add_argument("--resume", action="store_true")  # 从已有模型继续训练。
    p.add_argument("--resume-model-path", type=str, default="")  # 继续训练模型路径覆盖。
    p.add_argument("--resume-normalize-path", type=str, default="")  # 继续训练归一化路径覆盖。
    p.add_argument("--resume-replay-path", type=str, default="")  # 继续训练回放缓存路径覆盖。
    p.add_argument("--skip-replay-buffer", action="store_true")  # 继续训练时跳过旧回放缓存恢复。
    ns = p.parse_args()  # 解析命令行为 Namespace。
    return TrainArgs(  # 显式映射到 dataclass，避免隐藏字段错误。
        test=ns.test,  # 测试模式开关。
        algo=ns.algo,  # 算法名。
        timesteps=ns.timesteps,  # 训练步数。
        episodes=ns.episodes,  # 回合数。
        seed=ns.seed,  # 种子。
        n_envs=ns.n_envs,  # 并行环境数。
        device=ns.device,  # 设备。
        render=ns.render,  # 渲染开关。
        render_mode=ns.render_mode,  # 渲染模式。
        model_dir=ns.model_dir,  # 模型目录。
        log_dir=ns.log_dir,  # 日志目录。
        run_name=ns.run_name,  # 运行名。
        eval_freq=ns.eval_freq,  # 评估频率。
        n_eval_episodes=ns.n_eval_episodes,  # 每次评估回合数。
        save_best_model=ns.save_best_model,  # 评估时是否保存 best_model。
        render_freq=ns.render_freq,  # 渲染频率。
        log_interval=ns.log_interval,  # 日志间隔。
        batch_size=ns.batch_size,  # batch 大小。
        buffer_size=ns.buffer_size,  # 回放池容量。
        gradient_steps=ns.gradient_steps,  # 梯度步数。
        learning_starts=ns.learning_starts,  # 开始学习步数。
        action_noise_sigma=ns.action_noise_sigma,  # TD3 动作噪声。
        policy_delay=ns.policy_delay,  # TD3 actor 延迟更新。
        target_policy_noise=ns.target_policy_noise,  # TD3 目标策略噪声。
        target_noise_clip=ns.target_noise_clip,  # TD3 目标噪声裁剪。
        max_steps=ns.max_steps,  # 单回合最大步数。
        success_threshold=ns.success_threshold,  # 成功阈值。
        frame_skip=ns.frame_skip,  # frame-skip。
        physics_backend=ns.physics_backend,  # 物理后端。
        legacy_zero_ee_velocity=ns.legacy_zero_ee_velocity,  # 是否启用旧版末端速度读取。
        robot=ns.robot,  # 机械臂模型选择。
        lock_camera=ns.lock_camera,  # 是否锁定固定相机。
        ur5_target_x_min=ns.ur5_target_x_min,  # UR5 目标范围 x 最小值。
        ur5_target_x_max=ns.ur5_target_x_max,  # UR5 目标范围 x 最大值。
        ur5_target_y_min=ns.ur5_target_y_min,  # UR5 目标范围 y 最小值。
        ur5_target_y_max=ns.ur5_target_y_max,  # UR5 目标范围 y 最大值。
        ur5_target_z_min=ns.ur5_target_z_min,  # UR5 目标范围 z 最小值。
        ur5_target_z_max=ns.ur5_target_z_max,  # UR5 目标范围 z 最大值。
        zero_target_x_min=ns.zero_target_x_min,  # ZERO 目标范围 x 最小值。
        zero_target_x_max=ns.zero_target_x_max,  # ZERO 目标范围 x 最大值。
        zero_target_y_min=ns.zero_target_y_min,  # ZERO 目标范围 y 最小值。
        zero_target_y_max=ns.zero_target_y_max,  # ZERO 目标范围 y 最大值。
        zero_target_z_min=ns.zero_target_z_min,  # ZERO 目标范围 z 最小值。
        zero_target_z_max=ns.zero_target_z_max,  # ZERO 目标范围 z 最大值。
        curriculum_stage1_fixed_episodes=ns.curriculum_stage1_fixed_episodes,  # 阶段 1 回合数。
        curriculum_stage2_random_episodes=ns.curriculum_stage2_random_episodes,  # 阶段 2 回合数。
        curriculum_stage2_range_scale=ns.curriculum_stage2_range_scale,  # 阶段 2 范围缩放。
        fixed_target_x=ns.fixed_target_x,  # 固定目标 x。
        fixed_target_y=ns.fixed_target_y,  # 固定目标 y。
        fixed_target_z=ns.fixed_target_z,  # 固定目标 z。
        model_path=ns.model_path,  # 模型路径覆盖。
        normalize_path=ns.normalize_path,  # 归一化路径覆盖。
        resume=ns.resume,  # 是否继续训练。
        resume_model_path=ns.resume_model_path,  # 继续训练模型路径覆盖。
        resume_normalize_path=ns.resume_normalize_path,  # 继续训练归一化路径覆盖。
        resume_replay_path=ns.resume_replay_path,  # 继续训练回放缓存路径覆盖。
        skip_replay_buffer=ns.skip_replay_buffer,  # 是否跳过旧回放缓存恢复。
    )


def main():
    """脚本入口函数。"""
    args = parse_args()  # 解析命令行参数。
    if args.test:  # 根据模式选择测试或训练。
        test(args)  # 运行测试流程。
    else:
        train(args)  # 运行训练流程。


if __name__ == "__main__":
    main()  # 直接运行脚本时进入主函数。
