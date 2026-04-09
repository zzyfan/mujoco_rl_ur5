#!/usr/bin/env python3
# Warp 配置模块。
#
# 本模块负责：
# 1. 定义 Warp 环境参数和训练参数。
# 2. 定义 Warp 训练产物目录规则。
# 3. 提供参数说明字典，供文档和 notebook 直接引用。
#
# 主线和 Warp 线虽然共用同一套 UR5 模型文件，但训练后端不同，
# 因此需要单独维护一份更适合 JAX / Brax / Warp 的参数集合。

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WARP_ENV_PARAMETER_DOCS: dict[str, str] = {
    # 这张说明表专门给 Warp 线文档和 notebook 使用。
    "model_xml": "Warp 线使用的 MuJoCo XML。当前固定为 UR5 + Robotiq 的实验模型。",
    "sim_dt": "物理仿真步长。",
    "frame_skip": "一个策略动作对应多少个物理步。",
    "episode_length": "单回合最大决策步数。",
    "naconmax": "Warp 接触缓存大小。",
    "naccdmax": "Warp CCD 接触缓存大小。",
    "njmax": "Warp 约束缓存大小。",
    "target_sampling_mode": "目标采样模式，支持 `full_random`、`small_random`、`fixed`。默认使用更接近 zero-arm 的 `full_random`。",
    "target_range_scale": "小范围随机目标采样时的空间缩放比例。",
    "fixed_target_x": "固定目标点 x 坐标。",
    "fixed_target_y": "固定目标点 y 坐标。",
    "fixed_target_z": "固定目标点 z 坐标。",
    "success_threshold": "全随机阶段成功阈值。",
    "stage1_success_threshold": "固定目标阶段成功阈值。",
    "stage2_success_threshold": "小范围随机阶段成功阈值。",
    "controller_mode": "控制模式，支持 `torque` 和 `joint_position_delta`。当前默认使用更接近参考训练线的 `torque`。",
    "action_target_scale": "力矩模式下，策略输出映射成真实扭矩时的缩放比例。",
    "action_smoothing_alpha": "动作平滑系数。设为 0 表示不做平滑。",
    "joint_position_delta_scale": "位置增量控制时，每步允许的目标关节增量。",
    "position_control_kp": "位置控制比例增益。",
    "position_control_kd": "位置控制阻尼增益。",
    "goal_observation": "是否显式把 achieved/desired goal 拼到观测后面。",
    "reward_mode": "奖励模式，支持 `dense` 和 `sparse`。当前默认使用 zero 风格的 `dense`。",
}


WARP_TRAIN_PARAMETER_DOCS: dict[str, str] = {
    # 训练参数说明单独维护一份，避免和主线参数混在一起。
    "algo": "Warp 线支持的算法。当前只支持 `sac` 和 `ppo`。",
    "run_name": "实验名字，对应输出目录。",
    "seed": "随机种子。",
    "num_timesteps": "总训练步数。",
    "num_envs": "并行训练环境数量。Warp 线的吞吐主要就来自这里。",
    "num_eval_envs": "并行评估环境数量。",
    "num_evals": "训练过程中会评估多少次。",
    "learning_rate": "学习率。",
    "discounting": "折扣因子。",
    "reward_scaling": "Brax 训练器内部使用的奖励缩放。",
    "normalize_observations": "是否标准化观测。",
    "entropy_cost": "PPO 熵正则系数。",
    "unroll_length": "PPO rollout 片段长度。",
    "batch_size": "训练 batch 大小。",
    "num_minibatches": "PPO 每轮更新拆成多少个 mini-batch。",
    "num_updates_per_batch": "PPO 每批样本重复优化次数。",
    "sac_tau": "SAC 目标网络软更新系数。",
    "sac_min_replay_size": "SAC 开始更新前回放池最小大小。",
    "sac_max_replay_size": "SAC 回放池容量上限。",
    "sac_grad_updates_per_step": "SAC 每一步环境交互后做多少次梯度更新。",
}


@dataclass
class WarpUR5EnvConfig:
    # Warp 环境参数。
    #
    # 字段顺序按“仿真与缓存 -> 目标采样 -> 控制器 -> 观测与奖励 -> 诊断阈值”
    # 排列，方便顺着环境实现阅读。

    # MuJoCo 模型和基础仿真步长。`frame_skip` 会在环境里换算成控制周期 `ctrl_dt`。
    # Warp 线的环境步长不是靠外部训练器猜出来的，而是靠这里显式定义。
    model_xml: str = "assets/robotiq_cxy/lab_env.xml"
    sim_dt: float = 0.02
    frame_skip: int = 1
    episode_length: int = 3000

    # Warp / MJX 接触和约束缓存。环境并行数量上去以后，这些值会直接影响稳定性。
    # 如果并行规模增大但缓存太小，可能出现接触信息不完整或数值不稳定。
    naconmax: int = 128
    naccdmax: int = 128
    njmax: int = 64

    # 目标采样空间。
    target_x_min: float = -0.95
    target_x_max: float = -0.60
    target_y_min: float = 0.15
    target_y_max: float = 0.50
    target_z_min: float = 0.12
    target_z_max: float = 0.30

    # Warp 线通过 `target_sampling_mode` 控制固定目标 / 小范围随机 / 全空间随机。
    # 这组字段会在 `_sample_target()` 和 `_current_success_threshold()` 中联动使用。
    target_sampling_mode: str = "full_random"
    target_range_scale: float = 0.35
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None

    # 不同采样模式对应不同成功阈值。
    success_threshold: float = 0.010
    stage1_success_threshold: float = 0.010
    stage2_success_threshold: float = 0.010

    # 控制方式。`joint_position_delta` 会走 PD 控制器，`torque` 直接输出扭矩。
    # 这组参数最终会影响 `_scale_policy_action()` 和 `_compute_position_delta_torque()`。
    torque_low: float = -15.0
    torque_high: float = 15.0
    action_target_scale: float = 1.0
    action_smoothing_alpha: float = 0.0
    controller_mode: str = "torque"
    joint_position_delta_scale: float = 0.06
    position_control_kp: float = 55.0
    position_control_kd: float = 4.0
    goal_observation: bool = False
    reward_mode: str = "dense"

    # 与夹爪和目标滑块有关的固定控制值。
    fixed_gripper_ctrl: float = 0.0
    enable_gravity_motors: bool = True
    gravity_ctrl: float = -1.0

    # reset 初始姿态。
    home_joint1: float = 0.5183627878423158
    home_joint2: float = -1.4835298641951802
    home_joint3: float = 2.007128639793479

    # dense reward 的组成项和诊断阈值。`step()` 会逐项把这些系数拼进 reward。
    # phase_thresholds / phase_rewards 组合用于“第一次穿过某个距离阈值时发奖励”的机制。
    step_penalty: float = 0.10
    base_distance_weight: float = 0.80
    improvement_gain: float = 1.0
    regress_gain: float = 0.8
    speed_penalty_threshold: float = 0.5
    speed_penalty_value: float = 0.2
    direction_reward_gain: float = 1.0
    joint_vel_change_penalty_gain: float = 0.03
    action_magnitude_penalty_gain: float = 0.0
    action_change_penalty_gain: float = 0.0
    idle_distance_threshold: float = 0.08
    idle_speed_threshold: float = 0.015
    idle_penalty_value: float = 0.0
    phase_thresholds: tuple[float, ...] = (0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002)
    phase_rewards: tuple[float, ...] = (100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0)
    success_bonus: float = 10000.0
    success_remaining_step_gain: float = 4.0
    success_speed_bonus_very_slow: float = 2000.0
    success_speed_bonus_slow: float = 1000.0
    success_speed_bonus_medium: float = 500.0
    collision_penalty_value: float = 5000.0
    runaway_distance_threshold: float = 10.0
    runaway_ee_speed_threshold: float = 50.0
    runaway_joint_velocity_threshold: float = 100.0
    runaway_penalty_value: float = 0.0


@dataclass
class WarpTrainConfig:
    # Warp 训练参数。
    #
    # 这部分参数主要分成两类：
    # - Brax 训练器需要的超参数
    # - 训练产物与调试流程需要的运行参数

    # 实验标识和总体训练规模。
    # `num_envs` 和 `num_eval_envs` 是 Warp 线吞吐与评估成本的核心开关。
    algo: str = "sac"
    run_name: str = "ur5_warp_zero_aligned"
    seed: int = 42
    num_timesteps: int = 5_000_000
    num_envs: int = 256
    num_eval_envs: int = 128
    num_evals: int = 10
    learning_rate: float = 3e-4
    discounting: float = 0.99
    reward_scaling: float = 1.0
    normalize_observations: bool = True

    # PPO 特有参数。
    # 这些字段最终会传给 `brax.training.agents.ppo.train`。
    entropy_cost: float = 1e-4
    unroll_length: int = 10
    batch_size: int = 512
    num_minibatches: int = 8
    num_updates_per_batch: int = 4

    # SAC 特有参数。
    # 这些字段最终会传给 `brax.training.agents.sac.train`。
    sac_tau: float = 0.005
    sac_min_replay_size: int = 8192
    sac_max_replay_size: int = 3_000_000
    sac_grad_updates_per_step: int = 1

    # action_repeat 是 Brax 训练器一侧的动作重复次数，和环境 `frame_skip` 不是一回事。
    action_repeat: int = 1
    logdir: str = "runs/warp/logs"
    model_dir: str = "runs/warp/models"
    dry_run: bool = False


def project_root() -> Path:
    # 返回仓库根目录，不依赖调用时的当前工作目录。
    return Path(__file__).resolve().parent


def build_warp_run_dir(algo: str, run_name: str) -> Path:
    # 返回单个 Warp 实验的产物目录。
    #
    # Warp 训练线统一保存到 `runs/warp/{algo}/{run_name}`。
    # 路径结构和主线保持同一风格，只是训练线名从 `main` 换成 `warp`。
    return project_root() / "runs" / "warp" / algo / run_name


def save_warp_configuration(run_dir: Path, env_config: WarpUR5EnvConfig, train_config: WarpTrainConfig, runtime: str) -> None:
    # 保存 Warp 训练配置和运行时信息。
    #
    # `runtime` 会记录当前启用的 Warp/CUDA 设备，便于后续排查性能和兼容性问题。
    # 这里把环境参数、训练参数和运行时摘要打包成一个 JSON，
    # 后续调试时只看一个文件就能知道训练当时的上下文。
    payload = {
        "env_config": serialize_for_json(env_config),
        "train_config": serialize_for_json(train_config),
        "runtime": runtime,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_for_json(value: Any) -> Any:
    # 递归把 dataclass、Path、tuple 等对象转换成 JSON 可序列化结构。
    #
    # Warp 配置里有 tuple、Path 和 dataclass 混合嵌套，所以也需要这层清洗。
    if hasattr(value, "__dataclass_fields__"):
        return {key: serialize_for_json(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, list):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    return value
