#!/usr/bin/env python3
# 主线配置模块。
#
# 本模块的职责有三类：
# 1. 定义环境参数 dataclass，集中说明任务怎么构造。
# 2. 定义训练参数 dataclass，集中说明算法怎么运行。
# 3. 定义训练产物路径和参数说明字典，方便脚本、文档和 notebook 共用。
#
# 代码学习时建议先看这里，因为：
# - 参数名字会直接映射到命令行参数。
# - 训练脚本和环境脚本都会读取这里的 dataclass。
# - notebook 里的参数表也复用了这里的说明字典。

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ENV_PARAMETER_DOCS: dict[str, str] = {
    # 这张表供 README、参数文档和 notebook 直接复用。
    # 好处是参数解释只有一份来源，后续修改不会出现文档和代码不同步。
    "model_xml": "默认 MuJoCo XML 模型路径。保持仓库内相对路径，迁移到别的机器也能直接运行。",
    "disable_gripper_end_effector": "是否切换到不带夹爪的简化末端模型。打开后会自动使用简化 XML。",
    "frame_skip": "每个 RL step 对应多少个物理步。值越大训练更快，但动作更粗。",
    "episode_length": "单回合最多决策多少次。",
    "render_camera_name": "默认渲染相机名。",
    "target_x_min": "目标采样区域 x 最小值，单位米。",
    "target_x_max": "目标采样区域 x 最大值，单位米。",
    "target_y_min": "目标采样区域 y 最小值，单位米。",
    "target_y_max": "目标采样区域 y 最大值，单位米。",
    "target_z_min": "目标采样区域 z 最小值，单位米。",
    "target_z_max": "目标采样区域 z 最大值，单位米。",
    "curriculum_fixed_episodes": "课程学习第 1 阶段回合数。设为 0 表示关闭该阶段，直接使用随机目标。",
    "curriculum_local_random_episodes": "课程学习第 2 阶段回合数。设为 0 表示关闭该阶段，直接使用完整工作空间随机目标。",
    "curriculum_local_scale": "第 2 阶段采样范围相对全空间的缩放比例。",
    "fixed_target_x": "固定目标点 x 坐标。",
    "fixed_target_y": "固定目标点 y 坐标。",
    "fixed_target_z": "固定目标点 z 坐标。",
    "control_mode": "控制模式，支持 `joint_delta` 和 `torque`。当前默认使用更接近 zero-arm 训练逻辑的 `torque`。",
    "torque_low": "力矩控制最小值。",
    "torque_high": "力矩控制最大值。",
    "joint_delta_scale": "关节增量控制时，每步最多改多少弧度。",
    "action_smoothing_alpha": "动作平滑系数。越大越依赖上一时刻控制量；设为 0 表示不做平滑。",
    "position_kp": "关节位置 PD 的比例增益。",
    "position_kd": "关节位置 PD 的阻尼增益。",
    "gravity_compensation": "目标滑块的固定补偿控制值。",
    "fixed_gripper_ctrl": "夹爪固定控制值。",
    "home_joint1": "初始姿态第 1 关节角。",
    "home_joint2": "初始姿态第 2 关节角。",
    "home_joint3": "初始姿态第 3 关节角。",
    "success_threshold_stage1": "第 1 阶段成功阈值。",
    "success_threshold_stage2": "第 2 阶段成功阈值。",
    "success_threshold_stage3": "第 3 阶段成功阈值。",
    "target_contact_reward": "机器人与目标球保持接触时，每步给的小额正奖励；设计成略大于每步时间惩罚。",
    "target_hold_duration_seconds": "机器人与目标球连续接触多少秒后，判定为保持成功。",
    "target_hold_success_bonus": "连续接触达到保持时长后发放的一次性大额成功奖励。",
    "step_penalty": "每一步固定时间惩罚。",
    "distance_weight": "基础距离惩罚权重。",
    "progress_reward_gain": "距离变近时的奖励系数。",
    "regress_penalty_gain": "距离变远时的惩罚系数。",
    "phase_thresholds": "阶段性距离奖励的阈值列表。第一次穿过某个阈值时发一次奖励。",
    "phase_rewards": "和 `phase_thresholds` 一一对应的一次性奖励值。",
    "speed_penalty_threshold": "末端速度超过该阈值时施加速度惩罚。",
    "speed_penalty_value": "末端速度过快时的固定惩罚值。",
    "direction_reward_gain": "运动方向和目标方向一致时的奖励系数。",
    "action_l2_penalty": "动作幅值惩罚系数。",
    "action_smoothness_penalty": "动作变化惩罚系数。",
    "joint_velocity_penalty": "上一时刻与当前关节速度差值的惩罚系数。",
    "wrist_rotation_penalty": "UR5 后三轴 wrist 超出舒适微调速度后的线性惩罚系数，用于抑制持续旋转。",
    "wrist_action_smoothness_penalty": "UR5 后三轴 wrist 控制量跳变惩罚系数，用于抑制手腕来回抖动。",
    "wrist_speed_penalty_threshold": "UR5 后三轴 wrist 角速度阈值，超过后触发固定速度惩罚。",
    "wrist_speed_penalty_value": "UR5 后三轴 wrist 角速度过大时的固定惩罚值。",
    "wrist_direction_flip_penalty": "UR5 后三轴 wrist 角速度方向来回反转时的惩罚系数。",
    "wrist_micro_adjustment_speed_threshold": "UR5 后三轴 wrist 被视为正常微调时允许的角速度阈值。",
    "wrist_alignment_distance_threshold": "只有当末端距离目标足够近时，才启用 wrist 精细微调奖励。",
    "wrist_alignment_ee_speed_threshold": "只有当末端线速度足够小、确实在精细对准时，才启用 wrist 微调奖励。",
    "wrist_alignment_reward_gain": "接近目标时，wrist 小幅稳定调整并继续逼近目标的奖励系数。",
    "collision_penalty": "碰撞时的一次性惩罚。",
    "success_bonus": "成功时的一次性奖励。",
    "success_remaining_step_gain": "成功后根据剩余步数追加奖励的系数。",
    "success_speed_bonus_very_slow": "成功时末端速度非常小时的额外奖励。",
    "success_speed_bonus_slow": "成功时末端速度较小时的额外奖励。",
    "success_speed_bonus_medium": "成功时末端速度中等时的额外奖励。",
    "runaway_distance_threshold": "距离过大时判定为跑飞的阈值。",
    "runaway_penalty": "跑飞时的一次性惩罚。",
}


TRAIN_PARAMETER_DOCS: dict[str, str] = {
    # 训练参数说明表和环境参数说明表作用相同，只是面向训练脚本。
    "algo": "训练或测试算法，支持 `td3`、`sac`、`ppo`。默认值改成了更接近参考训练线的 `td3`。",
    "run_name": "实验名字。所有产物都会保存在这个名字对应的目录下。",
    "seed": "随机种子。",
    "total_timesteps": "总训练步数。",
    "n_envs": "并行环境数量。这个参数会直接影响采样速度、CPU 占用和显存占用。",
    "eval_freq": "每隔多少训练步做一次评估。",
    "eval_episodes": "每次评估多少回合。",
    "save_best_reward_threshold": "达到某个评估奖励后提前停训的阈值，不需要时设为 `None`。",
    "device": "训练设备，例如 `auto`、`cpu`、`cuda`。",
    "learning_rate": "学习率。",
    "buffer_size": "经验回放池大小。PPO 不使用这个参数，但统一保留方便对照。",
    "learning_starts": "收集多少步之后开始学习。PPO 不使用这个参数。",
    "batch_size": "梯度更新 batch 大小。",
    "tau": "目标网络软更新系数。PPO 不使用这个参数。",
    "gamma": "折扣因子。",
    "train_freq": "收集多少步后开始一次更新。PPO 不使用这个参数。",
    "gradient_steps": "每次触发训练时做多少次梯度更新。PPO 不使用这个参数。",
    "policy_delay": "TD3 actor 更新延迟。",
    "target_policy_noise": "TD3 目标策略平滑噪声。",
    "target_noise_clip": "TD3 目标噪声裁剪范围。",
    "action_noise_sigma": "TD3 探索动作噪声标准差。动作空间仍是归一化区间时，这个值表示归一化尺度下的噪声。",
    "actor_layers": "策略网络隐藏层结构。",
    "critic_layers": "价值网络隐藏层结构。",
    "ppo_n_steps": "PPO rollout 长度。",
    "ppo_n_epochs": "PPO 每轮 rollout 重复优化次数。",
    "ppo_gae_lambda": "PPO 的 GAE lambda。",
    "ppo_ent_coef": "PPO 的熵正则系数。",
    "ppo_vf_coef": "PPO 的 value loss 权重。",
    "ppo_clip_range": "PPO 的裁剪范围。",
    "normalize_observation": "是否对观测做 VecNormalize。",
    "normalize_reward": "是否对奖励做 VecNormalize。",
    "render_training": "训练时是否打开窗口。",
    "render_every": "训练渲染刷新间隔。",
    "spectator_render": "是否启用旁观模式。开启后，训练环境依旧无头并行，但主进程会单独开一个窗口旁观当前策略。",
    "spectator_render_every": "旁观环境每隔多少个训练 step 更新一次。",
    "spectator_deterministic": "旁观模式是否使用确定性动作。通常建议打开，方便观察策略是否稳定。",
}


@dataclass
class UR5ReachEnvConfig:
    # 主线环境参数。
    #
    # 字段顺序按“模型与时间步 -> 目标采样 -> 控制方式 -> 初始姿态 -> 奖励参数”
    # 排列，方便把任务定义从上往下读完。

    # 模型与仿真时间步设置。`model_xml` 使用仓库相对路径，保证跨机器迁移时不用改绝对路径。
    # `frame_skip` 决定“一个 RL step 里包含多少个 MuJoCo 物理子步”。
    model_xml: str = "assets/robotiq_cxy/lab_env.xml"
    disable_gripper_end_effector: bool = False
    frame_skip: int = 1
    episode_length: int = 3000
    render_camera_name: str = "workbench_camera"

    # 目标采样空间。前三个课程阶段都会从这里定义的工作空间中取子集。
    # 这组范围不是奖励参数，而是真正决定目标球会出现在哪里。
    target_x_min: float = -0.95
    target_x_max: float = -0.60
    target_y_min: float = 0.15
    target_y_max: float = 0.50
    target_z_min: float = 0.12
    target_z_max: float = 0.30

    # 课程学习参数。先固定目标，再局部随机，最后全空间随机。
    # 这组参数会在环境里决定 `_stage_name()` 和 `_sample_target_position()` 的行为。
    curriculum_fixed_episodes: int = 0
    curriculum_local_random_episodes: int = 0
    curriculum_local_scale: float = 0.35
    fixed_target_x: float = -0.72
    fixed_target_y: float = 0.28
    fixed_target_z: float = 0.20

    # 动作与控制参数。`joint_delta` 更稳定，`torque` 更接近直接力矩控制。
    # 这些字段会直接参与 `_action_to_control()` 和 `_compute_pd_torque()`。
    control_mode: str = "torque"
    # 默认把力矩范围收紧一些，减少策略早期学到“猛抽一下再纠正”的抖动习惯。
    torque_low: float = -8.0
    torque_high: float = 8.0
    joint_delta_scale: float = 0.06
    # 主线默认打开控制量低通滤波，让策略输出更连续。
    action_smoothing_alpha: float = 0.85
    position_kp: float = 55.0
    position_kd: float = 4.0
    gravity_compensation: float = -1.0
    fixed_gripper_ctrl: float = 0.0

    # 机械臂 reset 时使用的稳定初始姿态。后 3 个关节会在环境里按几何关系自动补齐。
    # 只需要显式给前三个主关节角，后面的耦合角度由环境推出来。
    home_joint1: float = math.radians(29.7)
    home_joint2: float = math.radians(-85.0)
    home_joint3: float = math.radians(115.0)

    # 课程学习不同阶段使用不同成功阈值，阶段越后要求越严格。
    success_threshold_stage1: float = 0.010
    success_threshold_stage2: float = 0.010
    success_threshold_stage3: float = 0.010
    target_contact_reward: float = 0.25
    target_hold_duration_seconds: float = 10.0
    target_hold_success_bonus: float = 12000.0

    # 奖励项参数。环境里的 reward 会逐项读取这些系数并写入 reward_terms。
    # 这些字段最终会进入 `_compute_reward()` 里的各个 reward term。
    # 适度提高每步时间成本，鼓励策略减少无效小抖动和拖延。
    step_penalty: float = 0.20
    # 提高距离项权重，让策略更明显地偏向持续逼近目标。
    distance_weight: float = 1.20
    # 统一抬高正向奖励时，先按比例放大“越靠近越奖励”的 shaping 项。
    progress_reward_gain: float = 1.5
    regress_penalty_gain: float = 0.8
    # 在接近目标的后半段把阈值拆得更密，让策略在“快到点”和“精细逼近”时
    # 都能收到更连续的阶段反馈，而不是只跨过几个大台阶。
    phase_thresholds: tuple[float, ...] = (
        0.5,
        0.3,
        0.1,
        0.09,
        0.07,
        0.05,
        0.03,
        0.02,
        0.01,
        0.008,
        0.005,
        0.003,
        0.002,
    )
    phase_rewards: tuple[float, ...] = (
        150.0,
        300.0,
        450.0,
        550.0,
        700.0,
        900.0,
        1150.0,
        1400.0,
        1800.0,
        2200.0,
        2700.0,
        3400.0,
        4200.0,
    )
    # 末端速度约束如果太弱，策略很容易学成高频来回摆动。
    speed_penalty_threshold: float = 0.25
    speed_penalty_value: float = 2.0
    direction_reward_gain: float = 1.5
    action_l2_penalty: float = 0.0
    # 这两项直接约束动作变化和关节速度突变，是抑制抖动最直接的奖励项。
    action_smoothness_penalty: float = 2.0
    joint_velocity_penalty: float = 0.08
    # 手腕三轴更容易在接近目标时出现“高频旋转找姿态”的抖动，所以单独加一层约束。
    # 这里不再惩罚所有 wrist 旋转，而是只惩罚“超过正常微调速度”的那一部分。
    # 设计意图是：
    # - 接近目标时允许 wrist 做小幅姿态修正
    # - 但不允许为了找姿态而长期高速空转
    wrist_rotation_penalty: float = 0.12
    wrist_action_smoothness_penalty: float = 1.5
    # 参考末端速度惩罚，给 wrist 三轴再加一层“转太快就直接罚”的阈值型约束。
    wrist_speed_penalty_threshold: float = 0.6
    wrist_speed_penalty_value: float = 3.0
    # 除了持续单向高速旋转，也显式惩罚 wrist 速度方向来回翻转。
    wrist_direction_flip_penalty: float = 2.5
    # 只有在接近目标时，才把 wrist 小幅稳定调整视为“正常姿态微调”并给小额奖励。
    # 这组参数相当于给 wrist 开了一条 gated reward：
    # - 离目标远时，不鼓励过早大量调整姿态
    # - 足够近时，如果动作稳定、末端速度小、距离还在继续变好，就给小奖励
    wrist_micro_adjustment_speed_threshold: float = 0.20
    wrist_alignment_distance_threshold: float = 0.03
    wrist_alignment_ee_speed_threshold: float = 0.10
    wrist_alignment_reward_gain: float = 1200.0
    collision_penalty: float = 5000.0
    success_bonus: float = 15000.0
    success_remaining_step_gain: float = 6.0
    success_speed_bonus_very_slow: float = 3000.0
    success_speed_bonus_slow: float = 1500.0
    success_speed_bonus_medium: float = 750.0
    runaway_distance_threshold: float = 10.0
    runaway_penalty: float = 0.0


@dataclass
class RLTrainConfig:
    # 主线训练参数。
    #
    # 字段顺序按“实验标识 -> 通用训练参数 -> 离策略算法参数 -> PPO 参数 -> 归一化与渲染”
    # 排列，方便对照 `train_ur5_reach.py` 的 CLI。

    # 实验标识与全局控制参数。
    # `run_name` 不只是日志名字，也直接决定模型保存目录名。
    algo: str = "td3"
    run_name: str = "ur5_zero_aligned_main"
    seed: int = 42
    total_timesteps: int = 5_000_000
    n_envs: int = 1
    eval_freq: int = 5_000
    eval_episodes: int = 5
    save_best_reward_threshold: float | None = None
    device: str = "auto"

    # 通用优化参数。SAC / TD3 / PPO 都会读取其中的一部分。
    # 虽然不是所有算法都使用这些字段，但统一放在一个 dataclass 里方便 CLI 管理。
    learning_rate: float = 3e-4
    buffer_size: int = 3_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 4
    target_policy_noise: float = 0.20
    target_noise_clip: float = 0.50
    action_noise_sigma: float = 0.1667

    # 网络结构参数。主线统一使用 MLP，actor / critic 层数在这里集中定义。
    # 这些字段会在 `build_policy_kwargs()` 里转换成 SB3 的网络结构描述。
    actor_layers: tuple[int, int, int] = (512, 512, 256)
    critic_layers: tuple[int, int, int] = (512, 512, 256)

    # PPO 特有参数。
    ppo_n_steps: int = 2048
    ppo_n_epochs: int = 10
    ppo_gae_lambda: float = 0.95
    ppo_ent_coef: float = 0.00
    ppo_vf_coef: float = 0.50
    ppo_clip_range: float = 0.20

    # 归一化、训练时渲染和旁观模式参数。
    # 这组参数影响的不是算法更新公式，而是训练外层的环境包装和可视化行为。
    normalize_observation: bool = True
    normalize_reward: bool = True
    render_training: bool = False
    render_every: int = 10
    spectator_render: bool = False
    spectator_render_every: int = 200
    spectator_deterministic: bool = True


@dataclass
class RunPaths:
    # 一组训练产物路径。
    #
    # 这里把目录层级显式展开成字段，目的是让训练脚本、测试脚本和文档都能用同一套路径规则。
    # 这样别的文件不用再自己手写字符串拼路径，减少分叉和出错概率。

    project_root: Path
    runs_root: Path
    line_root: Path
    algo_root: Path
    run_dir: Path
    final_dir: Path
    best_dir: Path
    interrupted_dir: Path
    tensorboard_dir: Path
    config_path: Path
    final_model_path: Path
    final_normalize_path: Path
    best_model_path: Path
    best_normalize_path: Path
    interrupted_model_path: Path
    interrupted_normalize_path: Path


def project_root() -> Path:
    # 返回仓库根目录。
    #
    # 这里不依赖当前 shell 的工作目录，而是从当前文件位置反推仓库根目录。
    # 这样脚本从别的目录启动时，模型路径和产物路径依旧稳定。
    return Path(__file__).resolve().parent


def artifact_scope() -> str:
    # 返回当前训练产物应该写入的环境分区。
    #
    # 规则：
    # 1. `UR5_ARTIFACT_SCOPE` 显式指定时优先使用。
    # 2. Windows 默认视为本地开发机，写入 `local`。
    # 3. 其他系统默认视为服务器训练机，写入 `server`。
    #
    # 这样本地和服务器跑出来的模型会自动落在不同目录下；
    # 如果未来有更复杂的部署方式，也可以通过环境变量手动覆盖。
    explicit = os.environ.get("UR5_ARTIFACT_SCOPE", "").strip().lower()
    if explicit in {"local", "server"}:
        return explicit
    return "local" if platform.system() == "Windows" else "server"


def build_run_paths(algo: str, run_name: str, root: Path | None = None) -> RunPaths:
    # 根据训练线、算法名和实验名构造一整套产物路径。
    #
    # 当前主线统一写到 `runs/{local|server}/main/{algo}/{run_name}`。
    # 先确定仓库根目录，再一层层往下拼装出分环境的产物目录结构。
    root_path = root or project_root()
    runs_root = root_path / "runs" / artifact_scope()
    line_root = runs_root / "main"
    algo_root = line_root / algo
    run_dir = algo_root / run_name
    # run_dir 下面再细分成 final_model / best_model / interrupted / tensorboard 四类产物目录。
    # `final_model` 和 `best_model` 采用显式命名，方便测试命令只传 `final` / `best`
    # 就能稳定定位到对应算法与实验下的模型文件夹。
    final_dir = run_dir / "final_model"
    best_dir = run_dir / "best_model"
    interrupted_dir = run_dir / "interrupted"
    tensorboard_dir = run_dir / "tensorboard"
    return RunPaths(
        project_root=root_path,
        runs_root=runs_root,
        line_root=line_root,
        algo_root=algo_root,
        run_dir=run_dir,
        final_dir=final_dir,
        best_dir=best_dir,
        interrupted_dir=interrupted_dir,
        tensorboard_dir=tensorboard_dir,
        config_path=run_dir / "run_config.json",
        final_model_path=final_dir / "final_model.zip",
        final_normalize_path=final_dir / "vec_normalize.pkl",
        best_model_path=best_dir / "best_model.zip",
        best_normalize_path=best_dir / "vec_normalize.pkl",
        interrupted_model_path=interrupted_dir / "interrupted_model.zip",
        interrupted_normalize_path=interrupted_dir / "vec_normalize.pkl",
    )


def ensure_run_directories(paths: RunPaths) -> None:
    # 创建训练需要的目录树。
    #
    # 这里逐个 mkdir，而不是只创建 run_dir，
    # 是为了让训练过程中的不同回调可以直接往各自目录写文件。
    for folder in (
        paths.runs_root,
        paths.line_root,
        paths.algo_root,
        paths.run_dir,
        paths.final_dir,
        paths.best_dir,
        paths.interrupted_dir,
        paths.tensorboard_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def serialize_for_json(value: Any) -> Any:
    # 递归把 dataclass、Path、tuple 等对象转换成 JSON 可序列化结构。
    #
    # 保存配置时可能混有 dataclass、Path、tuple、list、dict 等多种对象，
    # 所以这里统一做一层递归清洗。
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


def save_run_configuration(paths: RunPaths, env_config: UR5ReachEnvConfig, train_config: RLTrainConfig) -> None:
    # 保存实验配置。
    #
    # 这个 JSON 的作用有两个：
    # 1. 复现实验时能直接看到当时使用的环境参数和训练参数。
    # 2. notebook 或推理脚本需要读配置时，可以直接复用这个文件。
    payload = {
        "env_config": serialize_for_json(env_config),
        "train_config": serialize_for_json(train_config),
        "paths": serialize_for_json(paths),
    }
    paths.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
