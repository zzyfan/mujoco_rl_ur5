"""Gymnasium reach environment built on standard MuJoCo simulation."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
try:
    import mujoco_warp as mjwarp
    import warp as wp

    WARP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - 依赖本机运行环境
    mjwarp = None
    wp = None
    WARP_IMPORT_ERROR = exc

try:
    # `mujoco.viewer` 需要显式导入，不能假设 `mujoco` 顶层一定挂载 viewer 子模块。
    import mujoco.viewer as mj_viewer
except Exception:
    mj_viewer = None


ENV_ID = "UR5MujocoReach-v0"
_WARP_INITIALIZED = False


def _warp_available() -> bool:
    return mjwarp is not None and wp is not None


def _ensure_warp_runtime() -> None:
    global _WARP_INITIALIZED
    if not _warp_available():
        raise RuntimeError(f"请求使用 MuJoCo Warp，但当前不可用: {WARP_IMPORT_ERROR!r}")
    if _WARP_INITIALIZED:
        return
    wp.init()
    _WARP_INITIALIZED = True


def _resolve_physics_backend(requested: str) -> str:
    choice = str(requested or "auto").lower()
    if choice not in {"auto", "mujoco", "warp"}:
        raise ValueError(f"不支持的物理后端: {requested}")
    if choice == "mujoco":
        return "mujoco"
    if choice == "warp":
        if not _warp_available():
            raise RuntimeError(f"请求使用 MuJoCo Warp，但当前不可用: {WARP_IMPORT_ERROR!r}")
        return "warp"
    # auto：优先 warp，不可用则回退经典 MuJoCo
    return "warp" if _warp_available() else "mujoco"


@dataclass  # 这份配置同时定义目标采样、奖励计算、终止条件和渲染参数。
class MujocoEnvConfig:
    """到点任务参数配置。"""

    model_xml: str = "assets/robotiq_cxy/lab_env.xml"  # MuJoCo XML 模型路径。
    frame_skip: int = 1  # 每个环境 step 前向推进的物理步数。
    max_steps: int = 3000  # 单回合最大决策步数。
    success_threshold: float = 0.01  # 成功判定距离阈值。

    # 目标采样空间（按当前模型可达区设置）。
    target_x_min: float = -0.95
    target_x_max: float = -0.60
    target_y_min: float = 0.15
    target_y_max: float = 0.50
    target_z_min: float = 0.12
    target_z_max: float = 0.30

    # 课程学习（自动分阶段，不需要手动切换）：
    # 第 1 阶段：固定目标点，先学会“基本靠近”。
    curriculum_stage1_fixed_episodes: int = 200
    # 第 2 阶段：小范围随机，逐步增加泛化难度。
    curriculum_stage2_random_episodes: int = 800
    # 第 2 阶段随机范围缩放比例（相对全范围的一半宽度）。
    curriculum_stage2_range_scale: float = 0.35
    # 第 1 阶段固定目标点（None 表示自动使用采样空间中心）。
    fixed_target_x: Optional[float] = None
    fixed_target_y: Optional[float] = None
    fixed_target_z: Optional[float] = None
    zero_original_mode: bool = False  # 切换到兼容旧实验的目标采样和奖励参数。
    physics_backend: str = "mujoco"  # 物理后端：`mujoco`、`warp` 或 `auto`。
    legacy_zero_ee_velocity: bool = False  # 是否按兼容公式读取末端速度特征。

    # 六个机械臂关节的扭矩范围。
    torque_low: float = -15.0
    torque_high: float = 15.0
    # 到点任务默认保持夹爪打开。
    fixed_gripper_ctrl: float = 0.0
    enable_gravity_motors: bool = True  # 是否向重力补偿电机写入固定控制量。
    gravity_ctrl: float = -1.0  # 重力补偿电机的固定控制值。

    home_pose_mode: str = "ur5_coupled"  # 初始姿态写入方式。
    home_joint1: float = math.radians(29.7)  # 第 1 关节初始角度。
    home_joint2: float = math.radians(-85.0)  # 第 2 关节初始角度。
    home_joint3: float = math.radians(115.0)  # 第 3 关节初始角度。
    home_joint4: float = 0.0  # 第 4 关节初始角度。
    home_joint5: float = 0.0  # 第 5 关节初始角度。
    home_joint6: float = 0.0  # 第 6 关节初始角度。

    # 奖励项参数。
    step_penalty: float = 0.1
    base_distance_weight: float = 0.8
    improvement_gain: float = 1.0
    regress_gain: float = 0.8
    speed_penalty_threshold: float = 0.5
    speed_penalty_value: float = 0.2
    direction_reward_gain: float = 1.0
    joint_vel_change_penalty_gain: float = 0.03
    phase_thresholds: tuple[float, ...] = (0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002)
    phase_rewards: tuple[float, ...] = (100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0)
    success_bonus: float = 10000.0
    success_remaining_step_gain: float = 4.0
    success_speed_bonus_very_slow: float = 2000.0
    success_speed_bonus_slow: float = 1000.0
    success_speed_bonus_medium: float = 500.0
    collision_penalty_value: float = 5000.0
    # 渲染相机参数。
    render_camera_name: str = "workbench_camera"
    viewer_lock_camera: bool = False
    viewer_hide_ui: bool = False
    viewer_fallback_azimuth: float = 135.0
    viewer_fallback_elevation: float = -22.0
    viewer_fallback_distance: float = 1.8
    viewer_fallback_lookat_x: float = -0.18
    viewer_fallback_lookat_y: float = 0.25
    viewer_fallback_lookat_z: float = 0.28


class UR5MujocoEnv(gym.Env):
    """基于 CXY UR5+Robotiq 的 24 维状态到点环境。

    观测维度定义（24）：
    [0:3]   目标-末端 相对位置
    [3:9]   6 个关节角
    [9:15]  6 个关节角速度
    [15:21] 上一时刻扭矩动作
    [21:24] 末端线速度
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, config: Optional[MujocoEnvConfig] = None) -> None:
        super().__init__()
        self.config = config or MujocoEnvConfig()  # 环境行为从配置对象读取。
        self.render_mode = render_mode
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"不支持的 render_mode={self.render_mode}")
        self.viewer = None  # `human` 模式下的交互窗口。
        self.renderer = None  # `rgb_array` 模式下的离屏渲染器。
        self._target_viz_added = False  # viewer 里额外画目标球。
        self._target_geom_index: Optional[int] = None
        self._render_camera_id = -1
        self._render_error_logged = False

        xml_path = Path(__file__).resolve().parents[1] / self.config.model_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"未找到 MuJoCo 模型文件: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.physics_backend = _resolve_physics_backend(self.config.physics_backend)
        self.model_warp = None
        self.data_warp = None
        if self.physics_backend == "warp":
            _ensure_warp_runtime()
            self.model_warp = mjwarp.put_model(self.model)
            self.data_warp = mjwarp.put_data(self.model, self.data, nworld=1)  # 物理侧保持单环境实例。
        if self.config.render_camera_name:
            self._render_camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self.config.render_camera_name,
            )

        self.arm_joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
        self.arm_actuator_names = (
            "joint1_motor",
            "joint2_motor",
            "joint3_motor",
            "joint4_motor",
            "joint5_motor",
            "joint6_motor",
        )
        self.arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names],
            dtype=np.int32,
        )
        self.arm_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.arm_actuator_names],
            dtype=np.int32,
        )
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.arm_qvel_adr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        # 下面这些 adr 索引直接决定观测拼接和控制写入读的是哪几个关节。
        self.gripper_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "close_1"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "close_2"),
            ],
            dtype=np.int32,
        )
        self.gravity_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_1"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_2"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_3"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_4"),
            ],
            dtype=np.int32,
        )

        # reach 误差统一用两指中心点来算，这和 classic/MJX 现在的成功判定保持一致。
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_follower_link")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_follower_link")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body_1")
        self.target_x_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_x_1")]
        self.target_y_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_y_1")]
        self.target_z_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_z_1")]
        self.target_ball_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_ball_1")
        ]

        self.action_space = spaces.Box(
            low=np.full((6,), self.config.torque_low, dtype=np.float32),
            high=np.full((6,), self.config.torque_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)  # 24 维布局要和训练好的策略完全匹配。

        mujoco.mj_forward(self.model, self.data)
        self.home_qpos = self.data.qpos.copy()
        self.home_qvel = self.data.qvel.copy()  # reset 先回这个快照，再叠加机械臂初始姿态和目标点。

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.prev_torque = np.zeros(6, dtype=np.float32)
        self.prev_joint_vel = np.zeros(6, dtype=np.float32)
        self._warp_qpos: Optional[np.ndarray] = None
        self._warp_qvel: Optional[np.ndarray] = None
        self._warp_xpos: Optional[np.ndarray] = None
        self._warp_cvel: Optional[np.ndarray] = None
        self._warp_ncon: int = 0
        self.prev_ee_pos: Optional[np.ndarray] = None
        self.current_ee_vel = np.zeros(3, dtype=np.float32)
        self.prev_distance: Optional[float] = None
        self.min_distance: Optional[float] = None
        self.step_count = 0
        self.episode_count = 0  # 课程学习靠回合数推进，不靠成功率回推。
        self.curriculum_stage = "stage1_fixed"
        self._phase_rewards_given: set[float] = set()

    def _set_home_pose(self) -> None:
        """设置稳定的初始关节姿态。"""
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = self.home_qvel

        # 先设置前三个主关节角。
        self.data.qpos[self.arm_qpos_adr[0]] = float(self.config.home_joint1)
        self.data.qpos[self.arm_qpos_adr[1]] = float(self.config.home_joint2)
        self.data.qpos[self.arm_qpos_adr[2]] = float(self.config.home_joint3)

        if self.config.home_pose_mode == "direct6":
            self.data.qpos[self.arm_qpos_adr[3]] = float(self.config.home_joint4)
            self.data.qpos[self.arm_qpos_adr[4]] = float(self.config.home_joint5)
            self.data.qpos[self.arm_qpos_adr[5]] = float(self.config.home_joint6)
        else:
            # 按 `ur5_coupled` 模式计算后三个关节角。
            q1 = self.data.qpos[self.arm_qpos_adr[0]]
            q2 = self.data.qpos[self.arm_qpos_adr[1]]
            q3 = self.data.qpos[self.arm_qpos_adr[2]]
            self.data.qpos[self.arm_qpos_adr[3]] = 1.5 * math.pi - q2 - q3
            self.data.qpos[self.arm_qpos_adr[4]] = 1.5 * math.pi
            self.data.qpos[self.arm_qpos_adr[5]] = 1.25 * math.pi + q1

        # 到点任务中夹爪保持张开，减少抓取动力学干扰。
        self.data.qpos[6:14] = 0.0
        self.data.qvel[:] = 0.0

    def _set_target_xyz(self, x: float, y: float, z: float) -> None:
        """把 target_body_1 放到采样坐标，姿态固定为单位四元数。"""
        self.data.qpos[self.target_x_qpos_adr] = float(x)
        self.data.qpos[self.target_y_qpos_adr] = float(y)
        self.data.qpos[self.target_z_qpos_adr] = float(z)
        self.data.qpos[self.target_ball_qpos_adr:self.target_ball_qpos_adr + 4] = np.array([1.0, 0.0, 0.0, 0.0])

    def _sample_target_pos(self) -> np.ndarray:
        """在可达工作空间内均匀采样目标点。"""
        return np.array(
            [
                self.np_random.uniform(self.config.target_x_min, self.config.target_x_max),
                self.np_random.uniform(self.config.target_y_min, self.config.target_y_max),
                self.np_random.uniform(self.config.target_z_min, self.config.target_z_max),
            ],
            dtype=np.float32,
        )

    def _sample_target_pos_zero_original(self) -> np.ndarray:
        """按兼容模式采样目标点。"""
        target = self.np_random.uniform(-0.3, 0.3, size=3).astype(np.float32)
        if target[0] >= 0.0:
            target[0] = max(target[0], 0.1)
        else:
            target[0] = min(target[0], -0.1)
        if target[1] >= 0.0:
            target[1] = max(target[1], 0.1)
        else:
            target[1] = min(target[1], -0.1)
        target[2] = max(target[2], 0.05)
        return target

    def _sample_target_pos_curriculum(self) -> np.ndarray:
        """按课程学习阶段采样目标点（自动推进）。"""
        # 课程阶段按已完成回合数推进，而不是按单回合内部步数推进。
        episode_index = int(self.episode_count)

        # 第 1 阶段持续回合数，转为 int 避免外部传 float 带来比较歧义。
        stage1_episodes = max(int(self.config.curriculum_stage1_fixed_episodes), 0)
        # 第 2 阶段持续回合数，转为 int 并限制为非负。
        stage2_episodes = max(int(self.config.curriculum_stage2_random_episodes), 0)

        # 第 1 阶段：固定目标点。
        if episode_index < stage1_episodes:
            # 如果用户没显式指定固定目标，就用采样空间中心点。
            x = (
                float(self.config.fixed_target_x)
                if self.config.fixed_target_x is not None
                else 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))
            )
            # y 轴同理：优先用用户给定值，否则取采样范围中心。
            y = (
                float(self.config.fixed_target_y)
                if self.config.fixed_target_y is not None
                else 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))
            )
            # z 轴同理：优先用用户给定值，否则取采样范围中心。
            z = (
                float(self.config.fixed_target_z)
                if self.config.fixed_target_z is not None
                else 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))
            )
            # 记录阶段名字到实例属性，方便 reset 的 info 输出。
            self.curriculum_stage = "stage1_fixed"
            # 返回 float32，和 observation/action 的 dtype 保持一致。
            return np.array([x, y, z], dtype=np.float32)

        # 第 2 阶段：小范围随机（围绕中心点随机，难度比全范围低）。
        if episode_index < stage1_episodes + stage2_episodes:
            # 把缩放比例裁剪到 [1e-3, 1.0]，避免传入异常值。
            # - 太小会几乎固定点（学习会过拟合）
            # - 大于 1 会超出全范围（与设计目标不一致）
            scale = float(np.clip(self.config.curriculum_stage2_range_scale, 1e-3, 1.0))  # 限幅操作：调大调小时要关注稳定性和探索范围

            # 先计算 x 的中心与半宽，再按 scale 缩小随机范围。
            x_center = 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))
            x_half = 0.5 * (float(self.config.target_x_max) - float(self.config.target_x_min)) * scale
            x = self.np_random.uniform(x_center - x_half, x_center + x_half)

            # y 轴同样使用中心 + 缩小半宽的采样方式。
            y_center = 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))
            y_half = 0.5 * (float(self.config.target_y_max) - float(self.config.target_y_min)) * scale
            y = self.np_random.uniform(y_center - y_half, y_center + y_half)

            # z 轴也做同样处理，保证三维随机策略一致。
            z_center = 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))
            z_half = 0.5 * (float(self.config.target_z_max) - float(self.config.target_z_min)) * scale
            z = self.np_random.uniform(z_center - z_half, z_center + z_half)

            # 记录当前课程阶段名称。
            self.curriculum_stage = "stage2_small_random"
            # 返回阶段二采样结果。
            return np.array([x, y, z], dtype=np.float32)

        # 第 3 阶段：全范围随机（最终训练目标）。
        self.curriculum_stage = "stage3_full_random"
        return self._sample_target_pos()

    def _get_ee_pos(self) -> np.ndarray:
        """读取夹爪两指中心点的位置。"""
        if self.physics_backend == "warp" and self._warp_xpos is not None:
            left_pos = self._warp_xpos[self.left_finger_body_id]
            right_pos = self._warp_xpos[self.right_finger_body_id]
        else:
            left_pos = self.data.xpos[self.left_finger_body_id]
            right_pos = self.data.xpos[self.right_finger_body_id]
        center = 0.5 * (left_pos + right_pos)
        return center.copy().astype(np.float32)

    def _get_target_pos(self) -> np.ndarray:
        """读取目标体中心在世界坐标系下的位置。"""
        if self.physics_backend == "warp" and self._warp_xpos is not None:
            return self._warp_xpos[self.target_body_id].copy().astype(np.float32)
        return self.data.xpos[self.target_body_id].copy().astype(np.float32)

    def _get_legacy_ee_vel(self) -> np.ndarray:
        """按兼容公式读取末端速度。"""
        if self.physics_backend == "warp" and self._warp_cvel is not None:
            return self._warp_cvel[self.ee_body_id][:3].copy().astype(np.float32)
        return self.data.cvel[self.ee_body_id][:3].copy().astype(np.float32)

    def _get_ee_vel(self, _ee_pos: np.ndarray) -> np.ndarray:
        """读取末端速度。

        默认使用两指中心点的有限差分线速度，避免把 `cvel[:3]`
        的角速度误当作末端线速度。
        """
        if self.config.legacy_zero_ee_velocity:
            return self._get_legacy_ee_vel()
        return self.current_ee_vel.copy().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """组装 24 维观测向量。"""
        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        relative_pos = target_pos - ee_pos
        if self.physics_backend == "warp" and self._warp_qpos is not None and self._warp_qvel is not None:
            joint_pos = self._warp_qpos[self.arm_qpos_adr].copy().astype(np.float32)
            joint_vel = self._warp_qvel[self.arm_qvel_adr].copy().astype(np.float32)
        else:
            joint_pos = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float32)
            joint_vel = self.data.qvel[self.arm_qvel_adr].copy().astype(np.float32)
        ee_vel = self._get_ee_vel(ee_pos)
        # Gym/SB3 约定观测必须是 numpy 数组；
        # 统一为 float32 可以减少训练时 dtype 转换开销。
        return np.concatenate([relative_pos, joint_pos, joint_vel, self.prev_torque, ee_vel]).astype(np.float32)

    def _sync_warp_cache(self) -> None:
        """只拉取观测与奖励所需的小数组，减少 host<->device 往返。"""
        if self.physics_backend != "warp" or self.data_warp is None:
            return
        self._warp_qpos = self.data_warp.qpos.numpy()[0]
        self._warp_qvel = self.data_warp.qvel.numpy()[0]
        self._warp_xpos = self.data_warp.xpos.numpy()[0]
        self._warp_cvel = self.data_warp.cvel.numpy()[0]
        if getattr(self.data_warp, "nacon", None) is not None:
            self._warp_ncon = int(self.data_warp.nacon.numpy()[0])
        else:
            self._warp_ncon = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Gymnasium reset：重置场景并采样新目标。"""
        # Gymnasium 规范：
        # 1) 必须调用 super().reset(seed=seed)
        # 2) 返回值必须是 (obs, info)
        # super().reset 会帮你设置 self.np_random（可复现实验随机源）。
        super().reset(seed=seed)
        # 先做 MuJoCo 硬重置。
        mujoco.mj_resetData(self.model, self.data)
        # 再回到确定性的初始机械臂姿态。
        self._set_home_pose()

        # 根据开关选择目标采样方式。
        if self.config.zero_original_mode:
            self.target_pos = self._sample_target_pos_zero_original()
        else:
            self.target_pos = self._sample_target_pos_curriculum()
        self._set_target_xyz(*self.target_pos.tolist())
        mujoco.mj_forward(self.model, self.data)

        # 清空回合缓存。
        self.prev_torque[:] = 0.0
        self.prev_joint_vel[:] = 0.0
        self.prev_distance = None
        self.min_distance = None
        self.step_count = 0
        self._phase_rewards_given.clear()
        self.current_ee_vel[:] = 0.0

        # 重力补偿电机使用固定控制量。
        if self.config.enable_gravity_motors:
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)
        self.data.ctrl[self.arm_actuator_ids] = 0.0
        # 到点任务：夹爪固定为张开值。
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        self.prev_torque[:] = self.data.ctrl[self.arm_actuator_ids].astype(np.float32)

        # Gymnasium reset 返回 `(obs, info)`。
        # info 里保存距离、成功标记和课程阶段等额外状态。
        if self.physics_backend == "warp":
            # reset 后把 host 状态同步到 Warp 批量数据容器（nworld=1）。
            self.data_warp = mjwarp.put_data(self.model, self.data, nworld=1)
            self._sync_warp_cache()
        self.prev_ee_pos = self._get_ee_pos()
        obs = self._get_obs()
        # Python 字典语法：用键值对传递额外调试信息。
        info = {
            "target_pos": self.target_pos.copy(),
            "curriculum_stage": self.curriculum_stage,
            "episode_index": int(self.episode_count),
            "physics_backend": self.physics_backend,
        }
        # 课程计数器在 episode 初始化完成后 +1。
        self.episode_count += 1
        # 按 Gymnasium 规范返回 (obs, info)。
        return obs, info

    def step(self, action: np.ndarray):
        """Gymnasium step：输入 6 维关节扭矩动作。"""
        # 在写入新动作前先缓存上一时刻的控制量和关节速度。
        self.prev_torque = self.data.ctrl[self.arm_actuator_ids].copy().astype(np.float32)
        if self.physics_backend == "warp" and self._warp_qvel is not None:
            self.prev_joint_vel = self._warp_qvel[self.arm_qvel_adr].copy().astype(np.float32)
        else:
            self.prev_joint_vel = self.data.qvel[self.arm_qvel_adr].copy().astype(np.float32)

        # 1) 动作清洗与限幅。
        torque_cmd = np.asarray(action, dtype=np.float32).reshape(6)
        torque_cmd = np.clip(torque_cmd, float(self.config.torque_low), float(self.config.torque_high))  # 限幅操作：调大调小时要关注稳定性和探索范围

        # 2) 执行扭矩控制。
        # 注意：这一步是“策略输出 -> 物理引擎控制输入”的关键桥梁。
        self.data.ctrl[self.arm_actuator_ids] = torque_cmd
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        if self.config.enable_gravity_motors:
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)

        # 3) 前向仿真（按配置切后端）。
        # `frame_skip` 表示一个 RL step 对应多少个积分步。
        if self.physics_backend == "warp":
            ctrl_world = self.data.ctrl[None, :].astype(np.float32)
            wp.copy(self.data_warp.ctrl, wp.array(ctrl_world))
            for _ in range(int(self.config.frame_skip)):
                mjwarp.step(self.model_warp, self.data_warp)
            self._sync_warp_cache()
        else:
            for _ in range(int(self.config.frame_skip)):
                mujoco.mj_step(self.model, self.data)

        # 4) 读取新状态并计算奖励。
        self.step_count += 1
        ee_pos = self._get_ee_pos()
        if self.config.legacy_zero_ee_velocity:
            self.current_ee_vel = self._get_legacy_ee_vel()
        else:
            dt = max(float(self.model.opt.timestep) * max(int(self.config.frame_skip), 1), 1e-9)
            if self.prev_ee_pos is None:
                self.current_ee_vel[:] = 0.0
            else:
                self.current_ee_vel = ((ee_pos - self.prev_ee_pos) / dt).astype(np.float32)
        self.prev_ee_pos = ee_pos.copy()
        obs = self._get_obs()
        relative_pos = obs[0:3]
        joint_vel = obs[9:15]
        ee_vel = obs[21:24]

        distance = float(np.linalg.norm(relative_pos))
        ee_speed = float(np.linalg.norm(ee_vel))
        reward = 0.0

        if self.config.zero_original_mode:
            ee_pos = self._get_ee_pos().astype(np.float64)
            distance_to_target = float(np.linalg.norm(ee_pos - self.target_pos.astype(np.float64)))

            # 1) 密度奖励：靠近目标时更密集。
            density_reward = 1.2 * (1.0 - math.exp(-2.0 * (0.6 - distance_to_target)))
            if distance_to_target < 0.001:
                density_reward += (0.001 - distance_to_target) * 300.0
            elif distance_to_target < 0.005:
                density_reward += (0.005 - distance_to_target) * 100.0
            elif distance_to_target < 0.01:
                density_reward += (0.01 - distance_to_target) * 50.0
            elif distance_to_target < 0.05:
                density_reward += (0.05 - distance_to_target) * 10.0
            elif distance_to_target < 0.1:
                density_reward += (0.1 - distance_to_target) * 2.0
            elif distance_to_target < 0.3:
                density_reward += (0.3 - distance_to_target) * 0.5

            # 2) 分布奖励：速度与加速度分布。
            distribution_reward = 0.0
            if 0.01 <= ee_speed <= 0.1:
                distribution_reward += 0.5
            elif 0.005 <= ee_speed < 0.01:
                distribution_reward += 0.3
            elif 0.1 < ee_speed <= 0.2:
                distribution_reward += 0.2
            elif ee_speed < 0.005:
                distribution_reward += 0.1

            joint_velocities = joint_vel.astype(np.float64)
            joint_speeds = np.abs(joint_velocities)
            avg_joint_speed = float(np.mean(joint_speeds))
            if joint_speeds.size > 1:
                speed_std = float(np.std(joint_speeds))
                if speed_std < avg_joint_speed * 0.5:
                    distribution_reward += 0.3
                elif speed_std < avg_joint_speed:
                    distribution_reward += 0.1

            dt = max(float(self.model.opt.timestep), 1e-9)
            joint_accelerations = (joint_velocities - self.prev_joint_vel.astype(np.float64)) / dt
            joint_acc_magnitude = float(np.linalg.norm(joint_accelerations))
            if joint_acc_magnitude < 10.0:
                distribution_reward += 0.2
            elif joint_acc_magnitude < 20.0:
                distribution_reward += 0.1

            # 3) 平稳性奖励。
            smoothness_reward = 0.0
            action_change = float(np.linalg.norm(torque_cmd - self.prev_torque))
            if action_change < 2.0:
                smoothness_reward += 0.3
            elif action_change < 5.0:
                smoothness_reward += 0.1

            # 4) 惩罚项与其他项。
            joint_speed_penalty = -0.005 * float(np.sum(np.square(joint_velocities)))
            action_penalty = -0.001 * float(np.sum(np.square(torque_cmd)))

            collision_detected = False
            collision_contacts = self._warp_ncon if self.physics_backend == "warp" else int(self.data.ncon)
            collision_penalty = 0.0
            if collision_contacts > 0:
                collision_detected = True
                collision_penalty -= 100.0 * float(collision_contacts)

            height_penalty = 0.0
            if float(ee_pos[2]) < 0.05:
                height_penalty = -5.0 * (0.05 - float(ee_pos[2]))
            vertical_reward = 0.1 if 0.05 <= float(ee_pos[2]) <= 0.4 else 0.0
            step_penalty = -0.001 * float(self.step_count)

            reward = (
                density_reward
                + distribution_reward
                + smoothness_reward
                + joint_speed_penalty
                + action_penalty
                + collision_penalty
                + height_penalty
                + vertical_reward
                + step_penalty
            )

            success = distance_to_target < float(self.config.success_threshold)
            terminated = bool(success or collision_detected)
            truncated = self.step_count >= int(self.config.max_steps)
            if success:
                success_reward = 200.0
                if ee_speed < 0.01:
                    speed_reward_on_success = 100.0
                elif ee_speed < 0.05:
                    speed_reward_on_success = 50.0
                elif ee_speed < 0.1:
                    speed_reward_on_success = 20.0
                else:
                    speed_reward_on_success = 0.0
                reward += success_reward + speed_reward_on_success

            info = {
                "distance": distance_to_target,
                "success": success,
                "target_pos": self.target_pos.copy(),
                "physics_backend": self.physics_backend,
                "ee_speed": ee_speed,
                "collision": collision_detected,
                "collision_contacts": collision_contacts,
                "curriculum_stage": self.curriculum_stage,
                "episode_index": int(self.episode_count),
                "reward_info": {
                    "density_reward": density_reward,
                    "distribution_reward": distribution_reward,
                    "smoothness_reward": smoothness_reward,
                    "joint_speed_penalty": joint_speed_penalty,
                    "action_penalty": action_penalty,
                    "collision_penalty": collision_penalty,
                    "height_penalty": height_penalty,
                    "vertical_reward": vertical_reward,
                    "step_penalty": step_penalty,
                },
            }
            return obs, float(reward), terminated, truncated, info

        # ---- 以下是奖励项 ----
        # 时间惩罚：鼓励更快完成任务。
        reward -= float(self.config.step_penalty)
        # 基础距离惩罚：-0.8 * sqrt(distance)。
        reward += -float(self.config.base_distance_weight) * math.sqrt(max(distance, 1e-9))

        # 距离改进奖励 / 退步惩罚。
        if self.min_distance is None:
            self.min_distance = distance
        elif distance < self.min_distance:
            reward += float(self.config.improvement_gain) * float(self.min_distance - distance)
            self.min_distance = distance
        elif self.prev_distance is not None and distance > self.prev_distance:
            reward -= float(self.config.regress_gain) * float(distance - self.prev_distance)
        self.prev_distance = distance

        # 阶段奖励：首次进入阈值区间时一次性加分。
        for thresh, phase_reward in zip(self.config.phase_thresholds, self.config.phase_rewards):
            if distance < float(thresh) and thresh not in self._phase_rewards_given:
                reward += float(phase_reward)
                self._phase_rewards_given.add(thresh)

        # 速度惩罚：末端过快时扣分。
        if ee_speed > float(self.config.speed_penalty_threshold):
            reward -= float(self.config.speed_penalty_value)

        # 方向奖励：只奖励朝目标方向的运动（余弦 > 0）。
        if ee_speed > 1e-6 and distance > 1e-9:
            to_target = relative_pos / (distance + 1e-6)
            movement_dir = ee_vel / (ee_speed + 1e-6)
            direction_cos = float(np.dot(to_target, movement_dir))
            reward += max(0.0, direction_cos) ** 2 * float(self.config.direction_reward_gain)

        # 关节角速度变化惩罚：抑制抖动/抽动。
        joint_vel_change = np.abs(joint_vel - self.prev_joint_vel)
        reward -= float(self.config.joint_vel_change_penalty_gain) * float(np.sum(joint_vel_change))

        # 碰撞惩罚：按接触点数累计。
        collision_detected = False
        collision_contacts = self._warp_ncon if self.physics_backend == "warp" else int(self.data.ncon)
        if collision_contacts > 0:
            collision_detected = True
            reward -= float(self.config.collision_penalty_value) * float(collision_contacts)

        # 终止判定（Gymnasium 新接口）：
        # - terminated: 任务语义上的结束（成功/失败）。
        # - truncated: 时间上限等外部截断。
        success = distance <= float(self.config.success_threshold)
        terminated = bool(success or collision_detected)
        truncated = self.step_count >= int(self.config.max_steps)
        if success:
            # 成功奖励 + 剩余步数奖励。
            reward += float(self.config.success_bonus)
            reward += float(self.config.success_remaining_step_gain) * float(self.config.max_steps - self.step_count)
            # 成功时的速度奖励。
            if ee_speed < 0.01:
                reward += float(self.config.success_speed_bonus_very_slow)
            elif ee_speed < 0.05:
                reward += float(self.config.success_speed_bonus_slow)
            elif ee_speed < 0.1:
                reward += float(self.config.success_speed_bonus_medium)
        elif collision_detected:
            # 扣掉剩余步时间惩罚，避免策略学会“自杀式终止”。
            reward += -float(self.config.step_penalty) * float(self.config.max_steps - self.step_count)

        # 诊断信息：
        # 训练算法不直接用这些字段，但你可以在日志里观测学习状态。
        info = {
            "distance": distance,
            "success": success,
            "target_pos": self.target_pos.copy(),
            "physics_backend": self.physics_backend,
            "ee_speed": ee_speed,
            "collision": collision_detected,
            "collision_contacts": collision_contacts,
            "curriculum_stage": self.curriculum_stage,
            "episode_index": int(self.episode_count),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """按 Gymnasium 的 render_mode 渲染。"""
        if self.render_mode is None:
            return None
        if self.physics_backend == "warp" and self.data_warp is not None:
            # 仅在渲染时同步 host 数据，避免训练阶段每步全量回传。
            mjwarp.get_data_into(self.data, self.model, self.data_warp)
        if self.render_mode == "rgb_array":
            try:
                if self.renderer is None:
                    # 延迟初始化：无渲染训练时不占用额外上下文资源。
                    self.renderer = mujoco.Renderer(self.model)
                self.renderer.update_scene(self.data)
                return self.renderer.render()
            except Exception as e:
                if not self._render_error_logged:
                    print(f"[渲染] rgb_array 模式失败: {type(e).__name__}: {e}")
                    self._render_error_logged = True
                return None

        # 某些桌面/X11 场景下，窗口被关闭或 drawable 失效后继续 sync 可能触发 GLX 错误。
        # 先判断 viewer 是否仍然存活，避免向失效窗口继续提交渲染命令。
        if self.viewer is not None:
            is_running = getattr(self.viewer, "is_running", None)
            if callable(is_running):
                try:
                    if not bool(is_running()):
                        try:
                            self.viewer.close()
                        except Exception:
                            pass
                        self.viewer = None
                except Exception:
                    try:
                        self.viewer.close()
                    except Exception:
                        pass
                    self.viewer = None

        if self.viewer is None:
            if mj_viewer is None:
                if not self._render_error_logged:
                    print("[渲染] mujoco.viewer 导入失败，请检查 GLFW/OpenGL 运行环境。")
                    self._render_error_logged = True
                return None
            try:
                if self.config.viewer_hide_ui:
                    try:
                        self.viewer = mj_viewer.launch_passive(  # 隐藏 viewer 左右面板。  
                            self.model,
                            self.data,
                            show_left_ui=False,
                            show_right_ui=False,
                        )
                    except TypeError:
                        self.viewer = mj_viewer.launch_passive(self.model, self.data)
                else:
                    self.viewer = mj_viewer.launch_passive(self.model, self.data)  # 被动 viewer。  
                self._target_viz_added = False  # 新窗口需重新添加目标装饰球。  
                self._target_geom_index = None  # 索引重置，避免引用旧窗口对象。  
                self._apply_zero_style_camera()
            except Exception as e:
                if not self._render_error_logged:
                    print(f"[渲染] launch_passive 失败: {type(e).__name__}: {e}")
                    self._render_error_logged = True
                return None
        try:
            is_running = getattr(self.viewer, "is_running", None)
            if callable(is_running) and not bool(is_running()):
                self.viewer = None
                return None
            self._add_target_visualization()  # 每帧更新目标球位置。  
            self.viewer.sync()
        except Exception as e:
            if not self._render_error_logged:
                print(f"[渲染] viewer.sync 失败: {type(e).__name__}: {e}")
                self._render_error_logged = True
            self.viewer = None
        return None

    def _apply_zero_style_camera(self) -> None:
        """设置 viewer 相机。"""
        if self.viewer is None:
            return
        if self.config.viewer_lock_camera and self._render_camera_id >= 0:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = int(self._render_camera_id)
            return
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.azimuth = float(self.config.viewer_fallback_azimuth)
        self.viewer.cam.elevation = float(self.config.viewer_fallback_elevation)
        self.viewer.cam.distance = float(self.config.viewer_fallback_distance)
        self.viewer.cam.lookat[0] = float(self.config.viewer_fallback_lookat_x)
        self.viewer.cam.lookat[1] = float(self.config.viewer_fallback_lookat_y)
        self.viewer.cam.lookat[2] = float(self.config.viewer_fallback_lookat_z)

    def _add_target_visualization(self) -> None:
        """在 viewer.user_scn 里动态维护目标装饰球。"""
        if self.viewer is None:  # viewer 未初始化时直接返回。  
            return  # 避免空对象访问。  
        scn = self.viewer.user_scn  # 取 MuJoCo viewer 的用户场景对象。  
        if scn is None:  # 某些后端异常时 user_scn 可能为空。  
            return  # 保守退出，避免渲染异常中断训练。  

        if not self._target_viz_added:  # 首次渲染：创建装饰几何体。  
            if scn.ngeom >= scn.maxgeom:  # 无剩余几何槽位时无法添加。  
                return  # 不抛异常，保持训练可继续。  
            geom_index = int(scn.ngeom)  # 当前尾部索引就是新几何体位置。  
            geom = scn.geoms[geom_index]  # 取得待初始化的几何体引用。  
            scn.ngeom += 1  # 场景几何体计数 +1。  
            mujoco.mjv_initGeom(  # 初始化为可视化球体（不参与物理）。  
                geom,  # 目标几何体对象。  
                mujoco.mjtGeom.mjGEOM_SPHERE,  # 几何类型：球。  
                np.array([0.03, 0.03, 0.03], dtype=np.float64),  # 球半径（x/y/z 三轴一致）。  
                self.target_pos.astype(np.float64),  # 球心坐标：当前目标点。  
                np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # 单位旋转矩阵。  
                np.array([0.1, 1.0, 0.2, 0.8], dtype=np.float32),  # RGBA：绿色半透明。  
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR  # 仅装饰显示，不参与碰撞。  
            self._target_viz_added = True  # 标记已创建。  
            self._target_geom_index = geom_index  # 记录索引，后续只更新位置。  
            return  # 首次创建完成后返回即可。  

        if self._target_geom_index is None:  # 理论不该发生，但做保护。  
            return  # 防止无效索引访问。  
        if self._target_geom_index >= scn.ngeom:  # 场景变化导致索引失效时保护退出。  
            return  # 避免越界。  
        scn.geoms[self._target_geom_index].pos = self.target_pos.astype(np.float64)  # 每帧更新目标球位置。  

    def close(self):
        """释放渲染资源。"""
        if self.viewer is not None:
            viewer = self.viewer
            self.viewer = None
            # 某些驱动或 GLX 组合下 `viewer.close()` 可能阻塞，因此改成异步关闭。
            def _close_viewer():
                try:
                    viewer.close()
                except Exception:
                    pass

            t = threading.Thread(target=_close_viewer, daemon=True)
            t.start()
            t.join(timeout=0.5)
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None


def register_env() -> None:  # 环境注册入口，`gym.make` 和 SB3 都会通过它找到环境类。
    """注册 Gymnasium 环境 id。

    使用方式：
    1) 在训练脚本中先调用 `register_env()`
    2) 再通过 `make_vec_env(ENV_ID, ...)` 创建向量化环境
    3) 交给 SB3 的 SAC/PPO 直接训练
    """
    # 注册是幂等的：重复调用不会重复注册。
    if ENV_ID not in gym.registry:
        gym.register(
            id=ENV_ID,
            entry_point="classic.env:UR5MujocoEnv",
        )
