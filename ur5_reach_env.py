#!/usr/bin/env python3
# 主线 UR5 到点任务环境。
#
# 本模块把一个 UR5 到点任务包装成标准 Gymnasium 环境。
#
# 涉及的主要外部库：
# - `Gymnasium`：定义 `reset()`、`step()`、`render()` 接口。
# - `MuJoCo`：负责 XML 模型加载、物理推进、接触信息和渲染。
# - `NumPy`：负责观测、动作和奖励中的数组运算。
#
# 这是主线环境的“完全重写版”，结构参考 zero-robotic-arm 的直观写法，
# 但仍严格适配当前仓库的 UR5 模型与 actuator 命名。

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    import mujoco.viewer as mj_viewer
except Exception:
    mj_viewer = None


class UR5ReachEnv(gym.Env):
    # UR5 末端跟踪环境（MuJoCo）。
    #
    # 观测向量由 24 个量组成：
    # 1. 目标相对末端位置：3
    # 2. 六个机械臂关节角：6
    # 3. 六个机械臂关节速度：6
    # 4. 上一步实际施加的关节力矩：6
    # 5. 末端线速度：3

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # 训练参数默认值（与旧版配置保持一致）
    MODEL_XML = "assets/robotiq_cxy/lab_env.xml"
    FRAME_SKIP = 1
    EPISODE_LENGTH = 3000

    TARGET_X_MIN = -0.95
    TARGET_X_MAX = -0.60
    TARGET_Y_MIN = 0.15
    TARGET_Y_MAX = 0.50
    TARGET_Z_MIN = 0.12
    TARGET_Z_MAX = 0.30

    TORQUE_LOW = -15.0
    TORQUE_HIGH = 15.0

    STEP_PENALTY = 0.10
    DISTANCE_WEIGHT = 0.80
    PROGRESS_REWARD_GAIN = 1.0
    REGRESS_PENALTY_GAIN = 0.8
    PHASE_THRESHOLDS = (0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002)
    PHASE_REWARDS = (100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0)
    SPEED_PENALTY_THRESHOLD = 0.5
    SPEED_PENALTY_VALUE = 0.2
    DIRECTION_REWARD_GAIN = 1.0
    JOINT_VELOCITY_PENALTY = 0.03
    COLLISION_PENALTY = 5000.0
    SUCCESS_BONUS = 10000.0
    SUCCESS_REMAINING_STEP_GAIN = 4.0
    SUCCESS_SPEED_BONUS_VERY_SLOW = 2000.0
    SUCCESS_SPEED_BONUS_SLOW = 1000.0
    SUCCESS_SPEED_BONUS_MEDIUM = 500.0
    SUCCESS_THRESHOLD = 0.01

    GRAVITY_COMPENSATION = -1.0
    FIXED_GRIPPER_CTRL = 0.0

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        if not self.MODEL_XML:
            raise ValueError("MODEL_XML is empty.")

        # MuJoCo 负责解析 XML、构造动力学模型和维护运行时状态。
        xml_path = Path(__file__).resolve().parent / self.MODEL_XML
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 先把后续要反复查询的关节和执行器名字集中定义出来。
        self.arm_joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
        self.arm_actuator_names = (
            "joint1_motor",
            "joint2_motor",
            "joint3_motor",
            "joint4_motor",
            "joint5_motor",
            "joint6_motor",
        )
        self.gripper_actuator_names = ("close_1", "close_2")
        self.gravity_actuator_names = ("gravity_1", "gravity_2", "gravity_3", "gravity_4")

        self.arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names],
            dtype=np.int32,
        )
        self.arm_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.arm_actuator_names],
            dtype=np.int32,
        )
        self.gripper_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.gripper_actuator_names],
            dtype=np.int32,
        )
        self.gravity_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.gravity_actuator_names],
            dtype=np.int32,
        )
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)
        self.arm_qvel_adr = np.array([self.model.jnt_dofadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)

        # 这些索引会在 reset、step 和 reward 里反复使用。
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        self.target_x_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_x_1")
        ]
        self.target_y_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_y_1")
        ]
        self.target_z_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_z_1")
        ]
        self.target_ball_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_ball_1")
        ]

        # Gymnasium 需要显式声明动作空间和观测空间。
        # 动作空间：力矩
        self.action_space = spaces.Box(
            low=float(self.TORQUE_LOW),
            high=float(self.TORQUE_HIGH),
            shape=(len(self.arm_actuator_ids),),
            dtype=np.float32,
        )

        # 观测：相对位置(3) + 关节角(6) + 关节速度(6) + 上一时刻扭矩(6) + 末端速度(3) = 24
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        # 下面这组成员变量用于跨 step 保存历史信息，参与奖励和终止判定。
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.previous_distance = None
        self.min_distance = None
        self.previous_torque = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)
        self.previous_joint_velocities = np.zeros(len(self.arm_joint_ids), dtype=np.float32)
        self.step_count = 0

        # 渲染器句柄
        self.viewer = None
        self._target_viz_added = False
        self._target_geom = None

        # 重置环境
        self.reset()

    def _get_state(self) -> np.ndarray:
        # 拼出策略看到的观测向量。
        # 这里不直接返回 MuJoCo 全状态，只保留任务最相关的部分。
        # 先取末端位置与速度。
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        ee_vel = self.data.cvel[self.ee_body_id][:3].copy()

        # 相对位置向量
        relative_pos = self.target_pos - ee_pos

        # 关节状态
        joint_angles = self.data.qpos[self.arm_qpos_adr].copy()
        joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()

        state = np.concatenate(
            [
                relative_pos,
                joint_angles,
                joint_velocities,
                self.previous_torque.copy(),
                ee_vel,
            ]
        )
        return state.astype(np.float32)

    def _set_target_position(self, target_position: np.ndarray) -> None:
        # 把采样到的目标点写入 MuJoCo 的滑动关节。
        self.data.qpos[self.target_x_qpos_adr] = float(target_position[0])
        self.data.qpos[self.target_y_qpos_adr] = float(target_position[1])
        self.data.qpos[self.target_z_qpos_adr] = float(target_position[2])
        self.data.qpos[self.target_ball_qpos_adr : self.target_ball_qpos_adr + 4] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 重置环境并返回首个观测。
        super().reset(seed=seed)
        del options

        # 重置 MuJoCo 数据
        mujoco.mj_resetData(self.model, self.data)

        # 采样目标点并写入物理状态。
        self.target_pos = np.array(
            [
                self.np_random.uniform(self.TARGET_X_MIN, self.TARGET_X_MAX),
                self.np_random.uniform(self.TARGET_Y_MIN, self.TARGET_Y_MAX),
                self.np_random.uniform(self.TARGET_Z_MIN, self.TARGET_Z_MAX),
            ],
            dtype=np.float32,
        )
        self._set_target_position(self.target_pos)

        # 重置控制量（力矩 / 夹爪 / 重力补偿）
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.gripper_actuator_ids] = float(self.FIXED_GRIPPER_CTRL)
        self.data.ctrl[self.gravity_actuator_ids] = float(self.GRAVITY_COMPENSATION)

        # 缓存历史量
        self.step_count = 0
        self.previous_distance = None
        self.min_distance = None
        self.previous_torque = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)
        self.previous_joint_velocities = np.zeros(len(self.arm_joint_ids), dtype=np.float32)
        self._phase_rewards_given: set[float] = set()

        # 获取初始状态
        state = self._get_state()
        return state, {}

    def step(self, action: np.ndarray):
        # 推进一步物理并计算回报。
        # 这里保持和参考实现相同的奖励拆分结构，便于对照调参。
        # 第一步：保存上一时刻动作与速度。
        self.previous_torque = self.data.ctrl[self.arm_actuator_ids].copy()
        self.previous_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()

        # 第二步：应用动作（扭矩）。
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, float(self.TORQUE_LOW), float(self.TORQUE_HIGH))
        self.data.ctrl[self.arm_actuator_ids] = action
        self.data.ctrl[self.gripper_actuator_ids] = float(self.FIXED_GRIPPER_CTRL)
        self.data.ctrl[self.gravity_actuator_ids] = float(self.GRAVITY_COMPENSATION)

        # 第三步：推进物理。
        for _ in range(max(int(self.FRAME_SKIP), 1)):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 第四步：计算新状态。
        state = self._get_state()

        # 第五步：计算奖励。
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        distance = float(np.linalg.norm(ee_pos - self.target_pos))
        reward = 0.0

        # 时间惩罚
        reward -= float(self.STEP_PENALTY)

        # 距离进步/退步奖励
        if self.min_distance is None:
            self.min_distance = distance
            improvement_reward = 0.0
        elif distance < float(self.min_distance):
            improvement_reward = float(self.PROGRESS_REWARD_GAIN) * (float(self.min_distance) - distance)
            self.min_distance = distance
        elif self.previous_distance is not None and distance > float(self.previous_distance):
            improvement_reward = -float(self.REGRESS_PENALTY_GAIN) * (distance - float(self.previous_distance))
        else:
            improvement_reward = 0.0
        self.previous_distance = distance

        # 基础距离惩罚
        base_distance_penalty = -float(self.DISTANCE_WEIGHT) * float(np.sqrt(distance + 1e-8))

        # 阶段性距离奖励（一次性）
        phase_distance_reward = 0.0
        for thresh, phase_reward in zip(self.PHASE_THRESHOLDS, self.PHASE_REWARDS):
            if distance < float(thresh) and float(thresh) not in self._phase_rewards_given:
                phase_distance_reward += float(phase_reward)
                self._phase_rewards_given.add(float(thresh))

        # 综合距离奖励
        reward += improvement_reward + base_distance_penalty + phase_distance_reward

        # 速度惩罚
        ee_vel = self.data.cvel[self.ee_body_id][:3].copy()
        ee_speed = float(np.linalg.norm(ee_vel))
        if ee_speed > float(self.SPEED_PENALTY_THRESHOLD):
            reward -= float(self.SPEED_PENALTY_VALUE)

        # 方向奖励
        to_target = self.target_pos - ee_pos
        to_target /= (np.linalg.norm(to_target) + 1e-6)
        movement_dir = ee_vel / (np.linalg.norm(ee_vel) + 1e-6)
        direction_cos = float(np.dot(to_target, movement_dir))
        direction_reward = max(0.0, direction_cos) ** 2 * float(self.DIRECTION_REWARD_GAIN)
        reward += direction_reward

        # 碰撞惩罚（任意接触）
        collision_detected = self.data.ncon > 0
        if collision_detected:
            reward += -float(self.COLLISION_PENALTY)

        # 关节速度变化惩罚
        current_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()
        joint_velocity_change = np.abs(current_joint_velocities - self.previous_joint_velocities)
        reward += -float(self.JOINT_VELOCITY_PENALTY) * float(np.sum(joint_velocity_change))

        # 成功判定
        done = False
        if distance <= float(self.SUCCESS_THRESHOLD):
            done = True
            reward += float(self.SUCCESS_BONUS)
            reward += float(self.SUCCESS_REMAINING_STEP_GAIN) * float(max(int(self.EPISODE_LENGTH) - self.step_count, 0))
            if ee_speed < 0.01:
                reward += float(self.SUCCESS_SPEED_BONUS_VERY_SLOW)
            elif ee_speed < 0.05:
                reward += float(self.SUCCESS_SPEED_BONUS_SLOW)
            elif ee_speed < 0.1:
                reward += float(self.SUCCESS_SPEED_BONUS_MEDIUM)

        # 碰撞直接结束
        if collision_detected:
            reward += -float(self.STEP_PENALTY) * float(max(int(self.EPISODE_LENGTH) - self.step_count, 0))
            done = True

        truncated = self.step_count >= int(self.EPISODE_LENGTH)

        return state, float(reward), bool(done), bool(truncated), {}

    def render(self):
        # 渲染环境。
        # `human` 模式使用 `mujoco.viewer` 打开交互窗口。
        if self.render_mode is None:
            return None

        # 如果可视化器尚未创建，则创建一个
        if self.viewer is None:
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer is not available in this environment.")
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self._target_viz_added = False

        # 如果可视化器已经创建，同步更新场景
        if self.viewer:
            self._add_target_visualization()
            self.viewer.sync()
        return None

    def _add_target_visualization(self) -> None:
        # 在目标位置添加一个绿色小球用于可视化
        if not self.viewer:
            return
        scn = self.viewer.user_scn
        if not self._target_viz_added:
            if scn.ngeom < scn.maxgeom:
                self._target_geom = scn.ngeom
                geom = scn.geoms[scn.ngeom]
                scn.ngeom += 1
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.02, 0.02, 0.02]),
                    self.target_pos,
                    np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    np.array([0.0, 1.0, 0.0, 0.8]),
                )
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                self._target_viz_added = True
        elif self._target_geom is not None:
            geom = scn.geoms[self._target_geom]
            geom.pos = self.target_pos

    def close(self) -> None:
        # 释放 viewer 句柄。
        if self.viewer is not None:
            close_fn = getattr(self.viewer, "close", None)
            if callable(close_fn):
                close_fn()
            self.viewer = None
