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
    MODEL_XML = "assets/robotiq_cxy/lab_env.xml"  # MuJoCo XML 路径（相对当前文件）
    FRAME_SKIP = 1  # 每个 step 内部物理子步数
    EPISODE_LENGTH = 3000  # 每回合最大步数（超出即 truncated）

    TARGET_X_MIN = -0.95  # 目标球 x 最小值
    TARGET_X_MAX = -0.60  # 目标球 x 最大值
    TARGET_Y_MIN = 0.15  # 目标球 y 最小值
    TARGET_Y_MAX = 0.50  # 目标球 y 最大值
    TARGET_Z_MIN = 0.12  # 目标球 z 最小值
    TARGET_Z_MAX = 0.30  # 目标球 z 最大值
    TARGET_RANGE_SCALE = 0.35  # 小范围随机的采样缩放（0~1）
    TARGET_FIXED_X = None  # 固定目标 x（None 表示用工作空间中心）
    TARGET_FIXED_Y = None  # 固定目标 y
    TARGET_FIXED_Z = None  # 固定目标 z

    TORQUE_LOW = -15.0  # 力矩下界（单位与 MuJoCo actuator 一致）
    TORQUE_HIGH = 15.0  # 力矩上界
    # - PHASE_THRESHOLDS/PHASE_REWARDS: 首次跨越距离阈值的一次性奖励
    STEP_PENALTY = 0.10
    DISTANCE_WEIGHT = 0.80
    PROGRESS_REWARD_GAIN = 1.0
    REGRESS_PENALTY_GAIN = 0.8
    PHASE_THRESHOLDS = (0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002)
    PHASE_REWARDS = (100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0)
    SPEED_PENALTY_THRESHOLD = 0.5  # 末端速度惩罚阈值
    SPEED_PENALTY_VALUE = 0.2  # 末端速度惩罚强度
    DIRECTION_REWARD_GAIN = 1.0  # 朝目标运动方向奖励增益
    JOINT_VELOCITY_PENALTY = 0.03  # 关节速度变化惩罚增益
    COLLISION_PENALTY = 5000.0  # 碰撞惩罚（并触发 done）
    SUCCESS_BONUS = 10000.0  # 成功一次性奖励
    SUCCESS_REMAINING_STEP_GAIN = 4.0  # 剩余步数奖励权重
    SUCCESS_SPEED_BONUS_VERY_SLOW = 2000.0  # 成功且速度极慢奖励
    SUCCESS_SPEED_BONUS_SLOW = 1000.0  # 成功且速度较慢奖励
    SUCCESS_SPEED_BONUS_MEDIUM = 500.0  # 成功且速度中等奖励
    SUCCESS_THRESHOLD = 0.01  # 成功判定距离阈值

    GRAVITY_COMPENSATION = -1.0  # 重力补偿执行器控制值
    FIXED_GRIPPER_CTRL = 0.0  # 固定夹爪控制量（当前不训练夹爪）
    HOME_JOINT1 = 0.5183627878423158  # 初始关节1角度
    HOME_JOINT2 = -1.4835298641951802  # 初始关节2角度
    HOME_JOINT3 = 2.007128639793479  # 初始关节3角度
    SUCCESS_PER_STAGE = 5  # 每成功 N 次切换一次目标采样模式

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
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))  # 静态模型
        self.data = mujoco.MjData(self.model)  # 动态状态

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
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)  # qpos 索引
        self.arm_qvel_adr = np.array([self.model.jnt_dofadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)  # qvel 索引

        # 这些索引会在 reset、step 和 reward 里反复使用。
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")  # 末端 link
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
        # 动作空间：机械臂 6 关节力矩，范围由 TORQUE_LOW/HIGH 给出。
        self.action_space = spaces.Box(
            low=float(self.TORQUE_LOW),
            high=float(self.TORQUE_HIGH),
            shape=(len(self.arm_actuator_ids),),
            dtype=np.float32,
        )

        # 观测：相对位置(3) + 关节角(6) + 关节速度(6)
        # + 上一时刻扭矩(6) + 末端速度(3) = 24
        # 这里不包含目标绝对坐标，避免观测维度冗余。
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
        self.success_count = 0  # 累计成功次数（用于切换目标采样模式）
        self.target_stage = 0  # 0: fixed, 1: small_random, 2: full_random

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

    def _target_center(self) -> np.ndarray:
        # 固定目标中心点（未设置固定值时取工作空间中心）
        x = self.TARGET_FIXED_X if self.TARGET_FIXED_X is not None else 0.5 * (self.TARGET_X_MIN + self.TARGET_X_MAX)
        y = self.TARGET_FIXED_Y if self.TARGET_FIXED_Y is not None else 0.5 * (self.TARGET_Y_MIN + self.TARGET_Y_MAX)
        z = self.TARGET_FIXED_Z if self.TARGET_FIXED_Z is not None else 0.5 * (self.TARGET_Z_MIN + self.TARGET_Z_MAX)
        return np.array([x, y, z], dtype=np.float32)

    def _sample_target(self) -> np.ndarray:
        # 目标采样策略：
        # stage 0: 固定目标
        # stage 1: 小范围随机（以中心点为基准）
        # stage 2: 大范围随机（全随机）
        if self.target_stage <= 0:
            return self._target_center()
        if self.target_stage == 1:
            scale = float(np.clip(self.TARGET_RANGE_SCALE, 1e-3, 1.0))
            center = self._target_center()
            x_half = 0.5 * float(self.TARGET_X_MAX - self.TARGET_X_MIN) * scale
            y_half = 0.5 * float(self.TARGET_Y_MAX - self.TARGET_Y_MIN) * scale
            z_half = 0.5 * float(self.TARGET_Z_MAX - self.TARGET_Z_MIN) * scale
            return np.array(
                [
                    self.np_random.uniform(center[0] - x_half, center[0] + x_half),
                    self.np_random.uniform(center[1] - y_half, center[1] + y_half),
                    self.np_random.uniform(center[2] - z_half, center[2] + z_half),
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                self.np_random.uniform(self.TARGET_X_MIN, self.TARGET_X_MAX),
                self.np_random.uniform(self.TARGET_Y_MIN, self.TARGET_Y_MAX),
                self.np_random.uniform(self.TARGET_Z_MIN, self.TARGET_Z_MAX),
            ],
            dtype=np.float32,
        )

    def _home_joint_positions(self) -> np.ndarray:
        # 返回默认初始姿态关节角（与 Warp 线保持一致）。
        q1 = float(self.HOME_JOINT1)
        q2 = float(self.HOME_JOINT2)
        q3 = float(self.HOME_JOINT3)
        return np.array(
            [
                q1,
                q2,
                q3,
                1.5 * np.pi - q2 - q3,
                1.5 * np.pi,
                1.25 * np.pi + q1,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 重置环境并返回首个观测。
        super().reset(seed=seed)
        del options

        # 重置 MuJoCo 数据
        mujoco.mj_resetData(self.model, self.data)  # 清空动力学状态
        self.data.qpos[self.arm_qpos_adr] = self._home_joint_positions()  # 设置机械臂初始姿态
        self.data.qvel[self.arm_qvel_adr] = 0.0  # 清空机械臂关节速度

        # 采样目标点并写入物理状态。
        self.target_pos = self._sample_target()
        self._set_target_position(self.target_pos)  # 把目标球移动到采样位置
        mujoco.mj_forward(self.model, self.data)  # 让初始姿态和目标生效

        # 重置控制量（力矩 / 夹爪 / 重力补偿）
        self.data.ctrl[:] = 0.0  # 清空所有执行器控制量
        self.data.ctrl[self.gripper_actuator_ids] = float(self.FIXED_GRIPPER_CTRL)  # 固定夹爪
        self.data.ctrl[self.gravity_actuator_ids] = float(self.GRAVITY_COMPENSATION)  # 重力补偿

        # 缓存历史量
        self.step_count = 0  # 回合步数计数
        self.previous_distance = None  # 上一步距离（用于退步惩罚）
        self.min_distance = None  # 历史最小距离（用于进步奖励）
        self.previous_torque = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)  # 上一步扭矩
        self.previous_joint_velocities = np.zeros(len(self.arm_joint_ids), dtype=np.float32)  # 上一步关节速度
        self._phase_rewards_given: set[float] = set()  # 已触发的阶段奖励

        # 获取初始状态
        state = self._get_state()  # 组装观测向量
        info = {
            "distance": float(np.linalg.norm(self.target_pos - self.data.xpos[self.ee_body_id])),  # 相对距离
            "ee_speed": 0.0,  # 末端速度（初始为 0）
            "success": False,  # 是否成功命中目标
            "collision": False,  # 是否发生碰撞
        }
        return state, info

    def step(self, action: np.ndarray):
        # 推进一步物理并计算回报。
        # 这里保持和参考实现相同的奖励拆分结构，便于对照调参。
        # 第一步：保存上一时刻动作与速度。
        self.previous_torque = self.data.ctrl[self.arm_actuator_ids].copy()  # 保存旧扭矩
        self.previous_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()  # 保存旧关节速度

        # 第二步：应用动作（扭矩）。
        action = np.asarray(action, dtype=np.float32)  # 强制 float32
        action = np.clip(action, float(self.TORQUE_LOW), float(self.TORQUE_HIGH))  # 裁剪到合法范围
        self.data.ctrl[self.arm_actuator_ids] = action  # 写入机械臂执行器
        self.data.ctrl[self.gripper_actuator_ids] = float(self.FIXED_GRIPPER_CTRL)  # 固定夹爪
        self.data.ctrl[self.gravity_actuator_ids] = float(self.GRAVITY_COMPENSATION)  # 重力补偿

        # 第三步：推进物理。
        for _ in range(max(int(self.FRAME_SKIP), 1)):  # 每个 step 里执行若干子步
            mujoco.mj_step(self.model, self.data)  # MuJoCo 物理推进

        self.step_count += 1  # 更新回合步数

        # 第四步：计算新状态。
        state = self._get_state()  # 观测向量

        # 第五步：计算奖励。
        ee_pos = self.data.xpos[self.ee_body_id].copy()  # 末端位置
        distance = float(np.linalg.norm(ee_pos - self.target_pos))  # 与目标距离
        reward = 0.0  # 初始化奖励

        # 时间惩罚
        reward -= float(self.STEP_PENALTY)  # 每步固定惩罚

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
        self.previous_distance = distance  # 记录上一步距离

        # 基础距离惩罚
        base_distance_penalty = -float(self.DISTANCE_WEIGHT) * float(np.sqrt(distance + 1e-8))  # 距离越大惩罚越大

        # 阶段性距离奖励（一次性）
        phase_distance_reward = 0.0  # 只在首次穿越阈值时给
        for thresh, phase_reward in zip(self.PHASE_THRESHOLDS, self.PHASE_REWARDS):
            if distance < float(thresh) and float(thresh) not in self._phase_rewards_given:
                phase_distance_reward += float(phase_reward)
                self._phase_rewards_given.add(float(thresh))

        # 综合距离奖励
        reward += improvement_reward + base_distance_penalty + phase_distance_reward  # 距离类项相加

        # 速度惩罚
        ee_vel = self.data.cvel[self.ee_body_id][:3].copy()  # 末端线速度
        ee_speed = float(np.linalg.norm(ee_vel))  # 末端速度模长
        if ee_speed > float(self.SPEED_PENALTY_THRESHOLD):
            reward -= float(self.SPEED_PENALTY_VALUE)

        # 方向奖励
        to_target = self.target_pos - ee_pos  # 指向目标的方向
        to_target /= (np.linalg.norm(to_target) + 1e-6)  # 归一化
        movement_dir = ee_vel / (np.linalg.norm(ee_vel) + 1e-6)  # 当前移动方向
        direction_cos = float(np.dot(to_target, movement_dir))  # 方向一致性
        direction_reward = max(0.0, direction_cos) ** 2 * float(self.DIRECTION_REWARD_GAIN)  # 仅奖励朝向目标
        reward += direction_reward  # 添加方向奖励

        # 碰撞惩罚（任意接触）
        collision_detected = self.data.ncon > 0  # 只要有接触就认为碰撞
        if collision_detected:
            reward += -float(self.COLLISION_PENALTY)

        # 关节速度变化惩罚
        current_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()  # 当前关节速度
        joint_velocity_change = np.abs(current_joint_velocities - self.previous_joint_velocities)  # 变化量
        reward += -float(self.JOINT_VELOCITY_PENALTY) * float(np.sum(joint_velocity_change))  # 总变化惩罚

        # 成功判定
        done = False  # 先置为 False
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
            self.success_count += 1
            if self.success_count % int(self.SUCCESS_PER_STAGE) == 0:
                self.target_stage = min(self.target_stage + 1, 2)

        # 碰撞直接结束
        if collision_detected:
            reward += -float(self.STEP_PENALTY) * float(max(int(self.EPISODE_LENGTH) - self.step_count, 0))
            done = True

        truncated = self.step_count >= int(self.EPISODE_LENGTH)  # 超过最大步数即截断

        info = {
            "distance": distance,  # 相对距离
            "ee_speed": ee_speed,  # 末端速度
            "success": bool(distance <= float(self.SUCCESS_THRESHOLD)),  # 成功标记
            "collision": bool(collision_detected),  # 碰撞标记
        }
        return state, float(reward), bool(done), bool(truncated), info

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
