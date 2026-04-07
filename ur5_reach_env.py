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

from ur5_reach_config import UR5ReachEnvConfig, project_root

OBSERVATION_SCHEMA = (
    ("obs[00:03]", "relative_position_xyz", "目标位置减末端位置，单位米；依次对应 x/y/z 三个方向。"),
    ("obs[03:09]", "joint_positions", "UR5 六个关节当前角度，单位弧度；顺序是 joint1 到 joint6。"),
    ("obs[09:15]", "joint_velocities", "UR5 六个关节当前角速度，单位弧度每秒；顺序是 joint1 到 joint6。"),
    ("obs[15:21]", "previous_torque", "上一决策步实际施加到六个关节的力矩，单位牛米；顺序是 joint1 到 joint6。"),
    ("obs[21:24]", "ee_velocity_xyz", "末端执行器当前线速度，单位米每秒；依次对应 x/y/z 三个方向。"),
)


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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        # 下面这组成员变量用于跨 step 保存历史信息，参与动作平滑、速度估计和奖励计算。
        self.episode_index = 0
        self.step_count = 0
        self.current_stage = "fixed"
        self.target_position = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(6, dtype=np.float32)
        self.previous_control = np.zeros(6, dtype=np.float32)
        self.previous_joint_velocities = np.zeros(6, dtype=np.float32)
        self.previous_ee_position = np.zeros(3, dtype=np.float32)
        self.current_ee_velocity = np.zeros(3, dtype=np.float32)
        self.previous_distance: float | None = None
        self.best_distance: float | None = None
        self.phase_rewards_given: set[float] = set()
        self.last_reward_terms: dict[str, float] = {}
        self.episode_return = 0.0
        self.episode_collision_count = 0
        self.episode_success_count = 0
        self.lifetime_success_count = 0

    @classmethod
    def observation_schema(cls) -> tuple[tuple[str, str, str], ...]:
        # 返回观测向量的切片定义，便于训练脚本把每一段的物理含义直接打印出来。
        return OBSERVATION_SCHEMA

    def _collect_descendant_body_ids(self, root_body_id: int) -> set[int]:
        # 收集机器人根节点下的所有 body id，用来过滤机器人内部自碰撞。
        result: set[int] = set()
        queue = [int(root_body_id)]
        while queue:
            # 这里做的是一个简单的广度/深度混合遍历：
            # 从根 body 出发，一层层找到所有子 body。
            current = queue.pop()
            result.add(current)
            for body_id in range(self.model.nbody):
                if int(self.model.body_parentid[body_id]) == current and body_id != current:
                    queue.append(body_id)
        return result

    def _collect_ignored_contact_geom_ids(self) -> set[int]:
        # 收集在碰撞判定中需要忽略的几何体，例如装饰灯光。
        ignored: set[int] = set()
        for geom_id in range(self.model.ngeom):
            # 这些 geom 主要用于可视化提示，不应该参与失败碰撞判定。
            name = self.model.geom(geom_id).name or ""
            if name.startswith("light_"):
                ignored.add(int(geom_id))
        return ignored

    def _set_home_pose(self) -> None:
        # 把机械臂 reset 到一个稳定且能覆盖工作空间的初始姿态。
        #
        # 做法是：
        # 1. 先恢复 MuJoCo 初始 qpos/qvel 模板
        # 2. 再覆盖机械臂 6 个关节
        # 3. 最后把夹爪和重力补偿控制量写进去
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = self.home_qvel

        q1 = float(self.config.home_joint1)
        q2 = float(self.config.home_joint2)
        q3 = float(self.config.home_joint3)
        arm_pose = np.array(
            [
                q1,
                q2,
                q3,
                1.5 * math.pi - q2 - q3,
                1.5 * math.pi,
                1.25 * math.pi + q1,
            ],
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
        return observation.astype(np.float32)

    def _compute_pd_torque(self, action: np.ndarray) -> np.ndarray:
        # 把归一化的关节增量动作通过一个小型 PD 控制器转换成力矩。
        #
        # 做法是：
        # 1. 读出当前关节角和关节速度
        # 2. 用动作生成目标关节角
        # 3. 通过 `kp * 位置误差 - kd * 速度` 生成力矩
        current_positions = self.data.qpos[self.arm_qpos_adr].astype(np.float32).copy()
        current_velocities = self.data.qvel[self.arm_qvel_adr].astype(np.float32).copy()
        desired_positions = np.clip(
            current_positions + action * float(self.config.joint_delta_scale),
            self.arm_joint_low,
            self.arm_joint_high,
        )
        torques = (
            float(self.config.position_kp) * (desired_positions - current_positions)
            - float(self.config.position_kd) * current_velocities
        )
        return np.clip(torques, float(self.config.torque_low), float(self.config.torque_high)).astype(np.float32)

    def _action_to_control(self, action: np.ndarray) -> np.ndarray:
        # 把策略动作映射成真实执行器控制量。
        #
        # 两种控制方式：
        # - `torque`：直接把动作缩放成力矩。
        # - `joint_delta`：先解释成关节目标增量，再通过 PD 控制器转成力矩。
        # 第一步：把策略动作裁剪到动作空间范围内。
        normalized_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.config.control_mode == "torque":
            # torque 模式：直接线性映射到真实力矩区间。
            torque_center = 0.5 * (float(self.config.torque_high) + float(self.config.torque_low))
            torque_scale = 0.5 * (float(self.config.torque_high) - float(self.config.torque_low))
            raw_control = torque_center + torque_scale * normalized_action
        else:
            # joint_delta 模式：先转目标关节角，再通过 PD 算法转成力矩。
            raw_control = self._compute_pd_torque(normalized_action)
        # 平滑项的作用是抑制动作抖动，减少训练初期的高频力矩震荡。
        smoothed = (
            float(self.config.action_smoothing_alpha) * self.previous_control
            + (1.0 - float(self.config.action_smoothing_alpha)) * raw_control
        )
        smoothed = np.clip(smoothed, float(self.config.torque_low), float(self.config.torque_high)).astype(np.float32)
        self.previous_control = smoothed
        return smoothed

    def _has_external_collision(self) -> bool:
        # 把机器人与外部环境的碰撞视为失败，但忽略机器人自碰撞和纯装饰 geom。
        geom_body_ids = self.model.geom_bodyid
        for contact_index in range(self.data.ncon):
            # 逐条读取当前活跃 contact，再按名字和 body 归属过滤掉不危险的碰撞。
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.ignored_contact_geom_ids or geom2 in self.ignored_contact_geom_ids:
                continue
            body1 = int(geom_body_ids[geom1])
            body2 = int(geom_body_ids[geom2])
            body1_is_robot = body1 in self.robot_body_ids
            body2_is_robot = body2 in self.robot_body_ids
            if body1_is_robot and body2_is_robot:
                continue
            if body1_is_robot or body2_is_robot:
                return True
        return False

    def _update_ee_velocity(self, ee_position: np.ndarray) -> None:
        # 用有限差分估计末端执行器线速度。
        dt = max(float(self.model.opt.timestep) * int(self.config.frame_skip), 1e-6)
        self.current_ee_velocity = ((ee_position - self.previous_ee_position) / dt).astype(np.float32)
        self.previous_ee_position = ee_position.astype(np.float32)

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
        distance: float,
        collision: bool,
        ee_speed: float,
        remaining_steps: int,
    ) -> tuple[float, bool, bool, dict[str, float], str]:
        # 计算 reward，并把各个奖励项拆开返回。
        #
        # 这里同时返回 `reward_terms`，目的是让训练调试和 notebook 学习时可以直接看到：
        # - 每一项奖励是怎么来的
        # - 哪个终止条件先触发
        # - 当前策略是卡在“碰撞、超时、跑飞”还是“快要成功”
        terminated = False
        truncated = False
        done_reason = "running"

        # 先算和上一时刻相比是否更接近目标，并保留“历史最优距离”和“上一时刻距离”两种基准。
        previous_distance = float(self.previous_distance) if self.previous_distance is not None else float(distance)
        if self.best_distance is None:
            self.best_distance = float(distance)
        best_distance_before = float(self.best_distance)

        # 奖励拆分成若干明确的物理含义项，便于调参和解释。
        reward_terms = {
            "step_penalty": -float(self.config.step_penalty),
            "distance_penalty": -float(self.config.distance_weight) * float(np.sqrt(distance + 1e-8)),
            "progress_reward": 0.0,
            "regress_penalty": 0.0,
            "phase_reward": 0.0,
            "speed_penalty": 0.0,
            "direction_reward": 0.0,
            "action_penalty": -float(self.config.action_l2_penalty) * float(np.mean(np.square(action))),
            "smoothness_penalty": -float(self.config.action_smoothness_penalty) * float(np.mean(np.square(action - previous_action))),
            "joint_velocity_penalty": 0.0,
            "collision_penalty": 0.0,
            "success_bonus": 0.0,
            "runaway_penalty": 0.0,
        }

        # 奖励主体和 zero-arm 一样，优先看是否刷新历史最优距离，其次再看是否比上一时刻更远。
        if distance < best_distance_before:
            reward_terms["progress_reward"] = float(self.config.progress_reward_gain) * (best_distance_before - float(distance))
            self.best_distance = float(distance)
        elif distance > previous_distance:
            reward_terms["regress_penalty"] = -float(self.config.regress_penalty_gain) * (float(distance) - previous_distance)

        # 阶段奖励只在第一次穿过阈值时触发一次，鼓励先学会逐步逼近目标。
        for threshold, phase_reward in zip(self.config.phase_thresholds, self.config.phase_rewards):
            if distance < float(threshold) and float(threshold) not in self.phase_rewards_given:
                reward_terms["phase_reward"] += float(phase_reward)
                self.phase_rewards_given.add(float(threshold))

        # 速度太快时给予固定惩罚，抑制冲向目标时的剧烈摆动。
        if ee_speed > float(self.config.speed_penalty_threshold):
            reward_terms["speed_penalty"] = -float(self.config.speed_penalty_value)

        # 方向奖励需要当前末端速度和“朝向目标的方向”同时存在。
        to_target = self.target_position - self._finger_center()
        to_target_norm = float(np.linalg.norm(to_target))
        if ee_speed > 1e-6 and to_target_norm > 1e-6:
            movement_direction = self.current_ee_velocity / (ee_speed + 1e-6)
            target_direction = to_target / (to_target_norm + 1e-6)
            direction_cos = max(float(np.dot(movement_direction, target_direction)), 0.0)
            reward_terms["direction_reward"] = float(self.config.direction_reward_gain) * (direction_cos**2)

        # 关节速度变化惩罚使用“当前关节速度 - 上一时刻关节速度”的绝对值和。
        joint_velocity_change = np.abs(self.data.qvel[self.arm_qvel_adr] - self.previous_joint_velocities)
        reward_terms["joint_velocity_penalty"] = -float(self.config.joint_velocity_penalty) * float(np.sum(joint_velocity_change))

        # 碰撞、成功、跑飞和超时共同决定回合是否结束。
        if collision:
            reward_terms["collision_penalty"] = -float(self.config.collision_penalty)
            reward_terms["collision_penalty"] += -float(self.config.step_penalty) * float(max(remaining_steps, 0))
            terminated = True
            done_reason = "collision"

        success = distance <= self._success_threshold()
        if success:
            reward_terms["success_bonus"] = float(self.config.success_bonus)
            reward_terms["success_bonus"] += float(self.config.success_remaining_step_gain) * float(max(remaining_steps, 0))
            if ee_speed < 0.01:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_very_slow)
            elif ee_speed < 0.05:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_slow)
            elif ee_speed < 0.10:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_medium)
            terminated = True
            done_reason = "success"

        if distance > float(self.config.runaway_distance_threshold):
            reward_terms["runaway_penalty"] = -float(self.config.runaway_penalty)
            terminated = True
            done_reason = "runaway"

        if self.step_count >= int(self.config.episode_length):
            truncated = True
            done_reason = "timeout"

        self.previous_distance = float(distance)
        reward = float(sum(reward_terms.values()))
        self.last_reward_terms = reward_terms
        return reward, terminated, truncated, reward_terms, done_reason

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 重置环境并返回首个观测。
        super().reset(seed=seed)
        del options

        # 重置 MuJoCo 数据
        mujoco.mj_resetData(self.model, self.data)  # 清空动力学状态
        self.data.qpos[self.arm_qpos_adr] = self._home_joint_positions()  # 设置机械臂初始姿态
        self.data.qvel[self.arm_qvel_adr] = 0.0  # 清空机械臂关节速度

        # 第三步：初始化和历史相关的缓存量。
        ee_position = self._finger_center()
        self.previous_action[:] = 0.0
        self.previous_control[:] = 0.0
        self.previous_joint_velocities[:] = 0.0
        self.previous_ee_position = ee_position.astype(np.float32)
        self.current_ee_velocity[:] = 0.0
        self.previous_distance = float(np.linalg.norm(self.target_position - ee_position))
        self.best_distance = self.previous_distance
        self.phase_rewards_given = set()
        self.step_count = 0
        self.last_reward_terms = {}
        self.episode_return = 0.0
        self.episode_collision_count = 0
        self.episode_success_count = 0
        current_episode_index = self.episode_index + 1

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
            "episode_index": current_episode_index,
            "target_position": self.target_position.copy(),
            "curriculum_stage": self.current_stage,
            "success_threshold": self._success_threshold(),
            "distance": float(self.previous_distance),
            "relative_distance": float(self.previous_distance),
            "ee_speed": 0.0,
            "relative_speed": 0.0,
            "episode_return": 0.0,
            "episode_collision_count": 0,
            "episode_success_count": 0,
            "lifetime_success_count": int(self.lifetime_success_count),
        }
        self.episode_index = current_episode_index
        return observation, info

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

        # 第四步：根据新状态计算速度、距离、碰撞和 reward。
        self.step_count += 1
        ee_position = self._finger_center()
        self._update_ee_velocity(ee_position)
        distance = float(np.linalg.norm(self.target_position - ee_position))
        ee_speed = float(np.linalg.norm(self.current_ee_velocity))
        collision = self._has_external_collision()
        remaining_steps = int(self.config.episode_length) - self.step_count
        reward, terminated, truncated, reward_terms, done_reason = self._compute_reward(
            normalized_action,
            previous_action,
            distance,
            collision,
            ee_speed,
            remaining_steps,
        )
        self.episode_return += float(reward)
        if collision:
            self.episode_collision_count += 1
        if done_reason == "success":
            self.episode_success_count += 1
            self.lifetime_success_count += 1
        # 第五步：更新历史动作，构造下一步需要的观测和 info。
        self.previous_action = np.clip(normalized_action, -1.0, 1.0).astype(np.float32)
        observation = self._compose_observation()
        episode_summary = None
        if terminated or truncated:
            episode_summary = {
                "episode_index": int(self.episode_index),
                "episode_steps": int(self.step_count),
                "episode_return": float(self.episode_return),
                "episode_collision_count": int(self.episode_collision_count),
                "episode_success_count": int(self.episode_success_count),
                "lifetime_success_count": int(self.lifetime_success_count),
                "final_distance": float(distance),
                "min_distance": float(self.best_distance if self.best_distance is not None else distance),
                "final_speed": float(ee_speed),
                "curriculum_stage": self.current_stage,
                "done_reason": done_reason,
                "reward_terms": dict(reward_terms),
            }
        info = {
            "episode_index": int(self.episode_index),
            "step_in_episode": int(self.step_count),
            "distance": distance,
            "relative_distance": distance,
            "ee_speed": ee_speed,
            "relative_speed": ee_speed,
            "success": done_reason == "success",
            "collision": collision,
            "runaway": done_reason == "runaway",
            "done_reason": done_reason,
            "curriculum_stage": self.current_stage,
            "success_threshold": self._success_threshold(),
            "episode_return": float(self.episode_return),
            "episode_collision_count": int(self.episode_collision_count),
            "episode_success_count": int(self.episode_success_count),
            "lifetime_success_count": int(self.lifetime_success_count),
            "reward_terms": reward_terms,
            "episode_summary": episode_summary,
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
