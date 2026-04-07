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
# 主线实现流程：
# 1. 读取 MuJoCo XML 和网格模型。
# 2. 解析关节、执行器、目标物体和末端相关索引。
# 3. 在 `reset()` 中设置机械臂姿态并采样目标。
# 4. 在 `step()` 中把策略动作映射到控制量，再推进物理。
# 5. 计算观测、奖励、终止条件和调试信息。

from __future__ import annotations

import math
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
    # UR5 到点任务环境。
    #
    # 观测向量由 24 个量组成：
    # 1. 目标相对末端位置：3
    # 2. 六个机械臂关节角：6
    # 3. 六个机械臂关节速度：6
    # 4. 上一步实际施加的关节力矩：6
    # 5. 末端线速度：3

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: UR5ReachEnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        # 如果外部没有显式传配置，就使用主线默认环境配置。
        self.config = config or UR5ReachEnvConfig()
        self.render_mode = render_mode
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        # `model_xml` 使用相对仓库根目录的路径，避免在不同机器上硬编码绝对路径。
        xml_path = project_root() / self.config.model_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        # MuJoCo 负责解析 XML、构造动力学模型和维护运行时状态。
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None

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

        # 这些索引一旦找准，后面的 reset、step 和 reward 都会稳定落在同一组关节上。
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

        # 这些 body / joint / geom 索引会在后续 reset、奖励计算和渲染里反复使用。
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_follower_link")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_follower_link")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body_1")
        self.robot_root_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
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

        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_1")
        if self.target_geom_id >= 0:
            # 原始 XML 里目标球是透明的，这里把它调成可见，方便测试和学习。
            self.model.geom_rgba[self.target_geom_id] = np.array([0.10, 1.00, 0.20, 0.75], dtype=np.float32)

        self.robot_body_ids = self._collect_descendant_body_ids(self.robot_root_body_id)
        self.ignored_contact_geom_ids = self._collect_ignored_contact_geom_ids()

        # 先做一次 forward，拿到初始状态和关节范围。
        mujoco.mj_forward(self.model, self.data)
        self.home_qpos = self.data.qpos.copy()
        self.home_qvel = self.data.qvel.copy()
        self.arm_joint_low = self.model.jnt_range[self.arm_joint_ids, 0].astype(np.float32).copy()
        self.arm_joint_high = self.model.jnt_range[self.arm_joint_ids, 1].astype(np.float32).copy()

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
        self.data.qpos[self.arm_qpos_adr] = arm_pose
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_compensation)

    def _stage_name(self) -> str:
        # 根据当前回合编号判断课程学习阶段。
        if self.episode_index < int(self.config.curriculum_fixed_episodes):
            return "fixed"
        if self.episode_index < int(self.config.curriculum_fixed_episodes + self.config.curriculum_local_random_episodes):
            return "local_random"
        return "full_random"

    def _success_threshold(self) -> float:
        # 课程学习前期用宽松阈值，后期逐渐收紧成功判定。
        if self.current_stage == "fixed":
            return float(self.config.success_threshold_stage1)
        if self.current_stage == "local_random":
            return float(self.config.success_threshold_stage2)
        return float(self.config.success_threshold_stage3)

    def _sample_target_position(self) -> np.ndarray:
        # 按课程学习逻辑采样目标点。
        self.current_stage = self._stage_name()
        if self.current_stage == "fixed":
            # fixed 阶段直接返回一个固定目标点，先让策略学会“朝一个点稳定靠近”。
            return np.array(
                [self.config.fixed_target_x, self.config.fixed_target_y, self.config.fixed_target_z],
                dtype=np.float32,
            )
        if self.current_stage == "local_random":
            # local_random 阶段先以固定目标为中心，再在局部范围里随机扰动。
            center = np.array(
                [self.config.fixed_target_x, self.config.fixed_target_y, self.config.fixed_target_z],
                dtype=np.float32,
            )
            scale = float(np.clip(self.config.curriculum_local_scale, 1e-3, 1.0))
            half_range = 0.5 * np.array(
                [
                    self.config.target_x_max - self.config.target_x_min,
                    self.config.target_y_max - self.config.target_y_min,
                    self.config.target_z_max - self.config.target_z_min,
                ],
                dtype=np.float32,
            ) * scale
            low = np.maximum(
                center - half_range,
                np.array([self.config.target_x_min, self.config.target_y_min, self.config.target_z_min], dtype=np.float32),
            )
            high = np.minimum(
                center + half_range,
                np.array([self.config.target_x_max, self.config.target_y_max, self.config.target_z_max], dtype=np.float32),
            )
            return self.np_random.uniform(low, high).astype(np.float32)
        # full_random 阶段直接在整个工作空间范围里均匀采样。
        return self.np_random.uniform(
            np.array([self.config.target_x_min, self.config.target_y_min, self.config.target_z_min], dtype=np.float32),
            np.array([self.config.target_x_max, self.config.target_y_max, self.config.target_z_max], dtype=np.float32),
        ).astype(np.float32)

    def _set_target_position(self, target_position: np.ndarray) -> None:
        # 把采样到的目标点写入 MuJoCo 的滑动关节。
        #
        # 目标点的位置通过 3 个自由滑块控制，目标球自身姿态则固定成单位四元数。
        self.data.qpos[self.target_x_qpos_adr] = float(target_position[0])
        self.data.qpos[self.target_y_qpos_adr] = float(target_position[1])
        self.data.qpos[self.target_z_qpos_adr] = float(target_position[2])
        self.data.qpos[self.target_ball_qpos_adr : self.target_ball_qpos_adr + 4] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def _finger_center(self) -> np.ndarray:
        # 使用两个指尖中点作为任务空间中的末端执行器位置。
        left = self.data.xpos[self.left_finger_body_id]
        right = self.data.xpos[self.right_finger_body_id]
        return (left + right) * 0.5

    def _compose_observation(self) -> np.ndarray:
        # 拼出策略真正看到的观测向量。
        #
        # 这里故意不把 MuJoCo 全状态直接喂给策略，而是只保留和任务最相关的几类量，
        # 方便学习奖励设计，也能让 notebook 更容易解释每一维的含义。
        # 先分别取出各类观测分量。
        ee_position = self._finger_center()
        relative_position = self.target_position - ee_position
        joint_positions = self.data.qpos[self.arm_qpos_adr].astype(np.float32).copy()
        joint_velocities = self.data.qvel[self.arm_qvel_adr].astype(np.float32).copy()
        previous_torque = self.previous_control.astype(np.float32)
        # 再按固定顺序拼成一个一维向量，供策略网络直接读取。
        observation = np.concatenate(
            [
                relative_position.astype(np.float32),
                joint_positions,
                joint_velocities,
                previous_torque,
                self.current_ee_velocity.astype(np.float32),
            ]
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
        #
        # `reset()` 做的事情包括：
        # 1. 清空 MuJoCo 状态。
        # 2. 设置机械臂初始姿态。
        # 3. 按课程学习规则采样目标。
        # 4. 初始化上一时刻动作、速度和距离缓存。
        super().reset(seed=seed)
        del options
        # 第一步：把 MuJoCo 状态恢复到干净初始状态。
        mujoco.mj_resetData(self.model, self.data)
        self._set_home_pose()

        # 第二步：采样目标点并把目标位置写进物理状态。
        self.target_position = self._sample_target_position()
        self._set_target_position(self.target_position)
        mujoco.mj_forward(self.model, self.data)

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

        # 第四步：生成首个观测，并把调试信息放进 info。
        observation = self._compose_observation()
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
        #
        # `info` 里会保留距离、成功标记、碰撞标记、done_reason 和 reward_terms，
        # 这样测试脚本和 notebook 都能直接复用这些调试信息。
        # 第一步：把外部传入动作转成 NumPy 数组，并保留上一时刻动作给 smoothness reward 用。
        normalized_action = np.asarray(action, dtype=np.float32)
        previous_action = self.previous_action.copy()
        self.previous_joint_velocities = self.data.qvel[self.arm_qvel_adr].astype(np.float32).copy()
        # 第二步：把策略动作映射成真实控制量，并写入 MuJoCo 执行器。
        control = self._action_to_control(normalized_action)
        self.data.ctrl[self.arm_actuator_ids] = control
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_compensation)

        # 第三步：推进若干个物理子步。
        for _ in range(max(int(self.config.frame_skip), 1)):
            mujoco.mj_step(self.model, self.data)

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
        return observation, reward, terminated, truncated, info

    def render(self):
        # 渲染环境。
        #
        # - `human` 模式使用 `mujoco.viewer` 打开交互窗口。
        # - `rgb_array` 模式使用 `mujoco.Renderer` 返回一帧图像数组。
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer is not available in this environment.")
            if self.viewer is None:
                # 第一次 human 渲染时创建被动 viewer，后续只需要同步状态。
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.renderer is None:
            # 第一次 rgb_array 渲染时创建离屏渲染器。
            self.renderer = mujoco.Renderer(self.model, width=960, height=720)
        self.renderer.update_scene(self.data, camera=self.config.render_camera_name)
        return self.renderer.render()

    def close(self) -> None:
        # 释放 viewer 和 renderer 句柄。
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            close_fn = getattr(self.viewer, "close", None)
            if callable(close_fn):
                close_fn()
            self.viewer = None
