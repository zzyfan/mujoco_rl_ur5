#!/usr/bin/env python3
# 主线 UR5 到点任务环境。
#
# 这版文件以最近一版结构完整的实现为基线，
# 再补上后续约定的语义：
# - 成功判定按末端模型自动切换参考点：有夹爪时用两指中点，无夹爪时用 ee_link 原点
# - 目标球碰撞也计入碰撞惩罚
# - reset/step 都输出训练脚本期望的详细 info 字段
# - 课程学习按回合阶段切换 fixed -> local_random -> full_random

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
    ("obs[00:03]", "relative_position_xyz", "目标位置减当前末端参考点位置，单位米；依次对应 x/y/z 三个方向。"),
    ("obs[03:09]", "joint_positions", "UR5 六个关节当前角度，单位弧度；顺序是 joint1 到 joint6。"),
    ("obs[09:15]", "joint_velocities", "UR5 六个关节当前角速度，单位弧度每秒；顺序是 joint1 到 joint6。"),
    ("obs[15:21]", "previous_torque", "上一决策步实际施加到六个关节的力矩，单位牛米；顺序是 joint1 到 joint6。"),
    ("obs[21:24]", "ee_velocity_xyz", "当前末端参考点线速度，单位米每秒；依次对应 x/y/z 三个方向。"),
)


class UR5ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # 这几个类属性会被训练脚本直接读取。
    EPISODE_LENGTH = int(UR5ReachEnvConfig().episode_length)
    FRAME_SKIP = int(UR5ReachEnvConfig().frame_skip)

    def __init__(self, render_mode: str | None = None, config: UR5ReachEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or UR5ReachEnvConfig()
        self.render_mode = render_mode
        if self.render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        xml_path = self._resolve_model_xml()
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

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
        # 夹爪执行器现在允许在 XML 里直接注释掉。
        # 如果对应 actuator 不存在，就返回空数组，环境后面也不会再向这些 ctrl 槽写值。
        self.gripper_actuator_ids = np.array(
            [
                actuator_id
                for name in self.gripper_actuator_names
                if (actuator_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) != -1
            ],
            dtype=np.int32,
        )
        self.gravity_actuator_ids = np.array(
            [
                actuator_id
                for name in self.gravity_actuator_names
                if (actuator_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) != -1
            ],
            dtype=np.int32,
        )

        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)
        self.arm_qvel_adr = np.array([self.model.jnt_dofadr[idx] for idx in self.arm_joint_ids], dtype=np.int32)
        # UR5 的后三个关节主要承担 wrist / 末端姿态调整，单独留一组索引便于抑制手腕旋转抖动。
        self.wrist_joint_indices = np.array([3, 4, 5], dtype=np.int32)

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_follower_link")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_follower_link")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_1")
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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.arm_actuator_ids),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        self.arm_joint_low = np.array(
            [
                self.model.jnt_range[idx][0] if bool(self.model.jnt_limited[idx]) else -np.inf
                for idx in self.arm_joint_ids
            ],
            dtype=np.float32,
        )
        self.arm_joint_high = np.array(
            [
                self.model.jnt_range[idx][1] if bool(self.model.jnt_limited[idx]) else np.inf
                for idx in self.arm_joint_ids
            ],
            dtype=np.float32,
        )

        self.robot_body_ids = self._collect_descendant_body_ids(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        )
        self.ignored_contact_geom_ids = self._collect_ignored_contact_geom_ids()

        self.current_stage = "full_random"
        self.target_stage = 2
        self.episode_index = 0
        self.step_count = 0
        self.success_count = 0
        self.lifetime_success_count = 0
        self.episode_collision_count = 0
        self.episode_success_count = 0
        self.target_contact_steps = 0
        self.target_contact_achieved = False
        self.episode_return = 0.0

        self.target_position = np.zeros(3, dtype=np.float32)
        self.target_pos = self.target_position
        self.previous_action = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)
        self.previous_torque = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)
        self.previous_control = np.zeros(len(self.arm_actuator_ids), dtype=np.float32)
        self.previous_joint_velocities = np.zeros(len(self.arm_joint_ids), dtype=np.float32)
        self.previous_ee_position = np.zeros(3, dtype=np.float32)
        self.current_ee_velocity = np.zeros(3, dtype=np.float32)
        self.previous_distance: float | None = None
        self.best_distance: float | None = None
        self.min_distance: float | None = None
        self.phase_rewards_given: set[float] = set()
        self.last_reward_terms: dict[str, float] = {}

        self.viewer = None
        self._target_viz_added = False
        self._target_geom = None

        self.reset()

    @classmethod
    def observation_schema(cls) -> tuple[tuple[str, str, str], ...]:
        return OBSERVATION_SCHEMA

    def _resolve_model_xml(self) -> Path:
        # 末端夹爪作为一个可选开关处理：
        # - 默认使用原始带夹爪的 XML
        # - 打开开关后切到简化末端版本
        if bool(self.config.disable_gripper_end_effector):
            return project_root() / "assets/robotiq_cxy/lab_env_no_gripper.xml"
        return project_root() / self.config.model_xml

    def _collect_descendant_body_ids(self, root_body_id: int) -> set[int]:
        result: set[int] = set()
        queue = [int(root_body_id)]
        while queue:
            current = queue.pop()
            result.add(current)
            for body_id in range(self.model.nbody):
                if int(self.model.body_parentid[body_id]) == current and body_id != current:
                    queue.append(body_id)
        return result

    def _collect_ignored_contact_geom_ids(self) -> set[int]:
        ignored: set[int] = set()
        for geom_id in range(self.model.ngeom):
            name = self.model.geom(geom_id).name or ""
            if name.startswith("light_"):
                ignored.add(int(geom_id))
        return ignored

    def _home_joint_positions(self) -> np.ndarray:
        q1 = float(self.config.home_joint1)
        q2 = float(self.config.home_joint2)
        q3 = float(self.config.home_joint3)
        return np.array(
            [q1, q2, q3, 1.5 * np.pi - q2 - q3, 1.5 * np.pi, 1.25 * np.pi + q1],
            dtype=np.float32,
        )

    def _ee_reference_point(self) -> np.ndarray:
        # 参考点跟末端模型开关联动：
        # - 带夹爪模型：使用两指中点，贴近抓取中心语义
        # - 简化末端模型：直接使用 ee_link 原点
        if (
            not bool(self.config.disable_gripper_end_effector)
            and int(self.left_finger_body_id) != -1
            and int(self.right_finger_body_id) != -1
        ):
            left = self.data.xpos[self.left_finger_body_id].copy()
            right = self.data.xpos[self.right_finger_body_id].copy()
            return ((left + right) * 0.5).astype(np.float32)
        return self.data.xpos[self.ee_body_id].copy().astype(np.float32)

    def _stage_name(self, episode_index: int) -> str:
        fixed = max(int(self.config.curriculum_fixed_episodes), 0)
        local_random = max(int(self.config.curriculum_local_random_episodes), 0)
        if episode_index < fixed:
            return "fixed"
        if episode_index < fixed + local_random:
            return "local_random"
        return "full_random"

    def _success_threshold(self) -> float:
        if self.current_stage == "fixed":
            return float(self.config.success_threshold_stage1)
        if self.current_stage == "local_random":
            return float(self.config.success_threshold_stage2)
        return float(self.config.success_threshold_stage3)

    def _sample_target_position(self) -> np.ndarray:
        if self.current_stage == "fixed":
            return np.array(
                [self.config.fixed_target_x, self.config.fixed_target_y, self.config.fixed_target_z],
                dtype=np.float32,
            )
        if self.current_stage == "local_random":
            scale = float(np.clip(self.config.curriculum_local_scale, 1e-3, 1.0))
            center = np.array(
                [self.config.fixed_target_x, self.config.fixed_target_y, self.config.fixed_target_z],
                dtype=np.float32,
            )
            x_half = 0.5 * float(self.config.target_x_max - self.config.target_x_min) * scale
            y_half = 0.5 * float(self.config.target_y_max - self.config.target_y_min) * scale
            z_half = 0.5 * float(self.config.target_z_max - self.config.target_z_min) * scale
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
                self.np_random.uniform(self.config.target_x_min, self.config.target_x_max),
                self.np_random.uniform(self.config.target_y_min, self.config.target_y_max),
                self.np_random.uniform(self.config.target_z_min, self.config.target_z_max),
            ],
            dtype=np.float32,
        )

    def _set_target_position(self, target_position: np.ndarray) -> None:
        self.data.qpos[self.target_x_qpos_adr] = float(target_position[0])
        self.data.qpos[self.target_y_qpos_adr] = float(target_position[1])
        self.data.qpos[self.target_z_qpos_adr] = float(target_position[2])
        self.data.qpos[self.target_ball_qpos_adr : self.target_ball_qpos_adr + 4] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def _compose_observation(self) -> np.ndarray:
        ee_position = self._ee_reference_point()
        relative_position = self.target_position - ee_position
        joint_positions = self.data.qpos[self.arm_qpos_adr].copy()
        joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()
        observation = np.concatenate(
            [
                relative_position,
                joint_positions,
                joint_velocities,
                self.previous_torque.copy(),
                self.current_ee_velocity.copy(),
            ]
        )
        return observation.astype(np.float32)

    def _compute_pd_torque(self, action: np.ndarray) -> np.ndarray:
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
        normalized_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.config.control_mode == "torque":
            torque_center = 0.5 * (float(self.config.torque_high) + float(self.config.torque_low))
            torque_scale = 0.5 * (float(self.config.torque_high) - float(self.config.torque_low))
            raw_control = torque_center + torque_scale * normalized_action
        else:
            raw_control = self._compute_pd_torque(normalized_action)
        smoothed = (
            float(self.config.action_smoothing_alpha) * self.previous_control
            + (1.0 - float(self.config.action_smoothing_alpha)) * raw_control
        )
        smoothed = np.clip(smoothed, float(self.config.torque_low), float(self.config.torque_high)).astype(np.float32)
        self.previous_control = smoothed.copy()
        return smoothed

    def _contact_state(self) -> tuple[bool, bool]:
        # 把“碰到目标球”和“其他外部碰撞”分开：
        # - 目标球接触：用于接触奖励与保持成功逻辑
        # - 其他碰撞：继续按原来的碰撞惩罚处理
        target_contact = False
        external_collision = False
        geom_body_ids = self.model.geom_bodyid
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.ignored_contact_geom_ids or geom2 in self.ignored_contact_geom_ids:
                continue
            if geom1 == int(self.target_geom_id) or geom2 == int(self.target_geom_id):
                other_geom = geom2 if geom1 == int(self.target_geom_id) else geom1
                other_body = int(geom_body_ids[other_geom])
                if other_body in self.robot_body_ids:
                    target_contact = True
                continue
            body1 = int(geom_body_ids[geom1])
            body2 = int(geom_body_ids[geom2])
            body1_is_robot = body1 in self.robot_body_ids
            body2_is_robot = body2 in self.robot_body_ids
            if body1_is_robot and body2_is_robot:
                continue
            if body1_is_robot or body2_is_robot:
                external_collision = True
        return target_contact, external_collision

    def _update_ee_velocity(self, ee_position: np.ndarray) -> None:
        dt = max(float(self.model.opt.timestep) * int(self.config.frame_skip), 1e-6)
        self.current_ee_velocity = ((ee_position - self.previous_ee_position) / dt).astype(np.float32)
        self.previous_ee_position = ee_position.astype(np.float32)

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
        distance: float,
        target_contact: bool,
        target_contact_started: bool,
        target_hold_steps: int,
        collision: bool,
        ee_speed: float,
        remaining_steps: int,
    ) -> tuple[float, bool, bool, dict[str, float], str]:
        terminated = False
        truncated = False
        done_reason = "running"

        previous_distance = float(self.previous_distance) if self.previous_distance is not None else float(distance)
        if self.best_distance is None:
            self.best_distance = float(distance)
        best_distance_before = float(self.best_distance)

        # 这里把 reward 拆成显式字典而不是直接累加到一个标量里，
        # 有两个目的：
        # 1. `--print-reward-terms` 时可以直接看到每一项在干什么。
        # 2. 后面单独调 wrist / collision / phase reward 时，不容易把别的项一起改坏。
        reward_terms = {
            "step_penalty": -float(self.config.step_penalty),
            "distance_penalty": -float(self.config.distance_weight) * float(np.sqrt(distance + 1e-8)),
            "progress_reward": 0.0,
            "regress_penalty": 0.0,
            "phase_reward": 0.0,
            "speed_penalty": 0.0,
            "direction_reward": 0.0,
            "action_penalty": -float(self.config.action_l2_penalty) * float(np.mean(np.square(action))),
            "smoothness_penalty": -float(self.config.action_smoothness_penalty)
            * float(np.mean(np.square(action - previous_action))),
            "joint_velocity_penalty": 0.0,
            "target_contact_reward": 0.0,
            "target_hold_success_bonus": 0.0,
            "collision_penalty": 0.0,
            "success_bonus": 0.0,
            "runaway_penalty": 0.0,
        }

        if distance < best_distance_before:
            reward_terms["progress_reward"] = float(self.config.progress_reward_gain) * (best_distance_before - float(distance))
            self.best_distance = float(distance)
        elif distance > previous_distance:
            reward_terms["regress_penalty"] = -float(self.config.regress_penalty_gain) * (float(distance) - previous_distance)

        for threshold, phase_reward in zip(self.config.phase_thresholds, self.config.phase_rewards):
            if distance < float(threshold) and float(threshold) not in self.phase_rewards_given:
                reward_terms["phase_reward"] += float(phase_reward)
                self.phase_rewards_given.add(float(threshold))

        if ee_speed > float(self.config.speed_penalty_threshold):
            reward_terms["speed_penalty"] = -float(self.config.speed_penalty_value)

        to_target = self.target_position - self._ee_reference_point()
        to_target_norm = float(np.linalg.norm(to_target))
        if ee_speed > 1e-6 and to_target_norm > 1e-6:
            movement_direction = self.current_ee_velocity / (ee_speed + 1e-6)
            target_direction = to_target / (to_target_norm + 1e-6)
            direction_cos = max(float(np.dot(movement_direction, target_direction)), 0.0)
            reward_terms["direction_reward"] = float(self.config.direction_reward_gain) * (direction_cos**2)

        joint_velocity_change = np.abs(self.data.qvel[self.arm_qvel_adr] - self.previous_joint_velocities)
        reward_terms["joint_velocity_penalty"] = -float(self.config.joint_velocity_penalty) * float(np.sum(joint_velocity_change))
        # 末端执行器 / wrist 专项奖励与惩罚先整体关闭。
        # 当前先只保留通用的距离、进展、速度、碰撞和成功奖励，
        # 避免 wrist 特定 shaping 继续主导策略学习。
        if target_contact:
            # 接触到目标并保持接触时，每步给一个略高于时间惩罚的小正奖励，
            # 让策略学到“贴住目标并稳住”比马上弹开更划算。
            reward_terms["target_contact_reward"] = float(self.config.target_contact_reward)
            if target_contact_started:
                done_reason = "contact_success"

        if collision:
            reward_terms["collision_penalty"] = -float(self.config.collision_penalty)
            reward_terms["collision_penalty"] += -float(self.config.step_penalty) * float(max(remaining_steps, 0))
            terminated = True
            done_reason = "collision"

        required_hold_steps = max(
            int(round(float(self.config.target_hold_duration_seconds) / max(float(self.model.opt.timestep) * int(self.config.frame_skip), 1e-6))),
            1,
        )
        hold_success = bool(target_contact) and int(target_hold_steps) >= required_hold_steps
        if hold_success:
            reward_terms["target_hold_success_bonus"] = float(self.config.target_hold_success_bonus)
            reward_terms["success_bonus"] = float(self.config.success_bonus)
            reward_terms["success_bonus"] += float(self.config.success_remaining_step_gain) * float(max(remaining_steps, 0))
            if ee_speed < 0.01:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_very_slow)
            elif ee_speed < 0.05:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_slow)
            elif ee_speed < 0.10:
                reward_terms["success_bonus"] += float(self.config.success_speed_bonus_medium)
            terminated = True
            done_reason = "hold_success"

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
        super().reset(seed=seed)
        del options

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_adr] = self._home_joint_positions()
        self.data.qvel[self.arm_qvel_adr] = 0.0

        current_episode_index = self.episode_index + 1
        self.current_stage = self._stage_name(self.episode_index)
        self.target_stage = {"fixed": 0, "local_random": 1, "full_random": 2}[self.current_stage]
        self.target_position = self._sample_target_position()
        self.target_pos = self.target_position
        self._set_target_position(self.target_position)

        self.data.ctrl[:] = 0.0
        if self.gripper_actuator_ids.size > 0:
            self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        if self.gravity_actuator_ids.size > 0:
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_compensation)

        mujoco.mj_forward(self.model, self.data)
        ee_position = self._ee_reference_point()

        self.step_count = 0
        self.previous_action[:] = 0.0
        self.previous_torque[:] = 0.0
        self.previous_control[:] = 0.0
        self.previous_joint_velocities[:] = 0.0
        self.previous_ee_position = ee_position.astype(np.float32)
        self.current_ee_velocity[:] = 0.0
        self.previous_distance = float(np.linalg.norm(self.target_position - ee_position))
        self.best_distance = self.previous_distance
        self.min_distance = self.previous_distance
        self.phase_rewards_given = set()
        self.last_reward_terms = {}
        self.episode_return = 0.0
        self.episode_collision_count = 0
        self.episode_success_count = 0
        self.target_contact_steps = 0
        self.target_contact_achieved = False

        observation = self._compose_observation()
        info = {
            "episode_index": current_episode_index,
            "step_in_episode": 0,
            "target_position": self.target_position.copy(),
            "curriculum_stage": self.current_stage,
            "success_threshold": self._success_threshold(),
            "distance": float(self.previous_distance),
            "relative_distance": float(self.previous_distance),
            "ee_speed": 0.0,
            "relative_speed": 0.0,
            "success": False,
            "target_contact": False,
            "contact_success": False,
            "hold_success": False,
            "target_contact_steps": 0,
            "collision": False,
            "runaway": False,
            "done_reason": "reset",
            "episode_return": 0.0,
            "episode_collision_count": 0,
            "episode_success_count": 0,
            "lifetime_success_count": int(self.lifetime_success_count),
            "reward_terms": {},
            "episode_summary": None,
        }
        self.episode_index = current_episode_index
        return observation, info

    def step(self, action: np.ndarray):
        previous_action = self.previous_action.copy()
        normalized_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        control = self._action_to_control(normalized_action)

        self.data.ctrl[self.arm_actuator_ids] = control
        if self.gripper_actuator_ids.size > 0:
            self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        if self.gravity_actuator_ids.size > 0:
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_compensation)

        self.previous_torque = control.copy()
        self.previous_joint_velocities = self.data.qvel[self.arm_qvel_adr].copy()

        for _ in range(max(int(self.config.frame_skip), 1)):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        ee_position = self._ee_reference_point()
        self._update_ee_velocity(ee_position)
        distance = float(np.linalg.norm(self.target_position - ee_position))
        ee_speed = float(np.linalg.norm(self.current_ee_velocity))
        target_contact, collision = self._contact_state()
        target_contact_started = bool(target_contact) and self.target_contact_steps == 0
        if target_contact:
            self.target_contact_steps += 1
            self.target_contact_achieved = True
        else:
            self.target_contact_steps = 0
        remaining_steps = int(self.config.episode_length) - self.step_count
        reward, terminated, truncated, reward_terms, done_reason = self._compute_reward(
            normalized_action,
            previous_action,
            distance,
            target_contact,
            target_contact_started,
            int(self.target_contact_steps),
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
            self.success_count = self.lifetime_success_count

        self.previous_action = normalized_action.astype(np.float32)
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
                "target_contact_achieved": bool(self.target_contact_achieved),
                "hold_success": done_reason == "hold_success",
                "target_contact_steps": int(self.target_contact_steps),
                "reward_terms": dict(reward_terms),
            }

        info = {
            "episode_index": int(self.episode_index),
            "step_in_episode": int(self.step_count),
            "target_position": self.target_position.copy(),
            "distance": distance,
            "relative_distance": distance,
            "ee_speed": ee_speed,
            "relative_speed": ee_speed,
            "success": done_reason == "hold_success",
            "target_contact": bool(target_contact),
            "contact_success": bool(target_contact_started),
            "hold_success": done_reason == "hold_success",
            "target_contact_steps": int(self.target_contact_steps),
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
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return None
        if self.viewer is None:
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer is not available in this environment.")
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self._target_viz_added = False
        if self.viewer:
            self._add_target_visualization()
            self.viewer.sync()
        return None

    def _add_target_visualization(self) -> None:
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
                    self.target_position,
                    np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    np.array([0.0, 1.0, 0.0, 0.8]),
                )
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                self._target_viz_added = True
        elif self._target_geom is not None:
            geom = scn.geoms[self._target_geom]
            geom.pos = self.target_position

    def close(self) -> None:
        if self.viewer is not None:
            close_fn = getattr(self.viewer, "close", None)
            if callable(close_fn):
                close_fn()
            self.viewer = None
