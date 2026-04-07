#!/usr/bin/env python3
# mjlab 版 UR5 reach 任务。
#
# 这个模块的目标不是替换掉现有 `Gymnasium + SB3` 主线，
# 而是给当前仓库额外补一条 `mjlab` 训练线：
# 1. 继续复用仓库里的 UR5 MuJoCo 模型和工作空间定义
# 2. 改成 mjlab 的 manager-based 环境组织方式
# 3. 保留当前仓库一贯的中文、教学型注释风格，方便后续继续学习和维护

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import torch

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_root_state_uniform, time_out
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.viewer import ViewerConfig

from ur5_reach_config import UR5ReachEnvConfig, project_root

TASK_ID = "Mjlab-Reach-UR5"

_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_FINGER_BODY_NAMES = ("left_follower_link", "right_follower_link")
_CONTACT_GEOM_NAMES = ("floor", "workbench_down", "wall_1", "wall_2")
_ROBOT_XML = project_root() / "assets/robotiq_cxy/lab_env_mjlab.xml"


def _home_joint_map(cfg: UR5ReachEnvConfig) -> dict[str, float]:
    # 沿用主线环境的 home pose 逻辑，让两条训练线的初始机械臂姿态一致。
    q1 = float(cfg.home_joint1)
    q2 = float(cfg.home_joint2)
    q3 = float(cfg.home_joint3)
    return {
        "joint1": q1,
        "joint2": q2,
        "joint3": q3,
        "joint4": 1.5 * math.pi - q2 - q3,
        "joint5": 1.5 * math.pi,
        "joint6": 1.25 * math.pi + q1,
        "joint7_1": 0.0,
        "joint8_1": 0.0,
        "joint9_1": 0.0,
        "joint10_1": 0.0,
        "joint7_2": 0.0,
        "joint8_2": 0.0,
        "joint9_2": 0.0,
        "joint10_2": 0.0,
    }


def _robot_spec() -> mujoco.MjSpec:
    if not _ROBOT_XML.exists():
        raise FileNotFoundError(f"未找到 mjlab 版 UR5 XML: {_ROBOT_XML}")
    return mujoco.MjSpec.from_file(str(_ROBOT_XML))


def _target_spec() -> mujoco.MjSpec:
    # reach 任务只关心目标点位置，因此这里把目标建成一个独立的 3 自由度滑块实体。
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_body", pos=(0.0, 0.0, 0.025))
    body.add_joint(
        name="target_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(1.0, 0.0, 0.0),
        damping=0.0,
        frictionloss=0.0,
    )
    body.add_joint(
        name="target_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0.0, 1.0, 0.0),
        damping=0.0,
        frictionloss=0.0,
    )
    body.add_joint(
        name="target_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0.0, 0.0, 1.0),
        damping=0.0,
        frictionloss=0.0,
    )
    body.add_geom(
        name="target_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.03,),
        rgba=(0.10, 1.00, 0.20, 0.75),
        contype=0,
        conaffinity=0,
    )
    body.add_site(
        name="target_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.03,),
        rgba=(0.10, 1.00, 0.20, 0.15),
    )
    return spec


def _robot_entity_cfg(cfg: UR5ReachEnvConfig) -> EntityCfg:
    return EntityCfg(
        spec_fn=_robot_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=_ARM_JOINT_NAMES),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=_home_joint_map(cfg),
            joint_vel={".*": 0.0},
        ),
    )


def _target_entity_cfg(cfg: UR5ReachEnvConfig) -> EntityCfg:
    return EntityCfg(
        spec_fn=_target_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "target_x": float(cfg.fixed_target_x),
                "target_y": float(cfg.fixed_target_y),
                "target_z": float(cfg.fixed_target_z),
            },
            joint_vel={".*": 0.0},
        ),
    )


def _ensure_task_state(env: ManagerBasedRlEnv, defaults: UR5ReachEnvConfig | None = None) -> dict[str, torch.Tensor]:
    # mjlab 各个 manager term 彼此独立，所以这里挂一份轻量状态到 env 上，
    # 用来在 reset event、reward 和 termination 之间共享“当前阶段/阈值/目标点”。
    if hasattr(env, "_ur5_reach_state"):
        return env._ur5_reach_state

    cfg = defaults or UR5ReachEnvConfig()
    state = {
        "episode_count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        "stage_idx": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        "success_threshold": torch.full(
            (env.num_envs,),
            float(cfg.success_threshold_stage1),
            dtype=torch.float32,
            device=env.device,
        ),
        "target_pos": torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device),
    }
    env._ur5_reach_state = state
    return state


def _resolve_env_ids(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def _finger_center(
    robot: Entity,
    finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    finger_pos = robot.data.body_link_pos_w[:, finger_cfg.body_ids]
    return finger_pos.mean(dim=1)


def _finger_center_velocity(
    robot: Entity,
    finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    finger_vel = robot.data.body_link_lin_vel_w[:, finger_cfg.body_ids]
    return finger_vel.mean(dim=1)


def _target_world_position(
    target: Entity,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return target.data.body_link_pos_w[:, target_cfg.body_ids].squeeze(1)


def ee_to_target(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_FINGER_BODY_NAMES, preserve_order=True),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target", body_names=("target_body",)),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    target: Entity = env.scene[target_cfg.name]
    return _target_world_position(target, target_cfg) - _finger_center(robot, finger_cfg)


def arm_joint_positions(
    env: ManagerBasedRlEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    robot: Entity = env.scene[arm_cfg.name]
    return robot.data.joint_pos[:, arm_cfg.joint_ids]


def arm_joint_velocities(
    env: ManagerBasedRlEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    robot: Entity = env.scene[arm_cfg.name]
    return robot.data.joint_vel[:, arm_cfg.joint_ids]


def arm_joint_efforts(
    env: ManagerBasedRlEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    robot: Entity = env.scene[arm_cfg.name]
    return robot.data.joint_effort_target[:, arm_cfg.joint_ids]


def ee_linear_velocity(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_FINGER_BODY_NAMES, preserve_order=True),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    return _finger_center_velocity(robot, finger_cfg)


@dataclass(kw_only=True)
class UR5ReachActionCfg(ActionTermCfg):
    # 这类 action term 的输入仍然是 6 维归一化动作，
    # 但内部可以决定按“直接力矩”还是“关节增量 + PD”去解释它。
    control_mode: str = "torque"
    torque_low: float = -15.0
    torque_high: float = 15.0
    joint_delta_scale: float = 0.06
    action_smoothing_alpha: float = 0.0
    position_kp: float = 55.0
    position_kd: float = 4.0
    arm_joint_names: tuple[str, ...] = _ARM_JOINT_NAMES

    def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
        return UR5ReachAction(self, env)


class UR5ReachAction(ActionTerm):
    # 自定义动作项，把原仓库的 `_action_to_control()` 迁移到 mjlab 的 action manager 里。

    def __init__(self, cfg: UR5ReachActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, _names = self._entity.find_joints(cfg.arm_joint_names, preserve_order=True)
        self._arm_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, len(joint_ids), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_low = self._entity.data.default_joint_pos_limits[:, self._arm_joint_ids, 0]
        self._joint_high = self._entity.data.default_joint_pos_limits[:, self._arm_joint_ids, 1]

    @property
    def action_dim(self) -> int:
        return len(self.cfg.arm_joint_names)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        # 这里只负责把策略动作转换成“当前 step 目标力矩”，
        # 真正写入模拟器会在每个 decimation 子步的 `apply_actions()` 中执行。
        normalized = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self._raw_actions[:] = normalized

        if self.cfg.control_mode == "torque":
            torque_center = 0.5 * (float(self.cfg.torque_high) + float(self.cfg.torque_low))
            torque_scale = 0.5 * (float(self.cfg.torque_high) - float(self.cfg.torque_low))
            target_effort = torque_center + torque_scale * normalized
        else:
            current_pos = self._entity.data.joint_pos[:, self._arm_joint_ids]
            current_vel = self._entity.data.joint_vel[:, self._arm_joint_ids]
            desired_pos = torch.clamp(
                current_pos + normalized * float(self.cfg.joint_delta_scale),
                min=self._joint_low,
                max=self._joint_high,
            )
            target_effort = float(self.cfg.position_kp) * (desired_pos - current_pos)
            target_effort = target_effort - float(self.cfg.position_kd) * current_vel

        smoothed = (
            float(self.cfg.action_smoothing_alpha) * self._processed_actions
            + (1.0 - float(self.cfg.action_smoothing_alpha)) * target_effort
        )
        self._processed_actions[:] = torch.clamp(
            smoothed,
            min=float(self.cfg.torque_low),
            max=float(self.cfg.torque_high),
        )

    def apply_actions(self) -> None:
        self._entity.set_joint_effort_target(
            self._processed_actions,
            joint_ids=self._arm_joint_ids,
        )


class SampleReachTarget(ManagerTermBase):
    # reset event：按当前课程阶段为每个并行环境单独采样一个 reach 目标点。

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._cfg = cfg
        self._target: Entity = env.scene["target"]
        target_cfg = cfg.params["target_joint_cfg"]
        assert isinstance(target_cfg, SceneEntityCfg)
        self._target_joint_ids = target_cfg.joint_ids
        self._state = _ensure_task_state(env)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        target_joint_cfg: SceneEntityCfg,
        curriculum_fixed_episodes: int,
        curriculum_local_random_episodes: int,
        curriculum_local_scale: float,
        fixed_target_x: float,
        fixed_target_y: float,
        fixed_target_z: float,
        target_x_min: float,
        target_x_max: float,
        target_y_min: float,
        target_y_max: float,
        target_z_min: float,
        target_z_max: float,
        success_threshold_stage1: float,
        success_threshold_stage2: float,
        success_threshold_stage3: float,
    ) -> None:
        del target_joint_cfg
        env_ids = _resolve_env_ids(env, env_ids)
        episode_count = self._state["episode_count"][env_ids]

        stage_idx = torch.full_like(episode_count, 2)
        fixed_mask = episode_count < int(curriculum_fixed_episodes)
        local_mask = (~fixed_mask) & (
            episode_count
            < int(curriculum_fixed_episodes + curriculum_local_random_episodes)
        )
        stage_idx[fixed_mask] = 0
        stage_idx[local_mask] = 1

        target_pos = torch.empty((len(env_ids), 3), dtype=torch.float32, device=env.device)
        target_pos[:, 0] = float(fixed_target_x)
        target_pos[:, 1] = float(fixed_target_y)
        target_pos[:, 2] = float(fixed_target_z)

        if bool(torch.any(local_mask)):
            center = torch.tensor(
                [fixed_target_x, fixed_target_y, fixed_target_z],
                dtype=torch.float32,
                device=env.device,
            )
            scale = float(max(min(curriculum_local_scale, 1.0), 1.0e-3))
            half_range = 0.5 * torch.tensor(
                [
                    target_x_max - target_x_min,
                    target_y_max - target_y_min,
                    target_z_max - target_z_min,
                ],
                dtype=torch.float32,
                device=env.device,
            ) * scale
            low = torch.maximum(
                center - half_range,
                torch.tensor([target_x_min, target_y_min, target_z_min], dtype=torch.float32, device=env.device),
            )
            high = torch.minimum(
                center + half_range,
                torch.tensor([target_x_max, target_y_max, target_z_max], dtype=torch.float32, device=env.device),
            )
            local_sample = torch.rand((int(local_mask.sum().item()), 3), device=env.device)
            target_pos[local_mask] = low + (high - low) * local_sample

        full_mask = stage_idx == 2
        if bool(torch.any(full_mask)):
            full_low = torch.tensor(
                [target_x_min, target_y_min, target_z_min],
                dtype=torch.float32,
                device=env.device,
            )
            full_high = torch.tensor(
                [target_x_max, target_y_max, target_z_max],
                dtype=torch.float32,
                device=env.device,
            )
            full_sample = torch.rand((int(full_mask.sum().item()), 3), device=env.device)
            target_pos[full_mask] = full_low + (full_high - full_low) * full_sample

        success_threshold = torch.full(
            (len(env_ids),),
            float(success_threshold_stage3),
            dtype=torch.float32,
            device=env.device,
        )
        success_threshold[stage_idx == 0] = float(success_threshold_stage1)
        success_threshold[stage_idx == 1] = float(success_threshold_stage2)

        zero_vel = torch.zeros_like(target_pos)
        self._target.write_joint_position_to_sim(
            target_pos,
            joint_ids=self._target_joint_ids,
            env_ids=env_ids,
        )
        self._target.write_joint_velocity_to_sim(
            zero_vel,
            joint_ids=self._target_joint_ids,
            env_ids=env_ids,
        )

        self._state["target_pos"][env_ids] = target_pos
        self._state["stage_idx"][env_ids] = stage_idx
        self._state["success_threshold"][env_ids] = success_threshold
        self._state["episode_count"][env_ids] = episode_count + 1


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str | tuple[str, ...],
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor_names = (sensor_name,) if isinstance(sensor_name, str) else sensor_name
    collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        data = sensor.data
        if data.force is not None:
            force_mag = torch.norm(data.force, dim=-1)
            collision |= torch.any(force_mag > force_threshold, dim=-1)
        else:
            assert data.found is not None
            collision |= torch.any(data.found > 0, dim=-1)
    return collision


def reached_target(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_FINGER_BODY_NAMES, preserve_order=True),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target", body_names=("target_body",)),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    target: Entity = env.scene[target_cfg.name]
    distance = torch.linalg.norm(
        _target_world_position(target, target_cfg) - _finger_center(robot, finger_cfg),
        dim=-1,
    )
    state = _ensure_task_state(env)
    return distance <= state["success_threshold"]


def runaway_target(
    env: ManagerBasedRlEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_FINGER_BODY_NAMES, preserve_order=True),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target", body_names=("target_body",)),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    target: Entity = env.scene[target_cfg.name]
    distance = torch.linalg.norm(
        _target_world_position(target, target_cfg) - _finger_center(robot, finger_cfg),
        dim=-1,
    )
    return distance > float(threshold)


class UR5ReachReward(ManagerTermBase):
    # 奖励项整体迁移版。
    #
    # 这里没有把每一项拆成十几个独立 reward term，
    # 而是保留原项目“一个地方把 reach 奖励讲清楚”的风格，
    # 方便对照当前 `ur5_reach_env.py`。

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self._cfg = cfg
        self._phase_thresholds = torch.tensor(
            cfg.params["phase_thresholds"],
            dtype=torch.float32,
            device=self.device,
        )
        self._phase_rewards = torch.tensor(
            cfg.params["phase_rewards"],
            dtype=torch.float32,
            device=self.device,
        )
        self._prev_distance = torch.full(
            (self.num_envs,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self._best_distance = torch.full_like(self._prev_distance, float("nan"))
        self._prev_joint_vel = torch.zeros(
            (self.num_envs, len(_ARM_JOINT_NAMES)),
            dtype=torch.float32,
            device=self.device,
        )
        self._phase_awarded = torch.zeros(
            (self.num_envs, len(self._phase_thresholds)),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_distance[env_ids] = float("nan")
        self._best_distance[env_ids] = float("nan")
        self._prev_joint_vel[env_ids] = 0.0
        self._phase_awarded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_cfg: SceneEntityCfg,
        finger_cfg: SceneEntityCfg,
        target_cfg: SceneEntityCfg,
        arm_cfg: SceneEntityCfg,
        collision_sensor_name: str,
        collision_force_threshold: float,
        step_penalty: float,
        distance_weight: float,
        progress_reward_gain: float,
        regress_penalty_gain: float,
        speed_penalty_threshold: float,
        speed_penalty_value: float,
        direction_reward_gain: float,
        action_l2_penalty: float,
        action_smoothness_penalty: float,
        joint_velocity_penalty: float,
        collision_penalty: float,
        success_bonus: float,
        success_remaining_step_gain: float,
        success_speed_bonus_very_slow: float,
        success_speed_bonus_slow: float,
        success_speed_bonus_medium: float,
        runaway_distance_threshold: float,
        runaway_penalty: float,
        phase_thresholds: tuple[float, ...],
        phase_rewards: tuple[float, ...],
    ) -> torch.Tensor:
        del phase_thresholds, phase_rewards
        robot: Entity = env.scene[robot_cfg.name]
        target: Entity = env.scene[target_cfg.name]
        ee_pos = _finger_center(robot, finger_cfg)
        target_pos = _target_world_position(target, target_cfg)
        distance_vec = target_pos - ee_pos
        distance = torch.linalg.norm(distance_vec, dim=-1)
        ee_vel = _finger_center_velocity(robot, finger_cfg)
        ee_speed = torch.linalg.norm(ee_vel, dim=-1)
        current_joint_vel = robot.data.joint_vel[:, arm_cfg.joint_ids]

        prev_distance = torch.where(torch.isnan(self._prev_distance), distance, self._prev_distance)
        best_distance_before = torch.where(torch.isnan(self._best_distance), distance, self._best_distance)

        reward = torch.full_like(distance, -float(step_penalty))
        reward = reward - float(distance_weight) * torch.sqrt(distance + 1.0e-8)

        improved = distance < best_distance_before
        reward = reward + torch.where(
            improved,
            float(progress_reward_gain) * (best_distance_before - distance),
            torch.zeros_like(distance),
        )

        regressed = (~improved) & (distance > prev_distance)
        reward = reward - torch.where(
            regressed,
            float(regress_penalty_gain) * (distance - prev_distance),
            torch.zeros_like(distance),
        )

        crossed = (distance.unsqueeze(1) < self._phase_thresholds.unsqueeze(0)) & (~self._phase_awarded)
        reward = reward + torch.sum(crossed.to(torch.float32) * self._phase_rewards.unsqueeze(0), dim=1)
        self._phase_awarded |= crossed

        reward = reward - torch.where(
            ee_speed > float(speed_penalty_threshold),
            torch.full_like(distance, float(speed_penalty_value)),
            torch.zeros_like(distance),
        )

        distance_dir = distance_vec / torch.clamp(distance.unsqueeze(1), min=1.0e-6)
        speed_dir = ee_vel / torch.clamp(ee_speed.unsqueeze(1), min=1.0e-6)
        direction_cos = torch.clamp(torch.sum(speed_dir * distance_dir, dim=-1), min=0.0)
        direction_mask = (ee_speed > 1.0e-6) & (distance > 1.0e-6)
        reward = reward + torch.where(
            direction_mask,
            float(direction_reward_gain) * torch.square(direction_cos),
            torch.zeros_like(distance),
        )

        reward = reward - float(action_l2_penalty) * torch.mean(torch.square(env.action_manager.action), dim=1)
        reward = reward - float(action_smoothness_penalty) * torch.mean(
            torch.square(env.action_manager.action - env.action_manager.prev_action),
            dim=1,
        )

        joint_velocity_change = torch.sum(torch.abs(current_joint_vel - self._prev_joint_vel), dim=1)
        reward = reward - float(joint_velocity_penalty) * joint_velocity_change

        collision = illegal_contact(
            env,
            sensor_name=collision_sensor_name,
            force_threshold=float(collision_force_threshold),
        )
        remaining_steps = torch.clamp(
            env.max_episode_length - env.episode_length_buf,
            min=0,
        ).to(torch.float32)
        reward = reward + torch.where(
            collision,
            -float(collision_penalty) - float(step_penalty) * remaining_steps,
            torch.zeros_like(distance),
        )

        state = _ensure_task_state(env)
        success = distance <= state["success_threshold"]
        speed_bonus = torch.where(
            ee_speed < 0.01,
            torch.full_like(distance, float(success_speed_bonus_very_slow)),
            torch.where(
                ee_speed < 0.05,
                torch.full_like(distance, float(success_speed_bonus_slow)),
                torch.where(
                    ee_speed < 0.10,
                    torch.full_like(distance, float(success_speed_bonus_medium)),
                    torch.zeros_like(distance),
                ),
            ),
        )
        reward = reward + torch.where(
            success,
            float(success_bonus) + float(success_remaining_step_gain) * remaining_steps + speed_bonus,
            torch.zeros_like(distance),
        )

        runaway = distance > float(runaway_distance_threshold)
        reward = reward - torch.where(
            runaway,
            torch.full_like(distance, float(runaway_penalty)),
            torch.zeros_like(distance),
        )

        self._prev_distance[:] = distance
        self._best_distance[:] = torch.minimum(best_distance_before, distance)
        self._prev_joint_vel[:] = current_joint_vel

        log = env.extras.setdefault("log", {})
        log["Metrics/reach_distance_mean"] = torch.mean(distance).item()
        log["Metrics/ee_speed_mean"] = torch.mean(ee_speed).item()
        log["Metrics/success_threshold_mean"] = torch.mean(state["success_threshold"]).item()
        return reward


def make_mjlab_env_cfg(
    cfg: UR5ReachEnvConfig | None = None,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = cfg or UR5ReachEnvConfig()

    def robot_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot")

    def arm_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES, preserve_order=True)

    def finger_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot", body_names=_FINGER_BODY_NAMES, preserve_order=True)

    def target_body_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("target", body_names=("target_body",))

    def target_joint_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("target", joint_names=("target_x", "target_y", "target_z"))

    actor_terms = {
        "ee_to_target": ObservationTermCfg(
            func=ee_to_target,
            params={
                "robot_cfg": robot_cfg(),
                "finger_cfg": finger_cfg(),
                "target_cfg": target_body_cfg(),
            },
        ),
        "joint_pos": ObservationTermCfg(
            func=arm_joint_positions,
            params={"arm_cfg": arm_cfg()},
        ),
        "joint_vel": ObservationTermCfg(
            func=arm_joint_velocities,
            params={"arm_cfg": arm_cfg()},
        ),
        "joint_effort": ObservationTermCfg(
            func=arm_joint_efforts,
            params={"arm_cfg": arm_cfg()},
        ),
        "ee_velocity": ObservationTermCfg(
            func=ee_linear_velocity,
            params={
                "robot_cfg": robot_cfg(),
                "finger_cfg": finger_cfg(),
            },
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    actions = {
        "arm": UR5ReachActionCfg(
            entity_name="robot",
            control_mode=cfg.control_mode,
            torque_low=cfg.torque_low,
            torque_high=cfg.torque_high,
            joint_delta_scale=cfg.joint_delta_scale,
            action_smoothing_alpha=cfg.action_smoothing_alpha,
            position_kp=cfg.position_kp,
            position_kd=cfg.position_kd,
        )
    }

    collision_sensor_names = tuple(f"robot_env_contact_{geom_name}" for geom_name in _CONTACT_GEOM_NAMES)
    contact_sensor_cfgs = tuple(
        ContactSensorCfg(
            name=sensor_name,
            primary=ContactMatch(
                mode="subtree",
                pattern="base_link",
                entity="robot",
            ),
            secondary=ContactMatch(
                mode="geom",
                pattern=geom_name,
                entity="robot",
            ),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
        )
        for sensor_name, geom_name in zip(collision_sensor_names, _CONTACT_GEOM_NAMES, strict=True)
    )

    events = {
        # 参考 mjlab terrain / scene 的组织方式：
        # fixed-base 实体通过 reset_root_state_uniform 放到各自 env_origin。
        "reset_robot_base": EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_target_base": EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("target"),
            },
        ),
        "sample_target": EventTermCfg(
            func=SampleReachTarget,
            mode="reset",
            params={
                "target_joint_cfg": target_joint_cfg(),
                "curriculum_fixed_episodes": cfg.curriculum_fixed_episodes,
                "curriculum_local_random_episodes": cfg.curriculum_local_random_episodes,
                "curriculum_local_scale": cfg.curriculum_local_scale,
                "fixed_target_x": cfg.fixed_target_x,
                "fixed_target_y": cfg.fixed_target_y,
                "fixed_target_z": cfg.fixed_target_z,
                "target_x_min": cfg.target_x_min,
                "target_x_max": cfg.target_x_max,
                "target_y_min": cfg.target_y_min,
                "target_y_max": cfg.target_y_max,
                "target_z_min": cfg.target_z_min,
                "target_z_max": cfg.target_z_max,
                "success_threshold_stage1": cfg.success_threshold_stage1,
                "success_threshold_stage2": cfg.success_threshold_stage2,
                "success_threshold_stage3": cfg.success_threshold_stage3,
            },
        ),
    }

    rewards = {
        "reach_reward": RewardTermCfg(
            func=UR5ReachReward,
            weight=1.0,
            params={
                "robot_cfg": robot_cfg(),
                "finger_cfg": finger_cfg(),
                "target_cfg": target_body_cfg(),
                "arm_cfg": arm_cfg(),
                "collision_sensor_name": collision_sensor_names,
                "collision_force_threshold": 10.0,
                "step_penalty": cfg.step_penalty,
                "distance_weight": cfg.distance_weight,
                "progress_reward_gain": cfg.progress_reward_gain,
                "regress_penalty_gain": cfg.regress_penalty_gain,
                "speed_penalty_threshold": cfg.speed_penalty_threshold,
                "speed_penalty_value": cfg.speed_penalty_value,
                "direction_reward_gain": cfg.direction_reward_gain,
                "action_l2_penalty": cfg.action_l2_penalty,
                "action_smoothness_penalty": cfg.action_smoothness_penalty,
                "joint_velocity_penalty": cfg.joint_velocity_penalty,
                "collision_penalty": cfg.collision_penalty,
                "success_bonus": cfg.success_bonus,
                "success_remaining_step_gain": cfg.success_remaining_step_gain,
                "success_speed_bonus_very_slow": cfg.success_speed_bonus_very_slow,
                "success_speed_bonus_slow": cfg.success_speed_bonus_slow,
                "success_speed_bonus_medium": cfg.success_speed_bonus_medium,
                "runaway_distance_threshold": cfg.runaway_distance_threshold,
                "runaway_penalty": cfg.runaway_penalty,
                "phase_thresholds": cfg.phase_thresholds,
                "phase_rewards": cfg.phase_rewards,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "collision": TerminationTermCfg(
            func=illegal_contact,
            params={
                "sensor_name": collision_sensor_names,
                "force_threshold": 10.0,
            },
        ),
        "success": TerminationTermCfg(
            func=reached_target,
            params={
                "robot_cfg": robot_cfg(),
                "finger_cfg": finger_cfg(),
                "target_cfg": target_body_cfg(),
            },
        ),
        "runaway": TerminationTermCfg(
            func=runaway_target,
            params={
                "threshold": cfg.runaway_distance_threshold,
                "robot_cfg": robot_cfg(),
                "finger_cfg": finger_cfg(),
                "target_cfg": target_body_cfg(),
            },
        ),
    }

    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "robot": _robot_entity_cfg(cfg),
                "target": _target_entity_cfg(cfg),
            },
            sensors=contact_sensor_cfgs,
            num_envs=1,
            env_spacing=3.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="ee_link",
            distance=1.8,
            elevation=-15.0,
            azimuth=130.0,
        ),
        sim=SimulationCfg(
            nconmax=120,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.02,
                gravity=(0.0, 0.0, 0.0),
                iterations=20,
                ls_iterations=20,
                cone="elliptic",
            ),
        ),
        decimation=max(int(cfg.frame_skip), 1),
        episode_length_s=float(max(int(cfg.episode_length), 1)) * 0.02 * max(int(cfg.frame_skip), 1),
        scale_rewards_by_dt=False,
    )

    if play:
        env_cfg.episode_length_s = 1.0e9
        env_cfg.events["sample_target"].params["curriculum_fixed_episodes"] = 0
        env_cfg.events["sample_target"].params["curriculum_local_random_episodes"] = 0

    return env_cfg


def make_mjlab_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    # 当前先按 mjlab 官方最成熟的 PPO 训练线接入。
    # 这样能尽快把 UR5 任务迁过来，后面如果要继续扩成别的 runner 会更稳。
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="ur5_reach_mjlab",
        run_name="",
        logger="tensorboard",
        clip_actions=1.0,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=4_000,
        upload_model=False,
    )


def register_task() -> None:
    if TASK_ID in list_tasks():
        return
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_mjlab_env_cfg(),
        play_env_cfg=make_mjlab_env_cfg(play=True),
        rl_cfg=make_mjlab_rl_cfg(),
    )


register_task()
