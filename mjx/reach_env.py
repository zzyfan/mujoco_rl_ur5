"""MJX 版 UR5/zero reach 任务。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from mujoco_playground._src import mjx_env


_ROOT = Path(__file__).resolve().parents[1]
_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_ARM_ACTUATOR_NAMES = (
    "joint1_motor",
    "joint2_motor",
    "joint3_motor",
    "joint4_motor",
    "joint5_motor",
    "joint6_motor",
)
_GRIPPER_ACTUATOR_NAMES = ("close_1", "close_2")
_GRAVITY_ACTUATOR_NAMES = ("gravity_1", "gravity_2", "gravity_3", "gravity_4")


def normalize_impl_name(impl: str) -> str:
    """把用户习惯里的 `mjx` 映射到 MuJoCo 当前的 `jax` 实现名。"""
    lowered = str(impl or "warp").strip().lower()
    if lowered == "mjx":
        return "jax"
    if lowered not in {"warp", "jax", "c", "cpp"}:
        raise ValueError(f"不支持的 MJX impl: {impl}")
    return lowered


def default_config(robot: str = "ur5_cxy") -> config_dict.ConfigDict:
    """生成 reach 环境配置。"""
    cfg = config_dict.create(
        robot=robot,
        model_xml="assets/robotiq_cxy/lab_env.xml",
        ctrl_dt=0.02,
        sim_dt=0.02,
        episode_length=3000,
        action_repeat=1,
        frame_skip=1,
        impl="warp",
        naconmax=4096,  # 机械臂 + 夹爪的接触比 Cartpole 重得多，需要显式给接触缓存。
        naccdmax=4096,
        njmax=128,
        success_threshold=0.01,
        torque_low=-15.0,
        torque_high=15.0,
        fixed_gripper_ctrl=0.0,
        enable_gravity_motors=True,
        gravity_ctrl=-1.0,
        home_pose_mode="ur5_coupled",
        home_joint1=0.5183627878423158,
        home_joint2=-1.4835298641951802,
        home_joint3=2.007128639793479,
        home_joint4=0.0,
        home_joint5=0.0,
        home_joint6=0.0,
        target_x_min=-0.95,
        target_x_max=-0.60,
        target_y_min=0.15,
        target_y_max=0.50,
        target_z_min=0.12,
        target_z_max=0.30,
        fixed_target_x=None,
        fixed_target_y=None,
        fixed_target_z=None,
        step_penalty=0.1,
        base_distance_weight=0.8,
        improvement_gain=1.0,
        regress_gain=0.8,
        speed_penalty_threshold=0.5,
        speed_penalty_value=0.2,
        direction_reward_gain=1.0,
        joint_vel_change_penalty_gain=0.03,
        phase_thresholds=(0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002),
        phase_rewards=(100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0),
        success_bonus=10000.0,
        success_remaining_step_gain=4.0,
        success_speed_bonus_very_slow=2000.0,
        success_speed_bonus_slow=1000.0,
        success_speed_bonus_medium=500.0,
        collision_penalty_value=5000.0,
    )
    if robot == "zero_robotiq":
        cfg.model_xml = "assets/zero_arm/zero_with_robotiq_reach.xml"
        cfg.home_pose_mode = "direct6"  # zero 机械臂没有 UR5 那套 wrist 耦合初始姿态。
        cfg.home_joint1 = 0.0
        cfg.home_joint2 = -0.85
        cfg.home_joint3 = 1.35
        cfg.home_joint4 = -0.5
        cfg.home_joint5 = 0.0
        cfg.home_joint6 = 0.0
        cfg.target_x_min = -1.00
        cfg.target_x_max = -0.62
        cfg.target_y_min = 0.08
        cfg.target_y_max = 0.48
        cfg.target_z_min = 0.10
        cfg.target_z_max = 0.35
    return cfg


class UR5ReachMjxEnv(mjx_env.MjxEnv):
    """按 MuJoCo Playground 结构实现的 UR5/zero reach 环境。"""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        base_config = config.copy_and_resolve_references() if config is not None else default_config()
        if config_overrides:
            base_config.update_from_flattened_dict(config_overrides)
        base_config.impl = normalize_impl_name(base_config.impl)
        base_config.frame_skip = max(int(base_config.frame_skip), 1)
        base_config.ctrl_dt = float(base_config.sim_dt) * int(base_config.frame_skip)  # 让 MJX 的 dt 语义和 classic 一致。
        super().__init__(base_config, None)

        xml_path = (_ROOT / self._config.model_xml).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"未找到 MJX reach 模型文件: {xml_path}")
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt  # MJX 里直接把 XML timestep 设成训练使用的 sim_dt。
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._arm_joint_ids = np.array([self.mj_model.joint(name).id for name in _ARM_JOINT_NAMES], dtype=np.int32)
        self._arm_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _ARM_ACTUATOR_NAMES], dtype=np.int32)
        self._gripper_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _GRIPPER_ACTUATOR_NAMES], dtype=np.int32)
        self._gravity_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _GRAVITY_ACTUATOR_NAMES], dtype=np.int32)
        self._arm_qpos_adr = np.array([self.mj_model.jnt_qposadr[j] for j in self._arm_joint_ids], dtype=np.int32)
        self._arm_qvel_adr = np.array([self.mj_model.jnt_dofadr[j] for j in self._arm_joint_ids], dtype=np.int32)

        self._left_finger_body_id = self.mj_model.body("left_follower_link").id
        self._right_finger_body_id = self.mj_model.body("right_follower_link").id
        self._target_body_id = self.mj_model.body("target_body_1").id  # classic 也是用这个 body 作为 reach 目标。

        self._target_x_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_x_1").id]
        self._target_y_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_y_1").id]
        self._target_z_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_z_1").id]
        self._target_ball_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_ball_1").id]

        data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, data)
        self._home_qpos = jp.asarray(data.qpos.copy())
        self._home_qvel = jp.asarray(data.qvel.copy())
        self._zero_action = jp.zeros(6, dtype=jp.float32)  # 策略只管 6 个关节扭矩，夹爪和重力电机由环境补齐。
        self._zero_ee_vel = jp.zeros(3, dtype=jp.float32)
        self._phase_thresholds = jp.asarray(self._config.phase_thresholds, dtype=jp.float32)
        self._phase_rewards = jp.asarray(self._config.phase_rewards, dtype=jp.float32)
        self._identity_quat = jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)

    def _home_arm_pose(self) -> jax.Array:
        q1 = jp.asarray(self._config.home_joint1, dtype=jp.float32)
        q2 = jp.asarray(self._config.home_joint2, dtype=jp.float32)
        q3 = jp.asarray(self._config.home_joint3, dtype=jp.float32)
        if self._config.home_pose_mode == "direct6":
            return jp.asarray(
                [
                    self._config.home_joint1,
                    self._config.home_joint2,
                    self._config.home_joint3,
                    self._config.home_joint4,
                    self._config.home_joint5,
                    self._config.home_joint6,
                ],
                dtype=jp.float32,
            )
        return jp.asarray(
            [
                q1,
                q2,
                q3,
                1.5 * jp.pi - q2 - q3,
                1.5 * jp.pi,
                1.25 * jp.pi + q1,
            ],
            dtype=jp.float32,
        )

    def _sample_target(self, rng: jax.Array) -> jax.Array:
        if (
            self._config.fixed_target_x is not None
            and self._config.fixed_target_y is not None
            and self._config.fixed_target_z is not None
        ):
            return jp.asarray(
                [
                    self._config.fixed_target_x,
                    self._config.fixed_target_y,
                    self._config.fixed_target_z,
                ],
                dtype=jp.float32,
            )
        rx, ry, rz = jax.random.split(rng, 3)
        return jp.asarray(
            [
                jax.random.uniform(
                    rx, (), minval=self._config.target_x_min, maxval=self._config.target_x_max
                ),
                jax.random.uniform(
                    ry, (), minval=self._config.target_y_min, maxval=self._config.target_y_max
                ),
                jax.random.uniform(
                    rz, (), minval=self._config.target_z_min, maxval=self._config.target_z_max
                ),
            ],
            dtype=jp.float32,
        )

    def _build_reset_qpos(self, target_pos: jax.Array) -> jax.Array:
        qpos = self._home_qpos
        qpos = qpos.at[self._arm_qpos_adr].set(self._home_arm_pose())
        qpos = qpos.at[6:14].set(0.0)  # reach 任务里夹爪始终张开，减少抓取接触带来的额外变量。
        qpos = qpos.at[self._target_x_qpos_adr].set(target_pos[0])
        qpos = qpos.at[self._target_y_qpos_adr].set(target_pos[1])
        qpos = qpos.at[self._target_z_qpos_adr].set(target_pos[2])
        qpos = qpos.at[self._target_ball_qpos_adr : self._target_ball_qpos_adr + 4].set(self._identity_quat)
        return qpos

    def _compose_ctrl(self, arm_action: jax.Array) -> jax.Array:
        ctrl = jp.zeros(self.mj_model.nu, dtype=jp.float32)
        ctrl = ctrl.at[self._arm_actuator_ids].set(arm_action)  # 前 6 维策略动作映射到真实 12 维 ctrl 里的 arm 部分。
        ctrl = ctrl.at[self._gripper_actuator_ids].set(self._config.fixed_gripper_ctrl)
        if self._config.enable_gravity_motors:
            ctrl = ctrl.at[self._gravity_actuator_ids].set(self._config.gravity_ctrl)
        return ctrl

    def _get_ee_pos(self, data: mjx.Data) -> jax.Array:
        left_pos = data.xpos[self._left_finger_body_id]
        right_pos = data.xpos[self._right_finger_body_id]
        return 0.5 * (left_pos + right_pos)  # 和 classic 一样，末端位置取两指中心点而不是 wrist。

    def _get_target_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._target_body_id]

    def _get_obs(
        self,
        data: mjx.Data,
        prev_torque: jax.Array,
        ee_vel: jax.Array,
    ) -> jax.Array:
        ee_pos = self._get_ee_pos(data)
        target_pos = self._get_target_pos(data)
        relative_pos = target_pos - ee_pos
        joint_pos = data.qpos[self._arm_qpos_adr]
        joint_vel = data.qvel[self._arm_qvel_adr]
        return jp.concatenate([relative_pos, joint_pos, joint_vel, prev_torque, ee_vel]).astype(jp.float32)  # 24 维观测向量。

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, target_rng = jax.random.split(rng)
        target_pos = self._sample_target(target_rng)
        qpos = self._build_reset_qpos(target_pos)
        qvel = self._home_qvel
        ctrl = self._compose_ctrl(self._zero_action)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)  # reset 后先 forward 一次，保证 xpos/qvel 等派生量可读。

        ee_pos = self._get_ee_pos(data)
        distance = jp.linalg.norm(self._get_target_pos(data) - ee_pos)
        obs = self._get_obs(data, self._zero_action, self._zero_ee_vel)
        metrics = {
            "distance": distance,
            "ee_speed": jp.asarray(0.0, dtype=jp.float32),
            "success": jp.asarray(0.0, dtype=jp.float32),
            "collision": jp.asarray(0.0, dtype=jp.float32),
        }
        info = {
            "rng": rng,
            "prev_torque": self._zero_action,
            "prev_joint_vel": jp.zeros(6, dtype=jp.float32),
            "prev_distance": distance,
            "min_distance": distance,
            "phase_hits": jp.zeros(len(self._config.phase_thresholds), dtype=jp.bool_),  # MJX 里用 info 显式保存阶段奖励命中状态。
            "prev_ee_pos": ee_pos,
            "task_step": jp.asarray(0, dtype=jp.int32),
        }
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.asarray(0.0, dtype=jp.float32),
            done=jp.asarray(0.0, dtype=jp.float32),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        torque_cmd = jp.clip(action, self._config.torque_low, self._config.torque_high).astype(jp.float32)  # 先做和 classic 一样的扭矩限幅。
        ctrl = self._compose_ctrl(torque_cmd)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        task_step = state.info["task_step"] + 1
        ee_pos = self._get_ee_pos(data)
        ee_vel = (ee_pos - state.info["prev_ee_pos"]) / jp.maximum(jp.asarray(self.dt, dtype=jp.float32), 1e-6)  # 中心点有限差分速度。
        obs = self._get_obs(data, state.info["prev_torque"], ee_vel)
        relative_pos = obs[0:3]
        joint_vel = obs[9:15]
        distance = jp.linalg.norm(relative_pos)
        ee_speed = jp.linalg.norm(ee_vel)

        reward = jp.asarray(-self._config.step_penalty, dtype=jp.float32)
        reward += -self._config.base_distance_weight * jp.sqrt(jp.maximum(distance, 1e-9))  # 距离惩罚。

        improved = distance < state.info["min_distance"]
        regressed = distance > state.info["prev_distance"]
        reward += jp.where(
            improved,
            self._config.improvement_gain * (state.info["min_distance"] - distance),
            0.0,
        )
        reward -= jp.where(
            regressed,
            self._config.regress_gain * (distance - state.info["prev_distance"]),
            0.0,
        )
        next_min_distance = jp.minimum(state.info["min_distance"], distance)

        phase_hits = state.info["phase_hits"]
        new_phase_hits = jp.logical_and(distance < self._phase_thresholds, jp.logical_not(phase_hits))  # 阶段奖励只发一次，避免卡阈值刷分。
        reward += jp.sum(new_phase_hits.astype(jp.float32) * self._phase_rewards)
        phase_hits = jp.logical_or(phase_hits, new_phase_hits)

        reward -= jp.where(ee_speed > self._config.speed_penalty_threshold, self._config.speed_penalty_value, 0.0)

        to_target = relative_pos / jp.maximum(distance, 1e-6)
        move_dir = ee_vel / jp.maximum(ee_speed, 1e-6)
        direction_cos = jp.dot(to_target, move_dir)
        reward += jp.square(jp.maximum(direction_cos, 0.0)) * self._config.direction_reward_gain

        joint_vel_change = jp.abs(joint_vel - state.info["prev_joint_vel"])
        reward -= self._config.joint_vel_change_penalty_gain * jp.sum(joint_vel_change)

        collision_contacts = jp.asarray(data.ncon, dtype=jp.float32)  # 接触点数。
        collision = collision_contacts > 0
        reward -= self._config.collision_penalty_value * collision_contacts

        success = distance <= self._config.success_threshold
        remaining_steps = jp.maximum(jp.asarray(self._config.episode_length, dtype=jp.int32) - task_step, 0)
        reward += jp.where(success, self._config.success_bonus, 0.0)
        reward += jp.where(success, self._config.success_remaining_step_gain * remaining_steps.astype(jp.float32), 0.0)
        reward += jp.where(success & (ee_speed < 0.01), self._config.success_speed_bonus_very_slow, 0.0)
        reward += jp.where(
            success & (ee_speed >= 0.01) & (ee_speed < 0.05),
            self._config.success_speed_bonus_slow,
            0.0,
        )
        reward += jp.where(
            success & (ee_speed >= 0.05) & (ee_speed < 0.1),
            self._config.success_speed_bonus_medium,
            0.0,
        )
        reward += jp.where(
            jp.logical_and(collision, jp.logical_not(success)),
            -self._config.step_penalty * remaining_steps.astype(jp.float32),
            0.0,
        )

        nan_done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = jp.logical_or(jp.logical_or(success, collision), nan_done).astype(jp.float32)  # 成功、碰撞、数值炸掉都算终止。

        metrics = {
            "distance": distance,
            "ee_speed": ee_speed,
            "success": success.astype(jp.float32),
            "collision": collision.astype(jp.float32),
        }
        info = {
            "rng": state.info["rng"],
            "prev_torque": torque_cmd,
            "prev_joint_vel": joint_vel,
            "prev_distance": distance,
            "min_distance": next_min_distance,
            "phase_hits": phase_hits,
            "prev_ee_pos": ee_pos,
            "task_step": task_step,
        }
        return mjx_env.State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 6  # 策略动作空间仍然只暴露 6 个 arm torque。

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
