"""Warp GPU reach environment for UR5 and zero_robotiq."""

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
_WARP_IMPL = "warp"


def default_config(robot: str = "ur5_cxy") -> config_dict.ConfigDict:
    """Build the default environment configuration."""
    cfg = config_dict.create(
        robot=robot,
        model_xml="assets/robotiq_cxy/lab_env.xml",
        ctrl_dt=0.02,
        sim_dt=0.02,
        episode_length=3000,
        action_repeat=1,
        frame_skip=1,
        impl=_WARP_IMPL,  # MuJoCo Warp 通过这个字段选择底层物理实现。
        naconmax=128,  # 接触缓存上限会参与静态编译，过大时会明显拖慢首次启动。
        naccdmax=128,  # CCD 接触缓存控制连续碰撞检测的临时接触数量。
        njmax=64,
        success_threshold=0.01,  # 全随机阶段使用的最终成功阈值。
        stage1_success_threshold=0.05,  # 固定目标阶段成功阈值。
        stage2_success_threshold=0.03,  # 小范围随机阶段成功阈值。
        torque_low=-10.0,  # 真实扭矩下限；标准化动作会映射到这个范围。
        torque_high=10.0,  # 真实扭矩上限；标准化动作会映射到这个范围。
        action_target_scale=0.6,  # 标准化动作映射成目标扭矩时的缩放比例。
        action_smoothing_alpha=0.75,  # 动作低通滤波系数，越大越平滑。
        controller_mode="torque",  # 控制模式：`torque` 或 `joint_position_delta`。
        joint_position_delta_scale=0.08,  # `joint_position_delta` 模式下每步允许的关节目标增量。
        position_control_kp=45.0,  # 位置控制模式比例增益。
        position_control_kd=3.0,  # 位置控制模式阻尼增益。
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
        target_sampling_mode="full_random",  # 目标采样模式：`full_random` / `small_random` / `fixed`。
        target_range_scale=0.35,  # 小范围随机模式的目标范围缩放比例。
        fixed_target_x=None,
        fixed_target_y=None,
        fixed_target_z=None,
        step_penalty=0.02,
        base_distance_weight=1.0,
        improvement_gain=100.0,
        regress_gain=70.0,
        speed_penalty_threshold=0.28,
        speed_penalty_value=0.8,
        direction_reward_gain=2.0,
        joint_vel_change_penalty_gain=0.08,
        action_magnitude_penalty_gain=0.02,
        action_change_penalty_gain=0.015,
        idle_distance_threshold=0.08,
        idle_speed_threshold=0.015,
        idle_penalty_value=0.08,
        phase_thresholds=(1.6, 1.3, 1.0, 0.8, 0.6, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002),
        phase_rewards=(20.0, 35.0, 50.0, 70.0, 90.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0),
        success_bonus=10000.0,
        success_remaining_step_gain=4.0,
        success_speed_bonus_very_slow=2000.0,
        success_speed_bonus_slow=1000.0,
        success_speed_bonus_medium=500.0,
        collision_penalty_value=8000.0,
        runaway_distance_threshold=2.0,  # 跑飞阈值仅作为诊断信号，不再直接终止回合。
        runaway_ee_speed_threshold=4.0,
        runaway_joint_velocity_threshold=12.0,
        runaway_penalty_value=3000.0,  # 兼容旧配置字段，当前不再直接写入奖励。
    )
    if robot == "zero_robotiq":
        cfg.model_xml = "assets/zero_arm/zero_with_robotiq_reach.xml"
        cfg.home_pose_mode = "direct6"  # `direct6` 表示 6 个关节角按配置值直接写入 `qpos`。
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


class UR5ReachWarpEnv(mjx_env.MjxEnv):
    """Reach task implemented with MuJoCo Playground and Warp."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        base_config = config.copy_and_resolve_references() if config is not None else default_config()
        if config_overrides:
            base_config.update_from_flattened_dict(config_overrides)
        base_config.impl = _WARP_IMPL
        base_config.frame_skip = max(int(base_config.frame_skip), 1)
        base_config.ctrl_dt = float(base_config.sim_dt) * int(base_config.frame_skip)  # `ctrl_dt` 控制策略动作的生效周期。
        super().__init__(base_config, None)

        xml_path = (_ROOT / self._config.model_xml).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"未找到 Warp GPU 环境模型文件: {xml_path}")
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt  # 训练时统一使用配置里的物理步长。
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
        self._target_body_id = self.mj_model.body("target_body_1").id  # 目标点位置从这个 body 的 `xpos` 读取。
        self._robot_root_body_id = self.mj_model.body("base_link").id

        self._target_x_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_x_1").id]
        self._target_y_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_y_1").id]
        self._target_z_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_z_1").id]
        self._target_ball_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_ball_1").id]

        data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, data)
        self._home_qpos = jp.asarray(data.qpos.copy())
        self._home_qvel = jp.asarray(data.qvel.copy())
        self._zero_action = jp.zeros(6, dtype=jp.float32)  # 策略只输出 6 个关节的扭矩命令。
        self._zero_ee_vel = jp.zeros(3, dtype=jp.float32)
        self._phase_thresholds = jp.asarray(self._config.phase_thresholds, dtype=jp.float32)
        self._phase_rewards = jp.asarray(self._config.phase_rewards, dtype=jp.float32)
        self._identity_quat = jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        self._arm_joint_low = jp.asarray(self.mj_model.jnt_range[self._arm_joint_ids, 0], dtype=jp.float32)
        self._arm_joint_high = jp.asarray(self.mj_model.jnt_range[self._arm_joint_ids, 1], dtype=jp.float32)
        self._geom_body_ids = jp.asarray(np.asarray(self.mj_model.geom_bodyid, dtype=np.int32))
        self._robot_body_mask = jp.asarray(self._build_robot_body_mask(), dtype=jp.bool_)
        self._ignored_contact_geom_mask = jp.asarray(self._build_ignored_contact_geom_mask(), dtype=jp.bool_)

    def _build_robot_body_mask(self) -> np.ndarray:
        """构建机器人 body 掩码，用于忽略机器人内部自接触。"""
        mask = np.zeros(self.mj_model.nbody, dtype=bool)
        for body_id in range(self.mj_model.nbody):
            current = body_id
            while current >= 0:
                if current == self._robot_root_body_id:
                    mask[body_id] = True
                    break
                parent = int(self.mj_model.body_parentid[current])
                if parent == current:
                    break
                current = parent
        return mask

    def _build_ignored_contact_geom_mask(self) -> np.ndarray:
        """构建应忽略的可视化 geom 掩码。"""
        mask = np.zeros(self.mj_model.ngeom, dtype=bool)
        for geom_id in range(self.mj_model.ngeom):
            name = self.mj_model.geom(geom_id).name or ""
            if name.startswith("target_") or name.startswith("light_"):
                mask[geom_id] = True
        return mask

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
        mode = str(self._config.target_sampling_mode).lower()
        if mode == "fixed":
            return self._target_center()
        if mode == "small_random":
            scale = float(np.clip(self._config.target_range_scale, 1e-3, 1.0))
            center = self._target_center()
            rx, ry, rz = jax.random.split(rng, 3)
            x_half = 0.5 * float(self._config.target_x_max - self._config.target_x_min) * scale
            y_half = 0.5 * float(self._config.target_y_max - self._config.target_y_min) * scale
            z_half = 0.5 * float(self._config.target_z_max - self._config.target_z_min) * scale
            return jp.asarray(
                [
                    jax.random.uniform(rx, (), minval=center[0] - x_half, maxval=center[0] + x_half),
                    jax.random.uniform(ry, (), minval=center[1] - y_half, maxval=center[1] + y_half),
                    jax.random.uniform(rz, (), minval=center[2] - z_half, maxval=center[2] + z_half),
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

    def _target_center(self) -> jax.Array:
        """返回固定目标或采样空间中心点。"""
        return jp.asarray(
            [
                self._config.fixed_target_x
                if self._config.fixed_target_x is not None
                else 0.5 * (self._config.target_x_min + self._config.target_x_max),
                self._config.fixed_target_y
                if self._config.fixed_target_y is not None
                else 0.5 * (self._config.target_y_min + self._config.target_y_max),
                self._config.fixed_target_z
                if self._config.fixed_target_z is not None
                else 0.5 * (self._config.target_z_min + self._config.target_z_max),
            ],
            dtype=jp.float32,
        )

    def _current_success_threshold(self) -> jax.Array:
        """按目标采样模式返回当前成功判定阈值。"""
        mode = str(self._config.target_sampling_mode).lower()
        if mode == "fixed":
            return jp.asarray(self._config.stage1_success_threshold, dtype=jp.float32)
        if mode == "small_random":
            return jp.asarray(self._config.stage2_success_threshold, dtype=jp.float32)
        return jp.asarray(self._config.success_threshold, dtype=jp.float32)

    def _scale_policy_action(self, action: jax.Array, prev_torque: jax.Array) -> jax.Array:
        """把标准化动作映射成平滑后的真实扭矩。"""
        action = jp.clip(action.astype(jp.float32), -1.0, 1.0)
        target_scale = jp.asarray(np.clip(self._config.action_target_scale, 1e-3, 1.0), dtype=jp.float32)
        positive_limit = jp.asarray(self._config.torque_high, dtype=jp.float32) * target_scale
        negative_limit = jp.asarray(abs(self._config.torque_low), dtype=jp.float32) * target_scale
        target_torque = jp.where(action >= 0.0, action * positive_limit, action * negative_limit)
        smoothing_alpha = jp.asarray(np.clip(self._config.action_smoothing_alpha, 0.0, 0.999), dtype=jp.float32)
        torque_cmd = smoothing_alpha * prev_torque + (1.0 - smoothing_alpha) * target_torque
        return jp.clip(torque_cmd, self._config.torque_low, self._config.torque_high).astype(jp.float32)

    def _compute_position_delta_torque(
        self,
        action: jax.Array,
        prev_torque: jax.Array,
        prev_target_joint_pos: jax.Array,
        joint_pos: jax.Array,
        joint_vel: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """把动作解释成关节目标增量，再转换成平滑扭矩。"""
        action = jp.clip(action.astype(jp.float32), -1.0, 1.0)
        delta_scale = jp.asarray(max(float(self._config.joint_position_delta_scale), 1e-4), dtype=jp.float32)
        desired_joint_pos = jp.clip(
            prev_target_joint_pos + action * delta_scale,
            self._arm_joint_low,
            self._arm_joint_high,
        )
        kp = jp.asarray(float(self._config.position_control_kp), dtype=jp.float32)
        kd = jp.asarray(float(self._config.position_control_kd), dtype=jp.float32)
        target_torque = kp * (desired_joint_pos - joint_pos) - kd * joint_vel
        smoothing_alpha = jp.asarray(np.clip(self._config.action_smoothing_alpha, 0.0, 0.999), dtype=jp.float32)
        torque_cmd = smoothing_alpha * prev_torque + (1.0 - smoothing_alpha) * target_torque
        torque_cmd = jp.clip(torque_cmd, self._config.torque_low, self._config.torque_high).astype(jp.float32)
        return torque_cmd, desired_joint_pos.astype(jp.float32)

    def _compute_action_torque(
        self,
        action: jax.Array,
        prev_torque: jax.Array,
        prev_target_joint_pos: jax.Array,
        joint_pos: jax.Array,
        joint_vel: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """按控制模式把策略动作映射成真实扭矩。"""
        if str(self._config.controller_mode).lower() == "joint_position_delta":
            return self._compute_position_delta_torque(action, prev_torque, prev_target_joint_pos, joint_pos, joint_vel)
        return self._scale_policy_action(action, prev_torque), prev_target_joint_pos

    def _build_reset_qpos(self, target_pos: jax.Array) -> jax.Array:
        qpos = self._home_qpos
        qpos = qpos.at[self._arm_qpos_adr].set(self._home_arm_pose())
        qpos = qpos.at[6:14].set(0.0)  # 把夹爪展开，避免抓取状态影响到点任务。
        qpos = qpos.at[self._target_x_qpos_adr].set(target_pos[0])
        qpos = qpos.at[self._target_y_qpos_adr].set(target_pos[1])
        qpos = qpos.at[self._target_z_qpos_adr].set(target_pos[2])
        qpos = qpos.at[self._target_ball_qpos_adr : self._target_ball_qpos_adr + 4].set(self._identity_quat)
        return qpos

    def _compose_ctrl(self, arm_action: jax.Array) -> jax.Array:
        ctrl = jp.zeros(self.mj_model.nu, dtype=jp.float32)
        ctrl = ctrl.at[self._arm_actuator_ids].set(arm_action)  # 把 6 维策略动作写入机械臂电机控制槽位。
        ctrl = ctrl.at[self._gripper_actuator_ids].set(self._config.fixed_gripper_ctrl)
        if self._config.enable_gravity_motors:
            ctrl = ctrl.at[self._gravity_actuator_ids].set(self._config.gravity_ctrl)
        return ctrl

    def _get_ee_pos(self, data: mjx.Data) -> jax.Array:
        left_pos = data.xpos[self._left_finger_body_id]
        right_pos = data.xpos[self._right_finger_body_id]
        return 0.5 * (left_pos + right_pos)  # 末端位置定义为两指尖中心点。

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
        return jp.concatenate([relative_pos, joint_pos, joint_vel, prev_torque, ee_vel]).astype(jp.float32)  # 观测向量由相对位置、关节状态和速度组成。

    def _contact_count(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """返回 (危险接触数, 原始有效接触数)。"""
        contact = getattr(data, "contact", None)
        efc_address = getattr(contact, "efc_address", None)
        if efc_address is None:
            zero = jp.asarray(0.0, dtype=jp.float32)
            return zero, zero
        active_mask = jp.asarray(efc_address) >= 0
        raw_contacts = jp.sum(active_mask).astype(jp.float32)

        geom1 = jp.asarray(contact.geom1)
        geom2 = jp.asarray(contact.geom2)
        ignored = self._ignored_contact_geom_mask[geom1] | self._ignored_contact_geom_mask[geom2]
        body1 = self._geom_body_ids[geom1]
        body2 = self._geom_body_ids[geom2]
        body1_is_robot = self._robot_body_mask[body1]
        body2_is_robot = self._robot_body_mask[body2]
        internal_robot_contact = body1_is_robot & body2_is_robot
        hazardous_mask = active_mask & (~ignored) & (~internal_robot_contact) & (body1_is_robot | body2_is_robot)
        hazardous_contacts = jp.sum(hazardous_mask).astype(jp.float32)
        return hazardous_contacts, raw_contacts

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
        data = mjx.forward(self.mjx_model, data)  # `forward` 会刷新 `xpos`、接触和速度等派生量。

        ee_pos = self._get_ee_pos(data)
        distance = jp.linalg.norm(self._get_target_pos(data) - ee_pos)
        obs = self._get_obs(data, self._zero_action, self._zero_ee_vel)
        metrics = {
            "distance": distance,
            "ee_speed": jp.asarray(0.0, dtype=jp.float32),
            "success": jp.asarray(0.0, dtype=jp.float32),
            "collision": jp.asarray(0.0, dtype=jp.float32),
            "runaway": jp.asarray(0.0, dtype=jp.float32),
            "timeout": jp.asarray(0.0, dtype=jp.float32),
            "raw_collision_contacts": jp.asarray(0.0, dtype=jp.float32),
        }
        info = {
            "rng": rng,
            "prev_torque": self._zero_action,
            "prev_joint_vel": jp.zeros(6, dtype=jp.float32),
            "prev_target_joint_pos": qpos[self._arm_qpos_adr],
            "prev_distance": distance,
            "min_distance": distance,
            "phase_hits": jp.zeros(len(self._config.phase_thresholds), dtype=jp.bool_),  # 用布尔向量记录阶段奖励是否已经发放。
            "prev_ee_pos": ee_pos,
            "task_step": jp.asarray(0, dtype=jp.int32),
            "success_threshold": self._current_success_threshold(),
            "runaway_seen": jp.asarray(0.0, dtype=jp.float32),
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
        current_joint_pos = state.data.qpos[self._arm_qpos_adr]
        current_joint_vel = state.data.qvel[self._arm_qvel_adr]
        torque_cmd, next_target_joint_pos = self._compute_action_torque(
            action,
            state.info["prev_torque"],
            state.info["prev_target_joint_pos"],
            current_joint_pos,
            current_joint_vel,
        )
        ctrl = self._compose_ctrl(torque_cmd)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        task_step = state.info["task_step"] + 1
        ee_pos = self._get_ee_pos(data)
        ee_vel = (ee_pos - state.info["prev_ee_pos"]) / jp.maximum(jp.asarray(self.dt, dtype=jp.float32), 1e-6)  # 末端速度由相邻两步位置差分得到。
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
        new_phase_hits = jp.logical_and(distance < self._phase_thresholds, jp.logical_not(phase_hits))  # 每个距离阈值只触发一次奖励。
        reward += jp.sum(new_phase_hits.astype(jp.float32) * self._phase_rewards)
        phase_hits = jp.logical_or(phase_hits, new_phase_hits)

        reward -= jp.where(ee_speed > self._config.speed_penalty_threshold, self._config.speed_penalty_value, 0.0)

        to_target = relative_pos / jp.maximum(distance, 1e-6)
        move_dir = ee_vel / jp.maximum(ee_speed, 1e-6)
        direction_cos = jp.dot(to_target, move_dir)
        reward += jp.square(jp.maximum(direction_cos, 0.0)) * self._config.direction_reward_gain

        joint_vel_change = jp.abs(joint_vel - state.info["prev_joint_vel"])
        reward -= self._config.joint_vel_change_penalty_gain * jp.sum(joint_vel_change)

        reward -= self._config.action_magnitude_penalty_gain * jp.mean(jp.abs(torque_cmd))
        reward -= self._config.action_change_penalty_gain * jp.mean(jp.abs(torque_cmd - state.info["prev_torque"]))
        reward -= jp.where(
            jp.logical_and(distance > self._config.idle_distance_threshold, ee_speed < self._config.idle_speed_threshold),
            self._config.idle_penalty_value,
            0.0,
        )

        # 跑飞在 Warp 线里也只保留成诊断信号：
        # 训练主逻辑回到 success/collision/timeout，避免“防跑飞”盖过 reach 任务本身。
        runaway = jp.logical_or(
            distance > self._config.runaway_distance_threshold,
            jp.logical_or(
                ee_speed > self._config.runaway_ee_speed_threshold,
                jp.max(jp.abs(joint_vel)) > self._config.runaway_joint_velocity_threshold,
            ),
        )
        runaway_seen = jp.logical_or(state.info["runaway_seen"] > 0.5, runaway)

        collision_contacts, raw_collision_contacts = self._contact_count(data)  # 只统计机器人与外部危险几何的接触。
        collision = collision_contacts > 0
        reward -= self._config.collision_penalty_value * collision_contacts

        current_success_threshold = self._current_success_threshold()
        success = distance <= current_success_threshold
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
        timeout = task_step >= jp.asarray(self._config.episode_length, dtype=jp.int32)
        done = jp.logical_or(jp.logical_or(success, collision), jp.logical_or(nan_done, timeout)).astype(jp.float32)  # reach 任务主终止只保留 success/collision/timeout。

        metrics = {
            **state.metrics,  # 保留包装器维护的统计项，避免评估阶段的 metrics 结构变化。
            "distance": distance,
            "ee_speed": ee_speed,
            "success": success.astype(jp.float32),
            "collision": collision.astype(jp.float32),
            "runaway": runaway_seen.astype(jp.float32),
            "timeout": timeout.astype(jp.float32),
            "raw_collision_contacts": raw_collision_contacts,
        }
        info = {
            **state.info,  # 保留包装器写入的统计键，避免 JAX 扫描时字典结构变化。
            "rng": state.info["rng"],
            "prev_torque": torque_cmd,
            "prev_joint_vel": joint_vel,
            "prev_target_joint_pos": next_target_joint_pos,
            "prev_distance": distance,
            "min_distance": next_min_distance,
            "phase_hits": phase_hits,
            "prev_ee_pos": ee_pos,
            "task_step": task_step,
            "success_threshold": current_success_threshold,
            "runaway_seen": runaway_seen.astype(jp.float32),
        }
        return mjx_env.State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 6  # 策略只输出 6 个关节扭矩。

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
