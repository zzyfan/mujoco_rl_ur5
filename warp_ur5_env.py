#!/usr/bin/env python3
# Warp 训练线 UR5 到点任务环境。
#
# 本模块把同一个 UR5 到点任务改写成适合 JAX / MJX / Warp 批量训练的形式。
#
# 涉及的主要外部库：
# - `JAX` / `jax.numpy`：负责张量计算和函数式状态更新。
# - `MuJoCo`：负责 XML 模型解析。
# - `mujoco.mjx`：负责把 MuJoCo 模型转换成 MJX / Warp 可执行形式。
# - `mujoco_playground._src.mjx_env`：提供 Brax 兼容的环境基类和状态结构。

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

# 默认配置直接内联在本文件中，避免依赖额外配置模块。
PROJECT_ROOT = Path(__file__).resolve().parent

_DEFAULTS = dict(
    # 模型与仿真步长
    model_xml="assets/robotiq_cxy/lab_env.xml",  # MuJoCo XML 路径（相对当前文件）
    sim_dt=0.02,  # 物理仿真步长（秒）
    frame_skip=1,  # 每个环境 step 内执行的物理子步数
    episode_length=3000,  # 每回合最大步数
    naconmax=128,  # 最大接触数量（MJX buffer）
    naccdmax=128,  # 最大 CCD 接触数量（MJX buffer）
    njmax=64,  # 最大关节约束数量（MJX buffer）
    # 目标点采样范围（工作空间）
    target_x_min=-0.95,
    target_x_max=-0.60,
    target_y_min=0.15,
    target_y_max=0.50,
    target_z_min=0.12,
    target_z_max=0.30,
    # 目标点采样策略
    target_sampling_mode="full_random",  # fixed / small_random / full_random
    target_range_scale=0.35,  # small_random 时的采样范围缩放
    fixed_target_x=None,  # 固定目标 x（None 表示不固定）
    fixed_target_y=None,
    fixed_target_z=None,
    # 成功阈值（不同采样模式可用不同阈值）
    success_threshold=0.010,
    stage1_success_threshold=0.010,  # fixed 模式使用
    stage2_success_threshold=0.010,  # small_random 模式使用
    # 控制与动作映射
    torque_low=-15.0,  # 力矩下界
    torque_high=15.0,  # 力矩上界
    action_target_scale=1.0,  # 动作映射比例（1.0 为满量程）
    action_smoothing_alpha=0.0,  # 动作平滑系数（越大越平滑）
    controller_mode="torque",  # torque / joint_position_delta
    joint_position_delta_scale=0.06,  # 关节位置增量比例（position 模式）
    position_control_kp=55.0,  # 位置控制 P 增益
    position_control_kd=4.0,  # 位置控制 D 增益
    # 观测/奖励模式
    goal_observation=False,  # 是否拼接 achieved/desired goal
    reward_mode="dense",  # dense / sparse
    # 夹爪与重力补偿
    fixed_gripper_ctrl=0.0,  # 固定夹爪控制量
    enable_gravity_motors=True,  # 是否启用重力补偿执行器
    gravity_ctrl=-1.0,  # 重力补偿执行器控制值
    # 初始姿态（前三个关节角）
    home_joint1=0.5183627878423158,
    home_joint2=-1.4835298641951802,
    home_joint3=2.007128639793479,
    # 奖励权重与阈值
    step_penalty=0.10,  # 每步时间惩罚
    base_distance_weight=0.80,  # 距离惩罚权重
    improvement_gain=1.0,  # 靠近目标的增量奖励权重
    regress_gain=0.8,  # 远离目标的惩罚权重
    speed_penalty_threshold=0.5,  # 末端速度惩罚阈值
    speed_penalty_value=0.2,  # 超速惩罚强度
    direction_reward_gain=1.0,  # 朝目标运动方向奖励
    joint_vel_change_penalty_gain=0.03,  # 关节速度变化惩罚
    action_magnitude_penalty_gain=0.0,  # 动作幅度惩罚（默认关闭）
    action_change_penalty_gain=0.0,  # 动作变化惩罚（默认关闭）
    idle_distance_threshold=0.08,  # 停滞惩罚距离阈值
    idle_speed_threshold=0.015,  # 停滞惩罚速度阈值
    idle_penalty_value=0.0,  # 停滞惩罚强度（默认关闭）
    phase_thresholds=(0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002),  # 阶段距离阈值
    phase_rewards=(100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0),  # 阶段奖励
    success_bonus=10000.0,  # 成功一次性奖励
    success_remaining_step_gain=4.0,  # 剩余步数奖励权重
    success_speed_bonus_very_slow=2000.0,  # 成功且速度极慢奖励
    success_speed_bonus_slow=1000.0,  # 成功且速度较慢奖励
    success_speed_bonus_medium=500.0,  # 成功且速度中等奖励
    collision_penalty_value=5000.0,  # 碰撞惩罚
    # 失控判定（仅用于 metrics，不一定终止）
    runaway_distance_threshold=10.0,
    runaway_ee_speed_threshold=50.0,
    runaway_joint_velocity_threshold=100.0,
    runaway_penalty_value=0.0,
)

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


def default_config() -> config_dict.ConfigDict:
    # 将 dataclass 配置转换为 MuJoCo Playground 使用的 `ConfigDict`。
    #
    # Warp 训练线的环境类最终读取的是 `ConfigDict`，因此这里负责把普通 dataclass
    # 转换成 Playground 和 Brax 训练器都能识别的配置对象。
    # 这里额外补上两个字段：
    # - `ctrl_dt`：控制周期，等于单个物理步长乘以 frame_skip
    # - `impl`：明确告诉 MJX / Playground 使用 Warp 后端
    cfg = config_dict.create(**_DEFAULTS)  # 转成 ConfigDict，便于下游统一读取
    cfg.ctrl_dt = cfg.sim_dt * max(int(cfg.frame_skip), 1)  # 控制周期 = sim_dt * frame_skip
    cfg.action_repeat = 1  # Warp 线默认不额外重复动作
    cfg.impl = _WARP_IMPL  # 强制使用 warp 后端
    return cfg


class UR5WarpReachEnv(mjx_env.MjxEnv):
    # 运行在 Warp 兼容 JAX 数组上的 UR5 到点任务环境。

    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides: dict[str, Any] | None = None):
        # 初始化 Warp 环境。
        #
        # 这里会先解析配置，再加载 MuJoCo XML，然后调用 `mjx.put_model(..., impl="warp")`
        # 把模型切换到 Warp 后端。
        # 如果外部已经传了 ConfigDict，就复制一份并展开引用；
        # 否则直接从默认配置开始。
        base_config = config.copy_and_resolve_references() if config is not None else default_config()  # 基础配置
        if config_overrides:
            # `config_overrides` 允许训练脚本在不改默认配置对象的前提下，临时覆盖某些字段。
            base_config.update_from_flattened_dict(config_overrides)
        # 强制固定 Warp 实现和合法的 frame_skip / ctrl_dt，避免外部传入不一致状态。
        base_config.impl = _WARP_IMPL
        base_config.frame_skip = max(int(base_config.frame_skip), 1)
        base_config.ctrl_dt = float(base_config.sim_dt) * int(base_config.frame_skip)
        # `MjxEnv` 基类会保存配置、时间步和若干环境元信息。
        super().__init__(base_config, None)  # 调用 MjxEnv 基类初始化

        # 读取仓库里的 MuJoCo XML，并准备 host 侧 MuJoCo 模型。
        xml_path = (PROJECT_ROOT / self._config.model_xml).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"未找到 Warp GPU 环境模型文件: {xml_path}")
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        # 这里同步 MuJoCo 模型的物理步长，让 host 模型和 Warp 配置保持一致。
        self._mj_model.opt.timestep = self.sim_dt
        # 关键一步：把 MuJoCo host 模型转换成 MJX / Warp 可执行模型。
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # 解析一次性索引和常量缓存。
        #
        # 这一步的目的是把名字查索引、阈值数组转换、mask 构造这些只需做一次的工作
        # 提前处理掉，避免每个 step 里重复做 Python 侧开销。
        # 第一组缓存：机械臂关节、执行器、夹爪和重力补偿执行器的索引。
        self._arm_joint_ids = np.array([self.mj_model.joint(name).id for name in _ARM_JOINT_NAMES], dtype=np.int32)
        self._arm_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _ARM_ACTUATOR_NAMES], dtype=np.int32)
        self._gripper_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _GRIPPER_ACTUATOR_NAMES], dtype=np.int32)
        self._gravity_actuator_ids = np.array([self.mj_model.actuator(name).id for name in _GRAVITY_ACTUATOR_NAMES], dtype=np.int32)
        self._arm_qpos_adr = np.array([self.mj_model.jnt_qposadr[j] for j in self._arm_joint_ids], dtype=np.int32)
        self._arm_qvel_adr = np.array([self.mj_model.jnt_dofadr[j] for j in self._arm_joint_ids], dtype=np.int32)

        # 第二组缓存：末端、目标和机器人根节点的 body 索引。
        self._left_finger_body_id = self.mj_model.body("left_follower_link").id
        self._right_finger_body_id = self.mj_model.body("right_follower_link").id
        self._target_body_id = self.mj_model.body("target_body_1").id
        self._robot_root_body_id = self.mj_model.body("base_link").id

        # 第三组缓存：目标点对应自由关节在 qpos 里的地址，用来直接改目标位置。
        self._target_x_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_x_1").id]
        self._target_y_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_y_1").id]
        self._target_z_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_z_1").id]
        self._target_ball_qpos_adr = self.mj_model.jnt_qposadr[self.mj_model.joint("free_ball_1").id]

        # 用 host 侧 MuJoCo 跑一次前向传播，取出 reset 时需要复用的默认状态。
        data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, data)
        self._home_qpos = jp.asarray(data.qpos.copy())
        self._home_qvel = jp.asarray(data.qvel.copy())
        # 这些是 reset / step 中反复复用的常量张量，提前缓存成 JAX 数组更省事。
        self._zero_action = jp.zeros(6, dtype=jp.float32)
        self._zero_ee_vel = jp.zeros(3, dtype=jp.float32)
        self._phase_thresholds = jp.asarray(self._config.phase_thresholds, dtype=jp.float32)
        self._phase_rewards = jp.asarray(self._config.phase_rewards, dtype=jp.float32)
        self._identity_quat = jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        self._arm_joint_low = jp.asarray(self.mj_model.jnt_range[self._arm_joint_ids, 0], dtype=jp.float32)
        self._arm_joint_high = jp.asarray(self.mj_model.jnt_range[self._arm_joint_ids, 1], dtype=jp.float32)
        # 接触检测相关 mask 会在 reward 和 done 判断里频繁使用。
        self._geom_body_ids = jp.asarray(np.asarray(self.mj_model.geom_bodyid, dtype=np.int32))
        self._robot_body_mask = jp.asarray(self._build_robot_body_mask(), dtype=jp.bool_)
        self._ignored_contact_geom_mask = jp.asarray(self._build_ignored_contact_geom_mask(), dtype=jp.bool_)

    def _build_robot_body_mask(self) -> np.ndarray:
        # 构造机器人 body mask，用来过滤机器人内部自碰撞。
        mask = np.zeros(self.mj_model.nbody, dtype=bool)
        for body_id in range(self.mj_model.nbody):
            # 从当前 body 一直沿着 parent 指针往上找。
            # 如果祖先链上出现机器人根节点，就说明这个 body 属于机器人本体。
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
        # 构造被忽略的 geom mask，例如目标球和装饰灯光。
        mask = np.zeros(self.mj_model.ngeom, dtype=bool)
        for geom_id in range(self.mj_model.ngeom):
            # 这里按 geom 名字过滤，是因为这些物体本身只用于观察，不该被当成训练失败碰撞。
            name = self.mj_model.geom(geom_id).name or ""
            if name.startswith("target_") or name.startswith("light_"):
                mask[geom_id] = True
        return mask

    def _home_arm_pose(self) -> jax.Array:
        # 返回每次 reset 时使用的稳定 UR5 初始姿态。
        #
        # 前 3 个关节角直接来自配置；
        # 后 3 个关节角按几何关系补齐，让机械臂保持一个更稳定、可达空间更好的初始姿态。
        q1 = jp.asarray(self._config.home_joint1, dtype=jp.float32)
        q2 = jp.asarray(self._config.home_joint2, dtype=jp.float32)
        q3 = jp.asarray(self._config.home_joint3, dtype=jp.float32)
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

    def _target_center(self) -> jax.Array:
        # 返回固定目标点；如果没有设置固定目标，则返回工作空间中心。
        #
        # 这样做的意义是：
        # - fixed 模式直接用固定目标
        # - small_random 模式以固定目标或工作空间中心为采样中心
        return jp.asarray(
            [
                self._config.fixed_target_x if self._config.fixed_target_x is not None else 0.5 * (self._config.target_x_min + self._config.target_x_max),
                self._config.fixed_target_y if self._config.fixed_target_y is not None else 0.5 * (self._config.target_y_min + self._config.target_y_max),
                self._config.fixed_target_z if self._config.fixed_target_z is not None else 0.5 * (self._config.target_z_min + self._config.target_z_max),
            ],
            dtype=jp.float32,
        )

    def _sample_target(self, rng: jax.Array) -> jax.Array:
        # 按当前目标采样模式采样目标点。
        #
        # 这样做的好处是 Warp 线和主线在任务定义上保持一致，只是训练后端不同。
        mode = str(self._config.target_sampling_mode).lower()
        if mode == "fixed":
            # fixed 模式直接返回中心点，不做任何随机扰动。
            return self._target_center()
        if mode == "small_random":
            # small_random 模式先算出局部采样半径，再在中心点附近采样。
            scale = float(np.clip(self._config.target_range_scale, 1e-3, 1.0))
            center = self._target_center()
            # JAX 的随机数是函数式风格，所以要先拆出独立子 key。
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
        # full_random 模式直接在完整工作空间内采样。
        rx, ry, rz = jax.random.split(rng, 3)
        return jp.asarray(
            [
                jax.random.uniform(rx, (), minval=self._config.target_x_min, maxval=self._config.target_x_max),
                jax.random.uniform(ry, (), minval=self._config.target_y_min, maxval=self._config.target_y_max),
                jax.random.uniform(rz, (), minval=self._config.target_z_min, maxval=self._config.target_z_max),
            ],
            dtype=jp.float32,
        )

    def _current_success_threshold(self) -> jax.Array:
        # 根据目标采样模式选择对应的成功阈值。
        mode = str(self._config.target_sampling_mode).lower()
        if mode == "fixed":
            return jp.asarray(self._config.stage1_success_threshold, dtype=jp.float32)
        if mode == "small_random":
            return jp.asarray(self._config.stage2_success_threshold, dtype=jp.float32)
        return jp.asarray(self._config.success_threshold, dtype=jp.float32)

    def _scale_policy_action(self, action: jax.Array, prev_torque: jax.Array) -> jax.Array:
        # 把归一化策略动作映射成平滑后的执行器力矩。
        #
        # 处理步骤：
        # 1. 把动作裁到 [-1, 1]
        # 2. 根据正负方向分别映射到力矩上限
        # 3. 和上一步力矩做指数平滑
        # 4. 最后再裁回真实力矩边界
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
        # 把动作解释成关节位置增量，再通过 PD 控制器转换成力矩。
        #
        # 处理步骤：
        # 1. 动作先乘上 `joint_position_delta_scale`，得到目标关节变化量
        # 2. 用上一时刻目标关节位置累积得到新的目标关节位置
        # 3. 用 PD 控制器把“目标位置 - 当前关节状态”转换成力矩
        # 4. 再做一层动作平滑，减少高频抖动
        action = jp.clip(action.astype(jp.float32), -1.0, 1.0)
        delta_scale = jp.asarray(max(float(self._config.joint_position_delta_scale), 1e-4), dtype=jp.float32)
        desired_joint_pos = jp.clip(prev_target_joint_pos + action * delta_scale, self._arm_joint_low, self._arm_joint_high)
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
        # 根据控制模式把动作映射成扭矩。
        if str(self._config.controller_mode).lower() == "joint_position_delta":
            return self._compute_position_delta_torque(action, prev_torque, prev_target_joint_pos, joint_pos, joint_vel)
        return self._scale_policy_action(action, prev_torque), prev_target_joint_pos

    def _build_reset_qpos(self, target_pos: jax.Array) -> jax.Array:
        # 构造 reset 时使用的 qpos，包括机械臂初始姿态和目标点位置。
        #
        # 这里并不是从零拼一个新 qpos，而是先拿 `_home_qpos` 当模板，
        # 再把和当前任务有关的字段覆盖掉。
        qpos = self._home_qpos
        qpos = qpos.at[self._arm_qpos_adr].set(self._home_arm_pose())
        qpos = qpos.at[6:14].set(0.0)
        qpos = qpos.at[self._target_x_qpos_adr].set(target_pos[0])
        qpos = qpos.at[self._target_y_qpos_adr].set(target_pos[1])
        qpos = qpos.at[self._target_z_qpos_adr].set(target_pos[2])
        qpos = qpos.at[self._target_ball_qpos_adr : self._target_ball_qpos_adr + 4].set(self._identity_quat)
        return qpos

    def _compose_ctrl(self, arm_action: jax.Array) -> jax.Array:
        # 把 6 维机械臂动作写入完整 MuJoCo 控制向量。
        #
        # 完整控制向量里除了机械臂，还有夹爪执行器和目标滑块的重力补偿执行器。
        ctrl = jp.zeros(self.mj_model.nu, dtype=jp.float32)
        ctrl = ctrl.at[self._arm_actuator_ids].set(arm_action)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(self._config.fixed_gripper_ctrl)
        if self._config.enable_gravity_motors:
            ctrl = ctrl.at[self._gravity_actuator_ids].set(self._config.gravity_ctrl)
        return ctrl

    def _get_ee_pos(self, data: mjx.Data) -> jax.Array:
        # 把两个指尖中点定义为任务空间中的末端执行器位置。
        left_pos = data.xpos[self._left_finger_body_id]
        right_pos = data.xpos[self._right_finger_body_id]
        return 0.5 * (left_pos + right_pos)

    def _get_target_pos(self, data: mjx.Data) -> jax.Array:
        # 读取目标 body 的世界坐标位置。
        return data.xpos[self._target_body_id]

    def _get_obs(self, data: mjx.Data, prev_torque: jax.Array, ee_vel: jax.Array) -> jax.Array:
        # 构造 Warp 策略使用的扁平观测。
        #
        # 如果 `goal_observation=True`，会在基础观测后面额外拼接 achieved goal 和 desired goal。
        #
        # 基础观测顺序是：
        # 1. 目标相对末端位置
        # 2. 关节角
        # 3. 关节速度
        # 4. 上一步执行的扭矩
        # 5. 当前估计到的末端速度
        ee_pos = self._get_ee_pos(data)
        target_pos = self._get_target_pos(data)
        relative_pos = target_pos - ee_pos
        joint_pos = data.qpos[self._arm_qpos_adr]
        joint_vel = data.qvel[self._arm_qvel_adr]
        base_obs = jp.concatenate([relative_pos, joint_pos, joint_vel, prev_torque, ee_vel]).astype(jp.float32)
        if bool(self._config.goal_observation):
            return jp.concatenate([base_obs, ee_pos.astype(jp.float32), target_pos.astype(jp.float32)]).astype(jp.float32)
        return base_obs

    def _compute_goal_reward(self, achieved_goal: jax.Array, desired_goal: jax.Array, success_threshold: jax.Array) -> jax.Array:
        # 计算 goal-conditioned 风格的 dense / sparse reward。
        #
        # 这里主要给 sparse reward 模式用：
        # - 成功时返回 0
        # - 未成功时返回 -1
        # dense 模式则直接返回负距离。
        distance = jp.linalg.norm(achieved_goal - desired_goal)
        if str(self._config.reward_mode).lower() == "sparse":
            return jp.where(distance <= success_threshold, 0.0, -1.0).astype(jp.float32)
        return (-distance).astype(jp.float32)

    def _contact_count(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
        # 统计接触数量。
        #
        # 返回两个值：
        # - `hazardous_contact_count`：真正会被当作危险碰撞的接触数
        # - `raw_active_contact_count`：MuJoCo 当前所有活跃接触数
        contact = getattr(data, "contact", None)
        efc_address = getattr(contact, "efc_address", None)
        if efc_address is None:
            # 某些状态下没有接触缓存，这时直接返回 0。
            zero = jp.asarray(0.0, dtype=jp.float32)
            return zero, zero
        # `efc_address >= 0` 代表这一条 contact 当前是激活状态。
        active_mask = jp.asarray(efc_address) >= 0
        raw_contacts = jp.sum(active_mask).astype(jp.float32)

        # 下面这一段会把所有活跃接触进一步过滤成“真正危险的碰撞”：
        # 1. 目标球和灯光忽略
        # 2. 机器人内部自碰撞忽略
        # 3. 只保留“机器人 vs 外部物体”的碰撞
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
        # 重置一个 Warp 环境实例。
        #
        # Warp 线使用函数式状态，因此 `reset()` 会直接返回新的 `mjx_env.State`，
        # 而不是像 Gymnasium 一样把状态保存在类实例里。
        # 先拆随机数，再采样目标点。
        rng, target_rng = jax.random.split(rng)  # 拆随机数，保证纯函数式
        target_pos = self._sample_target(target_rng)  # 采样目标点
        # 用目标点构造新的 qpos，并用零动作生成 reset 时的控制量。
        qpos = self._build_reset_qpos(target_pos)  # 构造 reset 姿态
        qvel = self._home_qvel  # 初始速度清零
        ctrl = self._compose_ctrl(self._zero_action)  # 初始控制量
        # `mjx_env.make_data(...)` 会创建适合 MJX / Warp 执行的数据结构。
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
        # 先做一次 forward，让 `xpos`、接触缓存等派生量更新到最新状态。
        data = mjx.forward(self.mjx_model, data)  # 更新派生量

        # reset 后立刻构造首个观测和一组基础指标。
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
        # `info` 保存那些下一步还要继续用的状态缓存。
        # 这些量不会直接喂给策略，但会参与 reward、动作平滑和阶段奖励。
        info = {
            "rng": rng,
            "prev_torque": self._zero_action,
            "prev_joint_vel": jp.zeros(6, dtype=jp.float32),
            "prev_target_joint_pos": qpos[self._arm_qpos_adr],
            "prev_distance": distance,
            "min_distance": distance,
            "phase_hits": jp.zeros(len(self._config.phase_thresholds), dtype=jp.bool_),
            "prev_ee_pos": ee_pos,
            "task_step": jp.asarray(0, dtype=jp.int32),
            "success_threshold": self._current_success_threshold(),
            "runaway_seen": jp.asarray(0.0, dtype=jp.float32),
        }
        return mjx_env.State(data=data, obs=obs, reward=jp.asarray(0.0, dtype=jp.float32), done=jp.asarray(0.0, dtype=jp.float32), metrics=metrics, info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 推进一个 Warp 环境时间步。
        #
        # 实现顺序：
        # 1. 根据控制模式把动作转换成扭矩。
        # 2. 调用 `mjx_env.step` 推进一步动力学。
        # 3. 计算距离、速度、阶段奖励、碰撞和成功判定。
        # 4. 更新 metrics 和 info，返回新的函数式状态。
        # 先读出当前关节状态，为动作映射做准备。
        current_joint_pos = state.data.qpos[self._arm_qpos_adr]  # 当前关节角
        current_joint_vel = state.data.qvel[self._arm_qvel_adr]  # 当前关节速度
        # 这一段把策略动作转换成真正送进物理引擎的控制量。
        torque_cmd, next_target_joint_pos = self._compute_action_torque(
            action,
            state.info["prev_torque"],
            state.info["prev_target_joint_pos"],
            current_joint_pos,
            current_joint_vel,
        )
        ctrl = self._compose_ctrl(torque_cmd)  # 组装完整控制向量
        # `mjx_env.step(...)` 会执行若干个子步，并返回新的函数式物理状态。
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)  # 物理推进

        # 根据新状态计算观测、相对位置、关节速度和末端速度。
        task_step = state.info["task_step"] + 1  # 步数累加
        ee_pos = self._get_ee_pos(data)  # 末端位置
        ee_vel = (ee_pos - state.info["prev_ee_pos"]) / jp.maximum(jp.asarray(self.dt, dtype=jp.float32), 1e-6)  # 数值速度
        obs = self._get_obs(data, state.info["prev_torque"], ee_vel)  # 观测向量
        relative_pos = obs[0:3]  # 相对目标位置
        joint_vel = obs[9:15]  # 关节速度切片
        distance = jp.linalg.norm(relative_pos)  # 距离
        ee_speed = jp.linalg.norm(ee_vel)  # 末端速度

        # 先叠加 dense reward 的各个组成项，再视情况切到 sparse reward。
        #
        # 这里采用与主线相同的 zero 风格奖励结构：
        # - 每步固定时间惩罚
        # - 历史最优距离改进奖励
        # - 远离上一时刻目标时的惩罚
        # - 首次跨越距离阈值时的一次性阶段奖励
        # - 速度过快惩罚、朝向目标运动奖励
        # - 关节速度突变惩罚
        # - 成功奖励、剩余步数奖励和速度稳定奖励
        # - 碰撞大惩罚
        #
        # 第一组是“每步都存在”的基础项：时间惩罚和距离惩罚。
        reward = jp.asarray(-self._config.step_penalty, dtype=jp.float32)
        reward += -self._config.base_distance_weight * jp.sqrt(jp.maximum(distance, 1e-9))

        # 第二组是“是否比之前更好”的增量项：靠近历史最优点时发奖励，远离上一时刻时发惩罚。
        improved = distance < state.info["min_distance"]
        regressed = distance > state.info["prev_distance"]
        reward += jp.where(improved, self._config.improvement_gain * (state.info["min_distance"] - distance), 0.0)
        reward -= jp.where(regressed, self._config.regress_gain * (distance - state.info["prev_distance"]), 0.0)
        next_min_distance = jp.minimum(state.info["min_distance"], distance)

        # 第三组是阶段奖励：第一次穿过某个距离阈值时发一次奖励。
        phase_hits = state.info["phase_hits"]
        new_phase_hits = jp.logical_and(distance < self._phase_thresholds, jp.logical_not(phase_hits))
        reward += jp.sum(new_phase_hits.astype(jp.float32) * self._phase_rewards)
        phase_hits = jp.logical_or(phase_hits, new_phase_hits)

        # 第四组是运动质量项：速度过快惩罚、朝向目标运动奖励、关节速度变化惩罚。
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

        # 第五组是诊断项：如果距离、末端速度或关节速度明显失控，就标记为 runaway。
        # 这里默认不让 runaway 直接结束回合，只把它保存在 metrics 里做观察。
        runaway = jp.logical_or(
            distance > self._config.runaway_distance_threshold,
            jp.logical_or(
                ee_speed > self._config.runaway_ee_speed_threshold,
                jp.max(jp.abs(joint_vel)) > self._config.runaway_joint_velocity_threshold,
            ),
        )
        runaway_seen = jp.logical_or(state.info["runaway_seen"] > 0.5, runaway)

        # 第六组是碰撞项：危险碰撞会带来惩罚，并可能直接结束回合。
        collision_contacts, raw_collision_contacts = self._contact_count(data)
        collision = collision_contacts > 0
        reward -= self._config.collision_penalty_value * collision_contacts

        # 第七组是成功项：达到成功阈值后，根据剩余步数和稳定程度追加奖励。
        current_success_threshold = self._current_success_threshold()
        success = distance <= current_success_threshold
        remaining_steps = jp.maximum(jp.asarray(self._config.episode_length, dtype=jp.int32) - task_step, 0)
        reward += jp.where(success, self._config.success_bonus, 0.0)
        reward += jp.where(success, self._config.success_remaining_step_gain * remaining_steps.astype(jp.float32), 0.0)
        reward += jp.where(success & (ee_speed < 0.01), self._config.success_speed_bonus_very_slow, 0.0)
        reward += jp.where(success & (ee_speed >= 0.01) & (ee_speed < 0.05), self._config.success_speed_bonus_slow, 0.0)
        reward += jp.where(success & (ee_speed >= 0.05) & (ee_speed < 0.1), self._config.success_speed_bonus_medium, 0.0)
        reward += jp.where(
            jp.logical_and(collision, jp.logical_not(success)),
            -self._config.step_penalty * remaining_steps.astype(jp.float32),
            0.0,
        )

        # done 判定由四类条件组成：成功、碰撞、数值异常、超时。
        nan_done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        timeout = task_step >= jp.asarray(self._config.episode_length, dtype=jp.int32)
        done = jp.logical_or(jp.logical_or(success, collision), jp.logical_or(nan_done, timeout)).astype(jp.float32)

        # sparse 模式会覆盖前面的 dense reward，只保留 goal-conditioned 风格奖励。
        if str(self._config.reward_mode).lower() == "sparse":
            reward = self._compute_goal_reward(ee_pos, self._get_target_pos(data), current_success_threshold)

        # `metrics` 用来给训练器和日志系统看，强调“当前结果是什么”。
        metrics = {
            **state.metrics,
            "distance": distance,
            "ee_speed": ee_speed,
            "success": success.astype(jp.float32),
            "collision": collision.astype(jp.float32),
            "runaway": runaway_seen.astype(jp.float32),
            "timeout": timeout.astype(jp.float32),
            "raw_collision_contacts": raw_collision_contacts,
        }
        # `info` 用来给下一步计算继续用，强调“下一步还要带着什么状态往前走”。
        info = {
            **state.info,
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
        # 返回当前环境实际加载的 MuJoCo XML 路径。
        return self._xml_path

    @property
    def action_size(self) -> int:
        # 返回策略动作维度。
        #
        # 当前 UR5 任务只控制 6 个机械臂关节，因此固定为 6。
        return 6

    @property
    def mj_model(self) -> mujoco.MjModel:
        # 暴露底层 MuJoCo host 模型，供渲染和调试复用。
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        # 暴露底层 MJX / Warp 模型，供训练和状态推进使用。
        return self._mjx_model
