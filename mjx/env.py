"""MJX environment implementation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mjx.backend import ensure_warp_runtime, jax, jnp, mjx, mjwarp, resolve_physics_backend, wp
else:
    from .backend import ensure_warp_runtime, jax, jnp, mjx, mjwarp, resolve_physics_backend, wp

try:
    import mujoco.viewer as mj_viewer
except Exception:
    mj_viewer = None


@dataclass
class MJXEnvConfig:
    model_xml: str = "assets/zero_arm/zero_with_robotiq_reach.xml"
    frame_skip: int = 1
    max_steps: int = 3000
    success_threshold: float = 0.01
    target_x_min: float = -1.00
    target_x_max: float = -0.62
    target_y_min: float = 0.08
    target_y_max: float = 0.48
    target_z_min: float = 0.10
    target_z_max: float = 0.35
    curriculum_stage1_fixed_episodes: int = 200
    curriculum_stage2_random_episodes: int = 800
    curriculum_stage2_range_scale: float = 0.35
    fixed_target_x: Optional[float] = None
    fixed_target_y: Optional[float] = None
    fixed_target_z: Optional[float] = None
    torque_low: float = -15.0
    torque_high: float = 15.0
    fixed_gripper_ctrl: float = 0.0
    enable_gravity_motors: bool = True
    gravity_ctrl: float = -1.0
    home_pose_mode: str = "direct6"
    home_joint1: float = 0.0
    home_joint2: float = -0.85
    home_joint3: float = 1.35
    home_joint4: float = -0.5
    home_joint5: float = 0.0
    home_joint6: float = 0.0
    step_penalty: float = 0.1
    base_distance_weight: float = 0.8
    improvement_gain: float = 1.0
    regress_gain: float = 0.8
    success_bonus: float = 2000.0
    physics_backend: str = "auto"
    safe_disable_constraints: bool = True
    render_mode: Optional[str] = None
    render_camera_name: str = "workbench_camera"
    viewer_lock_camera: bool = False
    viewer_hide_ui: bool = False
    viewer_fallback_azimuth: float = 135.0
    viewer_fallback_elevation: float = -22.0
    viewer_fallback_distance: float = 1.8
    viewer_fallback_lookat_x: float = -0.18
    viewer_fallback_lookat_y: float = 0.25
    viewer_fallback_lookat_z: float = 0.28


class MJXReachEnv(gym.Env):
    """MJX/Warp-backed reaching environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[MJXEnvConfig] = None) -> None:
        super().__init__()
        self.config = config or MJXEnvConfig()
        self.render_mode = self.config.render_mode
        if self.config.render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode={self.config.render_mode}")

        root = Path(__file__).resolve().parents[1]
        xml_path = root / self.config.model_xml
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing MJX model file: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.physics_backend = resolve_physics_backend(self.config.physics_backend)
        if self.config.safe_disable_constraints:
            safe_flags = (
                mujoco.mjtDisableBit.mjDSBL_CONTACT
                | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                | mujoco.mjtDisableBit.mjDSBL_EQUALITY
                | mujoco.mjtDisableBit.mjDSBL_LIMIT
            )
            self.model.opt.disableflags = int(self.model.opt.disableflags) | int(safe_flags)

        self.data_host = mujoco.MjData(self.model)
        self.model_jax = None
        self.data_jax = None
        self.model_warp = None
        self.data_warp = None
        if self.physics_backend == "warp":
            ensure_warp_runtime()
            self.model_warp = mjwarp.put_model(self.model)
            self.data_warp = mjwarp.put_data(self.model, self.data_host, nworld=1)
        else:
            self.model_jax = mjx.put_model(self.model)
            self.data_jax = mjx.put_data(self.model, self.data_host)
        self.viewer = None
        self.renderer = None
        self._render_error_logged = False
        self._render_camera_id = -1
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

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_follower_link")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_follower_link")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body_1")
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

        self.arm_actuator_ids_jax = jnp.asarray(self.arm_actuator_ids, dtype=jnp.int32) if self.physics_backend == "mjx" else None
        self.gripper_actuator_ids_jax = (
            jnp.asarray(self.gripper_actuator_ids, dtype=jnp.int32) if self.physics_backend == "mjx" else None
        )
        self.gravity_actuator_ids_jax = (
            jnp.asarray(self.gravity_actuator_ids, dtype=jnp.int32) if self.physics_backend == "mjx" else None
        )
        self._step_fn = jax.jit(self._step_frame_skip) if self.physics_backend == "mjx" else None

        self.action_space = spaces.Box(
            low=np.full((6,), self.config.torque_low, dtype=np.float32),
            high=np.full((6,), self.config.torque_high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        mujoco.mj_forward(self.model, self.data_host)
        self.home_qpos = self.data_host.qpos.copy()
        self.home_qvel = self.data_host.qvel.copy()

        self.target_pos = np.zeros(3, dtype=np.float32)
        self.prev_torque = np.zeros(6, dtype=np.float32)
        self.prev_joint_vel = np.zeros(6, dtype=np.float32)
        self.prev_distance: Optional[float] = None
        self.step_count = 0
        self.episode_count = 0
        self.curriculum_stage = "stage1_fixed"

    def _step_frame_skip(self, data_jax, ctrl_jax):
        def body(_i, d):
            return mjx.step(self.model_jax, d.replace(ctrl=ctrl_jax))

        return jax.lax.fori_loop(0, int(self.config.frame_skip), body, data_jax)

    def _step_frame_skip_warp(self) -> None:
        for _ in range(int(self.config.frame_skip)):
            mjwarp.step(self.model_warp, self.data_warp)

    def _set_home_pose(self) -> None:
        self.data_host.qpos[:] = self.home_qpos
        self.data_host.qvel[:] = self.home_qvel
        self.data_host.qpos[self.arm_qpos_adr[0]] = float(self.config.home_joint1)
        self.data_host.qpos[self.arm_qpos_adr[1]] = float(self.config.home_joint2)
        self.data_host.qpos[self.arm_qpos_adr[2]] = float(self.config.home_joint3)
        if self.config.home_pose_mode == "direct6":
            self.data_host.qpos[self.arm_qpos_adr[3]] = float(self.config.home_joint4)
            self.data_host.qpos[self.arm_qpos_adr[4]] = float(self.config.home_joint5)
            self.data_host.qpos[self.arm_qpos_adr[5]] = float(self.config.home_joint6)
        else:
            q1 = self.data_host.qpos[self.arm_qpos_adr[0]]
            q2 = self.data_host.qpos[self.arm_qpos_adr[1]]
            q3 = self.data_host.qpos[self.arm_qpos_adr[2]]
            self.data_host.qpos[self.arm_qpos_adr[3]] = 1.5 * math.pi - q2 - q3
            self.data_host.qpos[self.arm_qpos_adr[4]] = 1.5 * math.pi
            self.data_host.qpos[self.arm_qpos_adr[5]] = 1.25 * math.pi + q1
        self.data_host.qvel[:] = 0.0

    def _set_target_xyz(self, x: float, y: float, z: float) -> None:
        self.data_host.qpos[self.target_x_qpos_adr] = float(x)
        self.data_host.qpos[self.target_y_qpos_adr] = float(y)
        self.data_host.qpos[self.target_z_qpos_adr] = float(z)
        self.data_host.qpos[self.target_ball_qpos_adr : self.target_ball_qpos_adr + 4] = np.array([1.0, 0.0, 0.0, 0.0])

    def _sample_target_pos(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(self.config.target_x_min, self.config.target_x_max),
                self.np_random.uniform(self.config.target_y_min, self.config.target_y_max),
                self.np_random.uniform(self.config.target_z_min, self.config.target_z_max),
            ],
            dtype=np.float32,
        )

    def _sample_target_pos_curriculum(self) -> np.ndarray:
        episode_index = int(self.episode_count)
        stage1_episodes = max(int(self.config.curriculum_stage1_fixed_episodes), 0)
        stage2_episodes = max(int(self.config.curriculum_stage2_random_episodes), 0)
        if episode_index < stage1_episodes:
            x = (
                float(self.config.fixed_target_x)
                if self.config.fixed_target_x is not None
                else 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))
            )
            y = (
                float(self.config.fixed_target_y)
                if self.config.fixed_target_y is not None
                else 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))
            )
            z = (
                float(self.config.fixed_target_z)
                if self.config.fixed_target_z is not None
                else 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))
            )
            self.curriculum_stage = "stage1_fixed"
            return np.array([x, y, z], dtype=np.float32)
        if episode_index < stage1_episodes + stage2_episodes:
            scale = float(np.clip(self.config.curriculum_stage2_range_scale, 1e-3, 1.0))
            x_center = 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))
            x_half = 0.5 * (float(self.config.target_x_max) - float(self.config.target_x_min)) * scale
            y_center = 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))
            y_half = 0.5 * (float(self.config.target_y_max) - float(self.config.target_y_min)) * scale
            z_center = 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))
            z_half = 0.5 * (float(self.config.target_z_max) - float(self.config.target_z_min)) * scale
            self.curriculum_stage = "stage2_small_random"
            return np.array(
                [
                    self.np_random.uniform(x_center - x_half, x_center + x_half),
                    self.np_random.uniform(y_center - y_half, y_center + y_half),
                    self.np_random.uniform(z_center - z_half, z_center + z_half),
                ],
                dtype=np.float32,
            )
        self.curriculum_stage = "stage3_full_random"
        return self._sample_target_pos()

    def _get_ee_pos(self) -> np.ndarray:
        left_pos = self.data_host.xpos[self.left_finger_body_id]
        right_pos = self.data_host.xpos[self.right_finger_body_id]
        return (0.5 * (left_pos + right_pos)).astype(np.float32)

    def _get_target_pos(self) -> np.ndarray:
        return self.data_host.xpos[self.target_body_id].copy().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._get_ee_pos()
        target = self._get_target_pos()
        relative = target - ee_pos
        joint_pos = self.data_host.qpos[self.arm_qpos_adr].copy().astype(np.float32)
        joint_vel = self.data_host.qvel[self.arm_qvel_adr].copy().astype(np.float32)
        ee_vel = self.data_host.cvel[self.ee_body_id][:3].copy().astype(np.float32)
        return np.concatenate([relative, joint_pos, joint_vel, self.prev_torque, ee_vel]).astype(np.float32)

    def _sync_host_from_jax(self) -> None:
        if self.physics_backend == "warp":
            mjwarp.get_data_into(self.data_host, self.model, self.data_warp)
        else:
            mjx.get_data_into(self.data_host, self.model, self.data_jax)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data_host)
        self._set_home_pose()
        self.target_pos = self._sample_target_pos_curriculum()
        self._set_target_xyz(*self.target_pos.tolist())
        mujoco.mj_forward(self.model, self.data_host)
        self.data_host.ctrl[self.arm_actuator_ids] = 0.0
        self.data_host.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        if self.config.enable_gravity_motors:
            self.data_host.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)
        if self.physics_backend == "warp":
            self.data_warp = mjwarp.put_data(self.model, self.data_host, nworld=1)
        else:
            self.data_jax = mjx.put_data(self.model, self.data_host)

        self.prev_torque[:] = 0.0
        self.prev_joint_vel[:] = 0.0
        self.prev_distance = None
        self.step_count = 0
        obs = self._get_obs()
        info = {
            "target_pos": self.target_pos.copy(),
            "curriculum_stage": self.curriculum_stage,
            "episode_index": int(self.episode_count),
        }
        self.episode_count += 1
        return obs, info

    def step(self, action: np.ndarray):
        self.prev_torque = self.data_host.ctrl[self.arm_actuator_ids].copy().astype(np.float32)
        self.prev_joint_vel = self.data_host.qvel[self.arm_qvel_adr].copy().astype(np.float32)

        torque_cmd = np.asarray(action, dtype=np.float32).reshape(6)
        torque_cmd = np.clip(torque_cmd, float(self.config.torque_low), float(self.config.torque_high))
        ctrl = self.data_host.ctrl.copy().astype(np.float32)
        ctrl[self.arm_actuator_ids] = torque_cmd
        ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)
        if self.config.enable_gravity_motors:
            ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)
        self.data_host.ctrl[:] = ctrl

        if self.physics_backend == "warp":
            wp.copy(self.data_warp.ctrl, wp.array(ctrl[None, :].astype(np.float32)))
            self._step_frame_skip_warp()
        else:
            ctrl_jax = self.data_jax.ctrl
            ctrl_jax = ctrl_jax.at[self.arm_actuator_ids_jax].set(jnp.asarray(torque_cmd, dtype=jnp.float32))
            ctrl_jax = ctrl_jax.at[self.gripper_actuator_ids_jax].set(float(self.config.fixed_gripper_ctrl))
            if self.config.enable_gravity_motors:
                ctrl_jax = ctrl_jax.at[self.gravity_actuator_ids_jax].set(float(self.config.gravity_ctrl))
            self.data_jax = self._step_fn(self.data_jax, ctrl_jax)
        self._sync_host_from_jax()
        self.step_count += 1

        obs = self._get_obs()
        relative_pos = obs[0:3]
        distance = float(np.linalg.norm(relative_pos))
        reward = 0.0
        reward -= float(self.config.step_penalty)
        reward += -float(self.config.base_distance_weight) * math.sqrt(max(distance, 1e-9))
        if self.prev_distance is not None:
            if distance < self.prev_distance:
                reward += float(self.config.improvement_gain) * float(self.prev_distance - distance)
            else:
                reward -= float(self.config.regress_gain) * float(distance - self.prev_distance)
        self.prev_distance = distance

        success = distance <= float(self.config.success_threshold)
        terminated = bool(success)
        truncated = self.step_count >= int(self.config.max_steps)
        if success:
            reward += float(self.config.success_bonus)
        info = {
            "distance": distance,
            "success": success,
            "target_pos": self.target_pos.copy(),
            "curriculum_stage": self.curriculum_stage,
            "episode_index": int(self.episode_count),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            try:
                if self.renderer is None:
                    self.renderer = mujoco.Renderer(self.model)
                self.renderer.update_scene(self.data_host)
                return self.renderer.render()
            except Exception as e:
                if not self._render_error_logged:
                    print(f"[MJX render] rgb_array mode failed: {type(e).__name__}: {e}")
                    self._render_error_logged = True
                return None

        if self.viewer is None:
            if mj_viewer is None:
                if not self._render_error_logged:
                    print("[MJX render] mujoco.viewer import failed; check GLFW/OpenGL.")
                    self._render_error_logged = True
                return None
            try:
                if self.config.viewer_hide_ui:
                    try:
                        self.viewer = mj_viewer.launch_passive(
                            self.model,
                            self.data_host,
                            show_left_ui=False,
                            show_right_ui=False,
                        )
                    except TypeError:
                        self.viewer = mj_viewer.launch_passive(self.model, self.data_host)
                else:
                    self.viewer = mj_viewer.launch_passive(self.model, self.data_host)
                self._apply_viewer_camera()
            except Exception as e:
                if not self._render_error_logged:
                    print(f"[MJX render] launch_passive failed: {type(e).__name__}: {e}")
                    self._render_error_logged = True
                return None

        try:
            self.viewer.sync()
        except Exception as e:
            if not self._render_error_logged:
                print(f"[MJX render] viewer.sync failed: {type(e).__name__}: {e}")
                self._render_error_logged = True
            self.viewer = None
        return None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None

    def _apply_viewer_camera(self) -> None:
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
