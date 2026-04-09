import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

class RobotArmEnv(gym.Env):
    """
    机械臂末端跟踪环境，使用MuJoCo作为物理引擎
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        """
        初始化机械臂环境
        
        参数:
        - render_mode: 渲染模式
        """
        # 加载 MuJoCo 模型与数据对象。
        self.model = mujoco.MjModel.from_xml_path("robot_arm_mujoco.xml")
        self.data = mujoco.MjData(self.model)
        
        # 获取关节数量
        self.nu = self.model.nu
        self.nq = self.model.nq
        
        # 动作空间：6 维关节力矩，单位 N*m。
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(self.nu,), dtype=np.float32
        )
        
        # 观测空间：24 维。
        # 维度定义：
        # 1) relative_pos(3) + 2) qpos(6) + 3) qvel(6) + 4) prev_torque(6) + 5) ee_vel(3)
        state_dim = 3 + 6 + 6 + 6 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 初始化目标点。真正训练中会在 reset() 里随机化。
        self.target_pos = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.previous_distance = None
        
        # 记录上一时刻控制量和速度，用于构建观测与平滑惩罚项。
        self.previous_torque = np.zeros(self.nu)
        self.previous_joint_velocities = np.zeros(self.nu)

        # 训练统计字段：用于把日志统一成“回合结束汇总”。
        self.episode_index = 0
        self.lifetime_success_count = 0
        self.episode_return = 0.0
        self.episode_distance_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_collision_count = 0
        self.episode_success_count = 0
        
        # 渲染器句柄（懒加载）。
        self.render_mode = render_mode
        self.viewer = None
                
        # 构造后先 reset 一次，保证状态变量完整初始化。
        self.reset()

    def _get_state(self):
        """
        获取当前状态
        """
        # 末端 site 位置。
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = self.data.site_xpos[ee_site_id].copy()

        # 目标相对位移。
        relative_pos = self.target_pos - ee_pos

        # 关节角与角速度。
        joint_angles = self.data.qpos[:self.nu].copy()
        joint_velocities = self.data.qvel[:self.nu].copy()

        # 上一时刻控制量。
        previous_torques = self.previous_torque.copy()

        # 末端线速度。
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_vel = self.data.cvel[ee_body_id][:3].copy()

        # 拼接观测向量。
        state = np.concatenate([
            relative_pos,
            joint_angles,
            joint_velocities,
            previous_torques,
            ee_vel
        ])
        return state.astype(np.float32)

    def _reset_episode_stats(self):
        """重置每回合统计字段。"""
        self.step_count = 0
        self.previous_distance = None
        self.min_distance = None
        self.previous_torque = np.zeros(self.nu, dtype=np.float32)
        self.previous_joint_velocities = np.zeros(self.nu, dtype=np.float32)
        self._phase_rewards_given = set()
        self.episode_return = 0.0
        self.episode_distance_sum = 0.0
        self.episode_speed_sum = 0.0
        self.episode_collision_count = 0
        self.episode_success_count = 0

    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        super().reset(seed=seed)
        
        # 重置仿真状态。
        mujoco.mj_resetData(self.model, self.data)

        # 随机目标空间（与旧版保持一致）。
        self.target_pos = np.array([
            self.np_random.uniform(-0.2, 0.2),
            self.np_random.uniform(-0.37, -0.17),
            self.np_random.uniform(0.2, 0.4),
        ], dtype=np.float32)

        # 成功阈值：1cm。
        self.success_threshold = 0.01
        self.episode_index += 1
        self._reset_episode_stats()

        # 初始距离写入统计，避免首步均值畸变。
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        init_distance = float(np.linalg.norm(ee_pos - self.target_pos))
        self.previous_distance = init_distance
        self.min_distance = init_distance
        self.episode_distance_sum = init_distance

        state = self._get_state()

        info = {
            "episode_index": int(self.episode_index),
            "step_count": 0,
            "target_pos": self.target_pos.copy(),
            "distance_to_target": init_distance,
            "episode_reward": 0.0,
            "episode_success": False,
            "collision_detected": False,
            "episode_summary": None,
        }
        return state, info

    def step(self, action):
        """
        执行动作并返回结果
        """
        # 保存上一时刻控制与速度供惩罚项使用。
        self.previous_torque = self.data.ctrl[:self.nu].copy()
        self.previous_joint_velocities = self.data.qvel[:self.nu].copy()

        # 动作裁剪后写入控制槽。
        action = np.clip(action, -15.0, 15.0)
        self.data.ctrl[:self.nu] = action

        # 物理仿真一步。
        mujoco.mj_step(self.model, self.data)

        # 更新回合步数。
        self.step_count += 1

        # 每回合最多 3000 步。
        max_steps = 3000

        # 计算新观测与末端距离。
        state = self._get_state()
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = 0.0

        # 每步时间惩罚：鼓励更快到达。
        setp_penalty = 0.1
        reward -= setp_penalty

        # 距离 shaping：变近奖励、变远惩罚。
        if self.min_distance is None:
            self.min_distance = distance
            improvement_reward = 0.0
        elif distance < self.min_distance:
            improvement_reward = 1 * (self.min_distance - distance)
            self.min_distance = distance
        elif distance > self.previous_distance:
            improvement_reward = -0.8 * (distance - self.previous_distance)
        else:
            improvement_reward = 0.0
        self.previous_distance = distance
        base_distance_penalty = -distance ** 0.5 * 0.8

        # 阶段阈值奖励：首次跨阈值时一次性发放。
        phase_distance_reward = 0.0
        phase_thresholds = [0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002]
        phase_rewards = [100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0]

        for thresh, phase_reward in zip(phase_thresholds, phase_rewards):
            if distance < thresh and thresh not in self._phase_rewards_given:
                phase_distance_reward += phase_reward
                self._phase_rewards_given.add(thresh)

        # 近距离持续奖励当前关闭，保留占位便于后续实验。
        close_range_bonus = 0.0

        # 距离总奖励。
        reward += improvement_reward + base_distance_penalty + phase_distance_reward + close_range_bonus

        # 速度项：速度过大给固定惩罚。
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_vel = self.data.cvel[ee_body_id][:3].copy()
        ee_speed = np.linalg.norm(ee_vel)

        if ee_speed > 0.5:
            reward -= 0.2

        # 方向奖励：速度方向与目标方向越一致越好。
        to_target = self.target_pos - ee_pos
        to_target /= (np.linalg.norm(to_target) + 1e-6)
        movement_dir = ee_vel / (np.linalg.norm(ee_vel) + 1e-6)
        direction_cos = np.dot(to_target, movement_dir)
        direction_reward = max(0, direction_cos) ** 2 * 1.0
        reward += direction_reward

        # 碰撞惩罚：检测到任意接触时给大惩罚并终止。
        collision_penalty = 0.0
        collision_detected = False
        for i in range(self.data.ncon):
            del i
            # 如果检测到碰撞，给予惩罚
            collision_penalty -= 5000.0
            collision_detected = True
        reward += collision_penalty

        # 关节速度变化惩罚：抑制抖动和突然抽动。
        current_joint_velocities = self.data.qvel[:self.nu].copy()
        joint_velocity_change = np.abs(current_joint_velocities - self.previous_joint_velocities)
        joint_velocity_change_penalty = -0.03 * np.sum(joint_velocity_change)
        reward += joint_velocity_change_penalty

        # 回合终止逻辑。
        distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
        done = False
        truncated = False
        done_reason = "running"

        if distance_to_target <= self.success_threshold:
            done = True
            done_reason = "success"
            success_reward = 10000.0
            reward += success_reward
            reward += (4 * (max_steps - self.step_count))
            if ee_speed < 0.01:
                speed_reward_on_success = 2000.0
            elif ee_speed < 0.05:
                speed_reward_on_success = 1000.0
            elif ee_speed < 0.1:
                speed_reward_on_success = 500.0
            else:
                speed_reward_on_success = 0.0
            reward += speed_reward_on_success

        if collision_detected:
            reward += -setp_penalty * (max_steps - self.step_count)
            done = True
            done_reason = "collision"

        if self.step_count >= max_steps:
            truncated = True
            done_reason = "timeout"

        # 更新回合统计。
        self.episode_return += float(reward)
        self.episode_distance_sum += float(distance_to_target)
        self.episode_speed_sum += float(ee_speed)
        if collision_detected:
            self.episode_collision_count += 1
        if done_reason == "success":
            self.episode_success_count += 1
            self.lifetime_success_count += 1

        episode_summary = None
        if done or truncated:
            episode_summary = {
                "episode_index": int(self.episode_index),
                "episode_steps": int(self.step_count),
                "episode_return": float(self.episode_return),
                "episode_collision_count": int(self.episode_collision_count),
                "episode_success_count": int(self.episode_success_count),
                "lifetime_success_count": int(self.lifetime_success_count),
                "final_distance": float(distance_to_target),
                "avg_distance": float(self.episode_distance_sum / max(self.step_count, 1)),
                "min_distance": float(self.min_distance if self.min_distance is not None else distance_to_target),
                "final_speed": float(ee_speed),
                "avg_speed": float(self.episode_speed_sum / max(self.step_count, 1)),
                "done_reason": done_reason,
            }

        info = {
            "episode_index": int(self.episode_index),
            "step_count": int(self.step_count),
            "distance_to_target": float(distance_to_target),
            "ee_speed": float(ee_speed),
            "episode_reward": float(self.episode_return),
            "episode_length": int(self.step_count),
            "episode_success": bool(done_reason == "success"),
            "collision_detected": bool(collision_detected),
            "done_reason": done_reason,
            "episode_summary": episode_summary,
        }
        return state, float(reward), bool(done), bool(truncated), info

    def render(self):
        """
        渲染环境
        """
        if self.render_mode is None:
            return
            
        # 可视化器懒加载，只在首次渲染时创建。
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._target_viz_added = False
            except Exception as e:
                print(f"无法启动可视化器: {e}")
                return

        # 同步渲染并更新目标可视化球。
        if self.viewer:
            try:
                self._add_target_visualization()
                self.viewer.sync()
            except Exception as e:
                print(f"可视化器同步失败: {e}")
                self.viewer = None

    def _add_target_visualization(self):
        """
        在目标位置添加一个绿色小球用于可视化
        """
        if self.viewer:
            scn = self.viewer.user_scn

            if not getattr(self, '_target_viz_added', False):
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
                        np.array([0.0, 1.0, 0.0, 0.8])
                    )
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    self._target_viz_added = True
            else:
                geom = scn.geoms[self._target_geom]
                geom.pos = self.target_pos

    def close(self):
        """
        关闭可视化器
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
