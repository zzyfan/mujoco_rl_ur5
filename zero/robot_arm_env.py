import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

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
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path("robot_arm_mujoco.xml")
        self.data = mujoco.MjData(self.model)
        
        # 获取关节数量
        self.nu = self.model.nu
        self.nq = self.model.nq
        
        # 设置动作空间 (扭矩)
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(self.nu,), dtype=np.float32
        )
        
        # 设置状态空间
        # 状态维度: 相对位置向量(3) + 关节角度(6) + 关节速度(6) + 上一时刻关节扭矩(6) + 末端速度(3) = 24维
        state_dim = 3 + 6 + 6 + 6 + 3  # 24维状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 设置目标位置 (末端执行器)
        self.target_pos = np.array([0.2, 0.2, 0.2])  # 初始目标位置
        self.previous_distance = None
        
        # 保存上一时刻的扭矩
        self.previous_torque = np.zeros(self.nu)
        
        # 保存上一时刻的关节速度
        self.previous_joint_velocities = np.zeros(self.nu)
        
        # 初始化可视化器为None
        self.render_mode = render_mode
        self.viewer = None
                
        # 重置环境
        self.reset()

    def _get_state(self):
        """
        获取当前状态
        """
        # 获取末端执行器位置（通过名称查找）
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        
        # 获取末端执行器位置
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        
        # 计算相对位置向量 (目标位置 - 当前末端位置)
        relative_pos = self.target_pos - ee_pos
        
        # 获取关节角度
        joint_angles = self.data.qpos[:self.nu].copy()
        
        # 获取关节速度
        joint_velocities = self.data.qvel[:self.nu].copy()
        
        # 获取上一时刻的关节扭矩
        previous_torques = self.previous_torque.copy()
        
        # 使用body速度获取末端执行器速度
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_vel = self.data.cvel[ee_body_id][:3].copy()
        
        # 拼接状态向量
        state = np.concatenate([
            relative_pos,          # 相对位置向量 (3,)
            joint_angles,          # 关节角度 (6,)
            joint_velocities,      # 关节角速度 (6,)
            previous_torques,      # 上一时刻关节扭矩 (6,)
            ee_vel                 # 末端速度 (3,)
        ])
        
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置随机目标位置
        # x轴: [-0.3, 0.3]
        # y轴: [-0.47, -0.17] 
        # z轴: [0.2, 0.5]
        self.target_pos = np.array([
            self.np_random.uniform(-0.2, 0.2),    # x轴
            self.np_random.uniform(-0.37, -0.17),   # y轴
            self.np_random.uniform(0.2, 0.4)        # z轴
        ])
        
        self.success_threshold = 0.01  # 1mm阈值
        
        # 重置步数计数器
        self.step_count = 0
        self.previous_distance = None
        self.min_distance = None
        self.previous_torque = np.zeros(self.nu)
        self.previous_joint_velocities = np.zeros(self.nu)
        
        # 重置阶段性奖励标记
        self._phase_rewards_given = set()
        
        # 获取初始状态
        state = self._get_state()
        
        return state, {}

 
    def step(self, action):
        """
        执行动作并返回结果
        统一奖励版本：适用于 PPO / SAC / TD3
        """

        # 保存上一时刻控制量和关节速度
        self.previous_torque = self.data.ctrl[:self.nu].copy()
        self.previous_joint_velocities = self.data.qvel[:self.nu].copy()

        # 应用动作（扭矩）
        action = np.clip(action, -15.0, 15.0)
        self.data.ctrl[:self.nu] = action

        # 仿真一步
        mujoco.mj_step(self.model, self.data)

        # 更新步数
        self.step_count += 1
        max_steps = 3000

        # 获取状态
        state = self._get_state()

        # =========================
        # 1. 计算末端位置与距离
        # =========================
        ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        distance = np.linalg.norm(ee_pos - self.target_pos)

        reward = 0.0

        # =========================
        # 2. 时间惩罚（轻量）
        # =========================
        step_penalty = 0.01
        reward -= step_penalty

        # =========================
        # 3. 距离主奖励（核心）
        # MuJoCo风格：距离越近越好
        # =========================
        distance_reward = -3.0 * distance
        reward += distance_reward

        # =========================
        # 4. 距离改善奖励（保留你原来最有价值的思路）
        # 用“相邻两步距离差”代替 min_distance 逻辑
        # =========================
        if self.previous_distance is None:
            delta_distance = 0.0
        else:
            delta_distance = self.previous_distance - distance

        improvement_reward = 5.0 * delta_distance
        reward += improvement_reward
        self.previous_distance = distance

        # =========================
        # 5. 动作惩罚（类似 MuJoCo control cost）
        # 扭矩越大，代价越大
        # =========================
        action_penalty = 0.001 * np.sum(np.square(action))
        reward -= action_penalty

        # =========================
        # 6. 末端速度与方向奖励
        # 保留你原来的方向引导思路，但降权
        # =========================
        ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link"
        )
        ee_vel = self.data.cvel[ee_body_id][:3].copy()
        ee_speed = np.linalg.norm(ee_vel)

        to_target = self.target_pos - ee_pos
        to_target = to_target / (np.linalg.norm(to_target) + 1e-6)

        movement_dir = ee_vel / (np.linalg.norm(ee_vel) + 1e-6)

        direction_cos = np.dot(to_target, movement_dir)
        direction_reward = 0.2 * max(0.0, direction_cos)
        reward += direction_reward

        # =========================
        # 7. 接近目标后鼓励减速
        # 防止冲过头
        # =========================
        close_speed_penalty = 0.0
        if distance < 0.05:
            close_speed_penalty = 0.1 * ee_speed
            reward -= close_speed_penalty

        # =========================
        # 8. 碰撞惩罚
        # 原来每个接触点 -5000 过大，改成一次性有限惩罚
        # =========================
        collision_detected = self.data.ncon > 0
        collision_penalty = 0.0
        if collision_detected:
            collision_penalty = 5.0
            reward -= collision_penalty

        # =========================
        # 9. 关节速度变化惩罚（平滑控制）
        # 保留你原来的思路，但减小权重
        # =========================
        current_joint_velocities = self.data.qvel[:self.nu].copy()
        joint_velocity_change = np.abs(
            current_joint_velocities - self.previous_joint_velocities
        )
        joint_velocity_change_penalty = 0.01 * np.sum(joint_velocity_change)
        reward -= joint_velocity_change_penalty

        # =========================
        # 10. 成功判定与成功奖励
        # 原来 +10000 过大，改为中等奖励
        # =========================
        done = False
        success = distance <= self.success_threshold

        if success:
            done = True
            success_reward = 20.0
            reward += success_reward
        else:
            success_reward = 0.0

        # =========================
        # 11. 碰撞终止（可保留）
        # =========================
        if collision_detected:
            done = True

        # 超步数截断
        truncated = self.step_count >= max_steps

        # 额外信息，方便你调试和画图
        info = {
            "distance": float(distance),
            "success": bool(success),
            "collision": bool(collision_detected),
            "distance_reward": float(distance_reward),
            "improvement_reward": float(improvement_reward),
            "action_penalty": float(action_penalty),
            "direction_reward": float(direction_reward),
            "close_speed_penalty": float(close_speed_penalty),
            "collision_penalty": float(collision_penalty),
            "joint_velocity_change_penalty": float(joint_velocity_change_penalty),
            "success_reward": float(success_reward),
            "total_reward": float(reward),
        }

        return state, reward, done, truncated, info

    def render(self):
        """
        渲染环境
        """
        if self.render_mode is None:
            return
            
        # 如果可视化器尚未创建，则创建一个
        if self.viewer is None:
            try:
                # 尝试创建被动式可视化器
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 初始化目标可视化添加标记
                self._target_viz_added = False
            except Exception as e:
                print(f"无法启动可视化器: {e}")
                return
        
        # 如果可视化器已经创建，同步更新场景
        if self.viewer:
            try:
                # 添加目标位置的可视化（绿色小球）
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
            # 获取用户场景对象
            scn = self.viewer.user_scn
            
            # 检查是否已经添加了目标可视化
            if not getattr(self, '_target_viz_added', False):
                # 增加一个几何体（小球）
                if scn.ngeom < scn.maxgeom:
                    # 获取新增几何体的引用
                    self._target_geom = scn.ngeom
                    geom = scn.geoms[scn.ngeom]
                    scn.ngeom += 1
                    # 初始化几何体为球体
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.02, 0.02, 0.02]),  # 球体半径
                        self.target_pos,  # 球体位置（目标位置）
                        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # 单位矩阵
                        np.array([0.0, 1.0, 0.0, 0.8])  # RGBA: 绿色，半透明
                    )
                    # 设置几何体类别为仅可视化，不参与物理碰撞
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    # 标记已添加目标可视化
                    self._target_viz_added = True
            else:
                # 更新已存在的几何体位置
                geom = scn.geoms[self._target_geom]
                geom.pos = self.target_pos
    def close(self):
        """
        关闭可视化器
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None