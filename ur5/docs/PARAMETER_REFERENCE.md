# Main Parameter Reference

本文档说明主线参数的含义、作用位置和主要影响。

## 环境参数

这些参数定义在 [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py) 的 `UR5ReachEnvConfig` 中。

| 参数名 | 作用 |
| --- | --- |
| `model_xml` | MuJoCo XML 路径。当前固定指向 `assets/robotiq_cxy/lab_env.xml`，并通过仓库相对路径加载。 |
| `frame_skip` | 一个 RL 动作会推进多少个物理步。 |
| `episode_length` | 一个回合最长决策步数。 |
| `render_camera_name` | 测试与学习笔记默认使用的相机名字。 |
| `target_x_min/max` | 目标采样区域 x 范围。 |
| `target_y_min/max` | 目标采样区域 y 范围。 |
| `target_z_min/max` | 目标采样区域 z 范围。 |
| `curriculum_fixed_episodes` | 固定目标点阶段持续多少回合。设为 `0` 时关闭该阶段，直接进入随机目标训练。 |
| `curriculum_local_random_episodes` | 局部随机阶段持续多少回合。设为 `0` 时关闭该阶段，直接使用完整工作空间采样。 |
| `curriculum_local_scale` | 局部随机阶段的采样半径比例。只有局部随机阶段启用时才会生效。 |
| `fixed_target_x/y/z` | 固定阶段使用的目标点。 |
| `control_mode` | 控制模式，`torque` 更接近参考训练线，`joint_delta` 提供更平滑的关节增量控制。 |
| `torque_low/high` | 力矩裁剪边界。 |
| `joint_delta_scale` | 关节增量控制每步最多改多少弧度。 |
| `action_smoothing_alpha` | 动作平滑系数，控制抖动。 |
| `position_kp/kd` | `joint_delta` 模式下的 PD 控制器参数。 |
| `gravity_compensation` | 给目标滑块和相关机构的补偿控制。 |
| `fixed_gripper_ctrl` | 夹爪保持张开时的固定控制值。 |
| `home_joint1/2/3` | 回合重置时前 3 个主关节的初始角度。 |
| `success_threshold_stage1/2/3` | 不同课程阶段的成功判定距离。 |
| `step_penalty` | 每步都会扣的时间代价。 |
| `distance_weight` | 基础距离惩罚的强度。 |
| `progress_reward_gain` | 比上一时刻更接近目标时的奖励强度。 |
| `regress_penalty_gain` | 比上一时刻更远离目标时的惩罚强度。 |
| `phase_thresholds` | 首次进入某个距离区间时触发的一次性阶段奖励阈值。 |
| `phase_rewards` | 与 `phase_thresholds` 对应的一次性奖励值。 |
| `speed_penalty_threshold/value` | 末端速度超过阈值时施加的固定惩罚。 |
| `direction_reward_gain` | 速度方向朝向目标时的奖励强度。 |
| `action_l2_penalty` | 抑制大动作。 |
| `action_smoothness_penalty` | 抑制动作突变。 |
| `joint_velocity_penalty` | 抑制关节速度在相邻时间步之间发生剧烈变化。 |
| `collision_penalty` | 碰撞时的一次性大惩罚。 |
| `success_bonus` | 成功时的一次性奖励。 |
| `success_remaining_step_gain` | 成功后根据剩余步数附加的奖励系数。 |
| `success_speed_bonus_very_slow/slow/medium` | 成功后按末端稳定程度附加的奖励。 |
| `runaway_distance_threshold` | 跑飞判定阈值。 |
| `runaway_penalty` | 跑飞时的一次性惩罚。 |

## 训练参数

这些参数定义在 [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py) 的 `RLTrainConfig` 中。

### 通用参数

| 参数名 | 作用 |
| --- | --- |
| `algo` | 选择训练算法，支持 `td3`、`sac`、`ppo`。 |
| `run_name` | 本次实验名称。 |
| `seed` | 随机种子。 |
| `total_timesteps` | 总训练步数。 |
| `n_envs` | 并行环境数量。 |
| `eval_freq` | 每隔多少训练步执行一次评估。 |
| `eval_episodes` | 每次评估跑多少个 episode。 |
| `device` | 使用 `cpu`、`cuda` 还是 `auto`。 |
| `learning_rate` | 优化器学习率。 |
| `batch_size` | 梯度更新 batch 大小。 |
| `gamma` | 折扣因子。 |
| `actor_layers` | 策略网络隐藏层。 |
| `critic_layers` | 价值网络隐藏层。 |
| `normalize_observation` | 是否归一化观测。 |
| `normalize_reward` | 是否归一化训练奖励。 |
| `render_training` | 训练时是否开窗口。 |
| `render_every` | 训练渲染刷新间隔。 |
| `spectator_render` | 是否启用旁观模式。训练环境仍然无头并行，但主进程会额外开一个窗口。 |
| `spectator_render_every` | 旁观窗口每隔多少个训练 step 更新一次。 |
| `spectator_deterministic` | 旁观模式是用确定性动作还是随机采样动作。 |

### TD3 / SAC 共用的离策略参数

| 参数名 | 作用 |
| --- | --- |
| `buffer_size` | 经验回放池容量。 |
| `learning_starts` | 收集多少步后开始学习。 |
| `tau` | 目标网络软更新系数。 |
| `train_freq` | 多久触发一次训练。 |
| `gradient_steps` | 每次训练做多少个梯度更新。 |

### 只对 TD3 生效

| 参数名 | 作用 |
| --- | --- |
| `policy_delay` | Actor 更新频率低于 Critic。 |
| `target_policy_noise` | 目标动作平滑噪声。 |
| `target_noise_clip` | 目标噪声裁剪。 |
| `action_noise_sigma` | 真实探索动作噪声。 |

### 只对 PPO 生效

| 参数名 | 作用 |
| --- | --- |
| `ppo_n_steps` | 每次 rollout 收集步数。 |
| `ppo_n_epochs` | 每轮 rollout 重复训练次数。 |
| `ppo_gae_lambda` | GAE lambda。 |
| `ppo_ent_coef` | 熵正则权重。 |
| `ppo_vf_coef` | Value loss 权重。 |
| `ppo_clip_range` | 策略裁剪范围。 |

## Recommended Reading Order

1. 先看环境参数，理解“目标怎么采样、动作怎么生效、奖励怎么构成”。
2. 再看训练参数，理解“算法怎么更新、多久评估一次、产物怎么保存”。
3. 最后结合 [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py) 看 CLI 参数如何映射到 dataclass。
