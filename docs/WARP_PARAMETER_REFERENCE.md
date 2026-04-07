# Warp Parameter Reference

本文档说明 Warp 训练线参数的含义、作用位置和主要影响。

## Environment Parameters

这些参数定义在 [warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py) 的 `WarpUR5EnvConfig` 中。

| 参数名 | 作用 |
| --- | --- |
| `model_xml` | Warp 线使用的 MuJoCo XML 路径。当前默认使用 `assets/robotiq_cxy/lab_env.xml`。 |
| `sim_dt` | 物理仿真步长。 |
| `frame_skip` | 单个策略动作推进多少个物理步。 |
| `episode_length` | 单回合最大决策步数。 |
| `naconmax` | Warp 接触缓存容量。 |
| `naccdmax` | Warp CCD 接触缓存容量。 |
| `njmax` | Warp 约束缓存容量。 |
| `target_x_min/max` | 全随机采样时目标点 x 范围。 |
| `target_y_min/max` | 全随机采样时目标点 y 范围。 |
| `target_z_min/max` | 全随机采样时目标点 z 范围。 |
| `target_sampling_mode` | 目标采样模式，可选 `full_random`、`small_random`、`fixed`。 |
| `target_range_scale` | 局部随机采样比例。 |
| `fixed_target_x/y/z` | 固定目标点坐标。 |
| `success_threshold` | 全随机阶段成功阈值。 |
| `stage1_success_threshold` | 固定目标阶段成功阈值。 |
| `stage2_success_threshold` | 局部随机阶段成功阈值。 |
| `torque_low/high` | 力矩裁剪边界。 |
| `action_target_scale` | 力矩模式下的动作缩放比例。 |
| `action_smoothing_alpha` | 动作平滑系数。 |
| `controller_mode` | 控制模式，可选 `torque` 或 `joint_position_delta`。 |
| `joint_position_delta_scale` | 位置增量控制时每步允许的目标关节增量。 |
| `position_control_kp/kd` | 位置控制器的比例与阻尼参数。 |
| `goal_observation` | 是否在观测后部显式拼接 achieved/desired goal。 |
| `reward_mode` | 奖励模式，可选 `dense` 或 `sparse`。 |
| `fixed_gripper_ctrl` | 夹爪固定控制值。 |
| `gravity_ctrl` | 目标相关滑块的重力补偿控制值。 |
| `home_joint1/2/3` | 机械臂 reset 时的初始主关节角度。 |
| `step_penalty` | 每一步的固定时间惩罚。 |
| `base_distance_weight` | 基础距离惩罚权重。 |
| `improvement_gain` | 逼近目标时的增量奖励系数。 |
| `regress_gain` | 远离目标时的惩罚系数。 |
| `speed_penalty_threshold` | 超过该末端速度时判定为过快。 |
| `speed_penalty_value` | 过快时的惩罚值。 |
| `direction_reward_gain` | 速度方向朝向目标时的奖励强度。 |
| `joint_vel_change_penalty_gain` | 关节速度变化惩罚系数。 |
| `action_magnitude_penalty_gain` | 动作幅值惩罚系数。 |
| `action_change_penalty_gain` | 相邻动作变化惩罚系数。 |
| `idle_distance_threshold` | 仍然远离目标时的距离阈值。 |
| `idle_speed_threshold` | 近似静止时的速度阈值。 |
| `idle_penalty_value` | 远离目标但近似静止时的惩罚值。 |
| `phase_thresholds` | 阶段性距离奖励阈值。 |
| `phase_rewards` | 各距离阈值对应的一次性奖励。 |
| `success_bonus` | 成功奖励。 |
| `success_remaining_step_gain` | 成功后根据剩余步数追加的奖励系数。 |
| `success_speed_bonus_very_slow/slow/medium` | 成功时按末端速度附加的稳定性奖励。 |
| `collision_penalty_value` | 碰撞时的惩罚。 |
| `runaway_distance_threshold` | 跑飞距离阈值。 |
| `runaway_ee_speed_threshold` | 跑飞末端速度阈值。 |
| `runaway_joint_velocity_threshold` | 跑飞关节速度阈值。 |
| `runaway_penalty_value` | 跑飞惩罚值。 |

## Training Parameters

这些参数定义在 [warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py) 的 `WarpTrainConfig` 中。

| 参数名 | 作用 |
| --- | --- |
| `algo` | 训练算法，可选 `sac` 或 `ppo`。 |
| `run_name` | 实验名称。 |
| `seed` | 随机种子。 |
| `num_timesteps` | 总训练步数。 |
| `num_envs` | 并行训练环境数。 |
| `num_eval_envs` | 并行评估环境数。 |
| `num_evals` | 训练期间评估次数。 |
| `learning_rate` | 学习率。 |
| `discounting` | 折扣因子。 |
| `reward_scaling` | Brax 奖励缩放。 |
| `normalize_observations` | 是否标准化观测。 |
| `entropy_cost` | PPO 熵正则系数。 |
| `unroll_length` | PPO rollout 长度。 |
| `batch_size` | 训练 batch 大小。 |
| `num_minibatches` | PPO mini-batch 数。 |
| `num_updates_per_batch` | PPO 每批数据重复优化次数。 |
| `sac_tau` | SAC 目标网络软更新系数。 |
| `sac_min_replay_size` | SAC 回放池预热大小。 |
| `sac_max_replay_size` | SAC 回放池容量。 |
| `sac_grad_updates_per_step` | SAC 每步环境交互后做多少次梯度更新。 |
| `action_repeat` | Brax 训练器里同一动作重复执行的次数。 |
| `dry_run` | 只初始化环境和配置，不正式训练。 |

## Recommended Reading Order

1. 先读 [warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py)
2. 再读 [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
3. 然后读 [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
4. 最后读 [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)
