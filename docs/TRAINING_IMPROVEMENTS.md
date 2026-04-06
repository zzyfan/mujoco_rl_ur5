# 训练改进记录

这份文档按迭代顺序整理当前项目从最初版本到现在的主要训练改进思路、对应实现方法，以及每一轮改动想解决的具体问题。

## 版本 1：先把两条训练线拆清楚

目标：
- 保留 `classic/` 作为 Gymnasium + SB3 主线。
- 单独整理 `warp_gpu/` 作为 Warp + Playground + Brax 训练线。

实现：
- 训练入口拆分成 [classic/train.py](/home/zzyfan/mujoco_ur5_rl/classic/train.py) 和 [warp_gpu/train.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/train.py)。
- `warp_gpu/` 单独维护配置、推理、自检和运行时检查。

作用：
- 先把“CPU 采样主线”和“GPU 并行主线”区分清楚，避免后续调参混在一起。

## 版本 2：统一任务定义和观测语义

目标：
- 让两条训练线尽量在同一个任务上训练。
- 修正末端速度和相对位置定义，避免奖励和观测读错语义。

实现：
- 统一两条线的 24 维观测结构：
  - 相对位置
  - 关节角
  - 关节速度
  - 上一时刻扭矩
  - 末端线速度
- 把末端速度改成相邻两步位置差分，而不是继续依赖旧的 `cvel[:3]` 兼容读法。

作用：
- 保证 `classic` 和 `warp_gpu` 对“目标方向”“接近速度”“成功距离”的理解一致。

## 版本 3：奖励结构向 zero-arm 对齐

目标：
- 把奖励主结构收成一套更清晰、更接近 zero-arm 的 dense reward。

实现：
- 引入基础距离惩罚、距离改善奖励、退步惩罚。
- 引入阶段性距离奖励。
- 保留成功奖励、成功剩余步数奖励、成功低速奖励。
- 保留碰撞惩罚和方向奖励。

作用：
- 先把训练目标定义清楚，再在此基础上调稳定性和可学性。

## 版本 4：按算法拆奖励系数

目标：
- 不改任务逻辑，但让 `PPO / SAC / TD3` 在同一任务下有更合适的奖励权重。

实现：
- `classic/train.py` 和 `warp_gpu/train.py` 里按算法自动套 reward preset。
- 保持主奖励结构一致，只改：
  - 距离改善权重
  - 方向奖励权重
  - 速度惩罚
  - 动作惩罚
  - 静止惩罚

作用：
- 让不同算法在不改变任务定义的前提下拥有更合理的训练接口。

## 版本 5：危险碰撞过滤和代理碰撞体

目标：
- 解决“视觉 mesh 直接参与碰撞”和“所有 contact 都直接判失败”的问题。

实现：
- 在 [assets/robotiq_cxy/lab_env.xml](/home/zzyfan/mujoco_ur5_rl/assets/robotiq_cxy/lab_env.xml) 中分离视觉体和碰撞体。
- 视觉 `mesh` 只显示，不参与碰撞。
- 使用 `capsule / cylinder / box` 代理体承担碰撞职责。
- 在 [classic/env.py](/home/zzyfan/mujoco_ur5_rl/classic/env.py) 和 [warp_gpu/env.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/env.py) 中只把“机器人与外部危险几何”的接触算作失败。

作用：
- 把训练从“所有接触都失败”的极端碰撞模型，收回到更接近真实任务约束的判定。

## 版本 6：跑飞终止

目标：
- 解决“既不成功也不碰撞，只是长时间发散，把回报滚到百万级负值”的坏解。

实现：
- 在两条线中新增：
  - 距离过大终止
  - 末端速度过大终止
  - 关节速度过大终止
- 新增 `runaway_penalty`
- 回合跑飞后同步扣掉剩余时间惩罚。

作用：
- 把失败类型从“只有碰撞失败”扩展成“碰撞失败 + 跑飞失败”，避免长回合坏解掩盖真正训练趋势。

## 版本 7：扩展外层阶段奖励

目标：
- 让策略在离目标还比较远时，也能更早感知“靠近目标是有好处的”。

实现：
- 把阶段性距离奖励从原来的内层阈值，扩展到 12 档。
- 新增的外层阶段包括：
  - `1.6m`
  - `1.3m`
  - `1.0m`
  - `0.8m`
  - `0.6m`

作用：
- 让“远距离接近”也能得到明确奖励，减少前期只有惩罚、缺少成就感的问题。

## 版本 8：课程学习、成功阈值调度、动作缩放和动作滤波

目标：
- 第一优先级：先让策略成功一次。
- 第二优先级：把失败类型分清楚。
- 第三优先级：让动作控制更容易学。

实现：

### 1. classic 自动课程学习

在 [classic/env.py](/home/zzyfan/mujoco_ur5_rl/classic/env.py) 中保留并强化三阶段目标采样：

- `stage1_fixed`
  - 固定目标点
  - 成功阈值默认放宽到 `0.05`
- `stage2_small_random`
  - 围绕中心点小范围随机
  - 成功阈值默认收紧到 `0.03`
- `stage3_full_random`
  - 回到完整随机范围
  - 成功阈值恢复到最终值 `0.01`

### 2. warp 的可配置训练阶段

在 [warp_gpu/env.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/env.py) 和 [warp_gpu/train.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/train.py) 中新增：

- `target_sampling_mode`
  - `fixed`
  - `small_random`
  - `full_random`
- `target_range_scale`
- `stage1_success_threshold`
- `stage2_success_threshold`

说明：
- `warp_gpu` 由于训练器是 JAX/Brax 批量环境，自动按 episode 递增的课程学习不如 `classic` 好做。
- 当前推荐用“分阶段启动训练”的方式做 `warp` 课程学习：
  1. 固定目标
  2. 小范围随机
  3. 全范围随机

### 3. 动作接口改成标准化输出

在两条线中都把策略动作统一成 `[-1, 1]` 的标准化空间，再映射成真实扭矩：

- 先按 `action_target_scale` 缩放到目标扭矩范围
- 再通过 `action_smoothing_alpha` 和上一时刻扭矩做低通滤波
- 最后裁剪到关节允许的真实扭矩范围

作用：
- 不再让策略直接面对“很硬的真实扭矩空间”
- 减少接近目标时的大扭矩突变
- 让“平稳靠近目标”更容易学

注意：
- 这一版动作接口改变后，旧版按“直接扭矩输出”训练的模型不再适合作为新配置的直接对照，后续训练建议全部重新开始。

### 4. 失败类型日志

两条线现在都会区分并统计：

- `success_rate`
- `collision_rate`
- `runaway_rate`
- `timeout_rate`

作用：
- 一眼看出当前坏解到底是：
  - 撞上去
  - 跑飞
  - 磨蹭到超时
  - 还是已经开始成功

## 当前推荐训练顺序

想先把任务学会时，推荐用下面这套顺序，而不是一上来就全随机：

1. 固定目标 + 放宽成功阈值
2. 小范围随机 + 中等成功阈值
3. 全范围随机 + 最终成功阈值

其中：
- `classic` 可以直接靠环境内课程学习完成
- `warp_gpu` 推荐分阶段跑三轮配置

## 当前仍值得继续观察的方向

- 夹爪和腕部代理碰撞体还能继续贴合
- 总队列训练建议使用单独 logfile，避免不同算法输出混写
- `warp` 的固定文本进度展示可以继续替代动态条，方便服务器日志查看

## 版本 9：把跑飞从主失败逻辑降回诊断信号

目标：
- 参考 `MuJoCo_RL_UR5`、`panda-gym`、`Gymnasium-Robotics` 等项目，把训练主线重新收回到 `success / collision / timeout`。
- 避免 `runaway` 大罚分和强制终止盖过“先学会成功一次”的任务主线。

实现：
- 在 [classic/env.py](/home/zzyfan/mujoco_ur5_rl/classic/env.py) 中，`runaway` 不再直接终止回合，也不再写入大额固定罚分。
- 在 [warp_gpu/env.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/env.py) 中，`runaway` 同样退回为“回合内是否出现过明显发散迹象”的诊断指标。
- 在 [classic/test.py](/home/zzyfan/mujoco_ur5_rl/classic/test.py) 中新增：
  - `--render-sleep`
  - `--fixed-target-x / y / z`
  - `done_reason` 输出

作用：
- 训练主逻辑重新聚焦到：
  - 成功到点
  - 危险碰撞失败
  - 超时失败
- `runaway` 继续保留可见性，方便判断“虽然没终止，但策略已经明显发散”的坏轨迹。
- 测试时可以固定目标点并放慢可视化速度，更容易定位第一步失败原因。

## 下一步最值得继续吸收的设计

- `goal-conditioned` 观测结构
- `HER` 作为 reach 任务强基线
- 比纯扭矩更容易学的控制模式（例如 `joint position delta`）
- 更清晰的 `done_reason` / success 指标输出

## 版本 10：goal-conditioned、HER 与更容易学的控制接口

目标：
- 吸收 `panda-gym`、`Gymnasium-Robotics`、`rl-baselines3-zoo` 和 `homestri-ur5e-rl` 这些项目里最值得借鉴的主线设计。
- 不再只靠 dense reward 和直接扭矩控制硬学 reach。

实现：

### 1. classic goal-conditioned 环境

在 [classic/env.py](/home/zzyfan/mujoco_ur5_rl/classic/env.py) 中新增可选的 goal-conditioned 观测结构：

- `observation`
- `achieved_goal`
- `desired_goal`

并补上 `compute_reward()`，满足 SB3 `HER` 对 goal-conditioned 环境的接口要求。

### 2. classic HER

在 [classic/train.py](/home/zzyfan/mujoco_ur5_rl/classic/train.py) 中新增：

- `--goal-conditioned`
- `--use-her`
- `--her-goal-selection-strategy`
- `--her-n-sampled-goal`
- `--reward-mode sparse`

说明：
- 目前 `HER` 只接到 `classic` 的 `SAC / TD3`。
- 启用 `HER` 时会自动切到：
  - `goal_conditioned=True`
  - `reward_mode=sparse`

### 3. 更容易学的控制接口

在 [classic/env.py](/home/zzyfan/mujoco_ur5_rl/classic/env.py) 和 [warp_gpu/env.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/env.py) 中新增：

- `controller_mode=torque`
- `controller_mode=joint_position_delta`

其中 `joint_position_delta` 的实现方式是：
- 动作表示“下一步的关节目标增量”
- 环境内部再通过简单的 PD 形式转换成真实扭矩

作用：
- 保留 torque 控制作对照
- 同时提供一个更容易学、对 reach 任务更友好的控制接口

### 4. 日志与测试诊断进一步加强

- 训练日志新增 `done_reasons=...`
- `classic` 训练日志新增：
  - `success_count`
  - `collision_count`
  - `runaway_count`
  - `timeout_count`
- `warp_gpu` 训练日志新增：
  - `eval_success_count`
  - `eval_collision_count`
  - `eval_runaway_count`
  - `eval_timeout_count`
  - `train_success_count`
  - `train_collision_count`
  - `train_runaway_count`
  - `train_timeout_count`
- 测试脚本支持：
  - 固定目标点
  - 慢速渲染
  - `done_reason`
  - goal-conditioned 模型推理参数

作用：
- 更快判断“为什么没成功”
- 更容易验证 curriculum / HER / 控制接口是否真正改善了首次成功率

## 版本 11：warp 奖励模式补齐

目标：
- 让 `warp_gpu/` 不再只有 dense reward 一种口径。
- 把 `warp_gpu/` 也收进 robotics 任务常见的 `success/fail` 训练接口。
- 给 `warp` 补上一条可直接和 `classic sparse + HER` 对照的实验线。

实现：

### 1. warp 新增 `reward_mode`

在 [warp_gpu/env.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/env.py) 中新增：

- `reward_mode=dense`
- `reward_mode=sparse`

其中：
- `dense` 继续保留当前距离 shaping 奖励
- `sparse` 改成：
  - 成功：`0`
  - 失败：`-1`

### 2. train/test 同步参数

在 [warp_gpu/train.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/train.py) 和 [warp_gpu/test.py](/home/zzyfan/mujoco_ur5_rl/warp_gpu/test.py) 中同步加入：

- `--reward-mode dense`
- `--reward-mode sparse`

这样训练、评估和推理就不会因为奖励模式不一致而产生歧义。

作用：
- `warp_gpu/` 现在也可以做：
  - dense shaping 训练
  - sparse success/fail 训练
- 这让它更接近：
  - `panda-gym`
  - `Gymnasium-Robotics`
  - `rl-baselines3-zoo`
 里常见的 robotics 任务设计方式。

说明：
- `warp_gpu/` 目前仍然没有 `goal-conditioned + HER`。
- 所以当前推荐分工是：
  - `classic`：goal-conditioned + HER 主线
  - `warp_gpu`：高吞吐 dense/sparse 对照线

## 版本 12：HER 启动保护与镜像式模型回传

目标：
- 修复 `HER + VecEnv` 在第一批完整 episode 落盘前就开始采样的问题。
- 让下载回本地的模型默认按用途归位，而不是全部平铺到同一层目录。

实现：

### 1. classic 的 HER 启动保护

在 [classic/train.py](/home/zzyfan/mujoco_ur5_rl/classic/train.py) 中，当启用 `--use-her` 时，自动保证：

- `learning_starts >= max_steps * n_envs`

说明：
- 这是因为 SB3 在 VecEnv 里统计的是“总环境步数”。
- 并行环境越多，想等到第一批完整 episode 全部写入回放池，需要的总步数也越高。

### 2. 模型回传脚本支持目录镜像

在 [scripts/auto_fetch_remote_models.py](/home/zzyfan/mujoco_ur5_rl/scripts/auto_fetch_remote_models.py) 中新增：

- `--layout mirror`
- `--layout artifact`

其中：
- `mirror`：按远端目录结构镜像到本地
- `artifact`：兼容旧版按预设名平铺目录

作用：
- 避免 `HerReplayBuffer` 在还没有完整 episode 时抛出采样异常。
- 让模型默认落到：
  - `downloads/remote_models/models/...`
  - `downloads/remote_models/logs/.../best_model`
- 默认改成“整轮全部产物齐了后统一回传”，避免训练中途先拿到半套模型。
