# 更新日志

这份日志按时间顺序整理当前仓库从最初主线拆分到现在的主要实现变化，便于快速回顾“做过什么、为什么这么做”。

## 2026-04 第一阶段：拆清训练线与任务语义

- 拆分 `classic/` 与 `warp_gpu/` 两条训练线。
- 统一两条线的 reach 任务、观测含义与目标采样范围。
- 修正末端速度语义，避免把角速度误当线速度。

## 2026-04 第二阶段：收奖励与碰撞逻辑

- 奖励结构向 zero-arm 思路对齐：
  - 距离惩罚
  - 距离改善/退步
  - 阶段性距离奖励
  - 成功奖励
  - 碰撞惩罚
- 奖励系数按算法拆分 preset。
- 视觉体与代理碰撞体分离。
- 危险碰撞过滤不再把目标球、灯光或机器人内部接触直接算失败。

## 2026-04 第三阶段：课程学习、动作缩放与诊断日志

- `classic` 支持自动课程学习：
  - 固定目标
  - 小范围随机
  - 全范围随机
- 成功阈值按课程阶段调度。
- 动作先标准化，再缩放成真实控制量。
- 加入动作低通滤波。
- 日志增加：
  - `success_rate`
  - `collision_rate`
  - `runaway_rate`
  - `timeout_rate`

## 2026-04 第四阶段：把 runaway 降回诊断信号

- `runaway` 不再作为主终止逻辑。
- 训练主线重新聚焦到：
  - `success`
  - `collision`
  - `timeout`
- `runaway` 继续保留为“回合内是否出现明显发散迹象”的诊断指标。
- 测试脚本新增：
  - `done_reason`
  - 固定目标点调试
  - 慢速渲染

## 2026-04 当前阶段：吸收外部项目思路

参考：
- `MuJoCo_RL_UR5`
- `panda-gym`
- `Gymnasium-Robotics`
- `rl-baselines3-zoo`
- `homestri-ur5e-rl`

落地改动：

- `classic` 新增 `goal-conditioned` 观测结构：
  - `observation`
  - `achieved_goal`
  - `desired_goal`
- `classic` 新增 `HER` 训练支持（`SAC/TD3`）。
- `classic` 新增 `dense / sparse` 奖励模式切换。
- `classic` 和 `warp_gpu` 都新增 `joint_position_delta` 控制模式。
- 训练日志增加 `done_reasons=...` 汇总，方便直接判断当前主要失败类型。

## 当前推荐优先级

1. 先用 `classic SAC + goal-conditioned + HER + joint_position_delta`
2. 先在固定目标阶段拿到第一次成功
3. 再切到小范围随机和全范围随机
4. `warp_gpu` 继续保留做高吞吐并行训练对照
