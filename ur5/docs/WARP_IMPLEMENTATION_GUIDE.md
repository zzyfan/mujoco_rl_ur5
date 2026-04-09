# Warp Implementation Guide

本文档介绍纯 GPU Warp 训练线的模块职责、实现流程和依赖关系。

## File Roles

[warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py)
- 定义环境参数与训练参数。
- 维护 Warp 参数说明字典。

[warp_ur5_runtime.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_runtime.py)
- 负责检查 Warp 依赖是否可用。
- 初始化 CUDA 设备。
- 输出运行时信息。

[warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
- 定义 `UR5WarpReachEnv`。
- 在 JAX / Warp 兼容数组上实现 reset、step、奖励和接触判断。

[train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
- 负责 Warp 训练命令行。
- 负责 `sac` 或 `ppo` 训练器的选择。
- 负责配置保存、进度条输出和最终参数导出。

[test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)
- 负责 Warp 推理。
- 负责最终参数或 checkpoint 的加载。
- 负责 human 窗口可视化。

## Workflow

### Runtime Checks

训练开始前会做三类检查：
- 是否安装了 `warp-lang`
- 是否安装了 `mujoco-warp`
- 是否安装了 `mujoco_playground`

之后会初始化 Warp，并锁定到第一块 CUDA 设备。

### Environment Construction

环境构建过程如下：

1. 根据 `WarpUR5EnvConfig` 生成 `ConfigDict`
2. 加载 UR5 MuJoCo XML
3. 用 `mjx.put_model(..., impl="warp")` 把模型放到 Warp 实现上
4. 在 reset 时生成目标点和初始关节姿态
5. 在 step 时执行动作映射、动力学推进和奖励计算

### Action Pipeline

Warp 线支持两种控制模式：

- `torque`
  直接把策略输出映射成平滑扭矩

- `joint_position_delta`
  先把策略输出解释为关节目标增量，再通过 PD 控制器转换成扭矩

这样设计的原因是：
- `torque` 更接近原始力控实验
- `joint_position_delta` 更稳定，更适合作为默认训练模式

### Reward Design

Warp 线的奖励逻辑与主线保持同一任务目标：
- 逼近目标
- 避免碰撞
- 避免过快和抖动
- 成功时鼓励稳定停留

具体实现包括：
- 距离惩罚
- 靠近奖励
- 远离惩罚
- 分阶段距离奖励
- 速度惩罚
- 方向奖励
- 动作平滑与关节速度惩罚
- 成功奖励
- 碰撞惩罚
- 跑飞诊断

### Trainer Integration

Warp 训练线直接使用 Brax 提供的训练器：

- `brax.training.agents.sac.train`
- `brax.training.agents.ppo.train`

训练脚本不会自行实现优化器，而是负责：
- 准备环境
- 拼接训练参数
- 组织日志输出
- 保存配置与最终参数

## Dependencies

### JAX

用途：
- 表达张量计算
- 让环境 step 与训练过程保持在 JAX 兼容路径上

位置：
- [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

### Brax

用途：
- 提供 `SAC` 与 `PPO` 训练器
- 提供参数保存与部分网络构造工具

位置：
- [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

### MuJoCo Playground

用途：
- 提供 `MjxEnv` 基类
- 提供 Warp 兼容的环境封装与训练包装器

位置：
- [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
- [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)

### warp-lang / mujoco-warp

用途：
- 提供纯 GPU 动力学实现
- 让 MuJoCo 模型在 Warp 后端运行

位置：
- [warp_ur5_runtime.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_runtime.py)

### Flax / Orbax

用途：
- 用于恢复 checkpoint 中的网络参数和标准化状态

位置：
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

## Recommended Reading Order

1. [warp_ur5_runtime.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_runtime.py)
2. [warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py)
3. [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
4. [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
5. [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)
