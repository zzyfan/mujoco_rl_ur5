# Main Implementation Guide

本文档介绍主线代码的模块职责、核心流程和依赖关系。

## File Roles

[ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py)
- 定义 `UR5ReachEnvConfig` 与 `RLTrainConfig`。
- 给主要参数提供文字说明字典。
- 统一训练产物目录和配置保存方式。

[ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
- 定义环境类 `UR5ReachEnv`。
- 负责 reset、step、奖励、碰撞和渲染。

[train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)
- 负责命令行参数解析。
- 负责根据 `algo` 选择 `TD3`、`SAC` 或 `PPO`。
- 负责训练、评估、模型保存和测试。

## Workflow

### Configuration Layer

配置层的目标是把“任务定义”和“训练配置”从脚本流程里拆出来。

这样做的好处有三个：
- 参数更集中，便于学习和查阅。
- 命令行参数与代码内部配置可以一一对应。
- 训练后的 `run_config.json` 可以直接用于复现实验。

### Environment Layer

环境层的实现顺序如下：

1. 加载 `assets/robotiq_cxy/lab_env.xml`
2. 解析关节、执行器、目标体和指尖 body 的索引
3. 在 `reset()` 中设置机械臂初始姿态并采样目标点
4. 在 `step()` 中把策略动作映射成控制量
5. 推进一步 MuJoCo 物理
6. 计算观测、奖励、终止条件和 `info`

### Reward Design

当前奖励由几类项组成：
- 时间惩罚
- 距离惩罚
- 靠近目标的增量奖励
- 远离目标的惩罚
- 朝目标方向运动的奖励
- 动作幅值与动作变化惩罚
- 关节速度惩罚
- 成功奖励
- 碰撞惩罚
- 跑飞惩罚

这些项的目的不是简单堆数值，而是让策略学会：
- 持续靠近目标
- 不要剧烈抖动
- 不要高速撞击
- 成功时尽量稳定

### Training Flow

训练入口的核心流程是：

1. 解析命令行参数
2. 生成 `UR5ReachEnvConfig` 和 `RLTrainConfig`
3. 构建训练环境与评估环境
4. 根据 `algo` 实例化 `TD3`、`SAC` 或 `PPO`
5. 注册回调：
   - `EvalCallback`
   - `SaveVecNormalizeCallback`
   - `ManualInterruptCallback`
   - `TrainRenderCallback` 或 `SpectatorRenderCallback`
6. 保存最终模型和归一化参数

## Dependencies

### Gymnasium

用途：
- 定义环境接口
- 统一 `reset()` / `step()` / `render()` 的行为

在项目中的位置：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)

### MuJoCo

用途：
- 负责物理仿真
- 负责 XML 模型加载
- 负责渲染与碰撞检测

在项目中的位置：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)

### Stable-Baselines3

用途：
- 提供 `TD3`、`SAC`、`PPO`
- 提供 `VecNormalize`、`EvalCallback` 等训练工具

在项目中的位置：
- [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)

### NumPy

用途：
- 处理观测、动作和奖励中的数组运算

在项目中的位置：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
- [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)

## Recommended Reading Order

1. 先读 [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py)
2. 再读 [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
3. 然后读 [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)
4. 最后结合 [docs/PARAMETER_REFERENCE.md](/home/zzyfan/mujoco_ur5_rl/docs/PARAMETER_REFERENCE.md) 和 notebook 一起看
