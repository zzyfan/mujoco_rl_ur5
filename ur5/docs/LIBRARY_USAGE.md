# Library Usage

本文档说明项目中主要外部库的职责、在代码中的使用位置，以及这些库是如何参与实现流程的。

## Main Pipeline

### Gymnasium

主要职责：
- 定义强化学习环境接口
- 统一 `reset()`、`step()`、`render()` 和 `spaces`

对应文件：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)

在本项目中的典型用法：
- `UR5ReachEnv` 继承 `gym.Env`
- 用 `spaces.Box` 定义动作空间和观测空间
- 按 Gymnasium 约定返回 `(observation, info)` 和 `(observation, reward, terminated, truncated, info)`

为什么这样用：
- 这样主线环境可以直接接入 Stable-Baselines3，无需额外适配层。

### MuJoCo

主要职责：
- 加载 XML 模型
- 负责动力学推进
- 负责碰撞检测
- 负责人类可视化与离屏渲染

对应文件：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)

在本项目中的典型用法：
- `mujoco.MjModel.from_xml_path(...)` 加载机器人和场景模型
- `mujoco.MjData(...)` 保存运行时物理状态
- `mujoco.mj_step(...)` 推进一步动力学
- `mujoco.viewer` 和 `mujoco.Renderer` 负责可视化

为什么这样用：
- MuJoCo 是主线环境的物理基础，环境里的动作映射、碰撞判断和渲染都直接依赖它。

### Stable-Baselines3

主要职责：
- 提供 `TD3`、`SAC`、`PPO`
- 提供 `VecNormalize`
- 提供 `EvalCallback` 等训练工具

对应文件：
- [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)

在本项目中的典型用法：
- `TD3`、`SAC`、`PPO` 负责策略训练
- `make_vec_env` 负责批量创建环境
- `EvalCallback` 负责周期性评估
- `VecNormalize` 负责观测与奖励归一化
- `NormalActionNoise` 只在 TD3 中用于连续动作探索

为什么这样用：
- SB3 已经提供成熟的单机训练流程，主线只需要把任务环境接进去，并把参数映射清楚。

### NumPy

主要职责：
- 处理观测、动作、奖励和日志中的数组运算

对应文件：
- [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
- [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)

在本项目中的典型用法：
- 观测拼接
- 动作裁剪与平滑
- 距离、速度和奖励项计算
- 训练日志中对数组结果做转换

为什么这样用：
- 主线环境是面向 MuJoCo 和 SB3 的单机流程，NumPy 是最直接的数组处理工具。

## Warp Pipeline

### JAX

主要职责：
- 表达 Warp 环境中的张量计算
- 在推理时生成和处理策略动作

对应文件：
- [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

在本项目中的典型用法：
- `jax.numpy` 构造和更新函数式状态
- `jax.random` 负责目标采样
- JAX 数组用于环境 `reset()` / `step()` 的输入输出

为什么这样用：
- Warp 训练线要和 Brax 训练器对接，因此环境状态和运算路径都需要保持在 JAX 兼容形式上。

### Brax

主要职责：
- 提供 `SAC` 与 `PPO` 训练器
- 提供参数保存与网络构造工具

对应文件：
- [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

在本项目中的典型用法：
- `brax.training.agents.sac.train`
- `brax.training.agents.ppo.train`
- `brax.io.model.save_params`
- 推理阶段使用 Brax 网络工厂重建策略函数

为什么这样用：
- Warp 训练线重点是高吞吐批量训练，因此直接使用 Brax 自带训练器更合适。

### MuJoCo Playground

主要职责：
- 提供 `MjxEnv` 基类
- 提供 Brax 训练包装器

对应文件：
- [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
- [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)

在本项目中的典型用法：
- `mjx_env.MjxEnv` 作为环境基类
- `mjx_env.make_data(...)` 构造初始状态
- `mjx_env.step(...)` 推进动力学
- `wrapper.wrap_for_brax_training(...)` 把环境包装成 Brax 训练器可用格式

为什么这样用：
- 它把 MuJoCo MJX 环境和 Brax 训练器连接起来，省去了大量自定义适配代码。

### warp-lang / mujoco-warp

主要职责：
- 提供纯 GPU 物理后端
- 让 MuJoCo 模型运行在 Warp 实现上

对应文件：
- [warp_ur5_runtime.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_runtime.py)

在本项目中的典型用法：
- `warp.init()` 初始化运行时
- `warp.get_cuda_devices()` 查询可用 GPU
- `warp.set_device(...)` 固定当前设备
- `mjx.put_model(..., impl="warp")` 让 MuJoCo 模型运行在 Warp 实现上

为什么这样用：
- 纯 GPU 训练线的核心目标就是把大规模并行环境放到 GPU 上执行。

### Flax / Orbax

主要职责：
- 加载中间 checkpoint
- 恢复标准化器状态和网络参数树

对应文件：
- [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)

在本项目中的典型用法：
- Orbax 用于恢复 checkpoint 目录中的参数树
- Flax 用于把标准化状态恢复回运行中的数据结构

为什么这样用：
- Warp 训练线的中间 checkpoint 不是单个 zip 文件，而是参数树和状态树的组合。

## Reading Order

建议按下面顺序理解依赖关系：

1. 先看环境依赖：`Gymnasium / MuJoCo / JAX / MuJoCo Playground`
2. 再看训练依赖：`Stable-Baselines3 / Brax`
3. 最后看运行时与参数恢复：`warp-lang / mujoco-warp / Flax / Orbax`

## Recommended Companion Files

如果你想把“库的职责”和“代码里的实现位置”一起对照着看，建议同时打开：

1. [docs/IMPLEMENTATION_GUIDE.md](/home/zzyfan/mujoco_ur5_rl/docs/IMPLEMENTATION_GUIDE.md)
2. [docs/WARP_IMPLEMENTATION_GUIDE.md](/home/zzyfan/mujoco_ur5_rl/docs/WARP_IMPLEMENTATION_GUIDE.md)
3. [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)
4. [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
5. [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
6. [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
