# UR5 Reach Reinforcement Learning Project

## Overview

本仓库实现 `UR5` 到点任务的强化学习训练与测试流程，包含主线 MuJoCo + Stable-Baselines3 实现，以及可选的纯 GPU Warp 训练线。

项目内容聚焦以下方向：

- `UR5` 到点任务环境与训练入口
- 面向学习的代码注释、参数说明和实现文档
- 兼顾本地调试、服务器训练和跨机器迁移的目录与配置设计

ubuntu系统推荐使用warp线，最大化减少硬件数据通讯损失，后期可以迁移使用mujoco的mjlab平台，这是一个结合isaac和mujoco的新平台

windows 因为jax不支持win系统使用另一条线

## Project Structure

```text
assets/
  robotiq_cxy/
    lab_env.xml
    meshes/
docs/
  IMPLEMENTATION_GUIDE.md
  LIBRARY_USAGE.md
  PARAMETER_REFERENCE.md
  PORTABILITY.md
  WARP_IMPLEMENTATION_GUIDE.md
  WARP_PARAMETER_REFERENCE.md
notebooks/
  01_code_learning_walkthrough.ipynb
  02_parameter_reference.ipynb
  03_warp_code_learning_walkthrough.ipynb
  04_cli_parameter_guide.ipynb
  05_library_usage_guide.ipynb
train_ur5_reach.py
ur5_reach_config.py
ur5_reach_env.py
train_ur5_reach_warp.py
test_ur5_reach_warp.py
warp_ur5_config.py
warp_ur5_env.py
warp_ur5_runtime.py
requirements.txt
requirements-warp.txt
Dockerfile
```

## Robot Model

当前机械臂模型保存在仓库内相对路径：

```text
assets/robotiq_cxy/lab_env.xml
assets/robotiq_cxy/meshes/
```

说明：

- `lab_env.xml` 是主线和 Warp 训练线共同使用的 MuJoCo 场景与机械臂模型入口。
- `meshes/` 目录保存 UR5 与 Robotiq 夹爪所需的网格文件。
- 训练脚本通过配置中的 `model_xml` 相对路径加载模型，不依赖当前终端所在目录。

## Main Pipeline

`ur5_reach_config.py`

- 定义环境参数、训练参数、产物路径规则。
- 维护参数说明字典，方便 Jupyter 文档和代码学习时直接引用。

`ur5_reach_env.py`

- 定义 Gymnasium + MuJoCo 的 `UR5ReachEnv`。
- 负责 reset、目标采样、动作到控制量的映射、奖励计算、碰撞判断和渲染。

`train_ur5_reach.py`

- 提供统一的训练与测试入口。
- 支持 `td3`、`sac`、`ppo`。
- 负责环境构建、模型初始化、回调注册、模型保存和评估流程。

## Warp Pipeline

`warp_ur5_config.py`

- 定义 Warp 训练线使用的环境参数、训练参数和参数说明字典。

`warp_ur5_env.py`

- 定义 `UR5WarpReachEnv`。
- 负责 Warp 环境中的 reset、目标采样、控制映射、奖励计算和接触判断。

`warp_ur5_runtime.py`

- 检查 `warp-lang`、`mujoco-warp` 和 `mujoco_playground` 是否可用。
- 负责初始化 CUDA 设备并输出运行时信息。

`train_ur5_reach_warp.py`

- Warp 训练入口。
- 支持 `sac` 与 `ppo`。

`test_ur5_reach_warp.py`

- Warp 推理与可视化入口。
- 支持加载最终参数或中间 checkpoint。

## Installation

推荐 Python `3.10` 或 `3.11`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

若在服务器或无显示环境上运行，可以先设置：

```bash
export MUJOCO_GL=egl
```

若在本地图形桌面运行，通常不需要额外设置。

## Warp Installation

若需要纯 GPU Warp 训练线，请在安装主线依赖后继续执行：

```bash
pip install -r requirements-warp.txt
```

Warp 训练线依赖：

- `jax`
- `brax`
- `flax`
- `orbax-checkpoint`
- `mujoco-playground`
- `warp-lang`
- `mujoco-warp`

## Training

### SAC

```bash
python train_ur5_reach.py \
  --algo sac \
  --run-name ur5_sac_main \
  --total-timesteps 1500000
```

### TD3

```bash
python train_ur5_reach.py \
  --algo td3 \
  --run-name ur5_td3_main \
  --total-timesteps 1500000
```

### PPO

```bash
python train_ur5_reach.py \
  --algo ppo \
  --run-name ur5_ppo_main \
  --total-timesteps 1500000
```

## Warp Training

### Warp SAC

```bash
python train_ur5_reach_warp.py \
  --algo sac \
  --run-name ur5_warp_sac \
  --num-envs 256 \
  --num-timesteps 5000000
```

### Warp PPO

```bash
python train_ur5_reach_warp.py \
  --algo ppo \
  --run-name ur5_warp_ppo \
  --num-envs 256 \
  --num-timesteps 5000000
```

## Spectator Mode

`--spectator-render` 用于在训练保持无头并行时，额外在主进程启动一个独立窗口进行旁观。

示例：

```bash
python train_ur5_reach.py \
  --algo sac \
  --run-name ur5_sac_watch \
  --n-envs 4 \
  --spectator-render
```

相关参数：

- `--spectator-render-every 200`：控制旁观窗口的更新间隔。
- `--no-spectator-deterministic`：旁观环境使用随机动作而不是确定性动作。

与 `--render-training` 的区别：

- `--render-training` 直接渲染训练环境本体，因此会退回单进程环境或 `DummyVecEnv`。
- `--spectator-render` 保持训练环境无头并行，窗口来自主进程中的独立旁观环境。

## Evaluation

### 测试最佳模型

```bash
python train_ur5_reach.py \
  --algo sac \
  --run-name ur5_sac_main \
  --test \
  --episodes 5 \
  --render
```

### 打印奖励分解

```bash
python train_ur5_reach.py \
  --algo td3 \
  --run-name ur5_td3_main \
  --test \
  --episodes 2 \
  --print-reward-terms
```

## Warp 测试命令

```bash
python test_ur5_reach_warp.py \
  --algo sac \
  --run-name ur5_warp_sac \
  --episodes 3 \
  --render
```

## Artifacts

所有训练产物都保存在仓库内相对路径，并按训练线、算法和实验名区分：

```text
runs/{local|server}/main/{algo}/{run_name}/
  run_config.json
  tensorboard/
  best_model/
    best_model.zip
    vec_normalize.pkl
  final/
    final_model.zip
    vec_normalize.pkl
  interrupted/
    interrupted_model.zip
    vec_normalize.pkl
```

Warp 训练线写入：

```text
runs/{local|server}/warp/{algo}/{run_name}/
  config.json
  checkpoints/
  final_policy.msgpack
```

说明：

- 本地开发机默认写入 `runs/local/...`。
- 服务器训练机默认写入 `runs/server/...`。
- 如需手动覆盖，可设置环境变量 `UR5_ARTIFACT_SCOPE=local` 或 `UR5_ARTIFACT_SCOPE=server`。

## Algorithms

主线保留以下算法：

- `td3`
- `sac`
- `ppo`

Warp 训练线当前提供：

- `sac`
- `ppo`

项目范围只保留 `UR5` 到点任务，不包含其他机器人任务分支。

## Task Semantics

- 成功判定使用“两个指尖中点”作为末端参考点，而不是 `ee_link` 原点。
- 主线里的参考点由 `left_follower_link` 和 `right_follower_link` 的中点构成；Warp 线保持相同定义。
- 目标相对位置统一定义为 `target_position - finger_center`，观测前 3 维、距离奖励和成功判定都使用这一参考系。
- 成功条件本质上是欧氏距离不大于当前阶段的成功阈值。
- 目标球本身不再被碰撞白名单忽略，因此机器人碰到目标球也会进入碰撞惩罚逻辑。

## Logging

- 主线训练会在开始时打印观测向量每一段的真实含义，便于直接对照 `obs` 的切片语义。
- 主线训练中的 `[train_step]` 会持续输出相对距离、相对速度、累计成功次数、当前回合回报和碰撞计数。
- 主线训练中的 `[episode_end]` 会在并行环境里按稳定顺序输出，每条日志都带有 `order=` 和 `env=`，便于复盘多环境回合结束顺序。
- Warp 训练线保留进度条，并在 Brax 回调可见的粒度上输出聚合指标；它不会像主线那样拿到每个并行子环境的逐回合事件。

## Documentation

Recommended reading order for the main pipeline:

1. [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py)
2. [ur5_reach_env.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_env.py)
3. [train_ur5_reach.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach.py)
4. [docs/IMPLEMENTATION_GUIDE.md](/home/zzyfan/mujoco_ur5_rl/docs/IMPLEMENTATION_GUIDE.md)
5. [docs/PARAMETER_REFERENCE.md](/home/zzyfan/mujoco_ur5_rl/docs/PARAMETER_REFERENCE.md)
6. [docs/LIBRARY_USAGE.md](/home/zzyfan/mujoco_ur5_rl/docs/LIBRARY_USAGE.md)
7. [docs/PORTABILITY.md](/home/zzyfan/mujoco_ur5_rl/docs/PORTABILITY.md)
8. [notebooks/01_code_learning_walkthrough.ipynb](/home/zzyfan/mujoco_ur5_rl/notebooks/01_code_learning_walkthrough.ipynb)
9. [notebooks/02_parameter_reference.ipynb](/home/zzyfan/mujoco_ur5_rl/notebooks/02_parameter_reference.ipynb)
10. [notebooks/04_cli_parameter_guide.ipynb](/home/zzyfan/mujoco_ur5_rl/notebooks/04_cli_parameter_guide.ipynb)
11. [notebooks/05_library_usage_guide.ipynb](/home/zzyfan/mujoco_ur5_rl/notebooks/05_library_usage_guide.ipynb)

Recommended reading order for the Warp pipeline:

1. [warp_ur5_config.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_config.py)
2. [warp_ur5_runtime.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_runtime.py)
3. [warp_ur5_env.py](/home/zzyfan/mujoco_ur5_rl/warp_ur5_env.py)
4. [train_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/train_ur5_reach_warp.py)
5. [test_ur5_reach_warp.py](/home/zzyfan/mujoco_ur5_rl/test_ur5_reach_warp.py)
6. [docs/WARP_IMPLEMENTATION_GUIDE.md](/home/zzyfan/mujoco_ur5_rl/docs/WARP_IMPLEMENTATION_GUIDE.md)
7. [docs/WARP_PARAMETER_REFERENCE.md](/home/zzyfan/mujoco_ur5_rl/docs/WARP_PARAMETER_REFERENCE.md)
8. [notebooks/03_warp_code_learning_walkthrough.ipynb](/home/zzyfan/mujoco_ur5_rl/notebooks/03_warp_code_learning_walkthrough.ipynb)

## Notes

- 默认任务逻辑对齐参考训练线的思路：随机目标、扭矩控制、24 维观测、阶段奖励和成功奖励。
- 机械臂模型、MuJoCo 场景和目标工作空间仍然保持当前仓库自己的 UR5 配置，不直接照搬其他机械臂的结构和坐标范围。
- 如果你要比较不同控制方式，可以切到 `--control-mode joint_delta`。
- 当前文档和代码注释优先服务于任务复现、参数理解和实现流程学习。
