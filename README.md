# UR5 Reach Reinforcement Learning Project

## Overview

本仓库实现 `UR5` 到点任务的强化学习训练与测试流程，当前包含三条训练线：主线 MuJoCo + Stable-Baselines3、可选的纯 GPU Warp 训练线，以及新增的 `mjlab` manager-based 训练线。

项目内容聚焦以下方向：
- `UR5` 到点任务环境与训练入口
- `mjlab` 版 manager-based 任务迁移
- 面向学习的代码注释、参数说明和实现文档
- 兼顾本地调试、服务器训练和跨机器迁移的目录与配置设计

## Project Structure

```text
assets/
  robotiq_cxy/
    lab_env.xml
    meshes/
docs/
  MJLAB_SETUP.md
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
play_ur5_reach_mjlab.py
mjlab_ur5_task.py
train_ur5_reach.py
train_ur5_reach_mjlab.py
ur5_reach_config.py
ur5_reach_env.py
train_ur5_reach_warp.py
test_ur5_reach_warp.py
warp_ur5_config.py
warp_ur5_env.py
warp_ur5_runtime.py
pyproject.toml
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

## MJLAB Pipeline

`mjlab_ur5_task.py`
- 定义 `mjlab` 版 UR5 reach 任务。
- 负责 scene、observation、action、event、reward 和 termination 的 manager-based 组织。

`train_ur5_reach_mjlab.py`
- `mjlab` 训练入口。
- 直接复用官方 `launch_training()`，同时保留仓库自己的 reach 任务参数定义。

`play_ur5_reach_mjlab.py`
- `mjlab` 推理与可视化入口。
- 支持自动定位最近一次训练的 checkpoint。

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

推荐 Python `3.10` 到 `3.12`。

### MJLAB 推荐安装方式

`mjlab` 官方当前更推荐使用 `uv` 管理环境与依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra cu128
```

如果只是 CPU 环境，可以改成：

```bash
uv sync --extra cpu
```

更完整说明见：
- [docs/MJLAB_SETUP.md](/home/zzyfan/mujoco_ur5_rl/docs/MJLAB_SETUP.md)

### pip 兼容安装方式

如果你希望继续沿用 `pip` 工作流，可以安装统一依赖文件：

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

## MJLAB Training

```bash
uv run train-ur5-reach-mjlab \
  --run-name ur5_reach_mjlab \
  --num-envs 1024
```

也可以直接运行 Python 脚本：

```bash
python train_ur5_reach_mjlab.py \
  --run-name ur5_reach_mjlab \
  --num-envs 1024
```

## MJLAB Play

```bash
uv run play-ur5-reach-mjlab
```

若要手动指定 checkpoint：

```bash
uv run play-ur5-reach-mjlab \
  --checkpoint-file logs/rsl_rl/ur5_reach_mjlab/你的运行目录/model_4000.pt
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
runs/main/{algo}/{run_name}/
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
runs/warp/{algo}/{run_name}/
  config.json
  checkpoints/
  final_policy.msgpack
```

## Algorithms

主线保留以下算法：
- `td3`
- `sac`
- `ppo`

Warp 训练线当前提供：
- `sac`
- `ppo`

项目范围只保留 `UR5` 到点任务，不包含其他机器人任务分支。

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
