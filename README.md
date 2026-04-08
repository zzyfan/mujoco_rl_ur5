# UR5 Reach RL

## Overview

本仓库聚焦一件事：使用同一套 UR5 + Robotiq MuJoCo 模型完成到点任务训练，并维护两条训练线。

- 主线：`Stable-Baselines3 + MuJoCo`
- Warp 线：`Brax + MJX/Warp`

这次整理后的目标是让代码、命令行、模型目录和文档保持同一套规则：

- 主线测试不再手写模型路径，只传 `--model best` 或 `--model final`
- 测试渲染统一使用 Gymnasium 官方风格的 `--render-mode human`
- 主线和 Warp 线都按“训练线 / 算法 / 实验名”分目录
- `best_model/` 和 `final_model/` 都放在各自算法实验目录下

## Project Layout

```text
assets/
  robotiq_cxy/
    lab_env.xml
    lab_env_no_gripper.xml
    meshes/
docs/
  TRAINING_GUIDE.md
notebooks/
  01_ur5_reach_tutorial.ipynb
  02_codebase_library_notes.ipynb
  03_main_and_warp_design_notes.ipynb
train_ur5_reach.py
train_ur5_reach_warp.py
ur5_reach_config.py
ur5_reach_env.py
warp_ur5_config.py
warp_ur5_env.py
warp_ur5_runtime.py
requirements.txt
requirements-warp.txt
```

## Installation

主线推荐 Python `3.10` 或 `3.11`。

```bash
pip install -r requirements.txt
```

若需要 Warp 训练线，再额外安装：

```bash
pip install -r requirements-warp.txt
```

说明：

- Windows 通常只跑主线，因为本地 JAX / Brax / Warp 运行时不稳定。
- Ubuntu 服务器更适合跑 Warp 线，能够减少 CPU/GPU 之间的数据搬运。

## Main Pipeline

主线入口是 [train_ur5_reach.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/train_ur5_reach.py)。

支持算法：

- `td3`
- `sac`
- `ppo`

训练示例：

```bash
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --total-timesteps 1500000
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --total-timesteps 1500000
python train_ur5_reach.py --algo ppo --run-name ur5_ppo_main --total-timesteps 1500000
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --total-timesteps 1500000 --disable-gripper-end-effector
```

测试示例：

```bash
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model best --episodes 1 --render-mode human
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model final --episodes 1 --render-mode human
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model best --episodes 1 --render-mode human --disable-gripper-end-effector
```

关键规则：

- `--model best|final` 由代码自动解析模型目录
- `--render-mode` 只接受 `none` 或 `human`
- `--disable-gripper-end-effector` 会切换到不带夹爪的简化末端模型
- 测试时不需要再手写 `--model-path` 或 `--normalize-path`
- 带夹爪和不带夹爪的实验会自动分开保存，避免同名目录互相覆盖

## Warp Pipeline

Warp 训练入口是 [train_ur5_reach_warp.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/train_ur5_reach_warp.py)。

支持算法：

- `sac`
- `ppo`

训练示例：

```bash
python train_ur5_reach_warp.py --algo sac --run-name ur5_warp_sac --num-envs 256 --num-timesteps 5000000
python train_ur5_reach_warp.py --algo ppo --run-name ur5_warp_ppo --num-envs 256 --num-timesteps 5000000
```

当前约定：

- 本地仓库先只维护 Warp 训练入口
- Warp 测试留到服务器环境执行
- Warp 线也统一写入 `best_model/` 和 `final_model/`

补充说明：

- Brax 公共训练接口当前稳定暴露的是“最终参数”
- 因此 Warp 线当前会把 `best_model/` 先镜像为最终导出，保证目录结构统一
- 后续若服务器训练链路能拿到独立最佳参数，再把这一层替换成真实 best export

## Artifact Layout

所有训练产物都放在仓库内相对路径，并按作用域、训练线、算法和实验名区分。

主线：

```text
runs/{local|server}/main/{algo}/{run_name}/
  run_config.json
  tensorboard/
  best_model/
    best_model.zip
    vec_normalize.pkl
  final_model/
    final_model.zip
    vec_normalize.pkl
  interrupted/
    interrupted_model.zip
    vec_normalize.pkl
  final_eval.json
```

若使用 `--disable-gripper-end-effector`，主线会自动改存到：

```text
runs/{local|server}/main/{algo}/{run_name}__no_gripper/
  run_config.json
  tensorboard/
  best_model/
    best_model.zip
    vec_normalize.pkl
  final_model/
    final_model.zip
    vec_normalize.pkl
  interrupted/
    interrupted_model.zip
    vec_normalize.pkl
  final_eval.json
```

Warp 线：

```text
runs/{local|server}/warp/{algo}/{run_name}/
  config.json
  checkpoints/
  best_model/
    best_policy.msgpack
  final_model/
    final_policy.msgpack
  final_eval.json
```

说明：

- Windows 默认写入 `runs/local/...`
- Linux 服务器默认写入 `runs/server/...`
- 可用 `UR5_ARTIFACT_SCOPE=local` 或 `UR5_ARTIFACT_SCOPE=server` 手动覆盖

## Task Semantics

- 主线参考点现在跟末端模型联动
- 带夹爪模型：成功判定和相对位置都基于两指中点
- 不带夹爪模型：成功判定和相对位置都基于 `ee_link` 原点
- 相对位置统一定义为 `target_position - reference_point`
- 目标球本身带碰撞体积；机器人碰到目标球会进入碰撞惩罚逻辑
- 主线课程学习按回合阶段走 `fixed -> local_random -> full_random`
- Warp 线当前仍使用显式采样模式：`fixed`、`small_random`、`full_random`

## Logging And Final Eval

主线训练会输出：

- `observation_schema`
- `[train_step]`：相对距离、相对速度、累计成功次数、当前回合回报、碰撞计数
- `[episode_end]`：并行环境单回合摘要，按稳定顺序打印
- `[final_eval]`：`min_distance`、`max_return`、`successes`、`success_rate`

Warp 训练会输出：

- 训练进度条
- `[warp_step]`：Brax 聚合指标
- `[warp_episode]`：训练期可见的回合聚合摘要
- `[final_eval]`：训练结束后的成功率、成功次数、最小距离和最大回报摘要

说明：

- 主线最终评估来自显式测试回合
- Warp 线当前最终评估来自 Brax 训练评估流汇总，方便在服务器训练完成时直接输出统一摘要

## Docs And Notes

推荐阅读顺序：

1. [ur5_reach_config.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/ur5_reach_config.py)
2. [ur5_reach_env.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/ur5_reach_env.py)
3. [train_ur5_reach.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/train_ur5_reach.py)
4. [warp_ur5_config.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/warp_ur5_config.py)
5. [warp_ur5_env.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/warp_ur5_env.py)
6. [train_ur5_reach_warp.py](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/train_ur5_reach_warp.py)
7. [TRAINING_GUIDE.md](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/docs/TRAINING_GUIDE.md)
8. [01_ur5_reach_tutorial.ipynb](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/notebooks/01_ur5_reach_tutorial.ipynb)
9. [02_codebase_library_notes.ipynb](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/notebooks/02_codebase_library_notes.ipynb)
10. [03_main_and_warp_design_notes.ipynb](/f:/mujoco_rl_ur5-main/mujoco_rl_ur5/notebooks/03_main_and_warp_design_notes.ipynb)

补充说明：

- `01_ur5_reach_tutorial.ipynb` 更偏命令和使用流程
- `02_codebase_library_notes.ipynb` 更偏 Python / 库用法
- `03_main_and_warp_design_notes.ipynb` 专门解释主线 class 和 Warp 两条线的代码设计思路，尤其重点讲奖励机制设计
