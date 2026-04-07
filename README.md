# UR5 Reach RL (Main + Warp)

<<<<<<< HEAD
## Quick Start

```bash
# 主线训练（SAC）
python train_ur5_reach.py --algo sac --total-timesteps 1500000

# 主线测试（渲染）
python train_ur5_reach.py --algo sac --test --episodes 1 --render

# Warp 训练（SAC）
python train_ur5_reach_warp.py --algo sac --num-envs 256 --num-timesteps 5000000
```

## Files

- `ur5_reach_env.py`：主线 Gymnasium + MuJoCo 环境（zero‑arm 风格）
=======
本仓库只保留 **主线（SB3 + MuJoCo）** 与 **Warp 线（Brax + MJX/Warp）** 两条训练路径。
目标是用同一套 UR5 模型完成到点任务训练，并提供清晰的训练/测试入口。

## 项目结构

- `ur5_reach_env.py`：主线 Gymnasium + MuJoCo 环境（zero-arm 风格）
>>>>>>> fa46e0e (Add learning notebooks and expand inline comments)
- `train_ur5_reach.py`：主线训练与测试入口（TD3/SAC/PPO）
- `warp_ur5_env.py`：Warp 线环境（JAX/MJX/Warp）
- `train_ur5_reach_warp.py`：Warp 线训练入口（SAC/PPO）
- `assets/robotiq_cxy/`：UR5 模型与 meshes
<<<<<<< HEAD

## Notes

- 主线与 Warp 线都使用同一套 UR5 模型文件。
- 主线训练默认保存到 `./logs` 与 `./models`。
- Warp 线训练默认保存到 `./warp_runs/{algo}/{run_name}`。

=======
- `docs/TRAINING_GUIDE.md`：训练与参数速查
- `notebooks/`：教学笔记（使用步骤 + 代码/库学习）

## 快速开始（主线）

### 1) 训练

```bash
python train_ur5_reach.py --algo td3 --total-timesteps 1500000
python train_ur5_reach.py --algo sac --total-timesteps 1500000
python train_ur5_reach.py --algo ppo --total-timesteps 1500000
```

### 2) 训练时渲染

```bash
python train_ur5_reach.py --algo td3 --total-timesteps 1500000 --render-mode human
```

### 3) 测试与渲染

```bash
python train_ur5_reach.py --algo td3 --test --episodes 1 --render-mode human
```

## 快速开始（Warp 线）

```bash
python train_ur5_reach_warp.py --algo sac --num-envs 256 --num-timesteps 5000000
python train_ur5_reach_warp.py --algo ppo --num-envs 256 --num-timesteps 5000000
```

## 训练产物位置

主线训练产物：
- `./logs/best_model/`（最佳模型）
- `./models/{algo}_ur5_final`（训练结束模型）
- `./models/vec_normalize.pkl`（归一化参数）

Warp 线训练产物：
- `./warp_runs/{algo}/{run_name}/`（含 `final_policy.msgpack`）

## 常用参数说明（主线）

- `--algo`：`td3` / `sac` / `ppo`
- `--total-timesteps`：训练总步数
- `--n-envs`：并行环境数
- `--render-mode`：`none` / `human`
- `--render-every`：训练渲染间隔
- `--device`：`auto` / `cpu` / `cuda`

## 常用参数说明（Warp 线）

- `--algo`：`sac` / `ppo`
- `--num-timesteps`：训练总步数
- `--num-envs`：并行训练环境数
- `--num-eval-envs`：并行评估环境数
- `--learning-rate`：学习率

## 教学笔记

- 使用步骤笔记：`notebooks/01_ur5_reach_tutorial.ipynb`
- 代码/库学习笔记：`notebooks/02_codebase_library_notes.ipynb`

## 说明

- 主线与 Warp 线使用同一套 UR5 模型文件。
- 渲染模式使用 Gymnasium 官方命名：`render_mode="human"`。
>>>>>>> fa46e0e (Add learning notebooks and expand inline comments)
