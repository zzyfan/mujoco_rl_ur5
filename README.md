# MuJoCo 机械臂项目说明（MTD3 笔记风格）

本项目是纯 MuJoCo 的机械臂强化学习工程，支持 `SAC / PPO / TD3`，支持 `UR5_CXY` 与 `zero+Robotiq` 两套模型，支持中断保存与恢复训练。

## 0. 结构分层（已区分）

- `classic/`：原始主训练线入口（完整约束 + 渲染）
- `mjx/`：独立 MJX / Warp 训练线入口（GPU 并行实验）
- `docs/`：结构说明与文档索引

## 1. 项目目标

训练机械臂完成到点任务：

- 环境：Gymnasium + MuJoCo
- 算法：Stable-Baselines3（SAC / PPO / TD3）
- 判定：两指中心点到目标点

## 2. 快速开始

### 2.1 安装依赖

```bash
cd /home/zzyfan/mujoco_ur5_rl
pip install -r requirements.txt
```

### 2.2 冒烟测试（随机策略）

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/test.py --random-policy --episodes 1 --render-mode human
```

### 2.3 开始训练（推荐先无渲染）

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py --algo sac --robot ur5_cxy --run-name exp_sac --timesteps 500000 --device cuda --no-render
```

默认行为（对齐 zero 原始 `train_robot_arm.py`）：
- 评估开启时（`--eval-freq > 0`）会自动保存 `logs/classic/{algo}/{robot}/{run_name}/best_model/`
- 训练结束会保存 `final/model.zip` 与 `final/vec_normalize.pkl`
- 如需关闭 best 保存可加 `--no-save-best-model`

### 2.4 Docker（GPU）

构建镜像：

```bash
cd /home/zzyfan/mujoco_ur5_rl
docker build -t mujoco-ur5-rl:latest .
```

进入容器（挂载当前项目目录，使用宿主 GPU）：

```bash
cd /home/zzyfan/mujoco_ur5_rl
docker run --rm -it --gpus all \
  -v /home/zzyfan/mujoco_ur5_rl:/workspace/mujoco_ur5_rl \
  -w /workspace/mujoco_ur5_rl \
  mujoco-ur5-rl:latest
```

容器内训练示例（classic）：

```bash
python classic/train.py --algo sac --robot ur5_cxy --run-name exp_sac_docker --timesteps 500000 --device cuda --no-render
```

容器内训练示例（mjx）：

```bash
python mjx/train.py --algo sac --robot zero_robotiq --run-name mjx_sac_docker --timesteps 500000 --n-envs 1 --device cuda --safe-disable-constraints
```

## 3. 常用训练命令

### 3.1 训练时显示 MuJoCo 界面

```bash
python classic/train.py --algo sac --robot ur5_cxy --run-name exp_sac_vis --timesteps 500000 --device cuda --render --render-mode human --n-envs 1
```

### 3.2 使用 zero 机械臂 + Robotiq 夹爪

```bash
python classic/train.py --algo sac --robot zero_robotiq --run-name exp_zero --timesteps 500000 --device cuda --no-render
```

### 3.3 关闭评估（训练结束更快）

```bash
python classic/train.py --algo sac --run-name exp_fast_exit --timesteps 500000 --eval-freq 0 --no-render
```

### 3.4 相机与目标采样范围（按机械臂独立）

- 训练渲染默认是自由相机（可鼠标拖动）
- 若你需要锁定 XML 固定相机，追加 `--lock-camera`
- UR5 与 ZERO 的目标采样范围可分别设置

```bash
python classic/train.py \
  --algo sac \
  --robot zero_robotiq \
  --run-name exp_zero_custom_range \
  --zero-target-x-min -0.95 --zero-target-x-max -0.65 \
  --zero-target-y-min 0.10 --zero-target-y-max 0.42 \
  --zero-target-z-min 0.12 --zero-target-z-max 0.30 \
  --render --render-mode human --free-camera
```

## 4. 模型保存结构（按后端+算法+机械臂独立）

```text
models/classic/{algo}/{robot}/{run_name}/
  final/
    model.zip
    vec_normalize.pkl
    replay_buffer.pkl        # SAC/TD3
  interrupted/
    model.zip
    vec_normalize.pkl
    replay_buffer.pkl        # SAC/TD3

models/mjx/{algo}/{robot}/{run_name}/
  final/
    model.zip
    vec_normalize.pkl
  interrupted/
    model.zip
    vec_normalize.pkl

logs/classic/{algo}/{robot}/{run_name}/
  best_model/
    best_model.zip
    vec_normalize.pkl

logs/mjx/{algo}/{robot}/{run_name}/
  best_model/
    best_model.zip
    vec_normalize.pkl
```

说明：

- `classic/train.py` 默认读写 `models/classic/...` 与 `logs/classic/...`
- 如果仓库里还存在旧的 `models/{algo}/...`、`logs/{algo}/...` 结果，脚本会在启动时自动同步缺失文件到 `classic` 分层目录
- 后续建议统一只看 `classic` 这套目录，避免继续依赖旧路径回退
- 默认会保存 `best_model`（除非显式传入 `--no-save-best-model`）

## 5. 中断与继续训练

### 5.1 中断训练

- 第一次 `Ctrl+C`：保存 `interrupted` 并退出
- 第二次 `Ctrl+C`：立即强制退出

### 5.2 继续训练（默认从 interrupted）

```bash
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac \
  --resume \
  --timesteps 300000 \
  --device cuda \
  --buffer-size 1000000 \
  --skip-replay-buffer
```

说明：

- `--resume` 会优先读取 `models/classic/{algo}/{robot}/{run_name}/interrupted/`
- `--buffer-size` 可显式限制 SAC/TD3 回放池大小，降低内存压力
- `--skip-replay-buffer` 会跳过旧 replay buffer 恢复，只保留模型权重和 `VecNormalize` 继续训练

### 5.3 指定恢复路径继续训练

```bash
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_resume \
  --resume \
  --resume-model-path models/classic/sac/ur5_cxy/exp_sac/interrupted/model \
  --resume-normalize-path models/classic/sac/ur5_cxy/exp_sac/interrupted/vec_normalize.pkl \
  --resume-replay-path models/classic/sac/ur5_cxy/exp_sac/interrupted/replay_buffer.pkl \
  --timesteps 300000 \
  --device cuda
```

如果你希望继续沿用旧缓存，也可以去掉 `--skip-replay-buffer`，并保留 `--resume-replay-path`。

## 6. 测试训练结果

### 6.1 用训练脚本测试

```bash
python classic/train.py --test --algo sac --robot ur5_cxy --run-name exp_sac --episodes 5 --render-mode human --device cuda
```

### 6.2 用独立脚本测试（推荐按 run-name 自动解析）

```bash
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac \
  --episodes 3 \
  --render-mode human
```

### 6.3 用独立脚本测试（手动指定路径）

```bash
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --model-path models/classic/sac/ur5_cxy/exp_sac/final/model.zip \
  --norm-path models/classic/sac/ur5_cxy/exp_sac/final/vec_normalize.pkl \
  --episodes 3 \
  --render-mode human
```

### 6.4 测试评估出的最佳模型（best_model）

```bash
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --model-path logs/classic/sac/ur5_cxy/exp_sac/best_model/best_model.zip \
  --norm-path logs/classic/sac/ur5_cxy/exp_sac/best_model/vec_normalize.pkl \
  --episodes 3 \
  --render-mode human
```

## 7. 关键文件

- `classic/train.py`：classic 训练入口（推荐）
- `classic/test.py`：classic 测试入口（推荐，支持 `--run-name` 自动找模型）
- `mjx/train.py`：MJX 训练入口（推荐）
- `mjx/benchmark.py`：MJX 并行压测入口（推荐）
- `mjx/backend.py`：MJX / MuJoCo Warp 后端选择与检测
- `docs/STRUCTURE.md`：分层结构说明
- `assets/robotiq_cxy/`：UR5 资源
- `assets/zero_arm/`：zero 与 zero+Robotiq 资源
- `PROJECT_USAGE_MTD3_STYLE.md`：完整使用手册（同风格）

## 8. 单独新线（MJX/Warp 预览）

这条线是“单独构建”的，不会影响 `classic/train.py` 主训练线。

```bash
cd /home/zzyfan/mujoco_ur5_rl
python mjx/benchmark.py --robot zero_robotiq --batch-size 256 --steps 1000 --safe-disable-constraints
```

说明：
- 这是后端并行步进压测脚本，不是完整 RL 训练脚本
- 默认 `--physics-backend auto` 优先走 `mjx`
- 如果你想强制走 MuJoCo Warp，显式加 `--physics-backend warp`

## 9. 单独新线（MJX 完整训练）

如果你要直接训练 MJX 独立线（不影响 `classic/train.py`）：

```bash
cd /home/zzyfan/mujoco_ur5_rl
python mjx/train.py --algo sac --robot zero_robotiq --run-name mjx_sac_zero --timesteps 500000 --n-envs 1 --device cuda --safe-disable-constraints
```

强制使用 MuJoCo Warp：

```bash
python mjx/train.py --algo sac --robot zero_robotiq --run-name mjx_sac_zero_warp --timesteps 500000 --n-envs 1 --device cuda --safe-disable-constraints --physics-backend warp
```

测试：

```bash
python mjx/train.py --test --algo sac --robot zero_robotiq --run-name mjx_sac_zero --episodes 3 --device cuda
```
