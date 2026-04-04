# MuJoCo UR5 RL

纯 MuJoCo 机械臂强化学习项目，当前包含两条训练线：

- `classic/`：主训练线，基于 Gymnasium + Stable-Baselines3
- `mjx/`：MuJoCo Playground + MJWarp 适配入口

当前仓库支持两套机械臂模型：

- `ur5_cxy`
- `zero_robotiq`

## 快速开始

安装依赖：

```bash
cd /home/zzyfan/mujoco_ur5_rl
pip install -r requirements.txt
```

随机策略冒烟测试：

```bash
python -m classic.test --random-policy --episodes 1 --max-steps 10 --no-render
```

开始训练：

```bash
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac \
  --timesteps 500000 \
  --device cuda \
  --no-render
```

测试训练结果：

```bash
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac \
  --episodes 3 \
  --render
```

## 训练线说明

### `classic/`

这条线是当前推荐主线。

- 环境：`classic/env.py`
- 训练入口：`classic/train.py`
- 测试入口：`classic/test.py`
- 算法：`SAC / PPO / TD3`
- 默认物理后端：`mujoco`

特点：

- 支持课程学习
- 支持 `VecNormalize`
- 支持 `best_model / final / interrupted`
- 支持 `zero_robotiq` 兼容参数
- 支持 `--legacy-zero-ee-velocity` 复现 zeroarm 旧速度语义

### `mjx/`

这条线用于接 Playground / MJWarp 实验：

- `mjx/train.py`
- `mjx/benchmark.py`
- `mjx/backend.py`

示例：

```bash
python mjx/benchmark.py --run-smoke --env-name CartpoleBalance
python mjx/train.py --trainer jax-ppo --env-name CartpoleBalance --impl warp
```

## 常用命令

训练 zero 机械臂：

```bash
python classic/train.py \
  --algo sac \
  --robot zero_robotiq \
  --run-name exp_zero \
  --timesteps 500000 \
  --device cuda \
  --no-render
```

继续训练：

```bash
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac \
  --resume \
  --timesteps 300000 \
  --device cuda \
  --skip-replay-buffer
```

测试最佳模型：

```bash
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --model-path logs/classic/sac/ur5_cxy/exp_sac/best_model/best_model.zip \
  --norm-path logs/classic/sac/ur5_cxy/exp_sac/best_model/vec_normalize.pkl \
  --episodes 3 \
  --render
```

兼容 zeroarm 旧末端速度语义：

```bash
python classic/train.py \
  --algo sac \
  --robot zero_robotiq \
  --run-name exp_zero_legacy \
  --legacy-zero-ee-velocity \
  --no-render
```

## 输出目录

`classic/` 默认输出：

```text
models/classic/{algo}/{robot}/{run_name}/
  final/
  interrupted/

logs/classic/{algo}/{robot}/{run_name}/
  best_model/
```

说明：

- `final/`：正常训练结束产物
- `interrupted/`：`Ctrl+C` 中断保存
- `best_model/`：评估回调保存
- 旧版输出会在运行时自动同步到 `classic` 分层目录

## 推荐工作流

1. 先跑 `python -m classic.test --random-policy --episodes 1 --max-steps 10 --no-render`
2. 再跑 `classic/train.py --no-render`
3. 需要恢复时使用 `--resume`
4. 需要复现 zeroarm 旧行为时再加 `--legacy-zero-ee-velocity`
5. 验收时用 `classic/test.py` 可视化

## 代码结构

更精简的结构说明见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

项目交付笔记见 `PROJECT_STATUS.ipynb`。
