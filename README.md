# MuJoCo UR5 RL

纯 MuJoCo 机械臂强化学习项目，当前包含两条训练线：

- `classic/`：主训练线，基于 Gymnasium + Stable-Baselines3
- `mjx/`：MuJoCo Playground / Brax 训练入口

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
- 支持 `zero_robotiq` 机器人参数
- 支持 `--legacy-zero-ee-velocity` 旧版速度读取开关

### `mjx/`

这条线提供 MuJoCo Playground / Brax 训练入口。

- 环境：`mjx/reach_env.py`
- 训练入口：`mjx/train.py`
- 自检入口：`mjx/benchmark.py`
- 后端检测：`mjx/backend.py`
- 算法：`PPO / SAC`
- 物理后端：`mjx(jax) / warp`

参数说明：

- `--algo ppo`：调用 `brax.training.agents.ppo`
- `--algo sac`：调用 `brax.training.agents.sac`
- `--algo td3`：当前不会启动训练，因为本地 Brax 版本没有 `td3` 训练入口
- `--impl mjx`：映射到 MuJoCo 的 `jax` 实现
- `--num-envs`：控制并行训练环境数量
- `--num-eval-envs`：控制并行评估环境数量

示例：

```bash
python -m mjx.benchmark --robot ur5_cxy --impl mjx --steps 2
python -m mjx.train --algo ppo --robot ur5_cxy --impl warp --num-envs 256
python -m mjx.train --algo sac --robot ur5_cxy --impl mjx --num-envs 16 --num-eval-envs 16
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

启用旧版末端速度读取：

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

`mjx/` 默认输出：

```text
models/mjx/{run_name}/
  checkpoints/
  config.json
  final_policy.msgpack
```

说明：

- `checkpoints/`：Brax 训练过程中保存的阶段参数
- `config.json`：训练参数、环境参数和后端信息
- `final_policy.msgpack`：训练结束时导出的最终策略参数

## 推荐工作流

1. 先跑 `python -m classic.test --random-policy --episodes 1 --max-steps 10 --no-render`
2. 再跑 `classic/train.py --no-render`
3. 需要恢复时使用 `--resume`
4. 需要启用旧版速度读取时再加 `--legacy-zero-ee-velocity`
5. 验收时用 `classic/test.py` 可视化

## 代码结构

更精简的结构说明见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

项目交付笔记见 `PROJECT_STATUS.ipynb`。
