# MuJoCo UR5 RL

纯 MuJoCo 机械臂强化学习项目，当前包含两条训练线：

- `classic/`：主训练线，基于 Gymnasium + Stable-Baselines3
- `warp_gpu/`：Warp GPU + MuJoCo Playground + Brax 训练入口

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

### `warp_gpu/`

这条线提供纯 GPU 的 Warp 训练入口。

- 环境：`warp_gpu/env.py`
- 训练入口：`warp_gpu/train.py`
- 测试入口：`warp_gpu/test.py`
- 自检入口：`warp_gpu/smoke.py`
- 运行时检测：`warp_gpu/runtime.py`
- 算法：`PPO / SAC`
- 物理后端：`warp`

参数说明：

- `--algo ppo`：调用 `brax.training.agents.ppo`
- `--algo sac`：调用 `brax.training.agents.sac`
- `TD3`：当前 Brax 官方安装包没有提供训练入口，因此 `warp_gpu/` 不支持
- `--num-envs`：控制并行训练环境数量
- `--num-eval-envs`：控制并行评估环境数量
- `--naconmax / --naccdmax / --njmax`：控制 Warp 接触缓存和约束缓存大小
- 运行前提：需要 `warp-lang`、`mujoco-warp` 和可用 CUDA 设备

示例：

```bash
python -m warp_gpu.smoke --robot ur5_cxy --steps 2
python -m warp_gpu.train --algo ppo --robot ur5_cxy --num-envs 256
python -m warp_gpu.train --algo sac --robot ur5_cxy --num-envs 16 --num-eval-envs 16
python -m warp_gpu.test --algo sac --robot ur5_cxy --run-name ur5_warp_sac --episodes 3
```

## 训练日志

`classic/` 和 `warp_gpu/` 现在都会在训练过程中打印阶段日志。

- `classic/`：按时间步输出最近窗口内的 `recent_reward / recent_ep_len / recent_distance / success_rate / collision_rate`，若开启评估还会附带 `eval_reward`
- `warp_gpu/`：除了进度条，还会打印 Brax 回调返回的关键指标，例如 `eval_episode_reward / episode_sum_reward / distance / success / collision`
- 训练结束时两条线都会额外打印最后一次可用回报

如果只想看新增日志，可以在服务器上配合：

```bash
tail -f classic_train.log
tail -f warp_gpu_train.log
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

`warp_gpu/` 默认输出：

```text
models/warp_gpu/{algo}/{robot}/{run_name}/
  checkpoints/
  config.json
  final_policy.msgpack
```

说明：

- `checkpoints/`：Brax 训练过程中保存的阶段参数
- `config.json`：训练参数、环境参数和 Warp 运行时信息
- `final_policy.msgpack`：训练结束时导出的最终策略参数
- `warp_gpu/` 当前只接入官方 Brax 自带的 `PPO / SAC`
- `warp_gpu/test.py` 默认读取 `final_policy.msgpack`，也支持指定 `latest-checkpoint` 或某个 checkpoint 目录

## 推荐工作流

1. 先跑 `python -m classic.test --random-policy --episodes 1 --max-steps 10 --no-render`
2. 再跑 `classic/train.py --no-render`
3. 需要恢复时使用 `--resume`
4. 需要启用旧版速度读取时再加 `--legacy-zero-ee-velocity`
5. 验收时用 `classic/test.py` 可视化

## 服务器更新代码

如果服务器本地没有手改文件，直接更新：

```bash
cd /home/zzyfan/mujoco_ur5_rl
git pull --rebase origin main
```

如果服务器本地已经有改动，先暂存再更新：

```bash
cd /home/zzyfan/mujoco_ur5_rl
git status
git stash push -u -m "server-local-changes"
git pull --rebase origin main
git stash pop
```

更新后建议先做一次最小检查：

```bash
python -m py_compile classic/*.py warp_gpu/*.py
python -m classic.test --random-policy --episodes 1 --max-steps 1 --no-render
python -m warp_gpu.smoke --robot ur5_cxy --steps 1
```

## 代码结构

更精简的结构说明见 [docs/STRUCTURE.md](docs/STRUCTURE.md)。

项目交付笔记见 `PROJECT_STATUS.ipynb`，代码学习笔记见 `CODE_LEARNING_NOTES.ipynb`。
