# MJLAB 训练线说明

## 目标

这条训练线的目标不是替换掉仓库里原来的 `Gymnasium + SB3` 主线，而是在保留原 reach 任务定义的前提下，额外补一套 `mjlab` manager-based 训练入口。

对应文件：

- `mjlab_ur5_task.py`
- `train_ur5_reach_mjlab.py`
- `play_ur5_reach_mjlab.py`
- `assets/robotiq_cxy/lab_env_mjlab.xml`

## 迁移思路

这次迁移刻意保持“任务本体不变，环境组织方式变化”：

- 机械臂仍然沿用仓库内的 `UR5 + Robotiq` MuJoCo 模型。
- 目标采样空间、课程学习、控制模式和奖励项，继续复用 [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py)。
- 训练调度改成 `mjlab` 官方推荐的 manager-based 结构。
- 训练器改成 `mjlab + RSL-RL` 的 PPO 训练线。

## 环境安装

### 官方推荐方式：使用 uv

`mjlab` 官方当前更推荐使用 `uv` 管理环境与依赖解析。

在仓库根目录执行：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra cu128
```

如果你只是 CPU 或没有合适的 CUDA 环境，可以改成：

```bash
uv sync --extra cpu
```

无头服务器通常还建议补上：

```bash
export MUJOCO_GL=egl
```

### pip 兼容方式

如果你当前工作流还是基于 `pip`，也可以直接执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 训练

最常用的最小命令：

```bash
uv run train-ur5-reach-mjlab --run-name ur5_reach_mjlab --num-envs 1024
```

如果你更习惯直接调用 Python：

```bash
python train_ur5_reach_mjlab.py --run-name ur5_reach_mjlab --num-envs 1024
```

说明：

- `num_envs` 控制并行环境数，越大越吃显存。
- 更细的任务参数继续挂在 `env` 下面，例如：

```bash
uv run train-ur5-reach-mjlab \
  --run-name ur5_reach_joint_delta \
  --env.control-mode joint_delta \
  --env.curriculum-fixed-episodes 200
```

## 推理与可视化

若不显式给 checkpoint，脚本会自动寻找最新一次训练的最新 `model_*.pt`：

```bash
uv run play-ur5-reach-mjlab
```

也可以手动指定：

```bash
uv run play-ur5-reach-mjlab \
  --checkpoint-file logs/rsl_rl/ur5_reach_mjlab/你的运行目录/model_4000.pt
```

如果只想检查任务通路而不加载策略：

```bash
uv run play-ur5-reach-mjlab --agent zero
uv run play-ur5-reach-mjlab --agent random
```

## 日志目录

`mjlab` 训练线默认输出到：

```text
logs/rsl_rl/ur5_reach_mjlab/
```

每次运行会自动再创建一层时间戳目录，并在里面保存：

- `params/`：环境与训练配置
- `events.out.tfevents.*`：TensorBoard 日志
- `model_*.pt`：中间 checkpoint
- `videos/`：训练或播放录屏

## 和 terrain 文档的对应关系

虽然当前 reach 任务不是地形任务，但这次环境组织方式刻意参考了 `mjlab` terrain 文档里那种 manager-based 拆分思路：

- `scene` 里只放实体和传感器
- `events` 里处理 reset 与目标采样
- `observations`、`rewards`、`terminations` 分开声明

这样后面如果你想继续加更多物体、障碍或者更复杂的场景随机化，结构会更容易扩展。
