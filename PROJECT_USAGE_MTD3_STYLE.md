# MuJoCo 机械臂项目使用说明（参考 `mtd3_robot_arm.ipynb` 风格）

这份说明按 `mtd3_robot_arm.ipynb` 的“从环境到训练到测试”的节奏写，适配当前 `mujoco_ur5_rl` 项目代码。

## 0. 分层入口（已做好区分）

- `classic/`：原始主训练线（推荐做主实验）
  - `classic/train.py`
  - `classic/test.py`
- `mjx/`：独立 MJX / Warp 训练线（推荐做并行后端实验）
  - `mjx/train.py`
  - `mjx/benchmark.py`

## 1. 项目目标

本项目用于训练机械臂完成“末端到目标点”的到点任务：

- 仿真引擎：MuJoCo
- 强化学习库：Stable-Baselines3（SAC / PPO / TD3）
- 环境接口：Gymnasium
- 任务判定：两指中心点到达目标点（`left_follower_link` 与 `right_follower_link` 中点）

## 2. 项目结构（你最常用的文件）

- `classic/env.py`：环境定义、奖励函数、渲染、目标采样、课程学习
- `classic/train.py`：训练与测试入口（支持中断保存、恢复训练）
- `classic/test.py`：推荐推理测试入口
- `mjx/backend.py`：MJX / MuJoCo Warp 后端切换
- `assets/robotiq_cxy/`：UR5+CXY 模型资源
- `assets/zero_arm/`：zero 机械臂与 zero+Robotiq 模型资源

## 3. 安装与环境准备

```bash
cd /home/zzyfan/mujoco_ur5_rl
pip install -r requirements.txt
```

快速确认环境可运行（随机动作）：

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/test.py --random-policy --episodes 1 --render-mode human
```

## 4. 训练（对应 notebook 里的 “创建环境和智能体并开始训练”）

### 4.1 推荐先跑无渲染（更快）

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5 \
  --timesteps 500000 \
  --device cuda \
  --no-render
```

### 4.2 训练时看 MuJoCo 画面

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5_vis \
  --timesteps 500000 \
  --device cuda \
  --render --render-mode human --n-envs 1
```

### 4.3 切换到 zero 机械臂 + 你的 Robotiq 夹爪

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --robot zero_robotiq \
  --run-name exp_sac_zero \
  --timesteps 500000 \
  --device cuda \
  --no-render
```

## 5. 模型保存规则（已按你的要求分开）

保存是“按算法 + 机械臂 + run_name 独立目录”：

```text
models/classic/{algo}/{robot}/{run_name}/
  final/
    model.zip
    vec_normalize.pkl
    replay_buffer.pkl        # SAC/TD3
  interrupted/
    model.zip
    vec_normalize.pkl
    replay_buffer.pkl        # SAC/TD3（中断时保存）

logs/classic/{algo}/{robot}/{run_name}/
  best_model/
    best_model.zip
    vec_normalize.pkl
```

说明：

- `classic/train.py` 默认把输出写到 `models/classic/...` 与 `logs/classic/...`
- 若仓库里还存在旧版 `models/{algo}/...`、`logs/{algo}/...` 结果，脚本会在启动时自动同步缺失文件到 `classic` 分层目录
- `final`：正常训练结束的模型
- `interrupted`：你按 `Ctrl+C` 中断时保存
- `best_model`：评估回调认为最优的模型（需开启评估）

## 6. 中断与继续训练（对应 notebook 的“加载已保存模型继续”）

### 6.1 中断训练

- 第一次 `Ctrl+C`：保存中断模型并退出
- 第二次 `Ctrl+C`：立即强制退出

### 6.2 从 interrupted 默认路径继续训练

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5 \
  --resume \
  --timesteps 300000 \
  --device cuda \
  --buffer-size 1000000 \
  --skip-replay-buffer
```

推荐说明：

- 当前机器内存紧张时，优先加上 `--buffer-size 1000000 --skip-replay-buffer`
- 这样会继续使用 `interrupted/model(.zip)` 和 `vec_normalize.pkl`，但不强制恢复旧 replay buffer

### 6.3 从指定路径继续训练

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5_resume2 \
  --resume \
  --resume-model-path models/classic/sac/ur5_cxy/exp_sac_ur5/interrupted/model \
  --resume-normalize-path models/classic/sac/ur5_cxy/exp_sac_ur5/interrupted/vec_normalize.pkl \
  --resume-replay-path models/classic/sac/ur5_cxy/exp_sac_ur5/interrupted/replay_buffer.pkl \
  --timesteps 300000 \
  --device cuda
```

如果你确定内存够用，可以不加 `--skip-replay-buffer`，直接恢复旧 replay buffer。

## 7. 测试训练好的模型（对应 notebook 的“测试智能体”）

### 7.1 用 `classic/train.py --test` 测

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --test \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5 \
  --episodes 5 \
  --render-mode human \
  --device cuda
```

### 7.2 用 `classic/test.py` 按 run-name 自动找模型

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name exp_sac_ur5 \
  --episodes 3 \
  --render-mode human
```

### 7.3 用 `classic/test.py` 指定路径测

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/test.py \
  --algo sac \
  --robot ur5_cxy \
  --model-path models/classic/sac/ur5_cxy/exp_sac_ur5/final/model.zip \
  --norm-path models/classic/sac/ur5_cxy/exp_sac_ur5/final/vec_normalize.pkl \
  --episodes 3 \
  --render-mode human
```

## 8. 评估与退出速度建议

如果你希望“训练退出更快”，可以关闭评估回调：

```bash
--eval-freq 0
```

如果要保留评估但减少评估耗时：

```bash
--n-eval-episodes 1
```

## 9. 常用参数速查

- `--algo {sac,ppo,td3}`：算法选择
- `--robot {ur5_cxy,zero_robotiq}`：机械臂模型选择
- `--timesteps`：训练总步数
- `--buffer-size`：SAC/TD3 回放池容量上限
- `--render / --no-render`：是否显示窗口
- `--lock-camera / --free-camera`：锁定固定相机或自由拖动相机
- `--run-name`：本次实验名（决定保存目录）
- `--eval-freq`：评估频率（`0` 表示关闭评估）
- `--resume`：继续训练开关
- `--skip-replay-buffer`：继续训练时跳过旧 replay buffer 恢复
- `--ur5-target-*`：UR5 目标采样范围
- `--zero-target-*`：ZERO 目标采样范围

## 10. 推荐工作流（最稳）

1. `classic/test.py --random-policy` 先冒烟
2. `--no-render` 训练主模型
3. 中断时自动保存到 `interrupted`
4. 需要继续时用 `--resume --buffer-size 1000000 --skip-replay-buffer`
5. 最后用 `classic/train.py --test` 或 `classic/test.py` 可视化验收

## 11. 单独新构建线（MJX/Warp 预览）

如果你要“单独构建一个新的后端线”，可以直接用：

```bash
cd /home/zzyfan/mujoco_ur5_rl
python mjx/benchmark.py --robot zero_robotiq --batch-size 256 --steps 1000 --safe-disable-constraints
```

如果你要强制走 MuJoCo Warp：

```bash
python mjx/benchmark.py --robot zero_robotiq --batch-size 256 --steps 1000 --safe-disable-constraints --physics-backend warp
```

这条线特点：
- 和 `classic/train.py` 完全分离，不会影响现有实验
- 主要用于 MJX/Warp 路线后端验证与吞吐压测
- 不是完整 RL 训练流程（后续可在这条线上继续接策略训练）
