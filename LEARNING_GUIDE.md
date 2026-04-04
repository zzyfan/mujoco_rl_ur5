# 学习导图（MTD3 笔记风格）

这份文档是“按步骤读懂项目”的学习路线，建议你边看代码边跑最小命令。

## 1. 学习目标

你需要先吃透三条主线：

1. Gym 环境接口：`reset / step / render`
2. MuJoCo 数据流：`ctrl -> mj_step -> qpos/qvel/xpos/contact`
3. SB3 训练链路：`VecEnv -> VecNormalize -> model.learn`

## 2. 推荐阅读顺序

1. `classic/env.py`：看环境定义和奖励设计
2. `classic/train.py`：看参数、模型构建、保存与恢复训练
3. `classic/test.py`：看推理、模型加载、归一化对齐

## 3. Python 语法在项目里的落点

- `@dataclass`：集中管理参数（`TrainArgs`、`MujocoEnvConfig`）
- `class`：环境类和回调类组织逻辑
- `if/else`：算法分支、设备分支、模型分支
- `for`：`frame_skip` 积分循环、回合循环

## 4. Gymnasium 接口关键点

### 4.1 `action_space`

动作维度和范围由环境定义，SB3 按此检查动作合法性。

### 4.2 `observation_space`

观测维度与 dtype 必须与 `step/reset` 返回一致。

### 4.3 `reset`

必须返回 `(obs, info)`，并在其中做目标采样、状态清零。

### 4.4 `step`

必须返回 `(obs, reward, terminated, truncated, info)`。

## 5. MuJoCo 控制链路

每个 step 发生的事情：

1. 策略输出动作
2. 写入 `data.ctrl`
3. 调用 `mujoco.mj_step`
4. 读取新状态组装观测
5. 计算奖励并判断终止

## 6. SB3 训练链路

训练主流程：

1. 注册环境 ID
2. `make_vec_env` 创建向量化环境
3. `VecNormalize` 处理观测/奖励归一化
4. 构建 `SAC/PPO/TD3`
5. `model.learn` 开始训练
6. 保存 `final / best / interrupted`

## 7. 最小命令（边学边跑）

### 7.1 先跑随机策略

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/test.py --random-policy --episodes 1 --render-mode human
```

### 7.2 再跑短训练

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py --algo sac --run-name learn_short --timesteps 20000 --device cuda --no-render --eval-freq 0
```

### 7.3 测试短训练模型

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py --test --algo sac --run-name learn_short --episodes 3 --render-mode human
```

## 8. 你下一步最值得做的实验

1. 只改 1 个奖励系数，观察收敛速度变化
2. 调整 `frame_skip`，比较动作平滑性
3. 对比 `ur5_cxy` 与 `zero_robotiq` 的学习曲线
4. 用 `--resume --buffer-size 1000000 --skip-replay-buffer` 做分阶段训练并保留实验记录

## 9. 配套文档

- `README.md`：项目总览与命令速查
- `PROJECT_USAGE_MTD3_STYLE.md`：完整操作手册
- `CURRICULUM_LINE_BY_LINE.md`：课程学习机制逐行解释
