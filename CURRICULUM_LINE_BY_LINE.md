# 课程学习机制说明（MTD3 笔记风格）

这份文档解释项目里“目标点自动分阶段采样”的实现逻辑，配合 `classic/env.py` 阅读效果最好。

## 1. 课程学习要解决什么问题

直接全范围随机目标会让早期训练很难收敛。  
课程学习把难度拆成三个阶段：

1. 固定目标：先学会基本靠近
2. 小范围随机：逐步增加泛化难度
3. 全范围随机：进入最终训练目标

## 2. 配置参数在哪里

课程参数定义在 `MujocoEnvConfig`：

- `curriculum_stage1_fixed_episodes`
- `curriculum_stage2_random_episodes`
- `curriculum_stage2_range_scale`
- `fixed_target_x / fixed_target_y / fixed_target_z`

这些参数由 `classic/train.py` 的命令行传入环境。

## 3. 阶段推进依据

环境内用 `episode_count` 推进课程，而不是 `step_count`。  
原因是“按回合推进”比“按步推进”更稳定。

## 4. 采样函数核心逻辑

课程采样函数会按 `episode_count` 分支：

### 4.1 阶段 1（固定点）

- 如果给了 `fixed_target_*`，使用固定值
- 否则使用目标采样空间中心点

### 4.2 阶段 2（小范围随机）

- 先计算每个轴中心与半宽
- 用 `range_scale` 缩小半宽
- 在缩小后的区间内均匀采样

### 4.3 阶段 3（全范围随机）

- 直接调用全范围采样函数

## 5. reset 里如何接入课程学习

每次 `reset`：

1. 先重置 MuJoCo 状态
2. 调用课程采样函数得到 `target_pos`
3. 写入目标 joint 的 `qpos`
4. 前向传播 `mj_forward`
5. `episode_count += 1`

## 6. step 里如何观察课程状态

`step` 的 `info` 字段里会输出：

- `curriculum_stage`
- `episode_index`

你可以在训练日志中追踪阶段切换点。

## 7. 命令行控制课程节奏

示例（SAC）：

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo sac \
  --run-name curriculum_exp \
  --timesteps 500000 \
  --curriculum-stage1-fixed-episodes 300 \
  --curriculum-stage2-random-episodes 1000 \
  --curriculum-stage2-range-scale 0.35 \
  --no-render
```

示例（TD3）：

```bash
cd /home/zzyfan/mujoco_ur5_rl
python classic/train.py \
  --algo td3 \
  --run-name curriculum_td3 \
  --timesteps 500000 \
  --curriculum-stage1-fixed-episodes 300 \
  --curriculum-stage2-random-episodes 1000 \
  --curriculum-stage2-range-scale 0.35 \
  --no-render
```

## 8. 调参建议

1. 收敛慢：增加阶段 1/2 回合数
2. 过拟合固定点：缩短阶段 1，增大阶段 2 范围
3. 后期不稳定：减小阶段切换跨度，逐步放大范围
4. 先跑 `--eval-freq 0` 快速迭代，再启用评估

## 9. 与项目其他文档关系

- `README.md`：总览与常用命令
- `PROJECT_USAGE_MTD3_STYLE.md`：完整使用手册
- `LEARNING_GUIDE.md`：代码学习路线
