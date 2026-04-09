# Zero Robot Arm RL (MuJoCo + SB3)

本项目是 Zero 机械臂的 MuJoCo 强化学习训练工程，提供三条独立训练线：

- TD3：`train_robot_arm.py`
- PPO：`train_robot_arm_ppo.py`
- SAC：`train_robot_arm_sac.py`

每条线都包含完整闭环：

1. 并行采样训练  
2. 周期性评估与 best 模型保存  
3. 中断恢复模型保存  
4. 训练结束统一 final 评估  
5. human 可视化测试  

---

## 1. 目录结构

```text
5. Deep_LR/
├─ robot_arm_env.py                  # 任务环境定义（状态、动作、奖励、终止）
├─ robot_arm_mujoco.xml              # MuJoCo 场景和机械臂模型
├─ meshes/                           # 网格资源
├─ train_robot_arm.py                # TD3 训练/测试
├─ train_robot_arm_ppo.py            # PPO 训练/测试
├─ train_robot_arm_sac.py            # SAC 训练/测试
├─ training_monitor.py               # 辅助监控（可选）
├─ PARAMETER_IMPACT.md               # 参数影响说明
├─ pull_after_queue_done.ps1         # 队列训练完成后自动拉回产物
├─ remote_artifacts/                 # 回传产物归档目录
└─ README.md
```

---

## 2. 环境定义（`robot_arm_env.py`）

### 2.1 观测空间

观测维度固定为 24 维：

- 相对目标位置 `relative_pos`：3
- 关节角 `joint_angles`：6
- 关节速度 `joint_velocities`：6
- 上一步扭矩 `previous_torques`：6
- 末端速度 `ee_vel`：3

### 2.2 动作空间

- 6 维连续动作（扭矩）
- 范围 `[-15, 15]`

### 2.3 奖励与终止机制

奖励项包含：

- 时间惩罚
- 距离改善奖励 / 退步惩罚
- 阶段阈值一次性奖励
- 方向奖励
- 速度惩罚
- 碰撞惩罚
- 成功奖励（含剩余步数与低速奖励）

回合结束触发：

- 成功：`distance <= success_threshold`（当前为 `0.01`）
- 碰撞：检测到接触
- 截断：`step_count >= 3000`

---

## 3. 三条训练线默认配置

### 3.1 TD3（`train_robot_arm.py`）

- `n_envs=12`
- `learning_rate=3e-4`
- `buffer_size=2_000_000`
- `learning_starts=25_000`
- `batch_size=2048`
- `train_freq=(16, "step")`
- `gradient_steps=16`
- `policy_delay=2`
- `target_policy_noise=0.2`
- `target_noise_clip=0.5`
- `action_noise_scale=0.1`

### 3.2 PPO（`train_robot_arm_ppo.py`）

- `n_envs=20`
- `learning_rate=3e-4`
- `batch_size=4096`
- `n_steps=1024`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.003`
- `vf_coef=0.5`
- `max_grad_norm=0.5`

### 3.3 SAC（`train_robot_arm_sac.py`）

- `n_envs=12`
- `learning_rate=3e-4`
- `buffer_size=2_000_000`
- `learning_starts=20_000`
- `batch_size=2048`
- `train_freq=(16, "step")`
- `gradient_steps=16`
- `tau=0.005`
- `gamma=0.99`
- `ent_coef="auto"`
- `use_sde=True`
- `sde_sample_freq=16`

---

## 4. 训练命令

在项目根目录（本 README 所在目录）下执行。

### 4.1 TD3

```powershell
python train_robot_arm.py --total-timesteps 5000000 --n-envs 12 --episode-log-interval 64 --eval-freq 200000 --eval-episodes 10 --final-eval-episodes 20
```

### 4.2 PPO

```powershell
python train_robot_arm_ppo.py --total-timesteps 5000000 --n-envs 20 --episode-log-interval 64 --eval-freq 200000 --eval-episodes 10 --final-eval-episodes 20
```

### 4.3 SAC

```powershell
python train_robot_arm_sac.py --total-timesteps 5000000 --n-envs 12 --episode-log-interval 64 --eval-freq 200000 --eval-episodes 10 --final-eval-episodes 20
```

---

## 5. 测试命令（human 模式）

### 5.1 TD3 最佳模型 5 回合

```powershell
python train_robot_arm.py --test --best --episodes 5
```

### 5.2 PPO 最佳模型 5 回合

```powershell
python train_robot_arm_ppo.py --test --best --episodes 5
```

### 5.3 SAC 最佳模型 5 回合

```powershell
python train_robot_arm_sac.py --test --best --episodes 5
```

### 5.4 测试 final 模型

TD3：

```powershell
python train_robot_arm.py --test --final --episodes 5
```

---

## 6. 日志与进度条

三条训练线已统一为和 UR5 主线一致的输出风格：

- `progress_bar=True`（SB3 原生进度条）
- `verbose=1`（SB3 标准训练信息）
- `EvalCallback` 周期评估日志（`Eval num_timesteps=...`）

终端会看到的核心信息包括：

1. 训练进度条（步数、速度、预计剩余时间）  
2. SB3 训练统计（loss、fps、time 等）  
3. 周期评估结果（平均回报、回合长度、best model 提示）  

---

## 7. 模型与日志产物目录

TD3：

```text
logs/td3/
models/td3/
├─ td3_robot_arm_final.zip
├─ vec_normalize.pkl
├─ best_model/
│  ├─ best_model.zip
│  └─ vec_normalize.pkl
└─ interrupted/
   ├─ td3_robot_arm_interrupted.zip
   └─ vec_normalize.pkl
```

PPO 与 SAC 同样结构，仅目录名和文件前缀替换为 `ppo` / `sac`。

---

## 8. 防乱码设置（screen/VSCode 终端）

训练脚本中已默认设置：

- `PYTHONUTF8=1`
- `PYTHONIOENCODING=UTF-8`
- `TQDM_ASCII=1`
- `RICH_NO_UNICODE=1`

远端 shell 仍建议执行：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

如果仍出现乱码，优先检查：

1. `screen` 会话编码  
2. VSCode 终端字体与编码  
3. 是否有外部命令向同一终端写入彩色/非 UTF-8 输出  

---

## 9. 远程顺序训练与自动回传

`pull_after_queue_done.ps1` 逻辑：

1. 轮询远端 `.done_128` 标记（`td3/ppo/sac`）
2. 三个算法全部完成后
3. 自动 `scp` 拉取：
   - `models/td3`, `models/ppo`, `models/sac`
   - `logs/td3`, `logs/ppo`, `logs/sac`
   - `train_queue.log`
4. 保存到本地 `remote_artifacts/queue128_时间戳/`

---

## 10. 依赖安装建议

推荐：

- Python 3.10 ~ 3.12
- 安装 MuJoCo、Gymnasium、SB3、PyTorch

示例：

```powershell
pip install --upgrade pip
pip install numpy gymnasium mujoco stable-baselines3[extra] torch
```

如需 CUDA 版 PyTorch：

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

---

## 11. 参数文档与调参入口

详细参数影响请看：

- `PARAMETER_IMPACT.md`

推荐调参顺序：

1. 先定并行数（不让 CPU 爆满）  
2. 再调学习强度（`batch_size` / `gradient_steps` / `n_steps`）  
3. 最后调奖励细节  

---

## 12. 常见问题

### Q1：进度条不显示？

- 确认 `progress_bar=True`
- 确认没有高频逐步 `print`
- `screen` 内尽量只跑一个训练进程占用该窗口

### Q2：训练看起来卡住？

通常是进入评估阶段、保存模型阶段或回放缓冲预热阶段。  
可观察 `Train Summary`、`Eval` 输出和进度条 `it/s` 是否持续刷新。

### Q3：best 模型和 final 模型怎么选？

- `best`：基于周期评估成绩保存，泛化通常更稳
- `final`：训练最后时刻状态，便于复现实验终点
