# Zero Robot Arm RL (MuJoCo + SB3)

本项目用于 Zero 机械臂在 MuJoCo 中的强化学习训练与测试，包含三条算法脚本：

- TD3: `train_robot_arm_td3.py`
- PPO: `train_robot_arm_ppo.py`
- SAC: `train_robot_arm_sac.py`

---

## 1. 快速开始

在项目根目录执行：

```powershell
python train_robot_arm_sac.py
```

测试（默认加载 final）：

```powershell
python train_robot_arm_sac.py --test
```

---

## 2. 统一参数（三个脚本一致）

- `--test`: 测试模式
- `--root-dir`: 固定保存根目录（默认 `./checkpoints`）
- `--model {final,best}`: 测试时选择 final 或 best（默认 `final`）
- `--model-path`: 测试模型路径（会覆盖 `--model`）
- `--normalize-path`: 测试归一化路径（会覆盖 `--model`）
- `--resume`: 继续训练
- `--resume-from {final,best,interrupted}`: 从哪类 checkpoint 继续
- `--resume-model-path`: 显式指定继续训练模型路径
- `--resume-normalize-path`: 显式指定继续训练归一化路径
- `--total-timesteps`: 本次训练总步数（新训练/续训都生效）
- `--train-render`: 训练时渲染
- `--episodes`: 测试回合数
- `--inference-report-dir`: 推理报告输出目录（默认按算法落到 root-dir 下）
- `--no-test-render`: 测试时关闭渲染

---

## 3. 训练命令示例

SAC 新训练 500 万步：

```powershell
python train_robot_arm_sac.py --total-timesteps 5000000
```

SAC 从 best 继续训练 250 万步：

```powershell
python train_robot_arm_sac.py --resume --resume-from best --total-timesteps 2500000
```

PPO/TD3 同理，只需替换脚本名。

---

## 4. 测试命令示例

测试 final：

```powershell
python train_robot_arm_td3.py --test --model final --episodes 5
```

测试 best：

```powershell
python train_robot_arm_ppo.py --test --model best --episodes 5
```

显式指定现有路径测试：

```powershell
python train_robot_arm_sac.py --test --model-path ./logs/best_model/best_model --normalize-path ./logs/sac/best_model/vec_normalize.pkl
```

---

## 5. 固定目录与分算法保存

默认 `--root-dir ./checkpoints`，目录结构：

```text
checkpoints/
├─ sac/
│  ├─ models/
│  │  ├─ sac_robot_arm_final.zip
│  │  ├─ vec_normalize.pkl
│  │  ├─ best_model/
│  │  │  ├─ best_model.zip
│  │  │  └─ vec_normalize.pkl
│  │  └─ interrupted/
│  │     ├─ sac_robot_arm_interrupted.zip
│  │     └─ vec_normalize.pkl
│  ├─ eval_logs/
│  ├─ metrics/
│  └─ inference/
├─ ppo/
└─ td3/
```

如果想改根目录：

```powershell
python train_robot_arm_sac.py --root-dir ./exp_001
```

---

## 6. 结果文件

训练阶段（`metrics/`）：

- `training_reward_curve.png`
- `training_episode_length_curve.png`
- `training_loss_curve.png`（有 loss 数据时）
- `training_episodes.csv`
- `training_losses.csv`（有 loss 数据时）
- `training_metrics.json`

推理阶段（`inference/`）：

- `inference_summary.json`
- `inference_per_episode.csv`
- `inference_metrics.png`

---

## 7. 推理指标

测试时会输出并保存：

- Success Rate
- Mean Final Distance
- Mean Episode Length
- Collision Rate
- Mean Smoothness
- Mean Reward

