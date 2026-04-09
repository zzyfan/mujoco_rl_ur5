# UR5 Reach RL (Main + Warp)

本项目用于训练 UR5 机械臂完成末端到点（reach）任务，包含两条训练线：

- 主线：`MuJoCo + Stable-Baselines3`（`TD3` / `SAC` / `PPO`）
- Warp 线：`Brax + JAX + MuJoCo Playground/Warp`（`SAC` / `PPO`）

目标是保证同一套 UR5 模型下，可以完成：

1. 本地可视化调试  
2. 服务器无头高吞吐训练  
3. 统一目录保存模型、归一化参数、配置与日志  

---

## 1. 项目结构

```text
mujoco_rl_ur5/
├─ assets/
│  └─ robotiq_cxy/
│     ├─ lab_env.xml                  # UR5 + 场景主模型
│     └─ meshes/                      # 网格资源
├─ docs/
│  ├─ IMPLEMENTATION_GUIDE.md
│  ├─ PARAMETER_REFERENCE.md
│  ├─ LIBRARY_USAGE.md
│  ├─ PORTABILITY.md
│  ├─ WARP_IMPLEMENTATION_GUIDE.md
│  └─ WARP_PARAMETER_REFERENCE.md
├─ notebooks/
│  ├─ 01_code_learning_walkthrough.ipynb
│  ├─ 02_parameter_reference.ipynb
│  ├─ 03_warp_code_learning_walkthrough.ipynb
│  ├─ 04_cli_parameter_guide.ipynb
│  └─ 05_library_usage_guide.ipynb
├─ ur5_reach_config.py                # 主线参数 dataclass 与路径规则
├─ ur5_reach_env.py                   # 主线环境定义（24 维观测、奖励、重置、碰撞）
├─ train_ur5_reach.py                 # 主线训练/测试统一入口
├─ warp_ur5_config.py                 # Warp 线参数 dataclass
├─ warp_ur5_env.py                    # Warp 环境定义
├─ warp_ur5_runtime.py                # Warp 依赖与运行时检查
├─ train_ur5_reach_warp.py            # Warp 训练入口
├─ test_ur5_reach_warp.py             # Warp 推理入口
├─ requirements.txt                   # 主线依赖
└─ requirements-warp.txt              # Warp 线额外依赖
```

---

## 2. 任务与环境说明（主线）

`ur5_reach_env.py` 的任务定义：

- 动作：6 维，归一化后映射到力矩或关节增量控制
- 观测：24 维  
  `relative_pos(3) + qpos(6) + qvel(6) + prev_torque(6) + ee_velocity(3)`
- 回合结束条件：
  - `success`：末端到目标距离 <= 成功阈值
  - `collision`：机器人与外部环境碰撞
  - `timeout`：达到 `episode_length`
  - `runaway`：距离过大

课程学习阶段（由 `ur5_reach_config.py` 控制）：

- `fixed` 固定目标
- `local_random` 小范围随机目标
- `full_random` 全范围随机目标

---

## 3. 环境准备

### 3.1 Python 版本建议

- 主线建议：`Python 3.10 ~ 3.12`
- Warp 线建议：Linux + CUDA 环境（Windows 兼容性明显更差）

### 3.2 Conda（推荐）

```powershell
conda create -n ur5-rl python=3.12 -y
conda activate ur5-rl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 可选：安装 PyTorch CUDA 版（环境里替换 CPU 版）

注意 PowerShell 不要用反斜杠续行，直接一行执行：

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

验证：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

---

## 4. 主线训练（SB3）

入口脚本：`train_ur5_reach.py`

### 4.1 三算法训练命令

TD3：

```powershell
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --total-timesteps 5000000
```

SAC：

```powershell
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --total-timesteps 5000000
```

PPO：

```powershell
python train_ur5_reach.py --algo ppo --run-name ur5_ppo_main --total-timesteps 5000000
```

### 4.2 常用训练参数

- `--n-envs`：并行环境数
- `--device cuda`：强制使用 CUDA（可改为 `auto`）
- `--eval-freq` / `--eval-episodes`：周期评估频率与回合数
- `--train-freq` / `--gradient-steps`：离策略学习强度（TD3/SAC）
- `--render-training`：训练时直接开窗口渲染
- `--spectator-render`：训练无头并行 + 主进程旁观窗口

### 4.3 当前代码默认值（主线）

来自 `RLTrainConfig` 与 CLI 默认：

- `algo=td3`
- `run_name=ur5_zero_aligned_main`
- `total_timesteps=5_000_000`
- `n_envs=1`
- `learning_rate=3e-4`
- `buffer_size=3_000_000`
- `learning_starts=10_000`
- `batch_size=256`
- `train_freq=1`
- `gradient_steps=1`
- `policy_delay=4`
- `ppo_n_steps=2048`
- `ppo_n_epochs=10`

---

## 5. 主线测试与可视化

### 5.1 5 回合 human 测试

```powershell
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --test --episodes 5 --render
```

### 5.2 指定模型与归一化参数测试

```powershell
python train_ur5_reach.py --algo sac --test --episodes 5 --render --model-path runs/main/sac/ur5_sac_main/best_model/best_model.zip --normalize-path runs/main/sac/ur5_sac_main/best_model/vec_normalize.pkl
```

### 5.3 打印奖励拆分（调参用）

```powershell
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --test --episodes 2 --print-reward-terms
```

---

## 6. Warp 线（可选）

Warp 线入口：

- 训练：`train_ur5_reach_warp.py`
- 推理：`test_ur5_reach_warp.py`

### 6.1 安装

```powershell
pip install -r requirements-warp.txt
```

### 6.2 训练

SAC：

```powershell
python train_ur5_reach_warp.py --algo sac --run-name ur5_warp_sac --num-envs 256 --num-timesteps 5000000
```

PPO：

```powershell
python train_ur5_reach_warp.py --algo ppo --run-name ur5_warp_ppo --num-envs 256 --num-timesteps 5000000
```

### 6.3 推理

```powershell
python test_ur5_reach_warp.py --algo sac --run-name ur5_warp_sac --episodes 3 --render
```

### 6.4 Windows 已知问题

在 Windows 上，Warp/JAX 路线常见失败点：

- `jax-cuda12-plugin` 无可用分发
- `orbax-checkpoint` 链路拉到 `uvloop`（Windows 不支持）
- `mujoco-playground` 安装命名与依赖冲突

如果你重点要跑 Warp，建议直接使用 Linux 服务器环境。

---

## 7. 产物目录

### 7.1 主线

```text
runs/main/{algo}/{run_name}/
├─ run_config.json
├─ tensorboard/
├─ best_model/
│  ├─ best_model.zip
│  └─ vec_normalize.pkl
├─ final/
│  ├─ final_model.zip
│  └─ vec_normalize.pkl
└─ interrupted/
   ├─ interrupted_model.zip
   └─ vec_normalize.pkl
```

### 7.2 Warp

```text
runs/warp/{algo}/{run_name}/
├─ config.json
├─ checkpoints/
└─ final_policy.msgpack
```

---

## 8. 服务器训练建议

无头服务器建议设置：

```bash
export MUJOCO_GL=egl
```

`screen` 启动训练时建议：

- 用 UTF-8 locale
- 不在训练输出里混入额外的 tree/彩色符号
- 保持 `progress_bar=True`，日志用聚合窗口输出

---

## 9. 常见问题

### Q1：GPU 利用率低？

主线 MuJoCo 物理仿真主要吃 CPU，GPU 主要用于网络前向与反向。  
提高 GPU 利用率通常靠：

- 增加并行环境数（`--n-envs`）
- 增加 batch size / gradient steps（离策略）
- 减少频繁渲染和过密日志

### Q2：训练窗口卡顿？

`--render-training` 会显著拖慢训练。  
建议训练无头跑，必要时只做短时可视化测试。

### Q3：找不到 best 模型？

只有触发评估后才会更新 best。  
检查 `--eval-freq`、`--eval-episodes` 与训练时长是否足够。

---

## 10. 进一步阅读

建议阅读顺序：

1. `ur5_reach_config.py`
2. `ur5_reach_env.py`
3. `train_ur5_reach.py`
4. `docs/IMPLEMENTATION_GUIDE.md`
5. `docs/PARAMETER_REFERENCE.md`
6. `docs/LIBRARY_USAGE.md`
7. `docs/WARP_IMPLEMENTATION_GUIDE.md`
8. `docs/WARP_PARAMETER_REFERENCE.md`
