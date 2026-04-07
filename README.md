# UR5 Reach RL (Main + Warp)

## Quick Start

```bash
# 主线训练（SAC）
python train_ur5_reach.py --algo sac --total-timesteps 1500000

# 主线测试（渲染）
python train_ur5_reach.py --algo sac --test --episodes 1 --render

# Warp 训练（SAC）
python train_ur5_reach_warp.py --algo sac --num-envs 256 --num-timesteps 5000000
```

## Files

- `ur5_reach_env.py`：主线 Gymnasium + MuJoCo 环境（zero‑arm 风格）
- `train_ur5_reach.py`：主线训练与测试入口（TD3/SAC/PPO）
- `warp_ur5_env.py`：Warp 线环境（JAX/MJX/Warp）
- `train_ur5_reach_warp.py`：Warp 线训练入口（SAC/PPO）
- `assets/robotiq_cxy/`：UR5 模型与 meshes

## Notes

- 主线与 Warp 线都使用同一套 UR5 模型文件。
- 主线训练默认保存到 `./logs` 与 `./models`。
- Warp 线训练默认保存到 `./warp_runs/{algo}/{run_name}`。

