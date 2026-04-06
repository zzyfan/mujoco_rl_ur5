#!/usr/bin/env bash
# SBX 实验线：
# - 目标：验证 JAX 算法层是否能在不改环境实现的前提下，吸收 classic/warp 的共同训练方法
# - 默认：SAC + goal-conditioned + sparse + joint_position_delta
# - 特意不启用 HER，先把“JAX 算法层 + 现有 MuJoCo/Gym 环境”跑稳

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
export PYTHONUNBUFFERED=1

python -m sbx_runner.train \
  --algo sac \
  --robot ur5_cxy \
  --run-name server_sbx_sac_gc \
  --n-envs 128 \
  --batch-size 1024 \
  --buffer-size 3000000 \
  --gradient-steps 4 \
  --frame-skip 2 \
  --device cuda \
  --goal-conditioned \
  --reward-mode sparse \
  --controller-mode joint_position_delta \
  --joint-position-delta-scale 0.08 \
  --position-control-kp 45 \
  --position-control-kd 3 \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  "$@"
