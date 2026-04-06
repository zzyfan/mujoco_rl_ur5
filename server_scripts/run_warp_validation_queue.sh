#!/usr/bin/env bash
# Warp 验证线：
# - 定位：GPU 高吞吐验证线
# - 目标：用 sparse reward + joint_position_delta + curriculum 快速验证 first success 是否出现
# - 说明：这里先不接 HER，而是保持 Warp 的 GPU 高并行优势

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
export PYTHONUNBUFFERED=1

echo "[warp-queue] stage1 fixed-target PPO start $(date '+%F %T')"
python -m warp_gpu.train \
  --algo ppo \
  --robot ur5_cxy \
  --run-name server_warp_ppo_gc \
  --num-timesteps 5000000 \
  --episode-length 3000 \
  --num-envs 512 \
  --num-eval-envs 64 \
  --batch-size 4096 \
  --controller-mode joint_position_delta \
  --joint-position-delta-scale 0.08 \
  --position-control-kp 45 \
  --position-control-kd 3 \
  --reward-mode sparse \
  --target-sampling-mode fixed \
  --fixed-target-x -0.775 \
  --fixed-target-y 0.325 \
  --fixed-target-z 0.21 \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --naconmax 4096 \
  --naccdmax 4096 \
  --njmax 1024

echo "[warp-queue] stage1 fixed-target SAC start $(date '+%F %T')"
python -m warp_gpu.train \
  --algo sac \
  --robot ur5_cxy \
  --run-name server_warp_sac_gc \
  --num-timesteps 5000000 \
  --episode-length 3000 \
  --num-envs 512 \
  --num-eval-envs 64 \
  --batch-size 4096 \
  --sac-max-replay-size 3000000 \
  --sac-grad-updates-per-step 16 \
  --controller-mode joint_position_delta \
  --joint-position-delta-scale 0.08 \
  --position-control-kp 45 \
  --position-control-kd 3 \
  --reward-mode sparse \
  --target-sampling-mode fixed \
  --fixed-target-x -0.775 \
  --fixed-target-y 0.325 \
  --fixed-target-z 0.21 \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --naconmax 4096 \
  --naccdmax 4096 \
  --njmax 1024

echo "[warp-queue] all done $(date '+%F %T')"

