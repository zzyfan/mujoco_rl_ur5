#!/usr/bin/env bash
# classic 成功率主线：
# - 定位：SB3 + HER 成功率主线
# - 目标：在 CPU 侧保持较高采样吞吐，同时复用成熟的 goal-conditioned + HER 生态
# - 说明：classic 经过实测更容易被 MuJoCo CPU step / IPC / HER 回放重标记限制，因此默认走 CPU

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
export PYTHONUNBUFFERED=1

echo "[classic-queue] SAC+HER start $(date '+%F %T')"
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name server_classic_sac_her \
  --n-envs 128 \
  --batch-size 1024 \
  --buffer-size 3000000 \
  --gradient-steps 4 \
  --frame-skip 2 \
  --device cpu \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --use-her \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[classic-queue] TD3+HER start $(date '+%F %T')"
python classic/train.py \
  --algo td3 \
  --robot ur5_cxy \
  --run-name server_classic_td3_her \
  --n-envs 128 \
  --batch-size 1024 \
  --buffer-size 3000000 \
  --gradient-steps 4 \
  --frame-skip 2 \
  --device cpu \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --use-her \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[classic-queue] PPO sparse control baseline start $(date '+%F %T')"
python classic/train.py \
  --algo ppo \
  --robot ur5_cxy \
  --run-name server_classic_ppo_gc \
  --n-envs 128 \
  --batch-size 1024 \
  --frame-skip 2 \
  --device cpu \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[classic-queue] all done $(date '+%F %T')"

