#!/usr/bin/env bash
# 远端 screen 队列脚本：
# 1. 先跑 Warp 线，验证高吞吐 dense/sparse 对照
# 2. 再跑 classic 线，先验证 HER 主线，再补一条 PPO 对照
#
# 这个脚本会在服务器端直接执行，所以只依赖：
# - 远端仓库代码
# - 远端 conda 环境 `mujoco`

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
export PYTHONUNBUFFERED=1

echo "[queue] warp PPO sparse start $(date '+%F %T')"
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

echo "[queue] warp SAC sparse start $(date '+%F %T')"
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

echo "[queue] classic SAC+HER start $(date '+%F %T')"
python classic/train.py \
  --algo sac \
  --robot ur5_cxy \
  --run-name server_classic_sac_her \
  --n-envs 256 \
  --batch-size 1024 \
  --buffer-size 3000000 \
  --gradient-steps 4 \
  --frame-skip 2 \
  --device cuda \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --use-her \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[queue] classic TD3+HER start $(date '+%F %T')"
python classic/train.py \
  --algo td3 \
  --robot ur5_cxy \
  --run-name server_classic_td3_her \
  --n-envs 256 \
  --batch-size 1024 \
  --buffer-size 3000000 \
  --gradient-steps 4 \
  --frame-skip 2 \
  --device cuda \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --use-her \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[queue] classic PPO goal-conditioned start $(date '+%F %T')"
python classic/train.py \
  --algo ppo \
  --robot ur5_cxy \
  --run-name server_classic_ppo_gc \
  --n-envs 256 \
  --batch-size 1024 \
  --frame-skip 2 \
  --device cuda \
  --controller-mode joint_position_delta \
  --goal-conditioned \
  --reward-mode sparse \
  --success-threshold 0.01 \
  --stage1-success-threshold 0.05 \
  --stage2-success-threshold 0.03 \
  --no-render

echo "[queue] all done $(date '+%F %T')"
