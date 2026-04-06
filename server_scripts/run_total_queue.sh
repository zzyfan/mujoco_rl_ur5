#!/usr/bin/env bash
# 远端 screen 队列脚本：
# 1. 先跑 Warp 线，保持 GPU 高吞吐验证定位
# 2. 再跑 classic 线，承担成功率主线（SB3 + HER）
# 3. SBX 作为独立实验线，单独用 `run_sbx_experiment.sh` 启动，不混进主队列
#
# 这个脚本会在服务器端直接执行，所以只依赖：
# - 远端仓库代码
# - 远端 conda 环境 `mujoco`

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5

echo "[queue] warp validation line start $(date '+%F %T')"
bash server_scripts/run_warp_validation_queue.sh

echo "[queue] classic success line start $(date '+%F %T')"
bash server_scripts/run_classic_success_queue.sh

echo "[queue] all done $(date '+%F %T')"
