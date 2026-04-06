#!/usr/bin/env bash
# 一键在服务器上用 screen 启动整轮训练。
# 用法：
#   bash server_scripts/start_total_queue_screen.sh
#
# 这个脚本负责：
# - 清理同名旧 screen 会话
# - 挂起一个新的 `total_queue` 会话
# - 在会话里激活远端 conda 环境后执行总队列脚本

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

SESSION_NAME="${SESSION_NAME:-total_queue}"
LOG_FILE="${LOG_FILE:-/root/autodl-tmp/mujoco_rl_ur5/server_logs/total_queue.log}"

if screen -ls | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  screen -S "${SESSION_NAME}" -X quit || true
fi

screen -U -L -Logfile "${LOG_FILE}" -dmS "${SESSION_NAME}" bash -lc '
  set -euo pipefail
  cd /root/autodl-tmp/mujoco_rl_ur5
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mujoco
  export PYTHONUNBUFFERED=1
  bash server_scripts/run_total_queue.sh
'

echo "[screen] started session=${SESSION_NAME}"
echo "[screen] attach with: screen -U -r ${SESSION_NAME}"
