#!/usr/bin/env bash
# 一键在服务器上用 screen 启动 classic 成功率主线。

set -euo pipefail

cd /root/autodl-tmp/mujoco_rl_ur5
mkdir -p server_logs

SESSION_NAME="${SESSION_NAME:-classic_success_queue}"
LOG_FILE="${LOG_FILE:-/root/autodl-tmp/mujoco_rl_ur5/server_logs/classic_success_queue.log}"

if screen -ls | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
  screen -S "${SESSION_NAME}" -X quit || true
fi

screen -U -L -Logfile "${LOG_FILE}" -dmS "${SESSION_NAME}" bash -lc '
  set -euo pipefail
  cd /root/autodl-tmp/mujoco_rl_ur5
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mujoco
  export PYTHONUNBUFFERED=1
  bash server_scripts/run_classic_success_queue.sh
'

echo "[screen] started session=${SESSION_NAME}"
echo "[screen] attach with: screen -U -r ${SESSION_NAME}"
