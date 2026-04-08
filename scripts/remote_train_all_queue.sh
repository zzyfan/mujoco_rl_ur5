#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPORT_ROOT="${REPO_ROOT}/exports"
TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${REPO_ROOT}/.full_suite/${TIMESTAMP}"

mkdir -p "${QUEUE_ROOT}" "${EXPORT_ROOT}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl-mujoco-env

cd "${REPO_ROOT}"

MASTER_LOG="${QUEUE_ROOT}/master.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MASTER_LOG}"
}

run_job() {
  local job_name="$1"
  shift
  local log_file="${QUEUE_ROOT}/${job_name}.log"
  log "start job=${job_name}"
  local cmd=()
  for arg in "$@"; do
    cmd+=("$(printf '%q' "$arg")")
  done
  script -q -f -c "${cmd[*]}" "${log_file}"
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    log "fail job=${job_name} exit_code=${exit_code}"
    exit "${exit_code}"
  fi
  log "done job=${job_name}"
}

log "queue_root=${QUEUE_ROOT}"
log "python=$(python --version 2>&1)"
log "hostname=$(hostname)"
log "repo=${REPO_ROOT}"
log "note=warp currently supports sac/ppo only; main line runs td3/sac/ppo"

run_job "warp_sac" python train_ur5_reach_warp.py \
  --algo sac \
  --run-name "${TIMESTAMP}_warp_sac" \
  --num-envs 256 \
  --num-eval-envs 128 \
  --num-timesteps 5000000 \
  --num-evals 10 \
  --learning-rate 1e-4 \
  --reward-scaling 0.1

run_job "warp_ppo" python train_ur5_reach_warp.py \
  --algo ppo \
  --run-name "${TIMESTAMP}_warp_ppo" \
  --num-envs 256 \
  --num-eval-envs 128 \
  --num-timesteps 5000000 \
  --num-evals 10 \
  --learning-rate 1e-4

run_job "main_td3" python train_ur5_reach.py \
  --algo td3 \
  --run-name "${TIMESTAMP}_main_td3" \
  --total-timesteps 5000000 \
  --n-envs 32 \
  --device cuda \
  --eval-freq 500000 \
  --eval-episodes 1

run_job "main_sac" python train_ur5_reach.py \
  --algo sac \
  --run-name "${TIMESTAMP}_main_sac" \
  --total-timesteps 5000000 \
  --n-envs 32 \
  --device cuda \
  --eval-freq 500000 \
  --eval-episodes 1

run_job "main_ppo" python train_ur5_reach.py \
  --algo ppo \
  --run-name "${TIMESTAMP}_main_ppo" \
  --total-timesteps 5000000 \
  --n-envs 32 \
  --device cuda \
  --eval-freq 500000 \
  --eval-episodes 1

ARTIFACT_TAR="${EXPORT_ROOT}/${TIMESTAMP}_artifacts.tar.gz"
tar -czf "${ARTIFACT_TAR}" \
  -C "${REPO_ROOT}" \
  "runs/server/warp/sac/${TIMESTAMP}_warp_sac" \
  "runs/server/warp/ppo/${TIMESTAMP}_warp_ppo" \
  "runs/server/main/td3/${TIMESTAMP}_main_td3" \
  "runs/server/main/sac/${TIMESTAMP}_main_sac" \
  "runs/server/main/ppo/${TIMESTAMP}_main_ppo"

touch "${QUEUE_ROOT}/queue_complete.ok"
log "queue_complete artifact_tar=${ARTIFACT_TAR}"
