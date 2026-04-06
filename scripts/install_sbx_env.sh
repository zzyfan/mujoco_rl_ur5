#!/usr/bin/env bash
# 安装 SBX 实验线依赖。
# 用法：
#   当前激活环境里执行：bash scripts/install_sbx_env.sh
#
# 这个脚本只操作“当前 python 对应的环境”，所以：
# - 本地请先激活 `mujoco_cuda`
# - 服务器请先激活 `mujoco`

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY_BIN="${PYTHON_BIN:-python}"

echo "[install] python=${PY_BIN}"
"${PY_BIN}" -m pip install -r requirements.txt
"${PY_BIN}" -m pip install sbx-rl

# SBX 会把 JAX 收到它当前支持的版本。若机器原来装过别的 JAX CUDA plugin，
# 很容易出现 `jaxlib 0.x` 和 `jax_cuda13_plugin 0.y` 版本不一致的警告。
# 这里把 plugin / pjrt 也对齐到当前 jax 版本，避免“SBX 装上了但 GPU JAX 退回 CPU plugin”的半残状态。
JAX_VERSION="$("${PY_BIN}" - <<'PY'
import importlib.metadata as m
print(m.version("jax"))
PY
)"
"${PY_BIN}" -m pip install \
  "jax-cuda13-plugin==${JAX_VERSION}" \
  "jax-cuda13-pjrt==${JAX_VERSION}"

"${PY_BIN}" - <<'PY'
import importlib.util
mods = ["sbx", "jax", "jaxlib", "flax", "optax", "stable_baselines3", "torch", "mujoco"]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"[install] missing modules after install: {missing}")
print("[install] SBX environment ready")
PY
