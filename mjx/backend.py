"""MuJoCo Playground + MJWarp backend helpers."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

try:
    import warp as wp
except Exception:
    wp = None

try:
    import mujoco_warp as mjwarp
except Exception:
    mjwarp = None

_WARP_INITIALIZED = False


def warp_available() -> bool:
    return wp is not None and mjwarp is not None


def ensure_warp_runtime() -> None:
    global _WARP_INITIALIZED
    if not warp_available():
        raise RuntimeError("MuJoCo Warp 不可用，请先安装 `mujoco-warp` 与 `warp-lang`。")
    if _WARP_INITIALIZED:
        return
    wp.init()
    _WARP_INITIALIZED = True


def describe_warp_runtime() -> str:
    if not warp_available():
        return "warp unavailable"
    ensure_warp_runtime()
    try:
        device = str(wp.get_device())
    except Exception:
        device = "unknown"
    try:
        cuda_devices = list(wp.get_cuda_devices())
    except Exception:
        cuda_devices = []
    return f"warp_device={device}, cuda_devices={cuda_devices}"


def detect_playground_root(explicit_root: str = "") -> str:
    """返回可用的 MuJoCo Playground 根目录（找不到则返回空字符串）。"""
    if explicit_root:
        p = Path(explicit_root).expanduser().resolve()
        if (p / "learning").exists() and (p / "mujoco_playground").exists():
            return str(p)
    env_root = os.getenv("MUJOCO_PLAYGROUND_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "learning").exists() and (p / "mujoco_playground").exists():
            return str(p)
    # 常见路径兜底
    for cand in (Path.cwd(), Path.cwd().parent, Path.home() / "mujoco_playground"):
        p = cand.resolve()
        if (p / "learning").exists() and (p / "mujoco_playground").exists():
            return str(p)
    return ""


def resolve_trainer_entry(trainer: str, playground_root: str = "") -> list[str]:
    """解析训练入口命令。优先使用 PATH 里的 CLI，回退到 `python learning/*.py`。"""
    if trainer not in {"jax-ppo", "rsl-ppo"}:
        raise ValueError(f"不支持的 trainer: {trainer}")
    bin_name = "train-jax-ppo" if trainer == "jax-ppo" else "train-rsl-ppo"
    bin_path = shutil.which(bin_name)
    if bin_path:
        return [bin_path]

    root = detect_playground_root(playground_root)
    if root:
        script = Path(root) / "learning" / ("train_jax_ppo.py" if trainer == "jax-ppo" else "train_rsl_ppo.py")
        if script.exists():
            return ["python", str(script)]
    raise RuntimeError(
        f"未找到 `{bin_name}`。请先安装 MuJoCo Playground，或设置 MUJOCO_PLAYGROUND_ROOT。"
    )


def playground_importable() -> bool:
    return importlib.util.find_spec("mujoco_playground") is not None
