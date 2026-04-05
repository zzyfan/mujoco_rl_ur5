"""Warp GPU runtime checks used by the Brax training entrypoints."""

from __future__ import annotations

import importlib.util
try:
    import warp as wp
except Exception:
    wp = None

try:
    import mujoco_warp as mjwarp
except Exception:
    mjwarp = None

_WARP_INITIALIZED = False  # 避免重复初始化 Warp 运行时。
_CUDA_DEVICE = ""


def warp_available() -> bool:
    """Return True when both Warp Python packages are importable."""
    return wp is not None and mjwarp is not None


def ensure_warp_runtime() -> str:
    """Initialize Warp and require at least one CUDA device."""
    global _WARP_INITIALIZED, _CUDA_DEVICE
    if not warp_available():
        raise RuntimeError("MuJoCo Warp 不可用，请先安装 `mujoco-warp` 与 `warp-lang`。")
    if _WARP_INITIALIZED:
        return _CUDA_DEVICE
    wp.init()  # 初始化 Warp 内核、设备列表和缓存目录。
    cuda_devices = list(wp.get_cuda_devices())
    if not cuda_devices:
        raise RuntimeError("未检测到可用的 CUDA 设备，纯 GPU 训练线无法启动。")
    _CUDA_DEVICE = str(cuda_devices[0])
    wp.set_device(_CUDA_DEVICE)  # 把默认执行设备固定到第一块 CUDA 卡。
    _WARP_INITIALIZED = True
    return _CUDA_DEVICE


def describe_warp_runtime() -> str:
    """Describe the active Warp device and detected CUDA devices."""
    if not warp_available():
        return "warp unavailable"
    active_device = ensure_warp_runtime()
    try:
        device = str(wp.get_device())
    except Exception:
        device = active_device or "unknown"
    try:
        cuda_devices = list(wp.get_cuda_devices())
    except Exception:
        cuda_devices = []
    return f"warp_device={device}, cuda_devices={cuda_devices}"


def playground_importable() -> bool:
    """Return True when `mujoco_playground` can be imported."""
    return importlib.util.find_spec("mujoco_playground") is not None
