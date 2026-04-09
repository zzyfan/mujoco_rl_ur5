#!/usr/bin/env python3
# Warp 运行时检查模块。
#
# 这个模块把运行时检查从训练脚本里拆出来，原因有两个：
# 1. 训练脚本只保留流程编排逻辑，减少入口文件的杂音。
# 2. notebook、推理脚本和训练脚本都能复用同一套依赖检查逻辑。
#
# 涉及的外部库：
# - `warp-lang`：负责纯 GPU 后端初始化与设备管理。
# - `mujoco-warp`：负责 MuJoCo 在 Warp 后端上的适配。
# - `mujoco_playground`：负责 Warp / MJX 环境的训练封装。

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

_WARP_INITIALIZED = False
_CUDA_DEVICE = ""


def warp_available() -> bool:
    # 检查 Warp 核心依赖是否可导入。
    return wp is not None and mjwarp is not None


def playground_importable() -> bool:
    # 检查 `mujoco_playground` 是否已安装。
    return importlib.util.find_spec("mujoco_playground") is not None


def ensure_warp_runtime() -> str:
    # 初始化 Warp 运行时并绑定第一块 CUDA 设备。
    #
    # 当前实现选择第一块可用 CUDA 设备，目的是让训练脚本在默认情况下行为稳定、可预测。
    global _WARP_INITIALIZED, _CUDA_DEVICE
    if not warp_available():
        raise RuntimeError("MuJoCo Warp 不可用，请先安装 `warp-lang` 与 `mujoco-warp`。")
    if _WARP_INITIALIZED:
        return _CUDA_DEVICE
    wp.init()
    cuda_devices = list(wp.get_cuda_devices())
    if not cuda_devices:
        raise RuntimeError("未检测到可用 CUDA 设备，Warp 纯 GPU 训练线无法启动。")
    _CUDA_DEVICE = str(cuda_devices[0])
    wp.set_device(_CUDA_DEVICE)
    _WARP_INITIALIZED = True
    return _CUDA_DEVICE


def describe_warp_runtime() -> str:
    # 返回当前 Warp 运行时摘要字符串。
    if not warp_available():
        return "warp unavailable"
    active = ensure_warp_runtime()
    try:
        device = str(wp.get_device())
    except Exception:
        device = active or "unknown"
    try:
        cuda_devices = list(wp.get_cuda_devices())
    except Exception:
        cuda_devices = []
    return f"warp_device={device}, cuda_devices={cuda_devices}"
