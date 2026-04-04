"""MJX/Warp backend helpers."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    MJX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local runtime.
    jax = None
    jnp = None
    mjx = None
    MJX_IMPORT_ERROR = exc

try:
    import mujoco_warp as mjwarp
    import warp as wp

    WARP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local runtime.
    mjwarp = None
    wp = None
    WARP_IMPORT_ERROR = exc

_WARP_INITIALIZED = False


def mjx_available() -> bool:
    return mjx is not None and jax is not None and jnp is not None


def warp_available() -> bool:
    return mjwarp is not None and wp is not None


def resolve_physics_backend(requested: str) -> str:
    choice = str(requested or "auto").lower()
    if choice not in {"auto", "mjx", "warp"}:
        raise ValueError(f"不支持的物理后端: {requested}")
    if choice == "auto":
        if mjx_available():
            return "mjx"
        if warp_available():
            return "warp"
        raise RuntimeError(
            "当前环境既不可用 MuJoCo Warp，也不可用 MuJoCo MJX；"
            f"warp_error={WARP_IMPORT_ERROR!r}, mjx_error={MJX_IMPORT_ERROR!r}"
        )
    if choice == "warp" and not warp_available():
        raise RuntimeError(f"请求使用 MuJoCo Warp，但当前不可用: {WARP_IMPORT_ERROR!r}")
    if choice == "mjx" and not mjx_available():
        raise RuntimeError(f"请求使用 MuJoCo MJX，但当前不可用: {MJX_IMPORT_ERROR!r}")
    return choice


def ensure_warp_runtime() -> None:
    global _WARP_INITIALIZED
    if not warp_available():
        raise RuntimeError(f"MuJoCo Warp 当前不可用: {WARP_IMPORT_ERROR!r}")
    if _WARP_INITIALIZED:
        return
    wp.init()
    _WARP_INITIALIZED = True
