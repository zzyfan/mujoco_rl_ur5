#!/usr/bin/env python3
"""独立的 MJX/Warp 预览脚本（不影响现有 SB3 训练线）。

用途：
1) 验证当前模型在 MJX 路线下是否可运行
2) 测试批量并行步进吞吐（steps/s）
3) 为后续“单独的新训练线”打基础

注意：
- 这是“独立新线”的后端压测脚本，不是完整 RL 训练脚本。
- 当前仓库未安装 warp（mujoco_warp），因此默认走 mujoco.mjx。
- 对 mesh-heavy 机械臂模型，MJX 可能在完整约束下报错；
  提供 --safe-disable-constraints 作为预览模式开关。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mjx.backend import ensure_warp_runtime, jax, jnp, mjwarp, mjx, resolve_physics_backend, wp
else:
    from .backend import ensure_warp_runtime, jax, jnp, mjwarp, mjx, resolve_physics_backend, wp


def _resolve_xml(robot: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    if robot == "zero_robotiq":
        return root / "assets/zero_arm/zero_with_robotiq_reach.xml"
    return root / "assets/robotiq_cxy/lab_env.xml"


def _maybe_apply_safe_flags(model: mujoco.MjModel, enable: bool) -> None:
    """预览模式：关闭一部分约束，提升 MJX 兼容性。"""
    if not enable:
        return
    safe_flags = (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        | mujoco.mjtDisableBit.mjDSBL_EQUALITY
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    model.opt.disableflags = int(model.opt.disableflags) | int(safe_flags)


def _batched_data(model: mujoco.MjModel, batch_size: int):
    """把单个 mjx.Data 扩展成批量维度 [B, ...]。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    one = mjx.put_data(model, data)

    def tile_leaf(x):
        if isinstance(x, jax.Array):
            return jnp.repeat(x[None, ...], repeats=batch_size, axis=0)
        return x

    return jax.tree_util.tree_map(tile_leaf, one)


def _ctrl_bounds(model: mujoco.MjModel, action_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """优先使用 actuator_ctrlrange；没有时回退到 [-action_scale, action_scale]。"""
    if model.nu <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if model.actuator_ctrlrange is None or model.actuator_ctrlrange.shape[0] != model.nu:
        low = -action_scale * np.ones((model.nu,), dtype=np.float32)
        high = action_scale * np.ones((model.nu,), dtype=np.float32)
        return low, high
    low_np = model.actuator_ctrlrange[:, 0].astype("float32")
    high_np = model.actuator_ctrlrange[:, 1].astype("float32")
    finite_mask = np.isfinite(low_np) & np.isfinite(high_np)
    low = np.where(finite_mask, low_np, -action_scale).astype(np.float32)
    high = np.where(finite_mask, high_np, action_scale).astype(np.float32)
    return low, high


def _run_mjx(model: mujoco.MjModel, args: argparse.Namespace, low: np.ndarray, high: np.ndarray) -> tuple[str, float]:
    mjx_model = mjx.put_model(model)
    data_batched = _batched_data(model, args.batch_size)

    def step_one(d, u):
        return mjx.step(mjx_model, d.replace(ctrl=u))

    step_batch = jax.jit(jax.vmap(step_one, in_axes=(0, 0)))
    key = jax.random.PRNGKey(args.seed)
    low_jax = jnp.asarray(low)
    high_jax = jnp.asarray(high)

    key, sub = jax.random.split(key)
    action0 = jax.random.uniform(sub, shape=(args.batch_size, model.nu), minval=low_jax, maxval=high_jax)
    data_batched = step_batch(data_batched, action0)
    jax.block_until_ready(data_batched.time)

    start = time.time()
    for _ in range(args.steps):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, shape=(args.batch_size, model.nu), minval=low_jax, maxval=high_jax)
        data_batched = step_batch(data_batched, action)
    jax.block_until_ready(data_batched.time)
    elapsed = max(time.time() - start, 1e-9)
    return f"mjx:{jax.default_backend()}", elapsed


def _run_warp(model: mujoco.MjModel, args: argparse.Namespace, low: np.ndarray, high: np.ndarray) -> tuple[str, float]:
    ensure_warp_runtime()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    warp_model = mjwarp.put_model(model)
    data_batched = mjwarp.put_data(model, data, nworld=args.batch_size)
    rng = np.random.default_rng(args.seed)

    action0 = rng.uniform(low, high, size=(args.batch_size, model.nu)).astype(np.float32)
    wp.copy(data_batched.ctrl, wp.array(action0))
    mjwarp.step(warp_model, data_batched)
    wp.synchronize()

    start = time.time()
    for _ in range(args.steps):
        action = rng.uniform(low, high, size=(args.batch_size, model.nu)).astype(np.float32)
        wp.copy(data_batched.ctrl, wp.array(action))
        mjwarp.step(warp_model, data_batched)
    wp.synchronize()
    elapsed = max(time.time() - start, 1e-9)
    return "warp", elapsed


def run(args: argparse.Namespace) -> None:
    xml_path = _resolve_xml(args.robot)
    if not xml_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    _maybe_apply_safe_flags(model, args.safe_disable_constraints)
    low, high = _ctrl_bounds(model, args.action_scale)
    backend = resolve_physics_backend(args.physics_backend)
    if backend == "warp":
        backend_name, elapsed = _run_warp(model, args, low, high)
    else:
        backend_name, elapsed = _run_mjx(model, args, low, high)

    total_env_steps = int(args.steps) * int(args.batch_size)
    sps = total_env_steps / elapsed
    print("=== MJX Separate Preview ===")
    print(f"robot={args.robot}")
    print(f"xml={xml_path}")
    print(f"physics_backend={backend_name}")
    print(f"batch_size={args.batch_size}")
    print(f"steps={args.steps}")
    print(f"safe_disable_constraints={args.safe_disable_constraints}")
    print(f"total_env_steps={total_env_steps}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"env_steps_per_sec={sps:.1f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立 MJX/Warp 预览脚本（批量并行步进压测）")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="zero_robotiq")
    p.add_argument("--physics-backend", choices=["auto", "mjx", "warp"], default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--safe-disable-constraints", action="store_true")
    p.add_argument("--strict-constraints", action="store_false", dest="safe_disable_constraints")
    p.set_defaults(safe_disable_constraints=True)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
