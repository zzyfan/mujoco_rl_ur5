#!/usr/bin/env python3
"""Smoke test for the Warp GPU reach environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from warp_gpu.env import UR5ReachWarpEnv, default_config
    from warp_gpu.runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable
else:
    from .env import UR5ReachWarpEnv, default_config
    from .runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warp GPU reach 环境自检")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")
    p.add_argument("--steps", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not playground_importable():
        raise SystemExit("未检测到 `mujoco_playground`。")
    ensure_warp_runtime()
    print(f"warp={describe_warp_runtime()}")

    cfg = default_config(args.robot)
    env = UR5ReachWarpEnv(config=cfg)

    state = env.reset(jax.random.PRNGKey(42))
    print(f"xml={env.xml_path}")
    print(f"obs_shape={state.obs.shape} action_size={env.action_size}")
    print(f"initial_distance={float(state.metrics['distance']):.4f}")

    action = jp.zeros((env.action_size,), dtype=jp.float32)
    for i in range(max(int(args.steps), 1)):
        state = env.step(state, action)
        print(
            f"step={i + 1} reward={float(state.reward):.3f} "
            f"distance={float(state.metrics['distance']):.4f} done={bool(state.done)}"
        )
        if bool(state.done):
            break


if __name__ == "__main__":
    main()
