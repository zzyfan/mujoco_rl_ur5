#!/usr/bin/env python3
"""本地 MJX reach 环境烟雾测试。"""

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
    from mjx.backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from mjx.reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name
else:
    from .backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from .reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 MJX reach 环境自检")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")
    p.add_argument("--impl", choices=["warp", "mjx", "jax"], default="warp")
    p.add_argument("--steps", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not playground_importable():
        raise SystemExit("未检测到 `mujoco_playground`。")
    impl = normalize_impl_name(args.impl)
    if impl == "warp":
        ensure_warp_runtime()
        print(f"warp={describe_warp_runtime()}")
    else:
        print("impl=jax")

    cfg = default_config(args.robot)
    cfg.impl = impl
    env = UR5ReachMjxEnv(config=cfg)

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
