#!/usr/bin/env python3
"""Playground + MJWarp 环境自检脚本（轻量）。"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mjx.backend import (
        describe_warp_runtime,
        detect_playground_root,
        ensure_warp_runtime,
        playground_importable,
        resolve_trainer_entry,
    )
else:
    from .backend import (
        describe_warp_runtime,
        detect_playground_root,
        ensure_warp_runtime,
        playground_importable,
        resolve_trainer_entry,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Playground + MJWarp 自检")
    p.add_argument("--trainer", choices=["jax-ppo", "rsl-ppo"], default="jax-ppo")
    p.add_argument("--playground-root", type=str, default="")
    p.add_argument("--run-smoke", action="store_true", help="执行一个极短训练命令做烟雾测试")
    p.add_argument("--env-name", type=str, default="CartpoleBalance")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_warp_runtime()
    root = detect_playground_root(args.playground_root)
    print(f"warp: {describe_warp_runtime()}")
    print(f"playground_root: {root or '(not found by path heuristic)'}")
    print(f"playground_importable: {playground_importable()}")
    entry = resolve_trainer_entry(args.trainer, root)
    print(f"trainer_entry: {' '.join(shlex.quote(x) for x in entry)}")

    if args.run_smoke:
        cmd = [
            *entry,
            "--env_name",
            args.env_name,
            "--impl",
            "warp",
            "--seed",
            "42",
            "--num_timesteps",
            "128",
        ]
        print(f"smoke_cmd: {' '.join(shlex.quote(x) for x in cmd)}")
        proc = subprocess.run(cmd, cwd=root or None, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
