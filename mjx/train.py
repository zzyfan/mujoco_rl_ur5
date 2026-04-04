#!/usr/bin/env python3
"""MuJoCo Playground 训练入口适配器（Brax training + MJWarp）。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class PlaygroundTrainArgs:
    trainer: str = "jax-ppo"
    env_name: str = "CartpoleBalance"
    impl: str = "warp"
    seed: int = 42
    playground_root: str = ""
    dry_run: bool = False
    test: bool = False
    test_command: str = ""


def _parse_args() -> tuple[PlaygroundTrainArgs, list[str]]:
    p = argparse.ArgumentParser(description="Playground 训练入口（默认 MJWarp）")
    p.add_argument("--trainer", choices=["jax-ppo", "rsl-ppo"], default="jax-ppo")
    p.add_argument("--env-name", type=str, default="CartpoleBalance")
    p.add_argument("--impl", choices=["warp", "mjx"], default="warp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--playground-root", type=str, default="")
    p.add_argument("--test", action="store_true")
    p.add_argument(
        "--test-command",
        type=str,
        default="",
        help="测试模式时执行的命令（示例: \"python learning/train_jax_ppo.py --env_name CartpoleBalance --impl warp --num_timesteps 0\"）",
    )
    p.add_argument("--dry-run", action="store_true")
    ns, passthrough = p.parse_known_args()
    return (
        PlaygroundTrainArgs(
            trainer=ns.trainer,
            env_name=ns.env_name,
            impl=ns.impl,
            seed=ns.seed,
            playground_root=ns.playground_root,
            dry_run=ns.dry_run,
            test=ns.test,
            test_command=ns.test_command,
        ),
        passthrough,
    )


def _run_train(args: PlaygroundTrainArgs, passthrough: list[str]) -> int:
    ensure_warp_runtime()
    playground_root = detect_playground_root(args.playground_root)
    if not playground_root and not playground_importable():
        raise RuntimeError(
            "未检测到 MuJoCo Playground。请先安装并设置 MUJOCO_PLAYGROUND_ROOT，或在其仓库目录中运行。"
        )
    entry = resolve_trainer_entry(args.trainer, playground_root)
    cmd = [
        *entry,
        "--env_name",
        args.env_name,
        "--impl",
        args.impl,
        "--seed",
        str(args.seed),
        *passthrough,
    ]

    cwd = playground_root if playground_root else None
    print(f"训练入口: {' '.join(shlex.quote(x) for x in entry)}")
    print(f"运行目录: {cwd or os.getcwd()}")
    print(f"Warp 运行时: {describe_warp_runtime()}")
    print(f"完整命令: {' '.join(shlex.quote(x) for x in cmd)}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


def _run_test(args: PlaygroundTrainArgs) -> int:
    if not args.test_command.strip():
        raise RuntimeError(
            "当前适配器无法自动推断 Playground 测试流程，请显式传入 --test-command。"
        )
    ensure_warp_runtime()
    playground_root = detect_playground_root(args.playground_root)
    cwd = playground_root if playground_root else None
    cmd = shlex.split(args.test_command)
    print(f"测试命令: {' '.join(shlex.quote(x) for x in cmd)}")
    print(f"运行目录: {cwd or os.getcwd()}")
    print(f"Warp 运行时: {describe_warp_runtime()}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


def main() -> None:
    args, passthrough = _parse_args()
    if args.test:
        code = _run_test(args)
    else:
        code = _run_train(args, passthrough)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
