#!/usr/bin/env python3
"""Playground + MJWarp 环境自检脚本（轻量）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import argparse  # 依赖导入：先认清这个文件需要哪些外部能力
import shlex  # 依赖导入：先认清这个文件需要哪些外部能力
import subprocess  # 依赖导入：先认清这个文件需要哪些外部能力
import sys  # 依赖导入：先认清这个文件需要哪些外部能力
from pathlib import Path  # 依赖导入：先认清这个文件需要哪些外部能力

if __package__ in (None, ""):  # 条件分支：学习时先看触发条件，再看两边行为差异
    ROOT = Path(__file__).resolve().parents[1]  # 状态或中间变量：调试时多观察它的值如何流动
    if str(ROOT) not in sys.path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        sys.path.insert(0, str(ROOT))  # 代码执行语句：结合上下文理解它对后续流程的影响
    from mjx.backend import (  # 依赖导入：先认清这个文件需要哪些外部能力
        describe_warp_runtime,  # 代码执行语句：结合上下文理解它对后续流程的影响
        detect_playground_root,  # 代码执行语句：结合上下文理解它对后续流程的影响
        ensure_warp_runtime,  # 代码执行语句：结合上下文理解它对后续流程的影响
        playground_importable,  # 代码执行语句：结合上下文理解它对后续流程的影响
        resolve_trainer_entry,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素
else:  # 兜底分支：当前面条件都不满足时走这里
    from .backend import (  # 依赖导入：先认清这个文件需要哪些外部能力
        describe_warp_runtime,  # 代码执行语句：结合上下文理解它对后续流程的影响
        detect_playground_root,  # 代码执行语句：结合上下文理解它对后续流程的影响
        ensure_warp_runtime,  # 代码执行语句：结合上下文理解它对后续流程的影响
        playground_importable,  # 代码执行语句：结合上下文理解它对后续流程的影响
        resolve_trainer_entry,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def parse_args() -> argparse.Namespace:  # 命令行解析入口：外部调参首先会影响这里
    p = argparse.ArgumentParser(description="Playground + MJWarp 自检")  # 参数解析器：脚本对外暴露的命令行接口从这里定义
    p.add_argument("--trainer", choices=["jax-ppo", "rsl-ppo"], default="jax-ppo")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--playground-root", type=str, default="")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--run-smoke", action="store_true", help="执行一个极短训练命令做烟雾测试")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--env-name", type=str, default="CartpoleBalance")  # 命令行参数：这里就是直接的调参入口
    return p.parse_args()  # 把当前结果返回给上层调用方


def main() -> None:  # 脚本入口：先看它如何把各个步骤串起来
    args = parse_args()  # 状态或中间变量：调试时多观察它的值如何流动
    ensure_warp_runtime()  # 代码执行语句：结合上下文理解它对后续流程的影响
    root = detect_playground_root(args.playground_root)  # 状态或中间变量：调试时多观察它的值如何流动
    print(f"warp: {describe_warp_runtime()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"playground_root: {root or '(not found by path heuristic)'}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"playground_importable: {playground_importable()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    entry = resolve_trainer_entry(args.trainer, root)  # 状态或中间变量：调试时多观察它的值如何流动
    print(f"trainer_entry: {' '.join(shlex.quote(x) for x in entry)}")  # 代码执行语句：结合上下文理解它对后续流程的影响

    if args.run_smoke:  # 条件分支：学习时先看触发条件，再看两边行为差异
        cmd = [  # 状态或中间变量：调试时多观察它的值如何流动
            *entry,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "--env_name",  # 代码执行语句：结合上下文理解它对后续流程的影响
            args.env_name,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "--impl",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "warp",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "--seed",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "42",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "--num_timesteps",  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "128",  # 代码执行语句：结合上下文理解它对后续流程的影响
        ]  # 收束上一段结构，阅读时回看上面的参数或元素
        print(f"smoke_cmd: {' '.join(shlex.quote(x) for x in cmd)}")  # 代码执行语句：结合上下文理解它对后续流程的影响
        proc = subprocess.run(cmd, cwd=root or None, check=False)  # 状态或中间变量：调试时多观察它的值如何流动
        if proc.returncode != 0:  # 条件分支：学习时先看触发条件，再看两边行为差异
            raise SystemExit(proc.returncode)  # 主动抛错：用来尽早暴露错误输入或不支持状态


if __name__ == "__main__":  # 条件分支：学习时先看触发条件，再看两边行为差异
    main()  # 代码执行语句：结合上下文理解它对后续流程的影响
