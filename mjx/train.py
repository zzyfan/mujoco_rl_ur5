#!/usr/bin/env python3
"""MuJoCo Playground 训练入口适配器（Brax training + MJWarp）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import argparse  # 依赖导入：先认清这个文件需要哪些外部能力
import os  # 依赖导入：先认清这个文件需要哪些外部能力
import shlex  # 依赖导入：先认清这个文件需要哪些外部能力
import subprocess  # 依赖导入：先认清这个文件需要哪些外部能力
import sys  # 依赖导入：先认清这个文件需要哪些外部能力
from dataclasses import dataclass  # 依赖导入：先认清这个文件需要哪些外部能力
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


@dataclass  # 用 dataclass 收拢配置，调参时优先回到这里
class PlaygroundTrainArgs:  # 类定义：先理解职责边界，再进入方法细节
    trainer: str = "jax-ppo"  # 状态或中间变量：调试时多观察它的值如何流动
    env_name: str = "CartpoleBalance"  # 状态或中间变量：调试时多观察它的值如何流动
    impl: str = "warp"  # 状态或中间变量：调试时多观察它的值如何流动
    seed: int = 42  # 状态或中间变量：调试时多观察它的值如何流动
    playground_root: str = ""  # 状态或中间变量：调试时多观察它的值如何流动
    dry_run: bool = False  # 状态或中间变量：调试时多观察它的值如何流动
    test: bool = False  # 状态或中间变量：调试时多观察它的值如何流动
    test_command: str = ""  # 状态或中间变量：调试时多观察它的值如何流动


def _parse_args() -> tuple[PlaygroundTrainArgs, list[str]]:  # 函数定义：先看输入输出，再理解内部控制流
    p = argparse.ArgumentParser(description="Playground 训练入口（默认 MJWarp）")  # 参数解析器：脚本对外暴露的命令行接口从这里定义
    p.add_argument("--trainer", choices=["jax-ppo", "rsl-ppo"], default="jax-ppo")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--env-name", type=str, default="CartpoleBalance")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--impl", choices=["warp", "mjx"], default="warp")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--seed", type=int, default=42)  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--playground-root", type=str, default="")  # 命令行参数：这里就是直接的调参入口
    p.add_argument("--test", action="store_true")  # 命令行参数：这里就是直接的调参入口
    p.add_argument(  # 命令行参数：这里就是直接的调参入口
        "--test-command",  # 代码执行语句：结合上下文理解它对后续流程的影响
        type=str,  # 状态或中间变量：调试时多观察它的值如何流动
        default="",  # 状态或中间变量：调试时多观察它的值如何流动
        help="测试模式时执行的命令（示例: \"python learning/train_jax_ppo.py --env_name CartpoleBalance --impl warp --num_timesteps 0\"）",  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    p.add_argument("--dry-run", action="store_true")  # 命令行参数：这里就是直接的调参入口
    ns, passthrough = p.parse_known_args()  # 状态或中间变量：调试时多观察它的值如何流动
    return (  # 把当前结果返回给上层调用方
        PlaygroundTrainArgs(  # 代码执行语句：结合上下文理解它对后续流程的影响
            trainer=ns.trainer,  # 状态或中间变量：调试时多观察它的值如何流动
            env_name=ns.env_name,  # 状态或中间变量：调试时多观察它的值如何流动
            impl=ns.impl,  # 状态或中间变量：调试时多观察它的值如何流动
            seed=ns.seed,  # 状态或中间变量：调试时多观察它的值如何流动
            playground_root=ns.playground_root,  # 状态或中间变量：调试时多观察它的值如何流动
            dry_run=ns.dry_run,  # 状态或中间变量：调试时多观察它的值如何流动
            test=ns.test,  # 状态或中间变量：调试时多观察它的值如何流动
            test_command=ns.test_command,  # 状态或中间变量：调试时多观察它的值如何流动
        ),  # 收束上一段结构，阅读时回看上面的参数或元素
        passthrough,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def _run_train(args: PlaygroundTrainArgs, passthrough: list[str]) -> int:  # 函数定义：先看输入输出，再理解内部控制流
    ensure_warp_runtime()  # 代码执行语句：结合上下文理解它对后续流程的影响
    playground_root = detect_playground_root(args.playground_root)  # 状态或中间变量：调试时多观察它的值如何流动
    if not playground_root and not playground_importable():  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise RuntimeError(  # 主动抛错：用来尽早暴露错误输入或不支持状态
            "未检测到 MuJoCo Playground。请先安装并设置 MUJOCO_PLAYGROUND_ROOT，或在其仓库目录中运行。"  # 代码执行语句：结合上下文理解它对后续流程的影响
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    entry = resolve_trainer_entry(args.trainer, playground_root)  # 状态或中间变量：调试时多观察它的值如何流动
    cmd = [  # 状态或中间变量：调试时多观察它的值如何流动
        *entry,  # 代码执行语句：结合上下文理解它对后续流程的影响
        "--env_name",  # 代码执行语句：结合上下文理解它对后续流程的影响
        args.env_name,  # 代码执行语句：结合上下文理解它对后续流程的影响
        "--impl",  # 代码执行语句：结合上下文理解它对后续流程的影响
        args.impl,  # 代码执行语句：结合上下文理解它对后续流程的影响
        "--seed",  # 代码执行语句：结合上下文理解它对后续流程的影响
        str(args.seed),  # 代码执行语句：结合上下文理解它对后续流程的影响
        *passthrough,  # 代码执行语句：结合上下文理解它对后续流程的影响
    ]  # 收束上一段结构，阅读时回看上面的参数或元素

    cwd = playground_root if playground_root else None  # 状态或中间变量：调试时多观察它的值如何流动
    print(f"训练入口: {' '.join(shlex.quote(x) for x in entry)}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"运行目录: {cwd or os.getcwd()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"Warp 运行时: {describe_warp_runtime()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"完整命令: {' '.join(shlex.quote(x) for x in cmd)}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    if args.dry_run:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return 0  # 把当前结果返回给上层调用方
    proc = subprocess.run(cmd, cwd=cwd, check=False)  # 状态或中间变量：调试时多观察它的值如何流动
    return int(proc.returncode)  # 把当前结果返回给上层调用方


def _run_test(args: PlaygroundTrainArgs) -> int:  # 函数定义：先看输入输出，再理解内部控制流
    if not args.test_command.strip():  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise RuntimeError(  # 主动抛错：用来尽早暴露错误输入或不支持状态
            "当前适配器无法自动推断 Playground 测试流程，请显式传入 --test-command。"  # 代码执行语句：结合上下文理解它对后续流程的影响
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    ensure_warp_runtime()  # 代码执行语句：结合上下文理解它对后续流程的影响
    playground_root = detect_playground_root(args.playground_root)  # 状态或中间变量：调试时多观察它的值如何流动
    cwd = playground_root if playground_root else None  # 状态或中间变量：调试时多观察它的值如何流动
    cmd = shlex.split(args.test_command)  # 状态或中间变量：调试时多观察它的值如何流动
    print(f"测试命令: {' '.join(shlex.quote(x) for x in cmd)}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"运行目录: {cwd or os.getcwd()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    print(f"Warp 运行时: {describe_warp_runtime()}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    if args.dry_run:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return 0  # 把当前结果返回给上层调用方
    proc = subprocess.run(cmd, cwd=cwd, check=False)  # 状态或中间变量：调试时多观察它的值如何流动
    return int(proc.returncode)  # 把当前结果返回给上层调用方


def main() -> None:  # 脚本入口：先看它如何把各个步骤串起来
    args, passthrough = _parse_args()  # 状态或中间变量：调试时多观察它的值如何流动
    if args.test:  # 条件分支：学习时先看触发条件，再看两边行为差异
        code = _run_test(args)  # 状态或中间变量：调试时多观察它的值如何流动
    else:  # 兜底分支：当前面条件都不满足时走这里
        code = _run_train(args, passthrough)  # 状态或中间变量：调试时多观察它的值如何流动
    raise SystemExit(code)  # 主动抛错：用来尽早暴露错误输入或不支持状态


if __name__ == "__main__":  # 条件分支：学习时先看触发条件，再看两边行为差异
    main()  # 代码执行语句：结合上下文理解它对后续流程的影响
