#!/usr/bin/env python3
# mjlab 版推理 / 可视化入口。
#
# 这个脚本主要做两件事：
# 1. 导入本仓库的自定义任务，确保 task registry 能看到 `Mjlab-Reach-UR5`
# 2. 帮忙定位最近一次 checkpoint，减少每次手敲完整路径的负担

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from mjlab.scripts.play import PlayConfig, run_play

from mjlab_ur5_task import TASK_ID, make_mjlab_rl_cfg


def _latest_checkpoint(experiment_name: str) -> Path:
    # mjlab 默认把日志写到 logs/rsl_rl/{experiment_name}/{timestamp_run_name}/
    # 这里沿用这套结构，自动挑最新 run 里的最新 model_*.pt。
    log_root = Path("logs") / "rsl_rl" / experiment_name
    if not log_root.exists():
        raise FileNotFoundError(f"未找到 mjlab 日志目录: {log_root}")

    run_dirs = sorted((path for path in log_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not run_dirs:
        raise FileNotFoundError(f"日志目录下还没有训练产物: {log_root}")

    latest_run = run_dirs[-1]
    checkpoints = sorted(latest_run.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"在 {latest_run} 里没有找到 model_*.pt checkpoint")
    return checkpoints[-1]


@dataclass
class PlayCliConfig:
    # 支持三种策略来源：
    # - trained: 加载训练好的 checkpoint
    # - zero: 全零动作，用来排查动作通路
    # - random: 随机动作，用来粗看任务是否有明显异常
    agent: Literal["trained", "zero", "random"] = "trained"
    checkpoint_file: str | None = None

    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    camera: int | str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    no_terminations: bool = False


def main() -> None:
    cli = tyro.cli(PlayCliConfig)

    checkpoint_file = cli.checkpoint_file
    if cli.agent == "trained" and checkpoint_file is None:
        checkpoint_file = str(_latest_checkpoint(make_mjlab_rl_cfg().experiment_name))

    play_cfg = PlayConfig(
        agent=cli.agent,
        checkpoint_file=checkpoint_file,
        num_envs=int(cli.num_envs),
        device=cli.device,
        video=bool(cli.video),
        video_length=int(cli.video_length),
        video_height=cli.video_height,
        video_width=cli.video_width,
        camera=cli.camera,
        viewer=cli.viewer,
        no_terminations=bool(cli.no_terminations),
    )
    run_play(task_id=TASK_ID, cfg=play_cfg)


if __name__ == "__main__":
    main()
