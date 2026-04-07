#!/usr/bin/env python3
# mjlab 版训练入口。
#
# 这个脚本不重新发明一套训练器，而是直接复用 mjlab 官方的
# `launch_training()` 流程，只把本仓库的 UR5 reach 任务配置接进去。
# 这样有两个好处：
# 1. 训练调度、日志目录和多 GPU 行为尽量保持和官方一致。
# 2. 当前仓库仍然保留自己的任务定义、中文注释和参数解释风格。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tyro
from mjlab.scripts.train import TrainConfig, launch_training

from mjlab_ur5_task import TASK_ID, make_mjlab_env_cfg, make_mjlab_rl_cfg
from ur5_reach_config import UR5ReachEnvConfig


@dataclass
class TrainCliConfig:
    # 任务本体参数继续沿用主线 dataclass。
    #
    # 这样 reach 任务的目标空间、奖励系数、课程学习和控制模式
    # 仍然只有一套来源，后面继续维护不会分叉。
    env: UR5ReachEnvConfig = field(default_factory=UR5ReachEnvConfig)

    # 训练器外层参数交给 mjlab 的 runner。
    run_name: str = "ur5_reach_mjlab"
    seed: int = 42
    num_envs: int = 1024
    num_steps_per_env: int = 24
    max_iterations: int = 4_000
    save_interval: int = 100

    # mjlab 官方训练脚本支持单 GPU / 多 GPU 两种模式，
    # 这里直接暴露同样的 GPU 选择接口。
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    # 可视化与调试开关。
    video: bool = False
    video_length: int = 200
    video_interval: int = 2_000
    enable_nan_guard: bool = False


def main() -> None:
    cli = tyro.cli(TrainCliConfig)

    env_cfg = make_mjlab_env_cfg(cli.env)
    env_cfg.scene.num_envs = int(cli.num_envs)
    env_cfg.seed = int(cli.seed)

    agent_cfg = make_mjlab_rl_cfg()
    agent_cfg.seed = int(cli.seed)
    agent_cfg.run_name = cli.run_name
    agent_cfg.num_steps_per_env = int(cli.num_steps_per_env)
    agent_cfg.max_iterations = int(cli.max_iterations)
    agent_cfg.save_interval = int(cli.save_interval)

    train_cfg = TrainConfig(
        env=env_cfg,
        agent=agent_cfg,
        video=bool(cli.video),
        video_length=int(cli.video_length),
        video_interval=int(cli.video_interval),
        enable_nan_guard=bool(cli.enable_nan_guard),
        gpu_ids=cli.gpu_ids,
    )
    launch_training(task_id=TASK_ID, args=train_cfg)


if __name__ == "__main__":
    main()
