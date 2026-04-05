#!/usr/bin/env python3
"""Warp GPU training entrypoint for the UR5 reach task."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from mujoco_playground._src import wrapper

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from warp_gpu.env import UR5ReachWarpEnv, default_config
    from warp_gpu.runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable
else:
    from .env import UR5ReachWarpEnv, default_config
    from .runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


@dataclass
class TrainArgs:
    algo: str = "ppo"  # Brax 训练器名称。
    robot: str = "ur5_cxy"  # 机器人型号决定 XML、初始姿态和目标采样范围。
    seed: int = 42
    num_timesteps: int = 5_000_000  # 总训练步数。
    num_envs: int = 256  # 并行训练环境数量。
    num_eval_envs: int = 128  # 并行评估环境数量。
    num_evals: int = 10
    learning_rate: float = 3e-4  # Brax 优化器学习率。
    entropy_cost: float = 1e-4  # PPO 熵正则系数。
    discounting: float = 0.99
    unroll_length: int = 10  # PPO rollout 片段长度。
    batch_size: int = 512  # 每次参数更新读取的样本数。
    num_minibatches: int = 8  # PPO 每轮更新拆分的 mini-batch 数。
    num_updates_per_batch: int = 4  # PPO 对同一批样本重复更新的次数。
    reward_scaling: float = 1.0  # Brax 训练器内部的奖励缩放因子。
    normalize_observations: bool = True  # 是否启用观测标准化。
    sac_tau: float = 0.005  # SAC 目标网络软更新系数。
    sac_min_replay_size: int = 8192  # SAC 开始更新前要求的最小回放池大小。
    sac_max_replay_size: int = 262_144  # SAC 回放池容量上限。
    sac_grad_updates_per_step: int = 1  # SAC 每轮采样后的梯度更新次数。
    action_repeat: int = 1  # 同一动作重复执行的物理步数。
    episode_length: int = 3000  # 单回合最大决策步数。
    frame_skip: int = 1  # 控制步长与仿真步长的倍率。
    success_threshold: float = 0.01  # 成功判定距离阈值，单位米。
    naconmax: int = 128  # 接触缓存容量。
    naccdmax: int = 128  # CCD 接触缓存容量。
    njmax: int = 64  # 约束缓存容量。
    logdir: str = "logs/warp_gpu"  # 文本日志输出目录。
    model_dir: str = "models/warp_gpu"  # 配置、checkpoint 和最终参数目录。
    run_name: str = "ur5_reach_warp_gpu"  # 当前实验名称。
    dry_run: bool = False  # 只构建环境和训练配置，不执行训练。
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="Warp GPU UR5/zero reach 训练入口")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-eval-envs", type=int, default=128)
    p.add_argument("--num-evals", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-cost", type=float, default=1e-4)
    p.add_argument("--discounting", type=float, default=0.99)
    p.add_argument("--unroll-length", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--num-updates-per-batch", type=int, default=4)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sac-tau", type=float, default=0.005)
    p.add_argument("--sac-min-replay-size", type=int, default=8192)
    p.add_argument("--sac-max-replay-size", type=int, default=262_144)
    p.add_argument("--sac-grad-updates-per-step", type=int, default=1)
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--episode-length", type=int, default=3000)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--success-threshold", type=float, default=0.01)
    p.add_argument("--naconmax", type=int, default=128)
    p.add_argument("--naccdmax", type=int, default=128)
    p.add_argument("--njmax", type=int, default=64)
    p.add_argument("--logdir", type=str, default="logs/warp_gpu")
    p.add_argument("--model-dir", type=str, default="models/warp_gpu")
    p.add_argument("--run-name", type=str, default="ur5_reach_warp_gpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fixed-target-x", type=float, default=None)
    p.add_argument("--fixed-target-y", type=float, default=None)
    p.add_argument("--fixed-target-z", type=float, default=None)
    ns = p.parse_args()
    return TrainArgs(
        algo=ns.algo,
        robot=ns.robot,
        seed=ns.seed,
        num_timesteps=ns.num_timesteps,
        num_envs=ns.num_envs,
        num_eval_envs=ns.num_eval_envs,
        num_evals=ns.num_evals,
        learning_rate=ns.learning_rate,
        entropy_cost=ns.entropy_cost,
        discounting=ns.discounting,
        unroll_length=ns.unroll_length,
        batch_size=ns.batch_size,
        num_minibatches=ns.num_minibatches,
        num_updates_per_batch=ns.num_updates_per_batch,
        reward_scaling=ns.reward_scaling,
        normalize_observations=ns.normalize_observations,
        sac_tau=ns.sac_tau,
        sac_min_replay_size=ns.sac_min_replay_size,
        sac_max_replay_size=ns.sac_max_replay_size,
        sac_grad_updates_per_step=ns.sac_grad_updates_per_step,
        action_repeat=ns.action_repeat,
        episode_length=ns.episode_length,
        frame_skip=ns.frame_skip,
        success_threshold=ns.success_threshold,
        naconmax=ns.naconmax,
        naccdmax=ns.naccdmax,
        njmax=ns.njmax,
        logdir=ns.logdir,
        model_dir=ns.model_dir,
        run_name=ns.run_name,
        dry_run=ns.dry_run,
        fixed_target_x=ns.fixed_target_x,
        fixed_target_y=ns.fixed_target_y,
        fixed_target_z=ns.fixed_target_z,
    )


def _build_env_config(args: TrainArgs):
    cfg = default_config(args.robot)
    cfg.frame_skip = max(int(args.frame_skip), 1)  # frame_skip 会同步影响 ctrl_dt。
    cfg.action_repeat = max(int(args.action_repeat), 1)
    cfg.episode_length = max(int(args.episode_length), 1)
    cfg.success_threshold = float(args.success_threshold)
    cfg.naconmax = max(int(args.naconmax), 1)
    cfg.naccdmax = max(int(args.naccdmax), 1)
    cfg.njmax = max(int(args.njmax), 1)
    cfg.fixed_target_x = args.fixed_target_x
    cfg.fixed_target_y = args.fixed_target_y
    cfg.fixed_target_z = args.fixed_target_z
    return cfg


def _run_train(args: TrainArgs) -> int:
    if not playground_importable():
        raise RuntimeError("未检测到 `mujoco_playground`，无法按 Playground 方式训练。")

    ensure_warp_runtime()

    env_cfg = _build_env_config(args)
    run_dir = Path(args.model_dir).resolve() / args.algo / args.robot / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    config_payload = {
        "train_args": asdict(args),
        "env_config": env_cfg.to_dict(),
        "runtime": describe_warp_runtime(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_payload, fp, indent=2, ensure_ascii=False)

    print(f"task=ur5_reach_warp_gpu algo={args.algo} robot={args.robot}")
    print(f"xml={(Path(env_cfg.model_xml).resolve())}")
    print(f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs} episode_length={args.episode_length}")
    print(f"run_dir={run_dir}")
    print(f"warp={describe_warp_runtime()}")
    print("building_train_env=true")
    env = UR5ReachWarpEnv(config=env_cfg)
    print(f"obs_dim={env.observation_size} action_dim={env.action_size}")
    if args.dry_run:  # dry-run 用来检查环境和训练参数能否完成初始化。
        return 0
    print("building_eval_env=true")
    eval_env = UR5ReachWarpEnv(config=env_cfg)
    wrap_env_fn = functools.partial(
        wrapper.wrap_for_brax_training,
        full_reset=True,  # full_reset 让回合结束后的目标点重新采样。
    )

    times = [time.monotonic()]

    def progress(step: int, metrics) -> None:
        times.append(time.monotonic())
        if "eval/episode_reward" in metrics:
            print(f"{step}: eval_reward={float(metrics['eval/episode_reward']):.3f}")
        elif "episode/sum_reward" in metrics:
            print(f"{step}: train_reward={float(metrics['episode/sum_reward']):.3f}")

    if args.algo == "ppo":
        train_fn = functools.partial(  # PPO 通过 rollout 收集样本后执行多轮策略更新。
            ppo.train,
            num_timesteps=args.num_timesteps,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            learning_rate=args.learning_rate,
            entropy_cost=args.entropy_cost,
            discounting=args.discounting,
            unroll_length=args.unroll_length,
            batch_size=args.batch_size,
            num_minibatches=args.num_minibatches,
            num_updates_per_batch=args.num_updates_per_batch,
            reward_scaling=args.reward_scaling,
            normalize_observations=args.normalize_observations,
            seed=args.seed,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            wrap_env_fn=wrap_env_fn,
            save_checkpoint_path=str(ckpt_dir),  # Brax 会把参数周期性写入 checkpoint 目录。
        )
    else:
        train_fn = functools.partial(  # SAC 依赖回放池，因此需要额外配置池容量和软更新参数。
            sac.train,
            num_timesteps=args.num_timesteps,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            learning_rate=args.learning_rate,
            discounting=args.discounting,
            batch_size=args.batch_size,
            reward_scaling=args.reward_scaling,
            normalize_observations=args.normalize_observations,
            tau=args.sac_tau,
            min_replay_size=args.sac_min_replay_size,
            max_replay_size=args.sac_max_replay_size,
            grad_updates_per_step=args.sac_grad_updates_per_step,
            seed=args.seed,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            wrap_env_fn=wrap_env_fn,
            checkpoint_logdir=str(ckpt_dir),
        )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )
    del make_inference_fn
    brax_model.save_params(str(run_dir / "final_policy.msgpack"), params)  # 训练结束后导出最终策略参数。

    print("Done training.")
    if len(times) > 1:
        print(f"jit_compile_s={times[1] - times[0]:.2f}")
        print(f"total_train_s={times[-1] - times[0]:.2f}")
    return 0


def main() -> None:
    raise SystemExit(_run_train(_parse_args()))


if __name__ == "__main__":
    main()
