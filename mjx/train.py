#!/usr/bin/env python3
"""本地 UR5/zero Playground PPO 训练入口。"""

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
from mujoco_playground._src import wrapper

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mjx.backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from mjx.reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name
else:
    from .backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from .reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name


@dataclass
class TrainArgs:
    robot: str = "ur5_cxy"  # 两套 XML 共用同一套 reach 逻辑，只在目标范围和 home pose 上分开。
    impl: str = "warp"  # 真正想跑出吞吐时优先用 warp；`mjx` 会映射到 MuJoCo 的 `jax` impl。
    seed: int = 42
    num_timesteps: int = 5_000_000  # Playground PPO 默认靠大样本吞吐推进，不走 replay buffer。
    num_envs: int = 256  # PPO 训练使用的并行环境数量。
    num_eval_envs: int = 128  # 评估使用的并行环境数量。
    num_evals: int = 10
    learning_rate: float = 3e-4  # PPO 常见起点；学不动先看 reward，再考虑改它。
    entropy_cost: float = 1e-4
    discounting: float = 0.99
    unroll_length: int = 10  # 单次 rollout 片段长度；太短会让优势估计更吵。
    batch_size: int = 512  # PPO update 的 batch，不是环境并行数。
    num_minibatches: int = 8
    num_updates_per_batch: int = 4
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    action_repeat: int = 1
    episode_length: int = 3000  # 单回合步数上限。
    frame_skip: int = 1
    success_threshold: float = 0.01  # 仍然沿用 classic 的“1 cm 内成功”定义。
    logdir: str = "logs/mjx"
    model_dir: str = "models/mjx"
    run_name: str = "ur5_reach_playground"
    dry_run: bool = False
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="UR5/zero Playground PPO 训练入口")
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")
    p.add_argument("--impl", choices=["warp", "mjx", "jax"], default="warp")
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
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--episode-length", type=int, default=3000)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--success-threshold", type=float, default=0.01)
    p.add_argument("--logdir", type=str, default="logs/mjx")
    p.add_argument("--model-dir", type=str, default="models/mjx")
    p.add_argument("--run-name", type=str, default="ur5_reach_playground")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fixed-target-x", type=float, default=None)
    p.add_argument("--fixed-target-y", type=float, default=None)
    p.add_argument("--fixed-target-z", type=float, default=None)
    ns = p.parse_args()
    return TrainArgs(
        robot=ns.robot,
        impl=ns.impl,
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
        action_repeat=ns.action_repeat,
        episode_length=ns.episode_length,
        frame_skip=ns.frame_skip,
        success_threshold=ns.success_threshold,
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
    cfg.impl = normalize_impl_name(args.impl)  # 这里把命令行里的 `mjx` 统一落成 MuJoCo 可识别的 impl。
    cfg.frame_skip = max(int(args.frame_skip), 1)  # frame_skip 会同步影响 ctrl_dt。
    cfg.action_repeat = max(int(args.action_repeat), 1)
    cfg.episode_length = max(int(args.episode_length), 1)
    cfg.success_threshold = float(args.success_threshold)
    cfg.fixed_target_x = args.fixed_target_x
    cfg.fixed_target_y = args.fixed_target_y
    cfg.fixed_target_z = args.fixed_target_z
    return cfg


def _run_train(args: TrainArgs) -> int:
    if not playground_importable():
        raise RuntimeError("未检测到 `mujoco_playground`，无法按 Playground 方式训练。")

    impl = normalize_impl_name(args.impl)
    if impl == "warp":
        ensure_warp_runtime()

    env_cfg = _build_env_config(args)
    run_dir = Path(args.model_dir).resolve() / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    config_payload = {
        "train_args": asdict(args),
        "env_config": env_cfg.to_dict(),
        "impl_runtime": describe_warp_runtime() if impl == "warp" else "jax",
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_payload, fp, indent=2, ensure_ascii=False)

    env = UR5ReachMjxEnv(config=env_cfg)
    eval_env = UR5ReachMjxEnv(config=env_cfg)
    wrap_env_fn = functools.partial(
        wrapper.wrap_for_brax_training,
        full_reset=True,  # reset 时重新采样目标点。
    )

    print(f"task=ur5_reach_playground robot={args.robot} impl={impl}")
    print(f"xml={env.xml_path}")
    print(f"obs_dim={env.observation_size} action_dim={env.action_size}")
    print(f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs} episode_length={args.episode_length}")
    print(f"run_dir={run_dir}")
    if impl == "warp":
        print(f"warp={describe_warp_runtime()}")
    if args.dry_run:  # dry-run 只验证“任务和训练参数能不能正确拼起来”。
        return 0

    times = [time.monotonic()]

    def progress(step: int, metrics) -> None:
        times.append(time.monotonic())
        if "eval/episode_reward" in metrics:
            print(f"{step}: eval_reward={float(metrics['eval/episode_reward']):.3f}")
        elif "episode/sum_reward" in metrics:
            print(f"{step}: train_reward={float(metrics['episode/sum_reward']):.3f}")

    train_fn = functools.partial(  # 训练入口直接调用 Brax PPO。
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
        save_checkpoint_path=ckpt_dir,  # checkpoint 仍按 Playground/Brax 的格式落盘。
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
    )
    del make_inference_fn
    brax_model.save_params(str(run_dir / "final_policy.msgpack"), params)  # 额外导出最终参数文件。

    print("Done training.")
    if len(times) > 1:
        print(f"jit_compile_s={times[1] - times[0]:.2f}")
        print(f"total_train_s={times[-1] - times[0]:.2f}")
    return 0


def main() -> None:
    raise SystemExit(_run_train(_parse_args()))


if __name__ == "__main__":
    main()
