#!/usr/bin/env python3
"""本地 UR5/zero Playground 训练入口。"""

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
    from mjx.backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from mjx.reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name
else:
    from .backend import describe_warp_runtime, ensure_warp_runtime, playground_importable
    from .reach_env import UR5ReachMjxEnv, default_config, normalize_impl_name


@dataclass
class TrainArgs:
    algo: str = "ppo"  # 训练算法选择；本文件当前接入 Brax 的 PPO 和 SAC。
    robot: str = "ur5_cxy"  # 机器人模型选择，会影响 XML、home pose 和目标采样范围。
    impl: str = "warp"  # 物理后端选择；`mjx` 会映射到 MuJoCo 的 `jax` 实现名。
    seed: int = 42
    num_timesteps: int = 5_000_000  # 总训练步数。
    num_envs: int = 256  # 并行训练环境数量；增大后通常能提高采样吞吐。
    num_eval_envs: int = 128  # 评估使用的并行环境数量。
    num_evals: int = 10
    learning_rate: float = 3e-4  # PPO 和 SAC 都常从 3e-4 起步。
    entropy_cost: float = 1e-4  # PPO 熵正则系数，用来控制策略探索强度。
    discounting: float = 0.99
    unroll_length: int = 10  # 单次 rollout 片段长度；太短会让优势估计更吵。
    batch_size: int = 512  # 参数更新使用的 batch 大小，不等于环境并行数。
    num_minibatches: int = 8  # PPO 每轮更新拆成多少个 mini-batch。
    num_updates_per_batch: int = 4  # PPO 每批采样重复更新的次数。
    reward_scaling: float = 1.0  # Brax 训练器内部的奖励缩放系数。
    normalize_observations: bool = True  # 是否启用观测归一化。
    sac_tau: float = 0.005  # SAC 软更新系数，越小越稳。
    sac_min_replay_size: int = 8192  # replay 至少累积到这个大小再开始更新。
    sac_max_replay_size: int = 262_144  # replay 上限，太大会额外占内存。
    sac_grad_updates_per_step: int = 1  # 每轮环境采样后做多少次梯度更新。
    action_repeat: int = 1  # 同一动作重复执行的步数。
    episode_length: int = 3000  # 单回合步数上限。
    frame_skip: int = 1  # MuJoCo 控制频率缩放参数，会同步影响 ctrl_dt。
    success_threshold: float = 0.01  # 末端与目标点的成功距离阈值，单位米。
    logdir: str = "logs/mjx"  # 训练日志目录。
    model_dir: str = "models/mjx"  # 配置、checkpoint 和最终参数输出目录。
    run_name: str = "ur5_reach_playground"  # 当前实验名称。
    dry_run: bool = False  # 只构建环境与参数，不进入正式训练。
    fixed_target_x: float | None = None
    fixed_target_y: float | None = None
    fixed_target_z: float | None = None


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="UR5/zero Playground 训练入口")
    p.add_argument("--algo", choices=["ppo", "sac", "td3"], default="ppo")
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
    p.add_argument("--sac-tau", type=float, default=0.005)
    p.add_argument("--sac-min-replay-size", type=int, default=8192)
    p.add_argument("--sac-max-replay-size", type=int, default=262_144)
    p.add_argument("--sac-grad-updates-per-step", type=int, default=1)
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
        algo=ns.algo,
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
        sac_tau=ns.sac_tau,
        sac_min_replay_size=ns.sac_min_replay_size,
        sac_max_replay_size=ns.sac_max_replay_size,
        sac_grad_updates_per_step=ns.sac_grad_updates_per_step,
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
    cfg.impl = normalize_impl_name(args.impl)  # 把命令行里的 `mjx` 转成 MuJoCo 使用的 `jax` 名称。
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
    if args.algo == "td3":
        raise RuntimeError("当前本地 Brax 版本未提供 TD3 训练入口，MJX 训练脚本目前只支持 `ppo` 和 `sac`。")

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

    print(f"task=ur5_reach_playground algo={args.algo} robot={args.robot} impl={impl}")
    print(f"xml={env.xml_path}")
    print(f"obs_dim={env.observation_size} action_dim={env.action_size}")
    print(f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs} episode_length={args.episode_length}")
    print(f"run_dir={run_dir}")
    if impl == "warp":
        print(f"warp={describe_warp_runtime()}")
    if args.dry_run:  # dry-run 用来检查任务、后端和训练参数能否正常初始化。
        return 0

    times = [time.monotonic()]

    def progress(step: int, metrics) -> None:
        times.append(time.monotonic())
        if "eval/episode_reward" in metrics:
            print(f"{step}: eval_reward={float(metrics['eval/episode_reward']):.3f}")
        elif "episode/sum_reward" in metrics:
            print(f"{step}: train_reward={float(metrics['episode/sum_reward']):.3f}")

    if args.algo == "ppo":
        train_fn = functools.partial(  # PPO 通过并行 rollout 收集轨迹，再做多轮策略更新。
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
            save_checkpoint_path=str(ckpt_dir),  # checkpoint 使用 Brax 默认格式保存。
        )
    else:
        train_fn = functools.partial(  # SAC 依赖 replay buffer，因此需要显式设置回放池与软更新参数。
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
