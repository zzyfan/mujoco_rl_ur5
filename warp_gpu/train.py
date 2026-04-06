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
from tqdm.auto import tqdm

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from warp_gpu.env import UR5ReachWarpEnv, default_config
    from warp_gpu.runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable
else:
    from .env import UR5ReachWarpEnv, default_config
    from .runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


def _metric_to_float(value) -> float | None:
    """把标量张量或普通数字转换成 Python float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_logged_metrics(metrics: dict) -> dict[str, float]:
    """从训练器回调字典里挑出最关键的可读指标。"""
    preferred_keys = (
        "eval/episode_reward",
        "episode/sum_reward",
        "eval/episode_length",
        "eval/avg_episode_length",
        "episode/length",
        "eval/distance",
        "episode/distance",
        "eval/success",
        "episode/success",
        "eval/collision",
        "episode/collision",
        "eval/runaway",
        "episode/runaway",
        "eval/timeout",
        "episode/timeout",
    )
    logged: dict[str, float] = {}
    for key in preferred_keys:
        if key not in metrics:
            continue
        value = _metric_to_float(metrics[key])
        if value is not None:
            logged[key] = value
    return logged


def _rate_count_summary(rate: float, total: int) -> str:
    """把 0~1 比例折成 `命中次数/窗口大小` 形式，方便和 classic 日志对齐。"""
    total = max(int(total), 1)
    hits = int(round(float(rate) * total))
    hits = max(0, min(hits, total))
    return f"{hits}/{total}"


@dataclass
class TrainArgs:
    algo: str = "ppo"  # Brax 训练器名称，可选 `ppo`、`sac`。
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
    sac_max_replay_size: int = 3_000_000  # SAC 回放池容量上限。
    sac_grad_updates_per_step: int = 1  # SAC 每轮采样后的梯度更新次数。
    action_repeat: int = 1  # 同一动作重复执行的物理步数。
    episode_length: int = 3000  # 单回合最大决策步数。
    frame_skip: int = 1  # 控制步长与仿真步长的倍率。
    success_threshold: float = 0.01  # 最终阶段成功判定距离阈值，单位米。
    stage1_success_threshold: float = 0.05  # 固定目标阶段成功阈值。
    stage2_success_threshold: float = 0.03  # 小范围随机阶段成功阈值。
    naconmax: int = 128  # 接触缓存容量。
    naccdmax: int = 128  # CCD 接触缓存容量。
    njmax: int = 64  # 约束缓存容量。
    target_sampling_mode: str = "full_random"  # 目标采样模式：`full_random` / `small_random` / `fixed`。
    target_range_scale: float = 0.35  # 小范围随机模式的目标范围缩放比例。
    action_target_scale: float = 0.6  # 标准化动作缩放成目标扭矩的比例。
    action_smoothing_alpha: float = 0.75  # 动作低通滤波系数。
    controller_mode: str = "torque"  # 控制模式：`torque` 或 `joint_position_delta`。
    joint_position_delta_scale: float = 0.08  # 位置增量控制时每步允许的关节目标增量。
    position_control_kp: float = 45.0  # 位置控制比例增益。
    position_control_kd: float = 3.0  # 位置控制阻尼增益。
    reward_mode: str = "dense"  # 奖励模式：`dense` 或 `sparse`。
    logdir: str = "logs/warp_gpu"  # 文本日志输出目录。
    model_dir: str = "models/warp_gpu"  # 配置、checkpoint 和最终参数目录。
    run_name: str = "ur5_reach_warp_gpu"  # 当前实验名称。
    dry_run: bool = False  # 只构建环境和训练配置，不执行训练。
    fixed_target_x: float | None = None  # 固定目标点 x 坐标；为空时按范围随机采样。
    fixed_target_y: float | None = None  # 固定目标点 y 坐标；为空时按范围随机采样。
    fixed_target_z: float | None = None  # 固定目标点 z 坐标；为空时按范围随机采样。


def _parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="Warp GPU UR5/zero reach 训练入口")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")  # 选择 Brax 训练器。
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 选择机器人模型。
    p.add_argument("--seed", type=int, default=42)  # 控制目标采样和网络初始化的随机种子。
    p.add_argument("--num-timesteps", type=int, default=5_000_000)  # 总训练步数。
    p.add_argument("--num-envs", type=int, default=256)  # 并行训练环境数。
    p.add_argument("--num-eval-envs", type=int, default=128)  # 并行评估环境数。
    p.add_argument("--num-evals", type=int, default=10)  # 训练过程中执行评估的次数。
    p.add_argument("--learning-rate", type=float, default=3e-4)  # 优化器学习率。
    p.add_argument("--entropy-cost", type=float, default=1e-4)  # PPO 熵正则系数。
    p.add_argument("--discounting", type=float, default=0.99)  # 奖励折扣因子。
    p.add_argument("--unroll-length", type=int, default=10)  # PPO rollout 片段长度。
    p.add_argument("--batch-size", type=int, default=512)  # 参数更新批大小。
    p.add_argument("--num-minibatches", type=int, default=8)  # PPO 每轮更新拆分的 mini-batch 数。
    p.add_argument("--num-updates-per-batch", type=int, default=4)  # PPO 每批样本重复更新次数。
    p.add_argument("--reward-scaling", type=float, default=1.0)  # 训练器内部使用的奖励缩放。
    p.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)  # 是否标准化观测。
    p.add_argument("--sac-tau", type=float, default=0.005)  # SAC 目标网络软更新系数。
    p.add_argument("--sac-min-replay-size", type=int, default=8192)  # SAC 回放池预热大小。
    p.add_argument("--sac-max-replay-size", type=int, default=3_000_000)  # SAC 回放池容量。
    p.add_argument("--sac-grad-updates-per-step", type=int, default=1)  # SAC 每轮采样后的梯度更新次数。
    p.add_argument("--action-repeat", type=int, default=1)  # 同一动作重复执行的物理步数。
    p.add_argument("--episode-length", type=int, default=3000)  # 单回合最大决策步数。
    p.add_argument("--frame-skip", type=int, default=1)  # 控制步长相对仿真步长的倍率。
    p.add_argument("--success-threshold", type=float, default=0.01)  # 成功判定距离阈值。
    p.add_argument("--stage1-success-threshold", type=float, default=0.05)  # 固定目标阶段成功阈值。
    p.add_argument("--stage2-success-threshold", type=float, default=0.03)  # 小范围随机阶段成功阈值。
    p.add_argument("--naconmax", type=int, default=128)  # 接触缓存容量。
    p.add_argument("--naccdmax", type=int, default=128)  # CCD 接触缓存容量。
    p.add_argument("--njmax", type=int, default=64)  # 约束缓存容量。
    p.add_argument("--target-sampling-mode", choices=["full_random", "small_random", "fixed"], default="full_random")  # 目标采样模式。
    p.add_argument("--target-range-scale", type=float, default=0.35)  # 小范围随机模式下的范围缩放。
    p.add_argument("--action-target-scale", type=float, default=0.6)  # 标准化动作缩放比例。
    p.add_argument("--action-smoothing-alpha", type=float, default=0.75)  # 动作滤波系数。
    p.add_argument("--controller-mode", choices=["torque", "joint_position_delta"], default="torque")  # 控制接口类型。
    p.add_argument("--joint-position-delta-scale", type=float, default=0.08)  # 位置增量控制每步允许的目标增量。
    p.add_argument("--position-control-kp", type=float, default=45.0)  # 位置控制比例增益。
    p.add_argument("--position-control-kd", type=float, default=3.0)  # 位置控制阻尼增益。
    p.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")  # Warp 线也支持稀疏成功奖励。
    p.add_argument("--logdir", type=str, default="logs/warp_gpu")  # 日志目录。
    p.add_argument("--model-dir", type=str, default="models/warp_gpu")  # 配置和模型输出目录。
    p.add_argument("--run-name", type=str, default="ur5_reach_warp_gpu")  # 当前实验名称。
    p.add_argument("--dry-run", action="store_true")  # 只构建环境和训练参数，不进入训练循环。
    p.add_argument("--fixed-target-x", type=float, default=None)  # 固定目标点 x 坐标。
    p.add_argument("--fixed-target-y", type=float, default=None)  # 固定目标点 y 坐标。
    p.add_argument("--fixed-target-z", type=float, default=None)  # 固定目标点 z 坐标。
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
        stage1_success_threshold=ns.stage1_success_threshold,
        stage2_success_threshold=ns.stage2_success_threshold,
        naconmax=ns.naconmax,
        naccdmax=ns.naccdmax,
        njmax=ns.njmax,
        target_sampling_mode=ns.target_sampling_mode,
        target_range_scale=ns.target_range_scale,
        action_target_scale=ns.action_target_scale,
        action_smoothing_alpha=ns.action_smoothing_alpha,
        controller_mode=ns.controller_mode,
        joint_position_delta_scale=ns.joint_position_delta_scale,
        position_control_kp=ns.position_control_kp,
        position_control_kd=ns.position_control_kd,
        reward_mode=ns.reward_mode,
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
    cfg.frame_skip = max(int(args.frame_skip), 1)  # `frame_skip` 同时决定 `ctrl_dt`。
    cfg.action_repeat = max(int(args.action_repeat), 1)  # `action_repeat` 控制同一动作连续执行多少个决策步。
    cfg.episode_length = max(int(args.episode_length), 1)  # 单回合最大决策步数。
    cfg.success_threshold = float(args.success_threshold)  # 奖励和终止逻辑共用这个成功阈值。
    cfg.stage1_success_threshold = float(args.stage1_success_threshold)  # 固定目标阶段成功阈值。
    cfg.stage2_success_threshold = float(args.stage2_success_threshold)  # 小范围随机阶段成功阈值。
    cfg.naconmax = max(int(args.naconmax), 1)  # Warp 接触缓存容量。
    cfg.naccdmax = max(int(args.naccdmax), 1)  # Warp CCD 接触缓存容量。
    cfg.njmax = max(int(args.njmax), 1)  # Warp 约束缓存容量。
    cfg.target_sampling_mode = str(args.target_sampling_mode)  # 目标采样模式。
    cfg.target_range_scale = float(args.target_range_scale)  # 小范围随机模式的范围缩放比例。
    cfg.action_target_scale = float(args.action_target_scale)  # 标准化动作缩放比例。
    cfg.action_smoothing_alpha = float(args.action_smoothing_alpha)  # 动作滤波系数。
    cfg.controller_mode = str(args.controller_mode)  # 控制接口类型。
    cfg.joint_position_delta_scale = float(args.joint_position_delta_scale)  # 位置增量控制每步的目标增量。
    cfg.position_control_kp = float(args.position_control_kp)  # 位置控制比例增益。
    cfg.position_control_kd = float(args.position_control_kd)  # 位置控制阻尼增益。
    cfg.reward_mode = str(args.reward_mode)  # `sparse` 时回到 success/fail 主导的 robotics 训练口径。
    cfg.fixed_target_x = args.fixed_target_x  # 固定目标点 x。
    cfg.fixed_target_y = args.fixed_target_y  # 固定目标点 y。
    cfg.fixed_target_z = args.fixed_target_z  # 固定目标点 z。
    # 与 classic 保持同一套奖励结构，并按算法套不同系数。
    if args.algo == "ppo":
        cfg.improvement_gain = 160.0  # PPO 保留较强逼近奖励，但避免因为奖励过强直接冲向目标发生碰撞。
        cfg.regress_gain = 80.0  # 提高退步惩罚，帮助 PPO 更快放弃碰撞型轨迹。
        cfg.direction_reward_gain = 4.0  # 方向奖励收小到中高水平，减少“只顾朝目标冲”的局部最优。
        cfg.speed_penalty_value = 0.35  # 提高超速惩罚，让 PPO 在接近目标前更愿意减速。
        cfg.action_magnitude_penalty_gain = 0.003  # 保留探索，同时抑制过大的扭矩输出。
        cfg.action_change_penalty_gain = 0.002  # 控制动作切换幅度，减少高噪声接触。
        cfg.idle_penalty_value = 0.35  # 继续惩罚发呆，但不再逼出过度探索。
    else:
        cfg.improvement_gain = 130.0  # SAC 仍以逼近为主，但减弱对“猛冲”的偏好。
        cfg.regress_gain = 75.0  # 略加强退步惩罚，帮助值函数更快区分偏离和逼近。
        cfg.direction_reward_gain = 3.0  # 降低方向奖励主导性，减少只管朝目标直冲。
        cfg.speed_penalty_value = 0.7  # 加强超速惩罚，压制接近目标时的过冲碰撞。
        cfg.action_magnitude_penalty_gain = 0.01  # 提高动作幅值惩罚，减少大扭矩碰撞。
        cfg.action_change_penalty_gain = 0.007  # 提高动作切换惩罚，减少目标附近急修正。
        cfg.idle_penalty_value = 0.12  # 保留轻度静止惩罚，避免又退回发呆策略。
    return cfg


def _run_train(args: TrainArgs) -> int:
    if not playground_importable():
        raise RuntimeError("未检测到 `mujoco_playground`，无法按 Playground 方式训练。")

    ensure_warp_runtime()

    env_cfg = _build_env_config(args)
    run_dir = Path(args.model_dir).resolve() / args.algo / args.robot / args.run_name  # 输出目录按算法、机器人和实验名分层。
    ckpt_dir = run_dir / "checkpoints"  # Brax 中间 checkpoint 写入这个目录。
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)  # 独立创建日志目录，便于追加训练记录。

    config_payload = {
        "train_args": asdict(args),  # 保存命令行参数，便于复现实验。
        "env_config": env_cfg.to_dict(),  # 保存环境配置，便于核对任务定义。
        "runtime": describe_warp_runtime(),  # 记录实际使用的 Warp 设备信息。
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config_payload, fp, indent=2, ensure_ascii=False)

    print(f"task=ur5_reach_warp_gpu algo={args.algo} robot={args.robot}")
    print(f"xml={(Path(env_cfg.model_xml).resolve())}")
    print(
        f"num_envs={args.num_envs} num_eval_envs={args.num_eval_envs} "
        f"episode_length={args.episode_length} target_sampling_mode={args.target_sampling_mode} "
        f"reward_mode={args.reward_mode} controller_mode={args.controller_mode}"
    )
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

    times = [time.monotonic()]  # 用于统计编译和训练耗时。
    total_steps = max(int(args.num_timesteps), 1)
    progress_bar = tqdm(total=total_steps, desc=f"{args.algo}:{args.robot}", unit="step")
    last_step = 0
    final_reward_label = ""
    final_reward_value: float | None = None

    def progress(step: int, metrics) -> None:
        nonlocal last_step, final_reward_label, final_reward_value
        times.append(time.monotonic())
        current_step = max(int(step), 0)
        visible_step = min(current_step, total_steps)  # 训练器可能按批次超出目标步数，进度条只显示到设定总步数。
        delta = max(visible_step - last_step, 0)
        if delta:
            progress_bar.update(delta)
        last_step = visible_step
        postfix: dict[str, str] = {}
        logged_metrics = _collect_logged_metrics(metrics)
        if "eval/episode_reward" in logged_metrics:
            final_reward_label = "最终评估回报"
            final_reward_value = logged_metrics["eval/episode_reward"]
            postfix["eval_reward"] = f"{final_reward_value:.3f}"  # 评估奖励用于观察当前策略整回合表现。
        elif "episode/sum_reward" in logged_metrics:
            final_reward_label = "最近训练回合回报"
            final_reward_value = logged_metrics["episode/sum_reward"]
            postfix["train_reward"] = f"{final_reward_value:.3f}"  # 训练奖励用于观察采样阶段的即时表现。
        if "eval/distance" in logged_metrics:
            postfix["eval_distance"] = f"{logged_metrics['eval/distance']:.4f}"
        elif "episode/distance" in logged_metrics:
            postfix["train_distance"] = f"{logged_metrics['episode/distance']:.4f}"
        if "eval/success" in logged_metrics:
            postfix["eval_success"] = f"{logged_metrics['eval/success']:.2%}"
            postfix["eval_success_count"] = _rate_count_summary(logged_metrics["eval/success"], args.num_eval_envs)
        elif "episode/success" in logged_metrics:
            postfix["train_success"] = f"{logged_metrics['episode/success']:.2%}"
            postfix["train_success_count"] = _rate_count_summary(logged_metrics["episode/success"], args.num_envs)
        if "eval/collision" in logged_metrics:
            postfix["eval_collision"] = f"{logged_metrics['eval/collision']:.2%}"
            postfix["eval_collision_count"] = _rate_count_summary(logged_metrics["eval/collision"], args.num_eval_envs)
        elif "episode/collision" in logged_metrics:
            postfix["train_collision"] = f"{logged_metrics['episode/collision']:.2%}"
            postfix["train_collision_count"] = _rate_count_summary(logged_metrics["episode/collision"], args.num_envs)
        if "eval/runaway" in logged_metrics:
            postfix["eval_runaway"] = f"{logged_metrics['eval/runaway']:.2%}"
            postfix["eval_runaway_count"] = _rate_count_summary(logged_metrics["eval/runaway"], args.num_eval_envs)
        elif "episode/runaway" in logged_metrics:
            postfix["train_runaway"] = f"{logged_metrics['episode/runaway']:.2%}"
            postfix["train_runaway_count"] = _rate_count_summary(logged_metrics["episode/runaway"], args.num_envs)
        if "eval/timeout" in logged_metrics:
            postfix["eval_timeout"] = f"{logged_metrics['eval/timeout']:.2%}"
            postfix["eval_timeout_count"] = _rate_count_summary(logged_metrics["eval/timeout"], args.num_eval_envs)
        elif "episode/timeout" in logged_metrics:
            postfix["train_timeout"] = f"{logged_metrics['episode/timeout']:.2%}"
            postfix["train_timeout_count"] = _rate_count_summary(logged_metrics["episode/timeout"], args.num_envs)
        if postfix:
            progress_bar.set_postfix(postfix)
        if logged_metrics:
            summary_parts = [f"[训练日志] step={current_step}"]
            for key, value in logged_metrics.items():
                label = key.replace("/", "_")
                if "success" in key or "collision" in key or "runaway" in key or "timeout" in key:
                    summary_parts.append(f"{label}={value:.2%}")
                    if key.startswith("eval/"):
                        summary_parts.append(
                            f"{label}_count={_rate_count_summary(value, args.num_eval_envs)}"
                        )
                    elif key.startswith("episode/"):
                        summary_parts.append(
                            f"{label}_count={_rate_count_summary(value, args.num_envs)}"
                        )
                elif "distance" in key:
                    summary_parts.append(f"{label}={value:.4f}")
                else:
                    summary_parts.append(f"{label}={value:.3f}")
            tqdm.write(" ".join(summary_parts))

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

    try:
        make_inference_fn, params, _ = train_fn(
            environment=env,
            eval_env=eval_env,
            progress_fn=progress,
        )
    finally:
        progress_bar.close()  # 无论训练正常结束还是异常退出，都主动关闭进度条。
    del make_inference_fn
    brax_model.save_params(str(run_dir / "final_policy.msgpack"), params)  # 训练结束后导出最终策略参数。

    print("Done training.")
    if len(times) > 1:
        print(f"jit_compile_s={times[1] - times[0]:.2f}")
        print(f"total_train_s={times[-1] - times[0]:.2f}")
    if final_reward_value is not None:
        print(f"{final_reward_label}: {final_reward_value:.3f}")  # 输出最后一次可用回报。
    return 0


def main() -> None:
    raise SystemExit(_run_train(_parse_args()))


if __name__ == "__main__":
    main()
