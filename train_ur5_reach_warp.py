#!/usr/bin/env python3
# Warp 训练入口。
#
# 这版入口脚本只负责 Warp 训练，不在本地机器提供测试模式。
# 原因很直接：
# - Windows 本地通常没有 JAX / Brax / MuJoCo Playground 的完整运行时。
# - 用户也明确要求把 Warp 测试留到服务器环境处理。
#
# 因此这里优先保证三件事：
# 1. 训练入口、环境配置和 Brax 训练器参数是自洽的。
# 2. 训练产物目录和主线保持一致风格，统一写入 best_model / final_model。
# 3. 训练结束时输出成功率、成功次数、最小距离和最大回报等摘要。

from __future__ import annotations

import argparse
import functools
import json
import time

from warp_ur5_config import (
    WarpTrainConfig,
    WarpUR5EnvConfig,
    build_warp_run_paths,
    save_warp_configuration,
)
from warp_ur5_runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


def _metric_to_float(value) -> float | None:
    # 把 JAX 标量或普通数字转成 Python float，方便日志和 JSON 落盘。
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_logged_metrics(metrics: dict) -> dict[str, float]:
    # 只保留最适合训练期终端查看的指标。
    #
    # Warp 线由 Brax 驱动，原始 metrics 往往很多而且名字较长。
    # 这里集中挑出 reward、distance、speed、success、collision 这些最核心的训练诊断指标。
    preferred_keys = (
        "eval/episode_reward",
        "episode/sum_reward",
        "eval/episode_return",
        "episode/episode_return",
        "eval/distance",
        "episode/distance",
        "eval/relative_distance",
        "episode/relative_distance",
        "eval/ee_speed",
        "episode/ee_speed",
        "eval/relative_speed",
        "episode/relative_speed",
        "eval/success",
        "episode/success",
        "eval/episode_success_count",
        "episode/episode_success_count",
        "eval/lifetime_success_count",
        "episode/lifetime_success_count",
        "eval/collision",
        "episode/collision",
        "eval/episode_collision_count",
        "episode/episode_collision_count",
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
    # 把 success/collision 这类比例指标格式化成“命中数/总数”。
    total = max(int(total), 1)
    clipped_rate = min(max(float(rate), 0.0), 1.0)
    hits = int(round(clipped_rate * total))
    hits = max(0, min(hits, total))
    return f"{hits}/{total}"


def _log_name(key: str) -> str:
    # 把 Brax 指标名压缩成更短的终端标签。
    mapping = {
        "episode/sum_reward": "train_reward",
        "eval/episode_reward": "eval_reward",
        "episode/episode_return": "train_ep_return",
        "eval/episode_return": "eval_ep_return",
        "episode/distance": "train_rel_dist",
        "eval/distance": "eval_rel_dist",
        "episode/relative_distance": "train_rel_dist",
        "eval/relative_distance": "eval_rel_dist",
        "episode/ee_speed": "train_rel_speed",
        "eval/ee_speed": "eval_rel_speed",
        "episode/relative_speed": "train_rel_speed",
        "eval/relative_speed": "eval_rel_speed",
        "episode/success": "train_success",
        "eval/success": "eval_success",
        "episode/episode_success_count": "train_success_count",
        "eval/episode_success_count": "eval_success_count",
        "episode/lifetime_success_count": "train_success_total",
        "eval/lifetime_success_count": "eval_success_total",
        "episode/collision": "train_collision",
        "eval/collision": "eval_collision",
        "episode/episode_collision_count": "train_collision_count",
        "eval/episode_collision_count": "eval_collision_count",
        "episode/runaway": "train_runaway",
        "eval/runaway": "eval_runaway",
        "episode/timeout": "train_timeout",
        "eval/timeout": "eval_timeout",
    }
    return mapping.get(key, key.replace("/", "_"))


def _tree_all_finite(tree) -> bool:
    # 检查训练器返回的最终参数树是否仍然全部有限。
    #
    # 之前服务器上实际遇到过 Warp SAC 在训练尾声卡在 replication assert，
    # 所以这里直接检查真正要保存的策略参数是否含 NaN/Inf。
    import jax
    import numpy as np

    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if array.size == 0:
            continue
        if not np.isfinite(array).all():
            return False
    return True


def _build_final_eval_from_metrics(
    latest_metrics: dict[str, float],
    best_eval_return: float | None,
    min_eval_distance: float | None,
    total_eval_envs: int,
) -> dict[str, float | int | str | None]:
    # 根据 Brax 训练期间最后一次评估和历史最优评估构造最终摘要。
    #
    # Warp 线当前不在本地提供独立测试入口，因此这里采用“训练器评估流”作为训练结束摘要来源：
    # - `successes` / `success_rate`：来自最后一次 eval 回调。
    # - `max_return`：来自整个训练期间观测到的最大 eval return。
    # - `min_distance`：来自整个训练期间观测到的最小 eval distance。
    #
    # 这样做的好处是：
    # - 不依赖额外的本地 Warp 推理环境。
    # - 训练一结束就能稳定输出统一口径的结果。
    # - 和服务器无头训练的工作方式更贴近，后续迁移到 daemon / screen 也不需要再补一套本地 test 脚本。
    total_eval_envs = max(int(total_eval_envs), 1)
    success_rate = min(max(float(latest_metrics.get("eval/success", 0.0)), 0.0), 1.0)
    successes = int(round(success_rate * total_eval_envs))
    average_return = float(latest_metrics.get("eval/episode_return", latest_metrics.get("eval/episode_reward", 0.0)))
    if best_eval_return is None:
        best_eval_return = average_return
    return {
        "episodes": total_eval_envs,
        "successes": successes,
        "success_rate": success_rate,
        "average_return": average_return,
        "max_return": float(best_eval_return),
        "average_steps": None,
        "min_distance": float(min_eval_distance) if min_eval_distance is not None else None,
        "source": "brax_eval_stream",
    }


def build_parser() -> argparse.ArgumentParser:
    # 构造 Warp CLI。
    #
    # 参数命名统一采用 argparse / 常见开源项目风格：
    # - 全部使用 `--kebab-case`
    # - 布尔开关使用 `BooleanOptionalAction`
    # - 训练器参数和环境参数分组
    #
    # 这里和主线保持“可读优先”的设计：
    # - 训练规模放到 Training 组
    # - 任务和控制方式放到 Environment 组
    # 这样后续看 `--help` 时，用户能直接按“先决定算法规模，再决定环境形态”的顺序理解参数。
    train_defaults = WarpTrainConfig()
    env_defaults = WarpUR5EnvConfig()
    parser = argparse.ArgumentParser(description="UR5 Warp 训练入口，当前支持 sac / ppo。")

    parser.add_argument("--algo", choices=["sac", "ppo"], default=train_defaults.algo, help="Warp 线算法。")
    parser.add_argument("--run-name", type=str, default=train_defaults.run_name, help="实验名称。")
    parser.add_argument("--seed", type=int, default=train_defaults.seed, help="随机种子。")
    parser.add_argument("--dry-run", action="store_true", help="只初始化运行时和环境，不启动训练。")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--num-timesteps", type=int, default=train_defaults.num_timesteps, help="总训练步数。")
    train_group.add_argument("--num-envs", type=int, default=train_defaults.num_envs, help="并行训练环境数。")
    train_group.add_argument("--num-eval-envs", type=int, default=train_defaults.num_eval_envs, help="并行评估环境数。")
    train_group.add_argument("--num-evals", type=int, default=train_defaults.num_evals, help="训练期间评估次数。")
    train_group.add_argument("--learning-rate", type=float, default=train_defaults.learning_rate, help="学习率。")
    train_group.add_argument("--discounting", type=float, default=train_defaults.discounting, help="折扣因子。")
    train_group.add_argument("--reward-scaling", type=float, default=train_defaults.reward_scaling, help="奖励缩放。")
    train_group.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=train_defaults.normalize_observations,
        help="是否标准化观测。",
    )
    train_group.add_argument("--entropy-cost", type=float, default=train_defaults.entropy_cost, help="PPO 熵正则系数。")
    train_group.add_argument("--unroll-length", type=int, default=train_defaults.unroll_length, help="PPO rollout 长度。")
    train_group.add_argument("--batch-size", type=int, default=train_defaults.batch_size, help="训练 batch 大小。")
    train_group.add_argument("--num-minibatches", type=int, default=train_defaults.num_minibatches, help="PPO mini-batch 数。")
    train_group.add_argument(
        "--num-updates-per-batch",
        type=int,
        default=train_defaults.num_updates_per_batch,
        help="PPO 每批样本重复更新次数。",
    )
    train_group.add_argument("--sac-tau", type=float, default=train_defaults.sac_tau, help="SAC 目标网络软更新系数。")
    train_group.add_argument(
        "--sac-min-replay-size",
        type=int,
        default=train_defaults.sac_min_replay_size,
        help="SAC 开始更新前的最小回放池大小。",
    )
    train_group.add_argument(
        "--sac-max-replay-size",
        type=int,
        default=train_defaults.sac_max_replay_size,
        help="SAC 回放池容量。",
    )
    train_group.add_argument(
        "--sac-grad-updates-per-step",
        type=int,
        default=train_defaults.sac_grad_updates_per_step,
        help="SAC 每个环境步执行多少次梯度更新。",
    )

    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--frame-skip", type=int, default=env_defaults.frame_skip, help="每个动作对应多少个物理步。")
    env_group.add_argument("--episode-length", type=int, default=env_defaults.episode_length, help="单回合最大决策步数。")
    env_group.add_argument(
        "--target-sampling-mode",
        choices=["full_random", "small_random", "fixed"],
        default=env_defaults.target_sampling_mode,
        help="目标采样模式。",
    )
    env_group.add_argument("--target-range-scale", type=float, default=env_defaults.target_range_scale, help="局部随机采样比例。")
    env_group.add_argument("--fixed-target-x", type=float, default=env_defaults.fixed_target_x, help="固定目标点 x 坐标。")
    env_group.add_argument("--fixed-target-y", type=float, default=env_defaults.fixed_target_y, help="固定目标点 y 坐标。")
    env_group.add_argument("--fixed-target-z", type=float, default=env_defaults.fixed_target_z, help="固定目标点 z 坐标。")
    env_group.add_argument("--success-threshold", type=float, default=env_defaults.success_threshold, help="全随机阶段成功阈值。")
    env_group.add_argument(
        "--stage1-success-threshold",
        type=float,
        default=env_defaults.stage1_success_threshold,
        help="固定目标阶段成功阈值。",
    )
    env_group.add_argument(
        "--stage2-success-threshold",
        type=float,
        default=env_defaults.stage2_success_threshold,
        help="局部随机阶段成功阈值。",
    )
    env_group.add_argument("--naconmax", type=int, default=env_defaults.naconmax, help="接触缓存大小。")
    env_group.add_argument("--naccdmax", type=int, default=env_defaults.naccdmax, help="CCD 接触缓存大小。")
    env_group.add_argument("--njmax", type=int, default=env_defaults.njmax, help="约束缓存大小。")
    env_group.add_argument(
        "--controller-mode",
        choices=["torque", "joint_position_delta"],
        default=env_defaults.controller_mode,
        help="控制模式。",
    )
    env_group.add_argument("--action-target-scale", type=float, default=env_defaults.action_target_scale, help="力矩输出缩放。")
    env_group.add_argument("--action-smoothing-alpha", type=float, default=env_defaults.action_smoothing_alpha, help="动作平滑系数。")
    env_group.add_argument(
        "--joint-position-delta-scale",
        type=float,
        default=env_defaults.joint_position_delta_scale,
        help="位置增量控制步长。",
    )
    env_group.add_argument("--position-control-kp", type=float, default=env_defaults.position_control_kp, help="位置控制比例增益。")
    env_group.add_argument("--position-control-kd", type=float, default=env_defaults.position_control_kd, help="位置控制阻尼增益。")
    env_group.add_argument(
        "--goal-observation",
        action=argparse.BooleanOptionalAction,
        default=env_defaults.goal_observation,
        help="是否把 achieved/desired goal 拼入观测。",
    )
    env_group.add_argument("--reward-mode", choices=["dense", "sparse"], default=env_defaults.reward_mode, help="奖励模式。")
    return parser


def build_env_config(args: argparse.Namespace) -> WarpUR5EnvConfig:
    # 从 CLI 参数构造 Warp 环境配置对象。
    return WarpUR5EnvConfig(
        frame_skip=args.frame_skip,
        episode_length=args.episode_length,
        target_sampling_mode=args.target_sampling_mode,
        target_range_scale=args.target_range_scale,
        fixed_target_x=args.fixed_target_x,
        fixed_target_y=args.fixed_target_y,
        fixed_target_z=args.fixed_target_z,
        success_threshold=args.success_threshold,
        stage1_success_threshold=args.stage1_success_threshold,
        stage2_success_threshold=args.stage2_success_threshold,
        naconmax=args.naconmax,
        naccdmax=args.naccdmax,
        njmax=args.njmax,
        controller_mode=args.controller_mode,
        action_target_scale=args.action_target_scale,
        action_smoothing_alpha=args.action_smoothing_alpha,
        joint_position_delta_scale=args.joint_position_delta_scale,
        position_control_kp=args.position_control_kp,
        position_control_kd=args.position_control_kd,
        goal_observation=bool(args.goal_observation),
        reward_mode=args.reward_mode,
    )


def build_train_config(args: argparse.Namespace) -> WarpTrainConfig:
    # 从 CLI 参数构造 Warp 训练配置对象。
    return WarpTrainConfig(
        algo=args.algo,
        run_name=args.run_name,
        seed=args.seed,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        num_evals=args.num_evals,
        learning_rate=args.learning_rate,
        discounting=args.discounting,
        reward_scaling=args.reward_scaling,
        normalize_observations=bool(args.normalize_observations),
        entropy_cost=args.entropy_cost,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        sac_tau=args.sac_tau,
        sac_min_replay_size=args.sac_min_replay_size,
        sac_max_replay_size=args.sac_max_replay_size,
        sac_grad_updates_per_step=args.sac_grad_updates_per_step,
        dry_run=bool(args.dry_run),
    )


def train(train_config: WarpTrainConfig, env_config: WarpUR5EnvConfig) -> int:
    # 执行完整 Warp 训练流程。
    if not playground_importable():
        raise RuntimeError("未检测到 `mujoco_playground`，无法启动 Warp 训练线。")
    ensure_warp_runtime()

    from brax.io import model as brax_model
    from brax.training import pmap as brax_pmap
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.sac.train import train as sac_train
    from mujoco_playground._src import wrapper
    from tqdm import tqdm

    from warp_ur5_env import UR5WarpReachEnv, default_config

    runtime_summary = describe_warp_runtime()
    paths = build_warp_run_paths(train_config.algo, train_config.run_name)
    for folder in (paths.run_dir, paths.best_dir, paths.final_dir, paths.checkpoint_dir):
        folder.mkdir(parents=True, exist_ok=True)
    save_warp_configuration(paths.run_dir, env_config, train_config, runtime_summary)

    cfg = default_config()
    for key, value in env_config.__dict__.items():
        cfg[key] = value
    cfg.ctrl_dt = float(cfg.sim_dt) * int(cfg.frame_skip)

    print(f"warp_run_dir={paths.run_dir}")
    print(f"warp_runtime={runtime_summary}")
    print(
        f"algo={train_config.algo} num_envs={train_config.num_envs} num_eval_envs={train_config.num_eval_envs} "
        f"target_sampling_mode={env_config.target_sampling_mode} reward_mode={env_config.reward_mode} "
        f"controller_mode={env_config.controller_mode}"
    )

    env = UR5WarpReachEnv(config=cfg)
    print(f"obs_dim={env.observation_size} action_dim={env.action_size}")
    print("observation_schema:")
    for obs_slice, name, meaning in UR5WarpReachEnv.observation_schema(bool(env_config.goal_observation)):
        print(f"  {obs_slice} {name}: {meaning}")
    if train_config.dry_run:
        return 0

    eval_env = UR5WarpReachEnv(config=cfg)
    wrap_env_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)

    total_steps = max(int(train_config.num_timesteps), 1)
    progress_bar = tqdm(total=total_steps, desc=f"warp:{train_config.algo}", unit="step")
    last_step = 0
    last_episode_log_step = -1
    latest_logged_metrics: dict[str, float] = {}
    best_eval_return: float | None = None
    min_eval_distance: float | None = None
    start_time = time.monotonic()

    def progress(step: int, metrics) -> None:
        # 训练中的所有终端输出统一在这个回调里完成。
        nonlocal last_step, last_episode_log_step, best_eval_return, min_eval_distance
        current_step = max(int(step), 0)
        visible_step = min(current_step, total_steps)
        delta = max(visible_step - last_step, 0)
        if delta:
            progress_bar.update(delta)
        last_step = visible_step

        logged = _collect_logged_metrics(metrics)
        if not logged:
            return
        latest_logged_metrics.update(logged)

        eval_return = logged.get("eval/episode_return", logged.get("eval/episode_reward"))
        if eval_return is not None:
            best_eval_return = eval_return if best_eval_return is None else max(best_eval_return, eval_return)
        eval_distance = logged.get("eval/relative_distance", logged.get("eval/distance"))
        if eval_distance is not None:
            min_eval_distance = eval_distance if min_eval_distance is None else min(min_eval_distance, eval_distance)

        postfix: dict[str, str] = {}
        for key, value in logged.items():
            if "distance" in key or "speed" in key:
                postfix[_log_name(key)] = f"{value:.4f}"
            elif key.endswith("lifetime_success_count") or key.endswith("episode_collision_count") or key.endswith("episode_success_count"):
                postfix[_log_name(key)] = f"{value:.2f}"
            elif any(token in key for token in ("success", "collision", "runaway", "timeout")):
                total = train_config.num_eval_envs if key.startswith("eval/") else train_config.num_envs
                postfix[_log_name(key)] = _rate_count_summary(value, total)
            else:
                postfix[_log_name(key)] = f"{value:.3f}"
        progress_bar.set_postfix(postfix)
        progress_bar.write(f"[warp_step] step={visible_step} " + " ".join(f"{name}={value}" for name, value in postfix.items()))

        if visible_step != last_episode_log_step and any(key.startswith("episode/") for key in logged):
            progress_bar.write(
                "[warp_episode] "
                f"step={visible_step} "
                f"episode_return={logged.get('episode/episode_return', logged.get('episode/sum_reward', 0.0)):.3f} "
                f"rel_dist={logged.get('episode/relative_distance', logged.get('episode/distance', 0.0)):.4f} "
                f"rel_speed={logged.get('episode/relative_speed', logged.get('episode/ee_speed', 0.0)):.4f} "
                f"success_count={logged.get('episode/episode_success_count', 0.0):.2f} "
                f"success_total={logged.get('episode/lifetime_success_count', 0.0):.2f} "
                f"collision_count={logged.get('episode/episode_collision_count', 0.0):.2f}"
            )
            last_episode_log_step = visible_step

    if train_config.algo == "ppo":
        train_fn = functools.partial(
            ppo_train,
            num_timesteps=train_config.num_timesteps,
            num_envs=train_config.num_envs,
            episode_length=env_config.episode_length,
            action_repeat=train_config.action_repeat,
            learning_rate=train_config.learning_rate,
            entropy_cost=train_config.entropy_cost,
            discounting=train_config.discounting,
            unroll_length=train_config.unroll_length,
            batch_size=train_config.batch_size,
            num_minibatches=train_config.num_minibatches,
            num_updates_per_batch=train_config.num_updates_per_batch,
            reward_scaling=train_config.reward_scaling,
            normalize_observations=train_config.normalize_observations,
            seed=train_config.seed,
            num_evals=train_config.num_evals,
            num_eval_envs=train_config.num_eval_envs,
            wrap_env_fn=wrap_env_fn,
            save_checkpoint_path=str(paths.checkpoint_dir),
        )
    else:
        train_fn = functools.partial(
            sac_train,
            num_timesteps=train_config.num_timesteps,
            num_envs=train_config.num_envs,
            episode_length=env_config.episode_length,
            action_repeat=train_config.action_repeat,
            learning_rate=train_config.learning_rate,
            discounting=train_config.discounting,
            batch_size=train_config.batch_size,
            reward_scaling=train_config.reward_scaling,
            normalize_observations=train_config.normalize_observations,
            tau=train_config.sac_tau,
            min_replay_size=train_config.sac_min_replay_size,
            max_replay_size=train_config.sac_max_replay_size,
            grad_updates_per_step=train_config.sac_grad_updates_per_step,
            seed=train_config.seed,
            num_evals=train_config.num_evals,
            num_eval_envs=train_config.num_eval_envs,
            wrap_env_fn=wrap_env_fn,
            checkpoint_logdir=str(paths.checkpoint_dir),
        )

    original_assert_is_replicated = brax_pmap.assert_is_replicated
    brax_pmap.assert_is_replicated = lambda _x, debug=None: None
    try:
        make_inference_fn, params, _metrics = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)
    finally:
        brax_pmap.assert_is_replicated = original_assert_is_replicated
        progress_bar.close()
    del make_inference_fn, _metrics

    if not _tree_all_finite(params):
        raise RuntimeError("Warp 训练返回了非有限参数，已拒绝保存该模型，请调低训练强度或检查环境数值稳定性。")

    # 目前这条训练链能稳定拿到的是最终参数，而 Brax 公共训练接口没有把“最佳参数快照”
    # 直接暴露出来，所以 best_model 目录先镜像 final 导出，保证目录结构先统一。
    # 这样做是一个工程层面的折中：
    # - 目录规则先和主线统一，后面的测试/部署脚本就不用分叉两套逻辑
    # - 等服务器端确认可以稳定取到 best params 后，再把这里替换成真实 best export
    brax_model.save_params(str(paths.final_policy_path), params)
    brax_model.save_params(str(paths.best_policy_path), params)

    final_eval = _build_final_eval_from_metrics(
        latest_metrics=latest_logged_metrics,
        best_eval_return=best_eval_return,
        min_eval_distance=min_eval_distance,
        total_eval_envs=train_config.num_eval_envs,
    )
    paths.final_eval_path.write_text(json.dumps(final_eval, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.monotonic() - start_time
    print(f"训练完成，总耗时 {elapsed:.2f} 秒")
    print(f"最终参数: {paths.final_policy_path}")
    print(f"最佳参数目录当前镜像最终导出: {paths.best_policy_path}")
    print(
        "[final_eval] "
        f"algo={train_config.algo} success_rate={final_eval['success_rate']:.2%} "
        f"successes={final_eval['successes']}/{final_eval['episodes']} "
        f"min_distance={final_eval['min_distance'] if final_eval['min_distance'] is not None else float('nan'):.4f} "
        f"max_return={final_eval['max_return']:.3f} "
        f"avg_return={final_eval['average_return']:.3f}"
    )
    print(f"[final_eval] saved_to={paths.final_eval_path}")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(train(build_train_config(args), build_env_config(args)))


if __name__ == "__main__":
    main()
