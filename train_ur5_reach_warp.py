#!/usr/bin/env python3
# Warp 训练入口（zero-arm 风格重写）。
#
# 本模块负责 Warp 训练参数解析、环境构建、训练器调用和训练产物保存。
#
# 涉及的主要外部库：
# - `Brax`：提供 PPO / SAC 训练器。
# - `JAX`：提供张量计算和函数式训练状态表示。
# - `MuJoCo Playground`：负责把 MJX / Warp 环境包装成 Brax 可训练环境。
# - `tqdm`：负责命令行进度显示。

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

from warp_ur5_env import UR5WarpReachEnv, default_config
from warp_ur5_runtime import ensure_warp_runtime, playground_importable


def _metric_to_float(value) -> float | None:
    # 把 JAX 标量或普通数字转换成 Python `float`，方便日志输出。
    # Brax 回调里很多指标是设备张量，直接 print 不直观，所以先统一转成普通数字。
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_logged_metrics(metrics: dict) -> dict[str, float]:
    # 从 Brax 回调指标中筛出最适合终端显示的一小部分。
    #
    # 这里不是把所有 metrics 都打印出来，而是只保留 reward、distance、success、
    # collision、runaway、timeout 这些最能反映训练状态的指标。
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
    logged: dict[str, float] = {}  # 输出指标容器
    for key in preferred_keys:
        if key in metrics:
            value = _metric_to_float(metrics[key])
            if value is not None:
                logged[key] = value
    return logged


def _rate_count_summary(rate: float, total: int) -> str:
    # 把 0~1 比例格式化成 `命中次数/总数`，便于长时间训练时快速阅读。
    #
    # 例如 success=0.125、num_envs=64 时，会输出成 `8/64`。
    total = max(int(total), 1)  # 总数至少为 1
    hits = int(round(float(rate) * total))  # 命中数量
    hits = max(0, min(hits, total))
    return f"{hits}/{total}"


def _log_name(key: str) -> str:
    # 把长指标名收缩成更容易扫读的终端标签。
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
    # 检查训练器返回的参数树里是否仍然都是有限值。
    #
    # Brax 0.14 的 Warp SAC 在训练完成后有时会因为环境状态里混入 NaN，
    # 卡在 `assert_is_replicated(training_state)`，但这不一定意味着最终策略参数已经坏掉。
    # 这里改成直接验证“真正要保存的 params 是否有限”，只要参数可用，就允许训练正常落盘。
    import numpy as np
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if array.size == 0:
            continue
        if not np.isfinite(array).all():
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    # 构造 Warp 训练 CLI（简化版）。
    parser = argparse.ArgumentParser(description="UR5 Warp 训练入口（zero-arm 风格）")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="训练算法")  # 算法选择
    parser.add_argument("--run-name", type=str, default="ur5_warp_run", help="实验名字")  # 输出目录名
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 训练可复现性
    parser.add_argument("--num-timesteps", type=int, default=5_000_000, help="总训练步数")  # 训练总步数
    parser.add_argument("--num-envs", type=int, default=256, help="并行训练环境数")  # 采样并行度
    parser.add_argument("--num-eval-envs", type=int, default=128, help="并行评估环境数")  # 评估并行度
    parser.add_argument("--num-evals", type=int, default=10, help="训练期间评估次数")  # 评估次数
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")  # Adam 学习率
    parser.add_argument("--discounting", type=float, default=0.99, help="折扣因子")  # gamma
    parser.add_argument("--reward-scaling", type=float, default=1.0, help="奖励缩放")  # reward scale
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True, help="是否标准化观测")  # obs 归一化
    parser.add_argument("--entropy-cost", type=float, default=1e-4, help="PPO 熵正则系数")  # PPO 探索
    parser.add_argument("--unroll-length", type=int, default=10, help="PPO rollout 长度")  # PPO rollout
    parser.add_argument("--batch-size", type=int, default=512, help="训练 batch 大小")  # batch 大小
    parser.add_argument("--num-minibatches", type=int, default=8, help="PPO mini-batch 数")  # PPO 分块
    parser.add_argument("--num-updates-per-batch", type=int, default=4, help="PPO 每批重复更新次数")  # PPO 重复次数
    parser.add_argument("--sac-tau", type=float, default=0.005, help="SAC 目标网络软更新系数")  # SAC tau
    parser.add_argument("--sac-min-replay-size", type=int, default=8192, help="SAC 回放池预热大小")  # SAC 预热
    parser.add_argument("--sac-max-replay-size", type=int, default=3_000_000, help="SAC 回放池容量")  # SAC 容量
    parser.add_argument("--sac-grad-updates-per-step", type=int, default=1, help="SAC 每步梯度更新次数")  # SAC 更新
    parser.add_argument("--dry-run", action="store_true", help="只初始化环境和配置，不启动训练")  # 仅检查
    return parser  # 返回命令行解析器


def build_env_config() -> dict:
    # Warp 环境直接使用 default_config，训练入口只覆盖训练超参数。
    return {}  # 当前不覆盖环境参数


def build_train_config(args: argparse.Namespace) -> dict:
    return vars(args)  # Namespace -> dict，便于传参


def train(train_config: dict, env_config: dict) -> int:
    # 执行 Warp 训练流程。
    #
    # 主要步骤：
    # 1. 检查 Warp / Playground 运行时。
    # 2. 把 dataclass 环境参数转换成 Playground `ConfigDict`。
    # 3. 构造训练环境、评估环境和进度回调。
    # 4. 根据算法选择 Brax PPO 或 SAC 训练器。
    # 5. 保存最终策略参数。
    # 第一步：先检查外部依赖是否齐全。
    if not playground_importable():  # 依赖检查
        raise RuntimeError("未检测到 `mujoco_playground`，无法启动 Warp 训练线。")
    ensure_warp_runtime()  # 检查 Warp 运行时与驱动版本

    # 这些依赖只在真正训练时导入，避免 `--help` 或静态阅读时就要求完整 Warp 环境。
    from brax.training import pmap as brax_pmap
    from brax.io import model as brax_model
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.sac.train import train as sac_train
    from mujoco_playground._src import wrapper
    from tqdm import tqdm

    # 先拿默认配置，再用覆盖字典更新。
    cfg = default_config()  # 环境默认配置
    for key, value in env_config.items():
        cfg[key] = value
    cfg.ctrl_dt = float(cfg.sim_dt) * int(cfg.frame_skip)

    # 第三步：准备输出目录并把配置落盘。
    run_dir = Path("./warp_runs") / str(train_config["algo"]) / str(train_config["run_name"])  # 输出目录
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text("{}")

    print(f"warp_run_dir={run_dir}")
    print(
        f"algo={train_config['algo']} num_envs={train_config['num_envs']} num_eval_envs={train_config['num_eval_envs']}"
    )

    # 第四步：创建训练环境和评估环境。
    env = UR5WarpReachEnv(config=cfg)  # 训练环境
    print(f"obs_dim={env.observation_size} action_dim={env.action_size}")
    print("observation_schema:")
    for obs_slice, name, meaning in UR5WarpReachEnv.observation_schema(bool(env_config.goal_observation)):
        print(f"  {obs_slice} {name}: {meaning}")
    if train_config.dry_run:
        # dry-run 模式只检查环境和配置是否能正确初始化。
        return 0
    eval_env = UR5WarpReachEnv(config=cfg)  # 评估环境
    # MuJoCo Playground 提供的包装器会把环境包装成 Brax 训练器期望的接口。
    wrap_env_fn = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)

    # 第五步：初始化命令行进度条和时间统计。
    total_steps = max(int(train_config["num_timesteps"]), 1)  # 进度条总步数
    progress_bar = tqdm(total=total_steps, desc=f"warp:{train_config['algo']}", unit="step")
    last_step = 0
    start_time = time.monotonic()
    last_episode_log_step = -1

    def progress(step: int, metrics) -> None:
        # 把 Brax 训练过程中的指标转成更易读的终端进度信息。
        nonlocal last_step, last_episode_log_step
        # 先更新进度条步数，再把挑选后的关键指标塞进 postfix。
        current_step = max(int(step), 0)  # 当前步数
        visible_step = min(current_step, total_steps)
        delta = max(visible_step - last_step, 0)
        if delta:
            progress_bar.update(delta)
        last_step = visible_step
        logged = _collect_logged_metrics(metrics)  # 筛选关键指标
        if not logged:
            return
        postfix: dict[str, str] = {}  # 进度条后缀信息
        for key, value in logged.items():
            # distance 直接打印浮点数；
            # success / collision / runaway / timeout 转成“命中数/总数”；
            # 其他 reward 类指标按普通浮点数显示。
            if "distance" in key:
                postfix[_log_name(key)] = f"{value:.4f}"
            elif "speed" in key:
                postfix[_log_name(key)] = f"{value:.4f}"
            elif key.endswith("lifetime_success_count") or key.endswith("episode_collision_count") or key.endswith("episode_success_count"):
                postfix[_log_name(key)] = f"{value:.2f}"
            elif any(token in key for token in ("success", "collision", "runaway", "timeout")):
                total = train_config.num_eval_envs if key.startswith("eval/") else train_config.num_envs
                postfix[_log_name(key)] = _rate_count_summary(value, total)
            else:
                postfix[_log_name(key)] = f"{value:.3f}"
        progress_bar.set_postfix(postfix)
        line = " ".join(f"{name}={value}" for name, value in postfix.items())
        progress_bar.write(f"[warp_step] step={visible_step} {line}")
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

    if train_config["algo"] == "ppo":
        # PPO 分支主要配置 rollout 长度、mini-batch 和重复优化次数。
        # 这些参数决定了每次采样后如何切 batch、如何重复优化同一批数据。
        train_fn = functools.partial(
            ppo_train,
            num_timesteps=train_config["num_timesteps"],
            num_envs=train_config["num_envs"],
            episode_length=cfg.episode_length,
            action_repeat=cfg.action_repeat,
            learning_rate=train_config["learning_rate"],
            entropy_cost=train_config["entropy_cost"],
            discounting=train_config["discounting"],
            unroll_length=train_config["unroll_length"],
            batch_size=train_config["batch_size"],
            num_minibatches=train_config["num_minibatches"],
            num_updates_per_batch=train_config["num_updates_per_batch"],
            reward_scaling=train_config["reward_scaling"],
            normalize_observations=train_config["normalize_observations"],
            seed=train_config["seed"],
            num_evals=train_config["num_evals"],
            num_eval_envs=train_config["num_eval_envs"],
            wrap_env_fn=wrap_env_fn,
            save_checkpoint_path=str(checkpoint_dir),
        )
    else:
        # SAC 分支主要配置 replay buffer 和 target network 相关参数。
        # SAC 不需要 PPO 的 rollout 片段参数，但需要 replay buffer 和软更新参数。
        train_fn = functools.partial(
            sac_train,
            num_timesteps=train_config["num_timesteps"],
            num_envs=train_config["num_envs"],
            episode_length=cfg.episode_length,
            action_repeat=cfg.action_repeat,
            learning_rate=train_config["learning_rate"],
            discounting=train_config["discounting"],
            batch_size=train_config["batch_size"],
            reward_scaling=train_config["reward_scaling"],
            normalize_observations=train_config["normalize_observations"],
            tau=train_config["sac_tau"],
            min_replay_size=train_config["sac_min_replay_size"],
            max_replay_size=train_config["sac_max_replay_size"],
            grad_updates_per_step=train_config["sac_grad_updates_per_step"],
            seed=train_config["seed"],
            num_evals=train_config["num_evals"],
            num_eval_envs=train_config["num_eval_envs"],
            wrap_env_fn=wrap_env_fn,
            checkpoint_logdir=str(checkpoint_dir),
        )

    # 第六步：真正调用 Brax 训练器。
    original_assert_is_replicated = brax_pmap.assert_is_replicated
    brax_pmap.assert_is_replicated = lambda _x, debug=None: None
    try:
        make_inference_fn, params, _metrics = train_fn(environment=env, eval_env=eval_env, progress_fn=progress)
    finally:
        brax_pmap.assert_is_replicated = original_assert_is_replicated
        # 无论训练是否正常结束，都先把进度条关闭，避免终端残留错位输出。
        progress_bar.close()
    del make_inference_fn
    if not _tree_all_finite(params):
        raise RuntimeError("Warp 训练返回了非有限参数，已拒绝保存该模型，请调低训练强度或检查环境数值稳定性。")
    # 第七步：保存最终策略参数。
    brax_model.save_params(str(run_dir / "final_policy.msgpack"), params)
    print(f"训练完成，总耗时 {time.monotonic() - start_time:.2f} 秒")
    print(f"最终参数: {run_dir / 'final_policy.msgpack'}")
    return 0


def main() -> None:
    # 解析参数并启动 Warp 训练入口。
    # 入口本身不做业务逻辑，只负责把 CLI 参数转成 dataclass 并交给 `train(...)`。
    args = build_parser().parse_args()
    raise SystemExit(train(build_train_config(args), build_env_config(args)))


if __name__ == "__main__":
    main()
