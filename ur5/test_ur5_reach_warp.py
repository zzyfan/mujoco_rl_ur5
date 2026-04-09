#!/usr/bin/env python3
# Warp 推理与测试入口。
#
# 本模块负责推理参数解析、模型参数加载、推理函数重建和测试可视化。
#
# 涉及的主要外部库：
# - `Brax`：负责根据保存的参数重建推理策略。
# - `JAX`：负责随机数、动作张量和环境状态更新。
# - `MuJoCo` / `mujoco.mjx`：负责把 Warp 环境状态同步到 human viewer。
# - `Orbax` / `Flax`：负责恢复 checkpoint 参数树和标准化状态。

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

from warp_ur5_config import WarpUR5EnvConfig, build_warp_run_dir
from warp_ur5_runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


def build_parser() -> argparse.ArgumentParser:
    # 构造 Warp 推理 CLI。
    #
    # 这里既支持读取最终导出的 `final_policy.msgpack`，
    # 也支持读取训练过程中的中间 checkpoint。
    defaults = WarpUR5EnvConfig()
    # 和训练入口一样，先从 dataclass 取默认值，保证 CLI 默认参数和代码默认参数一致。
    parser = argparse.ArgumentParser(description="UR5 Warp 纯 GPU 推理测试入口。")
    # 这组参数主要决定“加载哪份模型、以什么方式运行推理”。
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="推理时使用的训练算法。")
    parser.add_argument("--run-name", type=str, default="ur5_warp_gpu", help="实验名字。")
    parser.add_argument("--artifact", choices=["final", "latest-checkpoint", "checkpoint"], default="final", help="加载最终参数还是中间 checkpoint。")
    parser.add_argument("--params-path", type=str, default="", help="直接指定 final_policy.msgpack 路径。")
    parser.add_argument("--checkpoint-path", type=str, default="", help="直接指定 checkpoint 目录。")
    parser.add_argument("--checkpoint-step", type=str, default="", help="从 checkpoints 目录中按步数选择 checkpoint。")
    parser.add_argument("--episodes", type=int, default=3, help="测试回合数。")
    parser.add_argument("--max-steps", type=int, default=defaults.episode_length, help="每回合最大步数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="是否使用确定性动作。")
    # 这组参数允许在测试阶段临时覆盖环境配置，方便做控制模式和目标点对照实验。
    parser.add_argument("--fixed-target-x", type=float, default=None, help="固定目标点 x。")
    parser.add_argument("--fixed-target-y", type=float, default=None, help="固定目标点 y。")
    parser.add_argument("--fixed-target-z", type=float, default=None, help="固定目标点 z。")
    parser.add_argument("--controller-mode", choices=["torque", "joint_position_delta"], default=defaults.controller_mode, help="控制模式。")
    parser.add_argument("--joint-position-delta-scale", type=float, default=defaults.joint_position_delta_scale, help="位置增量控制步长。")
    parser.add_argument("--position-control-kp", type=float, default=defaults.position_control_kp, help="位置控制比例增益。")
    parser.add_argument("--position-control-kd", type=float, default=defaults.position_control_kd, help="位置控制阻尼增益。")
    parser.add_argument("--goal-observation", action=argparse.BooleanOptionalAction, default=defaults.goal_observation, help="是否显式拼接 goal 观测。")
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default=defaults.reward_mode, help="奖励模式。")
    parser.add_argument("--render", action="store_true", help="是否打开 human 窗口。")
    parser.add_argument("--print-step-reward", action="store_true", help="是否打印每一步的奖励和距离。")
    return parser


def _latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    # 找到数字编号最大的 checkpoint 目录。
    # Warp 训练器会按步数建目录，所以这里只要选数值最大的目录即可。
    candidates = [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"未找到 checkpoint 目录: {checkpoint_root}")
    return max(candidates, key=lambda path: int(path.name))


def _resolve_run_dir(algo: str, run_name: str) -> Path:
    # 解析 Warp 实验目录，并兼容旧版目录结构。
    #
    # 优先级：
    # 1. 新目录结构 `runs/warp/{algo}/{run_name}`
    # 2. 旧目录结构 `runs_warp/{algo}/{run_name}`
    run_dir = build_warp_run_dir(algo, run_name)
    if run_dir.exists():
        return run_dir
    legacy_run_dir = Path(__file__).resolve().parent / "runs_warp" / algo / run_name
    if legacy_run_dir.exists():
        return legacy_run_dir
    return run_dir


def _select_checkpoint_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    # 根据 CLI 参数选择要加载的 checkpoint 目录。
    # 这里支持三种入口：
    # 1. 直接给目录
    # 2. 取最新 checkpoint
    # 3. 按步数取某个 checkpoint
    if args.checkpoint_path:
        # 如果用户直接给了 checkpoint 目录，就优先使用这条路径。
        checkpoint_dir = Path(args.checkpoint_path).resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"未找到指定 checkpoint: {checkpoint_dir}")
        return checkpoint_dir
    checkpoint_root = run_dir / "checkpoints"
    if args.artifact == "latest-checkpoint" or not args.checkpoint_step:
        # latest-checkpoint 模式下，自动挑数值编号最大的目录。
        return _latest_checkpoint_dir(checkpoint_root)
    # 否则按用户给定的 checkpoint 步数拼目录名。
    selected = checkpoint_root / args.checkpoint_step
    if not selected.exists():
        raise FileNotFoundError(f"未找到指定 checkpoint: {selected}")
    return selected.resolve()


def _normalize_fn(enabled: bool) -> Callable:
    # 返回观测预处理函数。
    #
    # Brax 保存的策略可能依赖训练时的观测标准化，因此推理时需要根据保存配置
    # 决定是否继续使用 `running_statistics.normalize`。
    from brax.training.acme import running_statistics

    return running_statistics.normalize if enabled else (lambda x, y: x)


def _resolve_activation(name: str | None):
    # 把保存下来的激活函数名字映射回 JAX 可调用对象。
    from jax import nn as jnn

    mapping = {"relu": jnn.relu, "silu": jnn.silu, "swish": jnn.swish, None: None}
    if name not in mapping:
        raise ValueError(f"不支持的激活函数配置: {name}")
    return mapping[name]


def _resolve_initializer(name: str | None):
    # 把保存下来的初始化器名字映射回 JAX 初始化函数。
    from jax import nn as jnn

    mapping = {"lecun_uniform": jnn.initializers.lecun_uniform, None: None}
    if name not in mapping:
        raise ValueError(f"不支持的初始化器配置: {name}")
    return mapping[name]


def _load_checkpoint_params(checkpoint_dir: Path):
    # 用 Orbax 读取 checkpoint 参数树。
    # 不同版本的 Orbax 恢复结果可能是 list 或 tuple，这里统一归一。
    from orbax import checkpoint as ocp

    restored = ocp.PyTreeCheckpointer().restore(str(checkpoint_dir.resolve()))
    if isinstance(restored, list):
        return tuple(restored)
    return restored


def _restore_running_statistics(observation_size, state_dict: dict):
    # 从普通字典恢复 Brax 的 running-statistics 状态。
    # 先创建一份模板，再把保存的 state_dict 填回模板结构。
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from flax import serialization as flax_serialization

    template = running_statistics.init_state(jp.zeros(observation_size, dtype=jp.float32))
    return flax_serialization.from_state_dict(template, state_dict)


def _load_checkpoint_network_config(args: argparse.Namespace, checkpoint_dir: Path) -> dict:
    # 读取 checkpoint 对应的网络结构配置。
    config_path = checkpoint_dir / f"{args.algo}_network_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint 网络配置: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_final_policy(args: argparse.Namespace, run_dir: Path, env):
    # 加载最终策略参数并重建推理函数。
    #
    # 最终导出的 `final_policy.msgpack` 只保存了参数，不保存完整 Python 模型对象，
    # 所以这里需要：
    # 1. 先按算法类型重建对应网络结构。
    # 2. 再把参数加载进去。
    # 3. 最后生成真正可调用的 policy 函数。
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.sac import networks as sac_networks

    # 先解析最终参数文件路径。
    params_path = Path(args.params_path).resolve() if args.params_path else (run_dir / "final_policy.msgpack")
    if not params_path.exists():
        raise FileNotFoundError(f"未找到最终策略参数: {params_path}")

    # 推理时是否还要做观测标准化，取决于训练时的保存配置。
    config_path = run_dir / "config.json"
    normalize_observations = True
    if config_path.exists():
        train_args = json.loads(config_path.read_text(encoding="utf-8")).get("train_config", {})
        normalize_observations = bool(train_args.get("normalize_observations", True))

    preprocess = _normalize_fn(normalize_observations)
    # SAC 和 PPO 的网络工厂不同，因此要分支重建。
    if args.algo == "sac":
        networks = sac_networks.make_sac_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=preprocess,
        )
        make_policy = sac_networks.make_inference_fn(networks)
    else:
        networks = ppo_networks.make_ppo_networks(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=preprocess,
        )
        make_policy = ppo_networks.make_inference_fn(networks)

    # 最后把参数载入并构造真正可执行的 policy 函数。
    params = brax_model.load_params(str(params_path))
    policy = make_policy(params, deterministic=bool(args.deterministic))
    return policy, params_path


def _load_checkpoint_policy(args: argparse.Namespace, run_dir: Path):
    # 加载一个中间 checkpoint 并重建策略函数。
    #
    # 和最终参数不同，checkpoint 往往同时包含：
    # - 标准化统计量
    # - 网络参数树
    # - 额外训练状态
    # 因此恢复过程会更复杂一些。
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.sac import networks as sac_networks

    # 先选出要读取的 checkpoint 目录，再解析网络结构和参数树。
    checkpoint_dir = _select_checkpoint_dir(args, run_dir)
    network_config = _load_checkpoint_network_config(args, checkpoint_dir)
    # 先从保存配置里恢复“是否做观测标准化”这件事，再决定推理前处理函数。
    preprocess = _normalize_fn(bool(network_config.get("normalize_observations", True)))
    params = _load_checkpoint_params(checkpoint_dir)

    if args.algo == "sac":
        # SAC checkpoint 通常保存 `(running_stats, params)` 这样的结构。
        observation_size = int(network_config["observation_size"])
        kwargs = dict(network_config.get("network_factory_kwargs", {}))
        kwargs["activation"] = _resolve_activation(kwargs.get("activation"))
        kwargs["policy_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("policy_network_kernel_init_fn"))
        kwargs["q_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("q_network_kernel_init_fn"))
        params = (_restore_running_statistics(observation_size, params[0]), params[1])
        networks = sac_networks.make_sac_networks(
            observation_size=observation_size,
            action_size=int(network_config["action_size"]),
            preprocess_observations_fn=preprocess,
            **kwargs,
        )
        make_policy = sac_networks.make_inference_fn(networks)
    else:
        # PPO checkpoint 里除了 running-stats 和 policy 参数，还可能带有 value 网络相关状态。
        kwargs = dict(network_config.get("network_factory_kwargs", {}))
        kwargs["activation"] = _resolve_activation(kwargs.get("activation"))
        kwargs["policy_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("policy_network_kernel_init_fn"))
        kwargs["value_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("value_network_kernel_init_fn"))
        kwargs["mean_kernel_init_fn"] = _resolve_initializer(kwargs.get("mean_kernel_init_fn"))
        observation_size = network_config.get("observation_size")
        if isinstance(observation_size, dict):
            observation_size = tuple(observation_size.get("shape", ()))
        params = (_restore_running_statistics(observation_size, params[0]), params[1], params[2])
        networks = ppo_networks.make_ppo_networks(
            observation_size,
            int(network_config["action_size"]),
            preprocess_observations_fn=preprocess,
            **kwargs,
        )
        make_policy = ppo_networks.make_inference_fn(networks)
    # 无论 SAC 还是 PPO，最后都会统一生成一个 `policy(obs, rng)` 风格的函数。
    policy = make_policy(params, deterministic=bool(args.deterministic))
    return policy, checkpoint_dir


def main() -> None:
    # 运行 Warp 推理流程。
    #
    # 主要步骤：
    # 1. 检查运行时依赖。
    # 2. 构造测试环境并按需覆盖部分环境参数。
    # 3. 加载最终参数或中间 checkpoint。
    # 4. 循环执行若干测试回合，并按需可视化。
    # 第一步：解析 CLI 参数并检查外部依赖。
    args = build_parser().parse_args()
    if not playground_importable():
        raise SystemExit("未检测到 `mujoco_playground`，无法运行 Warp 推理线。")
    ensure_warp_runtime()

    # 这些库只在真正推理时导入，避免静态阅读代码时就要求完整图形和 Warp 环境。
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    try:
        import mujoco.viewer as mj_viewer
    except Exception:
        mj_viewer = None

    from warp_ur5_env import UR5WarpReachEnv, default_config

    # 第二步：准备测试环境配置，并允许 CLI 临时覆盖部分控制与目标参数。
    run_dir = _resolve_run_dir(args.algo, args.run_name)
    # 这里先取环境默认配置，再用 CLI 参数覆盖，保证推理逻辑和训练配置层保持一致。
    cfg = default_config()
    # 这一组覆盖项主要是“推理时可能经常想改”的参数，例如目标点、控制模式和奖励模式。
    cfg.episode_length = max(int(args.max_steps), 1)
    cfg.fixed_target_x = args.fixed_target_x
    cfg.fixed_target_y = args.fixed_target_y
    cfg.fixed_target_z = args.fixed_target_z
    cfg.controller_mode = args.controller_mode
    cfg.joint_position_delta_scale = float(args.joint_position_delta_scale)
    cfg.position_control_kp = float(args.position_control_kp)
    cfg.position_control_kd = float(args.position_control_kd)
    cfg.goal_observation = bool(args.goal_observation)
    cfg.reward_mode = str(args.reward_mode)

    # 第三步：创建环境，并按 `artifact` 选择最终参数或 checkpoint。
    env = UR5WarpReachEnv(config=cfg)
    if args.artifact == "final":
        # `final` 模式下，加载训练结束时导出的最终策略。
        policy, loaded_path = _load_final_policy(args, run_dir, env)
    else:
        # 其他模式下，加载训练过程中的阶段性 checkpoint。
        policy, loaded_path = _load_checkpoint_policy(args, run_dir)

    # 第四步：如果开启 human 渲染，就准备一份 host 侧 MuJoCo 数据结构。
    host_data = None
    viewer = None
    if args.render:
        if mj_viewer is None:
            raise SystemExit("当前环境无法导入 `mujoco.viewer`，不能使用 human 渲染。")
        # Warp 环境里的状态是 MJX / JAX 形式，human viewer 需要一份 MuJoCo host 侧数据结构。
        host_data = mujoco.MjData(env.mj_model)
        viewer = mj_viewer.launch_passive(env.mj_model, host_data)

    print(f"warp={describe_warp_runtime()}")
    print(f"xml={env.xml_path}")
    print(f"loaded_artifact={loaded_path}")

    # 第五步：循环执行测试回合。
    rewards: list[float] = []
    rng = jax.random.PRNGKey(args.seed)
    for episode_idx in range(max(int(args.episodes), 1)):
        # 每个 episode 都从主随机 key 中再拆一个 reset key，保证 reset 和 action 采样互不干扰。
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)
        if viewer is not None and host_data is not None:
            # 每次渲染前都要把 MJX 状态同步回 MuJoCo host 侧数据结构。
            mjx.get_data_into(host_data, env.mj_model, state.data)
            viewer.sync()
        total_reward = 0.0
        steps = 0
        while steps < max(int(args.max_steps), 1) and not bool(state.done):
            # 每一步都要：
            # 1. 拆随机数
            # 2. 用 policy 生成动作
            # 3. 推进一步环境
            # 4. 按需同步 viewer
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            action = jp.asarray(action, dtype=jp.float32)
            state = env.step(state, action)
            if viewer is not None and host_data is not None:
                mjx.get_data_into(host_data, env.mj_model, state.data)
                viewer.sync()
                time.sleep(0.01)
            steps += 1
            reward = float(state.reward)
            total_reward += reward
            if args.print_step_reward:
                # 这里直接读环境 metrics，而不是重新手算距离和成功标记。
                print(
                    f"[step {steps}] reward={reward:.6f} "
                    f"distance={float(state.metrics['distance']):.6f} "
                    f"success={bool(state.metrics['success'])} "
                    f"collision={bool(state.metrics['collision'])}"
                )
        rewards.append(total_reward)
        print(
            f"Episode {episode_idx + 1}: steps={steps}, reward={total_reward:.3f}, "
            f"distance={float(state.metrics['distance']):.4f}, "
            f"success={bool(state.metrics['success'])}, collision={bool(state.metrics['collision'])}"
        )

    # 第六步：输出测试汇总结果。
    if rewards:
        print(f"平均奖励: {sum(rewards) / len(rewards):.3f}")
    if args.render:
        # 主动退出进程，避免某些图形环境下 viewer 线程阻塞终端。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
