#!/usr/bin/env python3
"""Inference test entrypoint for Warp GPU reach policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jp
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from flax import serialization as flax_serialization
from jax import nn as jnn
from orbax import checkpoint as ocp

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from warp_gpu.env import UR5ReachWarpEnv, default_config
    from warp_gpu.runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable
else:
    from .env import UR5ReachWarpEnv, default_config
    from .runtime import describe_warp_runtime, ensure_warp_runtime, playground_importable


def parse_args() -> argparse.Namespace:
    """解析 Warp GPU 推理测试参数。"""
    p = argparse.ArgumentParser(description="Warp GPU 模型推理测试")
    p.add_argument("--algo", choices=["ppo", "sac"], default="sac")  # 决定策略结构和参数加载方式。
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 选择机器人模型和默认 XML。
    p.add_argument("--run-name", type=str, default="ur5_reach_warp_gpu")  # 未手动指定路径时按 run-name 解析产物目录。
    p.add_argument("--model-dir", type=str, default="models/warp_gpu")  # Warp GPU 产物根目录。
    p.add_argument("--artifact", choices=["final", "latest-checkpoint", "checkpoint"], default="final")  # 选择最终参数或中间 checkpoint。
    p.add_argument("--params-path", type=str, default="")  # 直接指定 final_policy.msgpack 路径。
    p.add_argument("--checkpoint-path", type=str, default="")  # 直接指定某个 checkpoint 目录。
    p.add_argument("--checkpoint-step", type=str, default="")  # 从 checkpoints 下按步数选择一个子目录。
    p.add_argument("--episodes", type=int, default=3)  # 推理回合数。
    p.add_argument("--max-steps", type=int, default=3000)  # 每回合最大步数。
    p.add_argument("--seed", type=int, default=42)  # 重置和动作采样的随机种子。
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)  # 是否使用确定性策略输出。
    p.add_argument("--fixed-target-x", type=float, default=None)  # 固定目标点 x，便于稳定观察策略表现。
    p.add_argument("--fixed-target-y", type=float, default=None)  # 固定目标点 y。
    p.add_argument("--fixed-target-z", type=float, default=None)  # 固定目标点 z。
    p.add_argument("--print-step-reward", action="store_true")  # 打印每一步奖励、距离和终止标记。
    return p.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    """按算法、机器人和运行名拼接标准输出目录。"""
    return Path(args.model_dir).resolve() / args.algo / args.robot / args.run_name


def _latest_checkpoint_dir(checkpoint_root: Path) -> Path:
    """从 checkpoints 目录里选出步数最大的 checkpoint。"""
    candidates = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"未找到 checkpoint 目录: {checkpoint_root}")
    return max(candidates, key=lambda p: int(p.name))


def _select_checkpoint_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    """解析用户指定的 checkpoint 路径或步数。"""
    if args.checkpoint_path:
        return Path(args.checkpoint_path).resolve()
    checkpoint_root = run_dir / "checkpoints"
    if args.artifact == "latest-checkpoint" or not args.checkpoint_step:
        return _latest_checkpoint_dir(checkpoint_root)
    selected = checkpoint_root / args.checkpoint_step
    if not selected.exists():
        raise FileNotFoundError(f"未找到指定 checkpoint: {selected}")
    return selected.resolve()


def _normalize_fn(enabled: bool) -> Callable:
    """根据训练配置决定是否在推理前执行观测标准化。"""
    return running_statistics.normalize if enabled else (lambda x, y: x)


def _resolve_activation(name: str | None):
    """把网络配置里的激活函数名映射成 JAX 可调用对象。"""
    mapping = {
        "relu": jnn.relu,
        "silu": jnn.silu,
        "swish": jnn.swish,
    }
    if name is None:
        return None
    if name not in mapping:
        raise ValueError(f"不支持的激活函数配置: {name}")
    return mapping[name]


def _resolve_initializer(name: str | None):
    """把网络配置里的初始化器名映射成 JAX 初始化函数。"""
    mapping = {
        "lecun_uniform": jnn.initializers.lecun_uniform,
        None: None,
    }
    if name not in mapping:
        raise ValueError(f"不支持的初始化器配置: {name}")
    return mapping[name]


def _load_checkpoint_params(checkpoint_dir: Path):
    """从 Orbax checkpoint 目录恢复参数树。"""
    restored = ocp.PyTreeCheckpointer().restore(str(checkpoint_dir.resolve()))
    if isinstance(restored, list):
        return tuple(restored)
    return restored


def _restore_running_statistics(observation_size, state_dict: dict):
    """把 checkpoint 里的 normalizer 字典恢复成 RunningStatisticsState。"""
    template = running_statistics.init_state(jp.zeros(observation_size, dtype=jp.float32))
    return flax_serialization.from_state_dict(template, state_dict)


def _load_checkpoint_network_config(args: argparse.Namespace, checkpoint_dir: Path) -> dict:
    """读取 checkpoint 目录里的网络结构配置。"""
    config_name = f"{args.algo}_network_config.json"
    config_path = checkpoint_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint 网络配置: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_final_policy(args: argparse.Namespace, run_dir: Path, env: UR5ReachWarpEnv):
    """加载 `final_policy.msgpack` 并重建推理函数。"""
    params_path = Path(args.params_path).resolve() if args.params_path else (run_dir / "final_policy.msgpack")
    if not params_path.exists():
        raise FileNotFoundError(f"未找到最终策略参数: {params_path}")

    config_path = run_dir / "config.json"
    normalize_observations = True
    if config_path.exists():
        train_args = json.loads(config_path.read_text(encoding="utf-8")).get("train_args", {})
        normalize_observations = bool(train_args.get("normalize_observations", True))

    preprocess = _normalize_fn(normalize_observations)
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

    params = brax_model.load_params(str(params_path))
    policy = make_policy(params, deterministic=bool(args.deterministic))
    return policy, params_path


def _load_checkpoint_policy(args: argparse.Namespace, run_dir: Path):
    """加载某个中间 checkpoint 对应的推理函数。"""
    checkpoint_dir = _select_checkpoint_dir(args, run_dir)
    network_config = _load_checkpoint_network_config(args, checkpoint_dir)
    preprocess = _normalize_fn(bool(network_config.get("normalize_observations", True)))
    params = _load_checkpoint_params(checkpoint_dir)

    if args.algo == "sac":
        observation_size = int(network_config["observation_size"])
        kwargs = dict(network_config.get("network_factory_kwargs", {}))
        kwargs["activation"] = _resolve_activation(kwargs.get("activation"))
        kwargs["policy_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("policy_network_kernel_init_fn"))
        kwargs["q_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("q_network_kernel_init_fn"))
        params = (
            _restore_running_statistics(observation_size, params[0]),
            params[1],
        )
        networks = sac_networks.make_sac_networks(
            observation_size=observation_size,
            action_size=int(network_config["action_size"]),
            preprocess_observations_fn=preprocess,
            **kwargs,
        )
        make_policy = sac_networks.make_inference_fn(networks)
    else:
        kwargs = dict(network_config.get("network_factory_kwargs", {}))
        kwargs["activation"] = _resolve_activation(kwargs.get("activation"))
        kwargs["policy_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("policy_network_kernel_init_fn"))
        kwargs["value_network_kernel_init_fn"] = _resolve_initializer(kwargs.get("value_network_kernel_init_fn"))
        kwargs["mean_kernel_init_fn"] = _resolve_initializer(kwargs.get("mean_kernel_init_fn"))
        observation_size = network_config.get("observation_size")
        if isinstance(observation_size, dict):
            observation_size = tuple(observation_size.get("shape", ()))
        params = (
            _restore_running_statistics(observation_size, params[0]),
            params[1],
            params[2],
        )
        networks = ppo_networks.make_ppo_networks(
            observation_size,
            int(network_config["action_size"]),
            preprocess_observations_fn=preprocess,
            **kwargs,
        )
        make_policy = ppo_networks.make_inference_fn(networks)
    policy = make_policy(params, deterministic=bool(args.deterministic))
    return policy, checkpoint_dir


def main() -> None:
    """Warp GPU 推理测试入口。"""
    args = parse_args()
    if not playground_importable():
        raise SystemExit("未检测到 `mujoco_playground`。")
    ensure_warp_runtime()

    run_dir = _run_dir(args)
    cfg = default_config(args.robot)
    cfg.episode_length = max(int(args.max_steps), 1)  # 测试时用命令行步数上限覆盖默认回合长度。
    cfg.fixed_target_x = args.fixed_target_x  # 固定目标点有利于横向比较不同策略输出。
    cfg.fixed_target_y = args.fixed_target_y
    cfg.fixed_target_z = args.fixed_target_z

    env = UR5ReachWarpEnv(config=cfg)
    if args.artifact == "final":
        policy, loaded_path = _load_final_policy(args, run_dir, env)
    else:
        policy, loaded_path = _load_checkpoint_policy(args, run_dir)

    print(f"warp={describe_warp_runtime()}")
    print(f"xml={env.xml_path}")
    print(f"loaded_artifact={loaded_path}")

    rewards: list[float] = []
    rng = jax.random.PRNGKey(args.seed)
    for ep in range(max(int(args.episodes), 1)):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)
        total_reward = 0.0
        steps = 0
        while steps < max(int(args.max_steps), 1) and not bool(state.done):
            rng, act_rng = jax.random.split(rng)
            action, _ = policy(state.obs, act_rng)
            action = jp.asarray(action, dtype=jp.float32)
            state = env.step(state, action)
            steps += 1
            reward = float(state.reward)
            total_reward += reward
            if args.print_step_reward:
                print(
                    f"[step {steps}] reward={reward:.6f} "
                    f"distance={float(state.metrics['distance']):.6f} "
                    f"success={bool(state.metrics['success'])} "
                    f"collision={bool(state.metrics['collision'])}"
                )
        rewards.append(total_reward)
        print(
            f"Episode {ep + 1}: steps={steps}, reward={total_reward:.3f}, "
            f"distance={float(state.metrics['distance']):.4f}, "
            f"success={bool(state.metrics['success'])}, collision={bool(state.metrics['collision'])}"
        )

    if rewards:
        print(f"平均奖励: {sum(rewards) / len(rewards):.3f}")


if __name__ == "__main__":
    main()
