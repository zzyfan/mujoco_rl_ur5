
"""
使用Stable-Baselines3的TD3算法训练机械臂进行位置跟踪
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
from robot_arm_env import RobotArmEnv
from rl_metrics import (
    TrainingMetricsCallback,
    evaluate_inference_metrics,
)
import argparse
import torch.nn as nn
import signal


class SaveVecNormalizeCallback(BaseCallback):
    """
    在保存最佳模型时同时保存VecNormalize参数的回调函数
    """
    def __init__(self, eval_callback, best_model_dir, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.eval_callback = eval_callback
        self.best_model_dir = best_model_dir
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 检查是否有新的最佳模型
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            
            # 保存VecNormalize参数到logs/best_model目录
            if self.verbose > 0:
                print("保存与最佳模型对应的VecNormalize参数")
            os.makedirs(self.best_model_dir, exist_ok=True)
            self.model.get_vec_normalize_env().save(os.path.join(self.best_model_dir, "vec_normalize.pkl"))
        
        return True


class ManualInterruptCallback(BaseCallback):
    """
    允许手动中断训练并保存模型的回调函数
    """
    def __init__(self, interrupted_model_path, interrupted_vec_path, verbose=0):
        super(ManualInterruptCallback, self).__init__(verbose)
        self.interrupted = False
        self.interrupted_model_path = interrupted_model_path
        self.interrupted_vec_path = interrupted_vec_path
        # 设置信号处理器来捕获Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        if self.interrupted:
            return
        print('\n接收到中断信号，正在保存模型并优雅停止训练...')
        self.interrupted = True
        # 保存当前模型
        self.save_model()
        print('模型已保存，将在当前 step 结束后停止训练')
        
    def save_model(self):
        """
        保存当前模型和环境归一化参数
        """
        if self.model is not None:
            # 创建保存目录
            os.makedirs(os.path.dirname(self.interrupted_model_path), exist_ok=True)
            
            # 保存模型
            self.model.save(self.interrupted_model_path)
            
            # 保存VecNormalize参数
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(self.interrupted_vec_path)
                
            print(f"已保存中断时的模型和参数到 {os.path.dirname(self.interrupted_model_path)}")
        
    def _on_step(self) -> bool:
        # 如果收到中断信号，停止训练
        if self.interrupted:
            return False
        return True


class TrainRenderCallback(BaseCallback):
    """
    训练期间主动调用渲染的回调函数
    """
    def __init__(self, verbose=0):
        super(TrainRenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.training_env.env_method("render", indices=0)
        except Exception as e:
            if self.verbose > 0:
                print(f"训练渲染失败: {e}")
        return True


def train_robot_arm(
    train_render=False,
    root_dir="./checkpoints",
    resume=False,
    resume_from="final",
    resume_model_path=None,
    resume_normalize_path=None,
    total_timesteps=5000000,
):
    """
    训练机械臂进行位置跟踪
    """
    print("创建机械臂环境...")
    
    paths = build_paths(root_dir)
    os.makedirs(paths["algo_dir"], exist_ok=True)
    os.makedirs(paths["models_dir"], exist_ok=True)
    os.makedirs(paths["best_model_dir"], exist_ok=True)
    os.makedirs(paths["eval_log_dir"], exist_ok=True)
    os.makedirs(paths["metrics_dir"], exist_ok=True)
    os.makedirs(paths["inference_dir"], exist_ok=True)

    # 创建环境并使用VecNormalize进行归一化
    train_render_mode = "human" if train_render else None
    base_env = make_vec_env(lambda: RobotArmEnv(render_mode=train_render_mode), n_envs=1)
    
    # 设置动作噪声
    n_actions = base_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.5 * np.ones(n_actions))
    
    # 自定义神经网络结构
    # 这里我们定义一个更复杂的网络结构：
    # Actor网络: [512, 512, 256]
    # Critic网络: [512, 512, 256] (每个Q网络)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],  # Actor网络结构
            qf=[512, 512, 256]   # Critic网络结构 (每个Q网络)
        ),
        activation_fn=nn.ReLU  # 使用ReLU激活函数
    )
    
    if resume:
        model_load_path, vec_load_path = resolve_resume_paths(
            paths,
            resume_from,
            resume_model_path,
            resume_normalize_path,
        )
        if not model_checkpoint_exists(model_load_path):
            raise FileNotFoundError(f"恢复训练失败，模型不存在: {model_load_path}")
        if os.path.exists(vec_load_path):
            env = VecNormalize.load(vec_load_path, base_env)
            env.training = True
            env.norm_reward = True
        else:
            print(f"警告: 未找到归一化参数 {vec_load_path}，将使用新的 VecNormalize。")
            env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
        model = TD3.load(model_load_path, env=env)
        print(f"继续训练: model={model_load_path}, vec={vec_load_path}")
    else:
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True)
        # 创建TD3模型
        model = TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=1,
            device="auto",  # 自动选择设备(CUDA/CPU)
            learning_rate=3e-4,
            buffer_size=3000000,
            learning_starts=20000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=4,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs  # 使用自定义网络结构
        )
    
    # 创建评估环境和回调函数
    eval_env = make_vec_env(lambda: RobotArmEnv(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # 加载训练环境的归一化参数到评估环境中
    eval_env.obs_rms = env.obs_rms
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=paths["best_model_dir"],
        log_path=paths["eval_log_dir"],
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 创建保存VecNormalize参数的回调函数
    save_vec_normalize_callback = SaveVecNormalizeCallback(
        eval_callback,
        best_model_dir=paths["best_model_dir"],
        verbose=1,
    )
    
    # 创建手动中断回调函数
    manual_interrupt_callback = ManualInterruptCallback(
        interrupted_model_path=paths["interrupted_model_path"],
        interrupted_vec_path=paths["interrupted_vec_path"],
        verbose=1,
    )
    training_metrics_callback = TrainingMetricsCallback(
        save_dir=paths["metrics_dir"],
        loss_log_freq=1000,
        record_to_sb3_logger=False,
        verbose=0,
    )
    
    print("开始训练...")
    print("提示: 按 Ctrl+C 可以中途停止训练并保存最后一次模型数据")
    start_time = time.time()
    
    callbacks = [
        eval_callback,
        save_vec_normalize_callback,
        manual_interrupt_callback,
        training_metrics_callback,
    ]
    if train_render:
        callbacks.append(TrainRenderCallback(verbose=1))
        print("训练渲染: 开启")
    else:
        print("训练渲染: 关闭")

    try:
        # 训练模型
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1000,
            progress_bar=True,
            reset_num_timesteps=not resume,
        )
        training_metrics_callback.save_and_plot()

        # 保存归一化环境和最终模型
        env.save(paths["final_vec_path"])
        model.save(paths["final_model_path"])
    finally:
        try:
            eval_env.close()
        except Exception as e:
            print(f"关闭评估环境失败: {e}")
        try:
            env.close()
        except Exception as e:
            print(f"关闭训练环境失败: {e}")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    
    return model, env


def test_robot_arm(model_path="./models/td3/td3_robot_arm_final", 
                   normalize_path="./models/td3/vec_normalize.pkl",
                   num_episodes=10,
                   report_dir="./logs/td3/inference",
                   render=True):
    """
    测试训练好的模型
    """
    print("加载模型并测试...")
    
    # 创建环境
    render_mode = "human" if render else None
    env = make_vec_env(lambda: RobotArmEnv(render_mode=render_mode), n_envs=1)
    
    # 加载归一化环境
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # 加载模型
    model = TD3.load(model_path, env=env)
    
    summary, _ = evaluate_inference_metrics(
        model=model,
        env=env,
        num_episodes=num_episodes,
        save_dir=report_dir,
        render=render,
        max_steps_per_episode=5000,
        sleep_seconds=0.01 if render else 0.0,
    )

    env.close()
    print("\n推理评估结果:")
    print(f"Success Rate: {summary['success_rate'] * 100:.2f}%")
    print(f"Mean Final Distance: {summary['mean_final_distance']:.6f}")
    print(f"Mean Episode Length: {summary['mean_episode_length']:.2f}")
    print(f"Collision Rate: {summary['collision_rate'] * 100:.2f}%")
    print(f"Mean Smoothness: {summary['mean_smoothness']:.6f}")
    print(f"Mean Reward: {summary['mean_reward']:.3f}")
    print(f"详细报告目录: {report_dir}")
    return summary


def build_paths(root_dir):
    algo_dir = os.path.join(root_dir, "td3")
    models_dir = os.path.join(algo_dir, "models")
    best_model_dir = os.path.join(models_dir, "best_model")
    interrupted_dir = os.path.join(models_dir, "interrupted")
    return {
        "algo_dir": algo_dir,
        "models_dir": models_dir,
        "best_model_dir": best_model_dir,
        "eval_log_dir": os.path.join(algo_dir, "eval_logs"),
        "metrics_dir": os.path.join(algo_dir, "metrics"),
        "inference_dir": os.path.join(algo_dir, "inference"),
        "final_model_path": os.path.join(models_dir, "td3_robot_arm_final"),
        "final_vec_path": os.path.join(models_dir, "vec_normalize.pkl"),
        "best_model_path": os.path.join(best_model_dir, "best_model"),
        "best_vec_path": os.path.join(best_model_dir, "vec_normalize.pkl"),
        "interrupted_model_path": os.path.join(interrupted_dir, "td3_robot_arm_interrupted"),
        "interrupted_vec_path": os.path.join(interrupted_dir, "vec_normalize.pkl"),
    }


def build_legacy_paths():
    return {
        "final_model_path": os.path.join(".", "models", "td3", "td3_robot_arm_final"),
        "final_vec_path": os.path.join(".", "models", "td3", "vec_normalize.pkl"),
        "best_model_path": os.path.join(".", "logs", "best_model", "best_model"),
        "best_vec_path": os.path.join(".", "logs", "td3", "best_model", "vec_normalize.pkl"),
        "interrupted_model_path": os.path.join(".", "models", "td3", "interrupted", "td3_robot_arm_interrupted"),
        "interrupted_vec_path": os.path.join(".", "models", "td3", "interrupted", "vec_normalize.pkl"),
    }


def model_checkpoint_exists(model_path):
    if os.path.exists(model_path):
        return True
    if model_path.endswith(".zip"):
        return False
    return os.path.exists(model_path + ".zip")


def first_existing_path(candidates, is_model=False):
    for path in candidates:
        if is_model:
            if model_checkpoint_exists(path):
                return path
        else:
            if os.path.exists(path):
                return path
    return candidates[0]


def resolve_test_paths(model_choice, root_dir, model_path_override=None, normalize_path_override=None):
    paths = build_paths(root_dir)
    legacy_paths = build_legacy_paths()
    if model_choice == "best":
        model_candidates = [paths["best_model_path"], legacy_paths["best_model_path"]]
        normalize_candidates = [paths["best_vec_path"], legacy_paths["best_vec_path"]]
    else:
        model_candidates = [paths["final_model_path"], legacy_paths["final_model_path"]]
        normalize_candidates = [paths["final_vec_path"], legacy_paths["final_vec_path"]]

    default_model_path = first_existing_path(model_candidates, is_model=True)
    default_normalize_path = first_existing_path(normalize_candidates, is_model=False)

    model_path = model_path_override if model_path_override else default_model_path
    normalize_path = normalize_path_override if normalize_path_override else default_normalize_path
    return model_path, normalize_path


def resolve_resume_paths(paths, resume_from, model_path_override=None, normalize_path_override=None):
    legacy_paths = build_legacy_paths()
    if resume_from == "best":
        model_candidates = [paths["best_model_path"], legacy_paths["best_model_path"]]
        normalize_candidates = [paths["best_vec_path"], legacy_paths["best_vec_path"]]
    elif resume_from == "interrupted":
        model_candidates = [paths["interrupted_model_path"], legacy_paths["interrupted_model_path"]]
        normalize_candidates = [paths["interrupted_vec_path"], legacy_paths["interrupted_vec_path"]]
    else:
        model_candidates = [paths["final_model_path"], legacy_paths["final_model_path"]]
        normalize_candidates = [paths["final_vec_path"], legacy_paths["final_vec_path"]]

    default_model_path = first_existing_path(model_candidates, is_model=True)
    default_normalize_path = first_existing_path(normalize_candidates, is_model=False)

    model_path = model_path_override if model_path_override else default_model_path
    normalize_path = normalize_path_override if normalize_path_override else default_normalize_path
    return model_path, normalize_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test robot arm with TD3")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--root-dir", type=str, default="./checkpoints",
                        help="Fixed root directory for saving models/logs by algorithm")
    parser.add_argument("--model", type=str, default="final", choices=["final", "best"],
                        help="Use final or best model for testing")
    parser.add_argument("--model-path", type=str, default=None, 
                        help="Optional explicit model path for testing (overrides --model)")
    parser.add_argument("--normalize-path", type=str, default=None,
                        help="Optional explicit normalization path (overrides --model)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from an existing checkpoint")
    parser.add_argument("--resume-from", type=str, default="final", choices=["final", "best", "interrupted"],
                        help="Which checkpoint to resume from")
    parser.add_argument("--resume-model-path", type=str, default=None,
                        help="Optional explicit model path for resume training")
    parser.add_argument("--resume-normalize-path", type=str, default=None,
                        help="Optional explicit normalization path for resume training")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to test")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
                        help="Total timesteps for this training run (also works with --resume)")
    parser.add_argument("--train-render", action="store_true",
                        help="Enable rendering during training")
    parser.add_argument("--inference-report-dir", type=str, default=None,
                        help="Directory to save inference metrics report")
    parser.add_argument("--no-test-render", action="store_true",
                        help="Disable rendering during test/inference")
    
    args = parser.parse_args()
    
    if args.test:
        model_path, normalize_path = resolve_test_paths(
            args.model, args.root_dir, args.model_path, args.normalize_path
        )
        report_dir = args.inference_report_dir or build_paths(args.root_dir)["inference_dir"]
        test_robot_arm(
            model_path,
            normalize_path,
            args.episodes,
            report_dir=report_dir,
            render=not args.no_test_render,
        )
    else:
        train_robot_arm(
            train_render=args.train_render,
            root_dir=args.root_dir,
            resume=args.resume,
            resume_from=args.resume_from,
            resume_model_path=args.resume_model_path,
            resume_normalize_path=args.resume_normalize_path,
            total_timesteps=max(1, args.total_timesteps),
        )
