#!/usr/bin/env python3
"""MuJoCo 独立训练脚本（SAC/PPO/TD3），不依赖 ROS/Gazebo。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import argparse  # 标准库：命令行参数解析。
import os  # 标准库：目录/路径操作。
import shutil  # 标准库：复制旧目录到新分层目录。
import signal  # 标准库：处理 Ctrl+C 信号。
import sys  # 依赖导入：先认清这个文件需要哪些外部能力
import time  # 标准库：计时。
from dataclasses import dataclass  # dataclass：简洁定义参数类。
from pathlib import Path  # 依赖导入：先认清这个文件需要哪些外部能力

import numpy as np  # 数值计算与数组操作。
import torch  # PyTorch：设备检查与张量。
import torch.nn as nn  # 神经网络模块（激活函数等）。
from stable_baselines3 import PPO, SAC, TD3  # 三种可选算法实现。
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback  # 训练回调基类与评估回调。
from stable_baselines3.common.env_util import make_vec_env  # 快速创建向量化环境。
from stable_baselines3.common.noise import NormalActionNoise  # TD3 常用动作噪声。
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # 向量环境与归一化封装。

if __package__ in (None, ""):  # 条件分支：学习时先看触发条件，再看两边行为差异
    ROOT = Path(__file__).resolve().parents[1]  # 状态或中间变量：调试时多观察它的值如何流动
    if str(ROOT) not in sys.path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        sys.path.insert(0, str(ROOT))  # 代码执行语句：结合上下文理解它对后续流程的影响
    from classic.env import ENV_ID, MujocoEnvConfig, register_env  # 本项目环境注册与配置。
else:  # 兜底分支：当前面条件都不满足时走这里
    from .env import ENV_ID, MujocoEnvConfig, register_env  # 本项目环境注册与配置。


@dataclass  # 用 dataclass 收拢配置，调参时优先回到这里
class TrainArgs:  # 类定义：先理解职责边界，再进入方法细节
    """训练/测试运行参数。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

    test: bool = False  # `True` 时运行测试流程。
    algo: str = "sac"  # 算法：`sac` / `ppo` / `td3`。
    timesteps: int = 5_000_000  # 总训练步数。
    episodes: int = 5000  # 测试回合数。
    seed: int = 42  # 随机种子。
    n_envs: int = 8  # 并行环境数量（UR5 默认优化值）。
    device: str = "cuda"  # 训练设备：`cuda` 或 `cpu`。
    render: bool = False  # 训练时是否显示窗口。
    render_mode: str = "human"  # 渲染模式：`human` 或 `rgb_array`。
    model_dir: str = "models/classic"  # 模型输出目录。
    log_dir: str = "logs/classic"  # 日志输出目录。
    run_name: str = "ur5_mujoco"  # 运行名（拼接到模型文件名）。
    eval_freq: int = 5000  # 评估频率（每多少步评估一次）。
    n_eval_episodes: int = 1  # 每次评估回合数（越小越快）。
    save_best_model: bool = True  # 评估时默认保存 best_model 与对应 VecNormalize（对齐 zero 原始脚本）。
    render_freq: int = 1  # 渲染频率（每多少个回调 step 渲染一次）。
    log_interval: int = 1000  # SB3 日志间隔。
    batch_size: int = 512  # 小批量大小（UR5 默认优化值）。
    buffer_size: int = 200_000  # SAC/TD3 回放池容量上限（16G 内存友好）。
    gradient_steps: int = 4  # 每次更新的梯度步数（UR5 默认优化值）。
    learning_starts: int = 20000  # 经验池至少积累多少步后开始学习。
    action_noise_sigma: float = 2.5  # TD3 动作噪声标准差。
    policy_delay: int = 4  # TD3 actor 延迟更新步数。
    target_policy_noise: float = 0.2  # TD3 目标策略噪声。
    target_noise_clip: float = 0.5  # TD3 目标噪声裁剪。
    max_steps: int = 3000  # 单回合最大步数。
    success_threshold: float = 0.01  # 到点成功阈值（米）。
    frame_skip: int = 1  # 每个 RL step 对应 MuJoCo 积分步数。
    physics_backend: str = "mujoco"  # 物理后端：mujoco（默认）/warp/auto（优先 warp）。
    legacy_zero_ee_velocity: bool = False  # 是否兼容 zero 原始 `cvel[:3]` 末端速度读取。
    robot: str = "ur5_cxy"  # 机械臂模型：`ur5_cxy` 或 `zero_robotiq`。
    lock_camera: bool = False  # 是否锁定到 XML 固定相机（默认 False，可鼠标拖动）。
    ur5_target_x_min: float = -0.95  # UR5 目标采样范围 x 最小值。
    ur5_target_x_max: float = -0.60  # UR5 目标采样范围 x 最大值。
    ur5_target_y_min: float = 0.15  # UR5 目标采样范围 y 最小值。
    ur5_target_y_max: float = 0.50  # UR5 目标采样范围 y 最大值。
    ur5_target_z_min: float = 0.12  # UR5 目标采样范围 z 最小值。
    ur5_target_z_max: float = 0.30  # UR5 目标采样范围 z 最大值。
    zero_target_x_min: float = -1.00  # ZERO 目标采样范围 x 最小值。
    zero_target_x_max: float = -0.62  # ZERO 目标采样范围 x 最大值。
    zero_target_y_min: float = 0.08  # ZERO 目标采样范围 y 最小值。
    zero_target_y_max: float = 0.48  # ZERO 目标采样范围 y 最大值。
    zero_target_z_min: float = 0.10  # ZERO 目标采样范围 z 最小值。
    zero_target_z_max: float = 0.35  # ZERO 目标采样范围 z 最大值。
    curriculum_stage1_fixed_episodes: int = 200  # 课程阶段 1 回合数（固定点）。
    curriculum_stage2_random_episodes: int = 800  # 课程阶段 2 回合数（小范围随机）。
    curriculum_stage2_range_scale: float = 0.35  # 阶段 2 随机范围缩放。
    fixed_target_x: float | None = None  # 阶段 1 固定目标 x。
    fixed_target_y: float | None = None  # 阶段 1 固定目标 y。
    fixed_target_z: float | None = None  # 阶段 1 固定目标 z。
    model_path: str = ""  # 测试时可手动指定模型路径。
    normalize_path: str = ""  # 测试时可手动指定归一化参数路径。
    resume: bool = False  # 是否从已有模型继续训练。
    resume_model_path: str = ""  # 继续训练的模型路径（为空时走默认 interrupted 路径）。
    resume_normalize_path: str = ""  # 继续训练的归一化路径（为空时走默认 interrupted 路径）。
    resume_replay_path: str = ""  # 继续训练的回放缓存路径（SAC/TD3 可用）。
    skip_replay_buffer: bool = False  # 继续训练时是否跳过旧回放缓存恢复。


class RenderDuringTrainingCallback(BaseCallback):  # 类定义：先理解职责边界，再进入方法细节
    """训练期间按频率调用渲染。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

    def __init__(self, render_freq: int = 1, verbose: int = 0):  # 构造函数：实例状态、依赖和默认值都从这里落地
        super().__init__(verbose)  # 初始化父类回调。
        self.render_freq = max(int(render_freq), 1)  # 保证渲染频率至少为 1。

    def _on_step(self) -> bool:  # 函数定义：先看输入输出，再理解内部控制流
        if self.n_calls % self.render_freq != 0:  # 未到渲染周期时直接继续训练。
            return True  # True 表示不中断训练。
        try:  # 异常保护：把高风险调用包起来，避免主流程中断
            if hasattr(self.training_env, "envs") and len(self.training_env.envs) > 0:  # VecEnv 中取第一个环境。
                env = self.training_env.envs[0]  # 先拿外层包装对象。
                while hasattr(env, "env"):  # 逐层解包到最内层 Gym 环境。
                    env = env.env  # unwrap 一层。
                if hasattr(env, "render"):  # 环境提供 render 接口才调用。
                    env.render()  # 刷新 MuJoCo 画面。
        except Exception as e:  # 渲染失败不应阻塞训练。
            if self.verbose > 0:  # verbose>0 时输出详细信息。
                print(f"渲染报错: {e}")  # 打印异常。
        return True  # 继续训练。


class SaveVecNormalizeCallback(BaseCallback):  # 类定义：先理解职责边界，再进入方法细节
    """当评估最优回报更新时，同步保存 VecNormalize 参数。"""  # 归一化相关：训练和测试必须尽量保持统计一致

    def __init__(self, eval_callback: EvalCallback, save_path: str, verbose: int = 0):  # 构造函数：实例状态、依赖和默认值都从这里落地
        super().__init__(verbose)  # 初始化父类。
        self.eval_callback = eval_callback  # EvalCallback 实例引用。
        self.save_path = save_path  # 归一化参数输出路径。
        self.best_mean_reward = -np.inf  # 记录历史最佳评估回报。

    def _on_step(self) -> bool:  # 函数定义：先看输入输出，再理解内部控制流
        if self.eval_callback.best_mean_reward > self.best_mean_reward:  # 评估最好成绩刷新时才保存。
            self.best_mean_reward = float(self.eval_callback.best_mean_reward)  # 更新最佳值缓存。
            vec_env = self.model.get_vec_normalize_env()  # 获取当前模型绑定的 VecNormalize。
            if vec_env is not None:  # 仅在使用 VecNormalize 时保存。
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)  # 自动创建目标目录。
                vec_env.save(self.save_path)  # 写入归一化统计文件。
                if self.verbose > 0:  # 控制台可选提示。
                    print("已为最佳模型保存 VecNormalize 参数")  # 打印成功信息。
        return True  # 不中断训练。


class ManualInterruptCallback(BaseCallback):  # 类定义：先理解职责边界，再进入方法细节
    """支持 Ctrl+C 中断保存（模型 + 归一化参数）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

    def __init__(self, interrupted_model_path: str, interrupted_norm_path: str, verbose: int = 0):  # 构造函数：实例状态、依赖和默认值都从这里落地
        super().__init__(verbose)  # 初始化父类。
        self.interrupted = False  # 中断标志位。
        self._saved_once = False  # 防止重复保存中断模型。
        self.interrupted_model_path = interrupted_model_path  # 中断模型保存路径。
        self.interrupted_norm_path = interrupted_norm_path  # 中断归一化参数保存路径。
        signal.signal(signal.SIGINT, self.signal_handler)  # 注册 Ctrl+C 信号处理函数。

    def signal_handler(self, _sig, _frame):  # 函数定义：先看输入输出，再理解内部控制流
        if self.interrupted:  # 第二次 Ctrl+C 直接强退。
            print("\n再次收到中断信号，立即强制退出。")  # 提示用户进入强退。
            os._exit(130)  # 不再等待任何清理逻辑。
        print("\n收到中断信号：将在当前 step 后保存并退出。")  # 第一次 Ctrl+C 走优雅停止。
        self.interrupted = True  # 标记为需要停止。

    def save_model(self):  # 函数定义：先看输入输出，再理解内部控制流
        if self.model is not None:  # 仅在模型已初始化时保存。
            os.makedirs(os.path.dirname(self.interrupted_model_path), exist_ok=True)  # 自动创建保存目录。
            self.model.save(self.interrupted_model_path)  # 保存 SB3 模型权重。
            vec_env = self.model.get_vec_normalize_env()  # 取归一化封装。
            if vec_env is not None:  # 存在归一化封装时同步保存。
                vec_env.save(self.interrupted_norm_path)  # 保存 VecNormalize 统计。

    def _on_step(self) -> bool:  # 函数定义：先看输入输出，再理解内部控制流
        if self.interrupted and not self._saved_once:  # 首次进入中断分支时执行保存。
            self.save_model()  # 保存中断模型与归一化统计。
            self._saved_once = True  # 标记已保存。
            print("已保存中断模型，正在退出训练...")  # 给用户反馈。
            return False  # 停止训练循环。
        return not self.interrupted  # 未中断时继续训练。


def _normalize_device_arg(device_raw: str) -> str:  # 函数定义：先看输入输出，再理解内部控制流
    """规范化并校验设备参数，尽早给出可读错误信息。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    device = str(device_raw).strip()  # 运行配置：这类参数会直接改变执行路径或调试体验
    lowered = device.lower()  # 运行配置：这类参数会直接改变执行路径或调试体验
    if lowered in {"cpu", "cuda", "mps"}:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return lowered  # 把当前结果返回给上层调用方
    if lowered.startswith("cuda:") and lowered.split(":", 1)[1].isdigit():  # 条件分支：学习时先看触发条件，再看两边行为差异
        return lowered  # 把当前结果返回给上层调用方
    if lowered.startswith("cud"):  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise ValueError(  # 主动抛错：用来尽早暴露错误输入或不支持状态
            f"无效的 --device 参数: {device_raw!r}。"  # 运行配置：这类参数会直接改变执行路径或调试体验
            "你可能想写的是 'cuda'。若你命令里还有 '--eval-freq 5000'，请确认参数之间有空格："  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "--device cuda --eval-freq 5000"  # 运行配置：这类参数会直接改变执行路径或调试体验
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    raise ValueError(  # 主动抛错：用来尽早暴露错误输入或不支持状态
        f"无效的 --device 参数: {device_raw!r}。可用值示例: cpu / cuda / cuda:0 / mps"  # 运行配置：这类参数会直接改变执行路径或调试体验
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def _apply_zero_original_preset(args: TrainArgs) -> None:  # 函数定义：先看输入输出，再理解内部控制流
    """在 zero_robotiq 场景下，按 zero 原始项目习惯自动补齐默认参数。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if args.robot != "zero_robotiq":  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程

    applied: list[str] = []  # 状态或中间变量：调试时多观察它的值如何流动
    if args.n_envs == 8:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.n_envs = 1  # 状态或中间变量：调试时多观察它的值如何流动
        applied.append("n_envs=1")  # 状态或中间变量：调试时多观察它的值如何流动
    if args.batch_size == 512:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.batch_size = 256  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        applied.append("batch_size=256")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    if args.buffer_size == 200_000:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.buffer_size = 1_000_000  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        applied.append("buffer_size=1000000")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    if args.gradient_steps == 4:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.gradient_steps = 1  # 状态或中间变量：调试时多观察它的值如何流动
        applied.append("gradient_steps=1")  # 状态或中间变量：调试时多观察它的值如何流动
    if args.learning_starts == 20000:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.learning_starts = 50000  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        applied.append("learning_starts=50000")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    if args.max_steps == 3000:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.max_steps = 5000  # 状态或中间变量：调试时多观察它的值如何流动
        applied.append("max_steps=5000")  # 状态或中间变量：调试时多观察它的值如何流动
    if abs(float(args.success_threshold) - 0.01) < 1e-12:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.success_threshold = 0.001  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        applied.append("success_threshold=0.001")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    if args.algo == "td3" and abs(float(args.action_noise_sigma) - 2.5) < 1e-12:  # 条件分支：学习时先看触发条件，再看两边行为差异
        args.action_noise_sigma = 0.12  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        applied.append("action_noise_sigma=0.12")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    if applied:  # 条件分支：学习时先看触发条件，再看两边行为差异
        print(f"已应用 zero 原始参数预设: {', '.join(applied)}")  # 代码执行语句：结合上下文理解它对后续流程的影响


def _make_env(args: TrainArgs, render_mode: str | None = None):  # 函数定义：先看输入输出，再理解内部控制流
    """构造 `make_vec_env` 需要的 `env_kwargs`。"""  # 向量环境创建：并行环境数和包装方式在这里生效
    cfg = MujocoEnvConfig(  # 把训练参数映射成环境配置对象。
        frame_skip=args.frame_skip,  # 仿真 frame-skip。
        physics_backend=args.physics_backend,  # 物理后端选择。
        legacy_zero_ee_velocity=args.legacy_zero_ee_velocity,  # 是否沿用 zero 原始末端速度读取。
        max_steps=args.max_steps,  # 单回合最大步数。
        success_threshold=args.success_threshold,  # 成功判定阈值。
        viewer_lock_camera=args.lock_camera,  # 是否锁定固定相机。
        curriculum_stage1_fixed_episodes=args.curriculum_stage1_fixed_episodes,  # 阶段 1 回合数。
        curriculum_stage2_random_episodes=args.curriculum_stage2_random_episodes,  # 阶段 2 回合数。
        curriculum_stage2_range_scale=args.curriculum_stage2_range_scale,  # 阶段 2 范围缩放。
        fixed_target_x=args.fixed_target_x,  # 固定目标 x。
        fixed_target_y=args.fixed_target_y,  # 固定目标 y。
        fixed_target_z=args.fixed_target_z,  # 固定目标 z。
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    if args.robot == "zero_robotiq":  # zero 机械臂 + Robotiq 夹爪配置分支。
        cfg.model_xml = "assets/zero_arm/zero_with_robotiq_reach.xml"  # 切换到 zero+夹爪模型。
        cfg.home_pose_mode = "direct6"  # zero 机械臂采用直接六关节初始角。
        cfg.home_joint1 = 0.0  # 初始肩关节角。
        cfg.home_joint2 = -0.85  # 初始大臂关节角。
        cfg.home_joint3 = 1.35  # 初始小臂关节角。
        cfg.home_joint4 = -0.5  # 初始腕关节 1 角。
        cfg.home_joint5 = 0.0  # 初始腕关节 2 角。
        cfg.home_joint6 = 0.0  # 初始腕关节 3 角。
        cfg.render_camera_name = "workbench_camera"  # zero 模型里也提供同名固定相机。
        cfg.target_x_min = float(args.zero_target_x_min)  # ZERO 目标范围 x 最小值。
        cfg.target_x_max = float(args.zero_target_x_max)  # ZERO 目标范围 x 最大值。
        cfg.target_y_min = float(args.zero_target_y_min)  # ZERO 目标范围 y 最小值。
        cfg.target_y_max = float(args.zero_target_y_max)  # ZERO 目标范围 y 最大值。
        cfg.target_z_min = float(args.zero_target_z_min)  # ZERO 目标范围 z 最小值。
        cfg.target_z_max = float(args.zero_target_z_max)  # ZERO 目标范围 z 最大值。
        # zero 原始项目默认动作范围是 [-20, 20] 扭矩。
        cfg.torque_low = -20.0  # 状态或中间变量：调试时多观察它的值如何流动
        cfg.torque_high = 20.0  # 状态或中间变量：调试时多观察它的值如何流动
        # 打开 zero 原始模式：目标采样与奖励逻辑切换到原始风格。
        cfg.zero_original_mode = True  # 状态或中间变量：调试时多观察它的值如何流动
        # 如果用户没有手动指定阶段 1 固定目标点，则给 zero 一个“离机身更远”的默认点，
        # 避免课程初期目标与机身重合导致学习不稳定。
        if args.fixed_target_x is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            cfg.fixed_target_x = float(np.clip(cfg.target_x_max - 0.06, cfg.target_x_min, cfg.target_x_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
        if args.fixed_target_y is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            cfg.fixed_target_y = float(np.clip(cfg.target_y_min + 0.65 * (cfg.target_y_max - cfg.target_y_min), cfg.target_y_min, cfg.target_y_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
        if args.fixed_target_z is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            cfg.fixed_target_z = float(np.clip(cfg.target_z_min + 0.55 * (cfg.target_z_max - cfg.target_z_min), cfg.target_z_min, cfg.target_z_max))  # 限幅操作：调大调小时要关注稳定性和探索范围
    else:  # 兜底分支：当前面条件都不满足时走这里
        cfg.target_x_min = float(args.ur5_target_x_min)  # UR5 目标范围 x 最小值。
        cfg.target_x_max = float(args.ur5_target_x_max)  # UR5 目标范围 x 最大值。
        cfg.target_y_min = float(args.ur5_target_y_min)  # UR5 目标范围 y 最小值。
        cfg.target_y_max = float(args.ur5_target_y_max)  # UR5 目标范围 y 最大值。
        cfg.target_z_min = float(args.ur5_target_z_min)  # UR5 目标范围 z 最小值。
        cfg.target_z_max = float(args.ur5_target_z_max)  # UR5 目标范围 z 最大值。
    return {"render_mode": render_mode, "config": cfg}  # make_vec_env 会把这个字典传给环境构造函数。


def _build_run_paths(args: TrainArgs) -> dict[str, str]:  # 函数定义：先看输入输出，再理解内部控制流
    """按算法/运行名构造分层保存路径。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    run_dir = os.path.join(args.model_dir, args.algo, args.robot, args.run_name)  # 每个算法+机械臂独立目录。
    log_run_dir = os.path.join(args.log_dir, args.algo, args.robot, args.run_name)  # 每个算法+机械臂独立日志目录。
    stem = f"{args.algo}_{args.robot}_{args.run_name}"  # 兼容旧版命名时的前缀。
    return {  # 把当前结果返回给上层调用方
        "run_dir": run_dir,  # 代码执行语句：结合上下文理解它对后续流程的影响
        "log_run_dir": log_run_dir,  # 代码执行语句：结合上下文理解它对后续流程的影响
        "final_model": os.path.join(run_dir, "final", "model"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "final_norm": os.path.join(run_dir, "final", "vec_normalize.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "final_replay": os.path.join(run_dir, "final", "replay_buffer.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "interrupted_model": os.path.join(run_dir, "interrupted", "model"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "interrupted_norm": os.path.join(run_dir, "interrupted", "vec_normalize.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "interrupted_replay": os.path.join(run_dir, "interrupted", "replay_buffer.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "best_model_dir": os.path.join(log_run_dir, "best_model"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "best_norm": os.path.join(log_run_dir, "best_model", "vec_normalize.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "legacy_final_model": os.path.join(args.model_dir, f"{stem}_final"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        "legacy_final_norm": os.path.join(args.model_dir, f"{stem}_vec_normalize.pkl"),  # 代码执行语句：结合上下文理解它对后续流程的影响
    }  # 收束上一段结构，阅读时回看上面的参数或元素


def _path_exists_with_optional_zip(path: str) -> bool:  # 函数定义：先看输入输出，再理解内部控制流
    """兼容检查 `model` 与 `model.zip` 两种路径写法。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    return os.path.exists(path) or os.path.exists(path + ".zip")  # 把当前结果返回给上层调用方


def _copy_tree_missing(src_dir: str, dst_dir: str) -> int:  # 函数定义：先看输入输出，再理解内部控制流
    """递归复制目录中的缺失文件，不覆盖当前分层目录里已有文件。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if not os.path.isdir(src_dir):  # 条件分支：学习时先看触发条件，再看两边行为差异
        return 0  # 把当前结果返回给上层调用方
    copied = 0  # 状态或中间变量：调试时多观察它的值如何流动
    for root, _dirs, files in os.walk(src_dir):  # 循环逻辑：关注迭代对象、次数和循环体副作用
        rel = os.path.relpath(root, src_dir)  # 状态或中间变量：调试时多观察它的值如何流动
        current_dst_dir = dst_dir if rel == "." else os.path.join(dst_dir, rel)  # 代码执行语句：结合上下文理解它对后续流程的影响
        os.makedirs(current_dst_dir, exist_ok=True)  # 状态或中间变量：调试时多观察它的值如何流动
        for name in files:  # 循环逻辑：关注迭代对象、次数和循环体副作用
            src_file = os.path.join(root, name)  # 状态或中间变量：调试时多观察它的值如何流动
            dst_file = os.path.join(current_dst_dir, name)  # 状态或中间变量：调试时多观察它的值如何流动
            if os.path.exists(dst_file):  # 条件分支：学习时先看触发条件，再看两边行为差异
                continue  # 代码执行语句：结合上下文理解它对后续流程的影响
            shutil.copy2(src_file, dst_file)  # 代码执行语句：结合上下文理解它对后续流程的影响
            copied += 1  # 状态或中间变量：调试时多观察它的值如何流动
    return copied  # 把当前结果返回给上层调用方


def _sync_legacy_run_artifacts(args: TrainArgs, paths: dict[str, str]) -> None:  # 函数定义：先看输入输出，再理解内部控制流
    """把旧输出目录里的产物同步到当前分层目录，避免继续走兼容回退。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    model_dir_parent = os.path.dirname(os.path.normpath(args.model_dir))  # 状态或中间变量：调试时多观察它的值如何流动
    log_dir_parent = os.path.dirname(os.path.normpath(args.log_dir))  # 状态或中间变量：调试时多观察它的值如何流动
    legacy_model_run_dir = os.path.join(model_dir_parent, args.algo, args.robot, args.run_name)  # 状态或中间变量：调试时多观察它的值如何流动
    legacy_log_run_dir = os.path.join(log_dir_parent, args.algo, args.robot, args.run_name)  # 状态或中间变量：调试时多观察它的值如何流动

    copied = 0  # 状态或中间变量：调试时多观察它的值如何流动
    if os.path.normpath(legacy_model_run_dir) != os.path.normpath(paths["run_dir"]):  # 条件分支：学习时先看触发条件，再看两边行为差异
        copied += _copy_tree_missing(os.path.join(legacy_model_run_dir, "final"), os.path.join(paths["run_dir"], "final"))  # 状态或中间变量：调试时多观察它的值如何流动
        copied += _copy_tree_missing(  # 状态或中间变量：调试时多观察它的值如何流动
            os.path.join(legacy_model_run_dir, "interrupted"),  # 代码执行语句：结合上下文理解它对后续流程的影响
            os.path.join(paths["run_dir"], "interrupted"),  # 代码执行语句：结合上下文理解它对后续流程的影响
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    if os.path.normpath(legacy_log_run_dir) != os.path.normpath(paths["log_run_dir"]):  # 条件分支：学习时先看触发条件，再看两边行为差异
        copied += _copy_tree_missing(legacy_log_run_dir, paths["log_run_dir"])  # 状态或中间变量：调试时多观察它的值如何流动

    if not _path_exists_with_optional_zip(paths["final_model"]) and _path_exists_with_optional_zip(paths["legacy_final_model"]):  # 条件分支：学习时先看触发条件，再看两边行为差异
        os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)  # 状态或中间变量：调试时多观察它的值如何流动
        legacy_model_file = paths["legacy_final_model"] if os.path.exists(paths["legacy_final_model"]) else paths["legacy_final_model"] + ".zip"  # 状态或中间变量：调试时多观察它的值如何流动
        final_model_file = paths["final_model"] if not legacy_model_file.endswith(".zip") else paths["final_model"] + ".zip"  # 状态或中间变量：调试时多观察它的值如何流动
        shutil.copy2(legacy_model_file, final_model_file)  # 代码执行语句：结合上下文理解它对后续流程的影响
        copied += 1  # 状态或中间变量：调试时多观察它的值如何流动
    if not os.path.exists(paths["final_norm"]) and os.path.exists(paths["legacy_final_norm"]):  # 条件分支：学习时先看触发条件，再看两边行为差异
        os.makedirs(os.path.dirname(paths["final_norm"]), exist_ok=True)  # 状态或中间变量：调试时多观察它的值如何流动
        shutil.copy2(paths["legacy_final_norm"], paths["final_norm"])  # 代码执行语句：结合上下文理解它对后续流程的影响
        copied += 1  # 状态或中间变量：调试时多观察它的值如何流动

    if copied > 0:  # 条件分支：学习时先看触发条件，再看两边行为差异
        print(f"已将 {copied} 个旧版产物同步到当前分层目录: {paths['run_dir']}")  # 代码执行语句：结合上下文理解它对后续流程的影响


def _resolve_test_artifact_paths(args: TrainArgs, paths: dict[str, str]) -> tuple[str, str]:  # 函数定义：先看输入输出，再理解内部控制流
    """为测试模式解析模型与归一化路径，优先使用规范分层目录。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    model_candidates = [paths["final_model"], paths["interrupted_model"], paths["legacy_final_model"]]  # 状态或中间变量：调试时多观察它的值如何流动
    norm_candidates = [paths["final_norm"], paths["interrupted_norm"], paths["legacy_final_norm"]]  # 状态或中间变量：调试时多观察它的值如何流动

    if args.model_path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        model_path = args.model_path  # 状态或中间变量：调试时多观察它的值如何流动
    else:  # 兜底分支：当前面条件都不满足时走这里
        model_path = next((path for path in model_candidates if _path_exists_with_optional_zip(path)), model_candidates[0])  # 状态或中间变量：调试时多观察它的值如何流动
    if args.normalize_path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        norm_path = args.normalize_path  # 状态或中间变量：调试时多观察它的值如何流动
    else:  # 兜底分支：当前面条件都不满足时走这里
        norm_path = next((path for path in norm_candidates if os.path.exists(path)), norm_candidates[0])  # 状态或中间变量：调试时多观察它的值如何流动
    return model_path, norm_path  # 把当前结果返回给上层调用方


def _make_train_vec_env(args: TrainArgs, render_mode: str | None):  # 函数定义：先看输入输出，再理解内部控制流
    """按并行数自动选择训练用 VecEnv 实现。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    vec_env_cls = DummyVecEnv  # 状态或中间变量：调试时多观察它的值如何流动
    vec_env_kwargs: dict[str, object] = {}  # 状态或中间变量：调试时多观察它的值如何流动
    selected_start_method: str | None = None  # 状态或中间变量：调试时多观察它的值如何流动
    if int(args.n_envs) > 1 and not args.render:  # 条件分支：学习时先看触发条件，再看两边行为差异
        start_methods = ("fork", "forkserver", "spawn")  # 状态或中间变量：调试时多观察它的值如何流动
        last_error: Exception | None = None  # 状态或中间变量：调试时多观察它的值如何流动
        for start_method in start_methods:  # 循环逻辑：关注迭代对象、次数和循环体副作用
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                env = make_vec_env(  # 向量环境创建：并行环境数和包装方式在这里生效
                    ENV_ID,  # 代码执行语句：结合上下文理解它对后续流程的影响
                    n_envs=args.n_envs,  # 状态或中间变量：调试时多观察它的值如何流动
                    seed=args.seed,  # 状态或中间变量：调试时多观察它的值如何流动
                    env_kwargs=_make_env(args, render_mode),  # 运行配置：这类参数会直接改变执行路径或调试体验
                    vec_env_cls=SubprocVecEnv,  # 状态或中间变量：调试时多观察它的值如何流动
                    vec_env_kwargs={"start_method": start_method},  # 状态或中间变量：调试时多观察它的值如何流动
                )  # 收束上一段结构，阅读时回看上面的参数或元素
                vec_env_cls = SubprocVecEnv  # 状态或中间变量：调试时多观察它的值如何流动
                vec_env_kwargs = {"start_method": start_method}  # 状态或中间变量：调试时多观察它的值如何流动
                selected_start_method = start_method  # 状态或中间变量：调试时多观察它的值如何流动
                print(f"训练环境包装器: SubprocVecEnv (n_envs={args.n_envs}, start_method={start_method})")  # 状态或中间变量：调试时多观察它的值如何流动
                return env  # 把当前结果返回给上层调用方
            except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
                last_error = e  # 状态或中间变量：调试时多观察它的值如何流动
                print(f"SubprocVecEnv 启动失败（start_method={start_method}）: {e}")  # 状态或中间变量：调试时多观察它的值如何流动
        print(f"SubprocVecEnv 初始化失败，自动回退到 DummyVecEnv: {last_error}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    elif int(args.n_envs) > 1 and args.render:  # 补充分支：用于细化前面条件没有覆盖的情况
        print("render 模式下为避免多进程窗口问题，训练环境保持 DummyVecEnv。")  # 代码执行语句：结合上下文理解它对后续流程的影响

    env = make_vec_env(  # 向量环境创建：并行环境数和包装方式在这里生效
        ENV_ID,  # 代码执行语句：结合上下文理解它对后续流程的影响
        n_envs=args.n_envs,  # 状态或中间变量：调试时多观察它的值如何流动
        seed=args.seed,  # 状态或中间变量：调试时多观察它的值如何流动
        env_kwargs=_make_env(args, render_mode),  # 运行配置：这类参数会直接改变执行路径或调试体验
        vec_env_cls=vec_env_cls,  # 状态或中间变量：调试时多观察它的值如何流动
        vec_env_kwargs=vec_env_kwargs,  # 状态或中间变量：调试时多观察它的值如何流动
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    if selected_start_method is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
        print(f"训练环境包装器: {vec_env_cls.__name__} (n_envs={args.n_envs})")  # 状态或中间变量：调试时多观察它的值如何流动
    return env  # 把当前结果返回给上层调用方


def _load_model(args: TrainArgs, model_path: str, env, device: str):  # 函数定义：先看输入输出，再理解内部控制流
    """按算法类型加载模型（训练恢复/测试共用）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if args.algo == "td3":  # TD3 模型加载分支。
        return TD3.load(  # 把当前结果返回给上层调用方
            model_path,  # 代码执行语句：结合上下文理解它对后续流程的影响
            env=env,  # 状态或中间变量：调试时多观察它的值如何流动
            device=device,  # 运行配置：这类参数会直接改变执行路径或调试体验
            custom_objects={  # 状态或中间变量：调试时多观察它的值如何流动
                "batch_size": args.batch_size,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "buffer_size": args.buffer_size,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "gradient_steps": args.gradient_steps,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "learning_starts": args.learning_starts,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "policy_delay": args.policy_delay,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "target_policy_noise": args.target_policy_noise,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "target_noise_clip": args.target_noise_clip,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            },  # 收束上一段结构，阅读时回看上面的参数或元素
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    if args.algo == "sac":  # SAC 模型加载分支。
        return SAC.load(  # 把当前结果返回给上层调用方
            model_path,  # 代码执行语句：结合上下文理解它对后续流程的影响
            env=env,  # 状态或中间变量：调试时多观察它的值如何流动
            device=device,  # 运行配置：这类参数会直接改变执行路径或调试体验
            custom_objects={  # 状态或中间变量：调试时多观察它的值如何流动
                "batch_size": args.batch_size,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "buffer_size": args.buffer_size,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "gradient_steps": args.gradient_steps,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "learning_starts": args.learning_starts,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            },  # 收束上一段结构，阅读时回看上面的参数或元素
        )  # 收束上一段结构，阅读时回看上面的参数或元素
    return PPO.load(  # PPO 模型加载分支。
        model_path,  # 代码执行语句：结合上下文理解它对后续流程的影响
        env=env,  # 状态或中间变量：调试时多观察它的值如何流动
        device=device,  # 运行配置：这类参数会直接改变执行路径或调试体验
        custom_objects={"batch_size": args.batch_size},  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def _rebuild_replay_buffer_if_needed(model, env) -> None:  # 函数定义：先看输入输出，再理解内部控制流
    """当回放缓存与当前并行环境数不一致时，重建空回放缓存。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if not hasattr(model, "replay_buffer") or model.replay_buffer is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程
    rb = model.replay_buffer  # 状态或中间变量：调试时多观察它的值如何流动
    rb_n_envs = getattr(rb, "n_envs", None)  # 状态或中间变量：调试时多观察它的值如何流动
    env_n_envs = getattr(env, "num_envs", None)  # 状态或中间变量：调试时多观察它的值如何流动
    if rb_n_envs is None or env_n_envs is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程
    if int(rb_n_envs) == int(env_n_envs):  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程
    print(  # 代码执行语句：结合上下文理解它对后续流程的影响
        f"检测到回放缓存并行数不匹配（replay n_envs={rb_n_envs}, current n_envs={env_n_envs}），"  # 状态或中间变量：调试时多观察它的值如何流动
        "将重建空回放缓存以继续训练。"  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    rb_class = getattr(model, "replay_buffer_class", None)  # 状态或中间变量：调试时多观察它的值如何流动
    if rb_class is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
        model.replay_buffer = None  # 状态或中间变量：调试时多观察它的值如何流动
        return  # 提前结束当前函数，回到上层流程
    rb_kwargs = dict(getattr(model, "replay_buffer_kwargs", {}) or {})  # 状态或中间变量：调试时多观察它的值如何流动
    rb_kwargs.pop("n_envs", None)  # 代码执行语句：结合上下文理解它对后续流程的影响
    model.replay_buffer = rb_class(  # 状态或中间变量：调试时多观察它的值如何流动
        model.buffer_size,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        model.observation_space,  # 代码执行语句：结合上下文理解它对后续流程的影响
        model.action_space,  # 代码执行语句：结合上下文理解它对后续流程的影响
        device=model.device,  # 运行配置：这类参数会直接改变执行路径或调试体验
        n_envs=int(env_n_envs),  # 状态或中间变量：调试时多观察它的值如何流动
        optimize_memory_usage=getattr(model, "optimize_memory_usage", False),  # 状态或中间变量：调试时多观察它的值如何流动
        handle_timeout_termination=getattr(model, "handle_timeout_termination", True),  # 状态或中间变量：调试时多观察它的值如何流动
        **rb_kwargs,  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def _build_model(args: TrainArgs, env, device: str):  # 函数定义：先看输入输出，再理解内部控制流
    """按参数创建 TD3/SAC/PPO 模型。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if args.algo == "td3":  # TD3 分支。
        n_actions = env.action_space.shape[-1]  # 动作维度（6）。
        action_noise = NormalActionNoise(  # TD3 训练时的高斯动作噪声。
            mean=np.zeros(n_actions, dtype=np.float32),  # 均值向量全 0。
            sigma=float(args.action_noise_sigma) * np.ones(n_actions, dtype=np.float32),  # 每维同样噪声标准差。
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        policy_kwargs = dict(  # 网络结构配置。
            net_arch=dict(  # 分开 actor 和 critic 网络结构。
                pi=[512, 512, 256],  # 策略网络层宽。
                qf=[512, 512, 256],  # Q 网络层宽。
            ),  # 收束上一段结构，阅读时回看上面的参数或元素
            activation_fn=nn.ReLU,  # 激活函数。
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        return TD3(  # 返回 TD3 模型实例。
            "MlpPolicy",  # 多层感知机策略。
            env,  # 训练环境。
            action_noise=action_noise,  # 动作噪声。
            verbose=1,  # 打印训练日志。
            seed=args.seed,  # 随机种子。
            device=device,  # 训练设备。
            learning_rate=3e-4,  # 学习率。
            buffer_size=args.buffer_size,  # 回放池大小。
            learning_starts=args.learning_starts,  # 多少步后开始学习。
            batch_size=args.batch_size,  # batch 大小。
            tau=0.005,  # 软更新系数。
            gamma=0.99,  # 折扣因子。
            train_freq=1,  # 每步训练一次。
            gradient_steps=args.gradient_steps,  # 每次训练做几步梯度更新。
            policy_delay=args.policy_delay,  # actor 延迟更新。
            target_policy_noise=args.target_policy_noise,  # 目标动作噪声。
            target_noise_clip=args.target_noise_clip,  # 目标噪声裁剪。
            policy_kwargs=policy_kwargs,  # 网络配置。
        )  # 收束上一段结构，阅读时回看上面的参数或元素

    if args.algo == "sac":  # SAC 分支。
        policy_kwargs = dict(  # 网络结构配置。
            net_arch=dict(  # SAC 中 pi/qf 分开定义。
                pi=[512, 512, 256],  # 策略网络层宽。
                qf=[512, 512, 256],  # Q 网络层宽。
            ),  # 收束上一段结构，阅读时回看上面的参数或元素
            activation_fn=nn.ReLU,  # 激活函数。
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        return SAC(  # 返回 SAC 模型实例。
            "MlpPolicy",  # MLP 策略。
            env,  # 训练环境。
            verbose=1,  # 输出日志。
            seed=args.seed,  # 随机种子。
            device=device,  # 训练设备。
            learning_rate=3e-4,  # 学习率。
            buffer_size=args.buffer_size,  # 回放池大小。
            learning_starts=args.learning_starts,  # 开始学习步数。
            batch_size=args.batch_size,  # batch 大小。
            tau=0.005,  # 目标网络软更新。
            gamma=0.99,  # 折扣因子。
            train_freq=1,  # 训练频率。
            gradient_steps=args.gradient_steps,  # 梯度步数。
            policy_kwargs=policy_kwargs,  # 网络配置。
        )  # 收束上一段结构，阅读时回看上面的参数或元素

    policy_kwargs = dict(  # PPO 分支网络配置。
        net_arch=dict(  # PPO 用 `vf` 表示 value function 分支。
            pi=[512, 512, 256],  # 策略网络层宽。
            vf=[512, 512, 256],  # 价值网络层宽。
        ),  # 收束上一段结构，阅读时回看上面的参数或元素
        activation_fn=nn.ReLU,  # 激活函数。
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    return PPO(  # 返回 PPO 模型实例。
        "MlpPolicy",  # MLP 策略。
        env,  # 训练环境。
        verbose=1,  # 输出日志。
        seed=args.seed,  # 随机种子。
        device=device,  # 训练设备。
        learning_rate=3e-4,  # 学习率。
        n_steps=2048,  # rollout 长度。
        batch_size=args.batch_size,  # batch 大小。
        gamma=0.99,  # 折扣因子。
        gae_lambda=0.95,  # GAE 参数。
        policy_kwargs=policy_kwargs,  # 网络配置。
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def train(args: TrainArgs):  # 训练主流程：排查训练问题优先顺着这里读
    """训练主流程。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    _apply_zero_original_preset(args)  # zero 机械臂默认套用原始项目预设。
    register_env()  # 注册 Gym 环境 id。
    os.makedirs(args.model_dir, exist_ok=True)  # 创建模型根目录。
    os.makedirs(args.log_dir, exist_ok=True)  # 创建日志根目录。
    paths = _build_run_paths(args)  # 生成按算法/运行名分层的保存路径。
    os.makedirs(paths["run_dir"], exist_ok=True)  # 确保运行目录存在。
    os.makedirs(paths["log_run_dir"], exist_ok=True)  # 确保运行日志目录存在。
    _sync_legacy_run_artifacts(args, paths)  # 先把旧目录补齐到当前分层目录。

    if args.render and args.n_envs > 1:  # 有头渲染时多环境会显著拖慢且显示混乱。
        print("render 时建议 n_envs=1，否则只会显示一个环境画面。")  # 给出提示。
    train_render_mode = args.render_mode if args.render else None  # 训练时是否启用渲染模式。

    device = _normalize_device_arg(args.device)  # 先读取并规范化设备参数。
    if device.startswith("cuda") and not torch.cuda.is_available():  # CUDA 不可用时回退 CPU。
        print("请求使用 cuda，但当前不可用，自动回退到 cpu")  # 打印设备回退提示。
        device = "cpu"  # 设置为 CPU。
    print(f"当前训练设备: {device}")  # 打印最终设备。
    print(f"classic 物理后端: {args.physics_backend}")  # 打印物理后端设置。

    base_env = _make_train_vec_env(args, train_render_mode)  # 创建训练用向量化环境。

    if args.resume:  # 继续训练模式。
        resume_model_path = args.resume_model_path or paths["interrupted_model"]  # 默认从 interrupted 模型恢复。
        resume_norm_path = args.resume_normalize_path or paths["interrupted_norm"]  # 默认从 interrupted 归一化恢复。
        resume_replay_path = args.resume_replay_path or paths["interrupted_replay"]  # 默认从 interrupted 回放缓存恢复。
        # 兼容路径迁移：若当前默认路径不存在，尝试从上一级 models 根目录回退查找旧路径。
        if not _path_exists_with_optional_zip(resume_model_path):  # 条件分支：学习时先看触发条件，再看两边行为差异
            model_dir_parent = os.path.dirname(os.path.normpath(args.model_dir))  # 状态或中间变量：调试时多观察它的值如何流动
            legacy_interrupted_dir = os.path.join(model_dir_parent, args.algo, args.robot, args.run_name, "interrupted")  # 状态或中间变量：调试时多观察它的值如何流动
            legacy_model_path = os.path.join(legacy_interrupted_dir, "model")  # 状态或中间变量：调试时多观察它的值如何流动
            if not args.resume_model_path and (  # 条件分支：学习时先看触发条件，再看两边行为差异
                _path_exists_with_optional_zip(legacy_model_path)  # 代码执行语句：结合上下文理解它对后续流程的影响
            ):  # 代码执行语句：结合上下文理解它对后续流程的影响
                print(f"当前分层路径不存在，已兼容回退到旧路径继续训练: {legacy_model_path}")  # 代码执行语句：结合上下文理解它对后续流程的影响
                resume_model_path = legacy_model_path  # 状态或中间变量：调试时多观察它的值如何流动
                if not args.resume_normalize_path:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    resume_norm_path = os.path.join(legacy_interrupted_dir, "vec_normalize.pkl")  # 状态或中间变量：调试时多观察它的值如何流动
                if not args.resume_replay_path:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    resume_replay_path = os.path.join(legacy_interrupted_dir, "replay_buffer.pkl")  # 状态或中间变量：调试时多观察它的值如何流动
        if not _path_exists_with_optional_zip(resume_model_path):  # 条件分支：学习时先看触发条件，再看两边行为差异
            raise FileNotFoundError(f"未找到继续训练模型: {resume_model_path}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
        if os.path.exists(resume_norm_path):  # 若存在归一化统计，先恢复统计。
            env = VecNormalize.load(resume_norm_path, base_env)  # 归一化相关：训练和测试必须尽量保持统计一致
            env.training = True  # 状态或中间变量：调试时多观察它的值如何流动
            env.norm_reward = True  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        else:  # 兜底分支：当前面条件都不满足时走这里
            print(f"未找到归一化文件，使用新统计开始继续训练: {resume_norm_path}")  # 代码执行语句：结合上下文理解它对后续流程的影响
            env = VecNormalize(base_env, norm_obs=True, norm_reward=True)  # 归一化相关：训练和测试必须尽量保持统计一致
        model = _load_model(args, resume_model_path, env, device)  # 加载模型权重并绑定当前环境。
        if args.skip_replay_buffer:  # 条件分支：学习时先看触发条件，再看两边行为差异
            print("已按参数跳过旧回放缓存恢复，将以空回放池继续训练。")  # 代码执行语句：结合上下文理解它对后续流程的影响
        elif hasattr(model, "load_replay_buffer") and os.path.exists(resume_replay_path):  # SAC/TD3 尝试恢复回放缓存。
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                model.load_replay_buffer(resume_replay_path)  # 代码执行语句：结合上下文理解它对后续流程的影响
                print(f"已恢复回放缓存: {resume_replay_path}")  # 代码执行语句：结合上下文理解它对后续流程的影响
            except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
                print(f"回放缓存恢复失败（继续训练）: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
        _rebuild_replay_buffer_if_needed(model, env)  # n_envs 改变时避免回放缓存形状不匹配。
        if hasattr(model, "buffer_size"):  # 条件分支：学习时先看触发条件，再看两边行为差异
            print(f"当前回放池容量上限: {int(model.buffer_size)}")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        print(f"继续训练模型: {resume_model_path}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    else:  # 兜底分支：当前面条件都不满足时走这里
        env = VecNormalize(base_env, norm_obs=True, norm_reward=True)  # 新训练模式初始化归一化封装。
        model = _build_model(args, env, device)  # 创建算法模型。
        if hasattr(model, "buffer_size"):  # 条件分支：学习时先看触发条件，再看两边行为差异
            print(f"当前回放池容量上限: {int(model.buffer_size)}")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    try:  # 异常保护：把高风险调用包起来，避免主流程中断
        probe = torch.zeros(1, device=device)  # 创建设备探针张量验证设备可写。
        print(f"设备探针张量所在设备: {probe.device}")  # 打印探针设备。
        del probe  # 释放探针变量。
    except Exception as e:  # 探针失败不阻塞训练。
        print(f"设备探针失败: {e}")  # 打印异常信息。

    interrupt_callback = ManualInterruptCallback(  # Ctrl+C 时保存中断模型。
        interrupted_model_path=paths["interrupted_model"],  # 中断模型路径（按算法独立）。
        interrupted_norm_path=paths["interrupted_norm"],  # 中断归一化路径（按算法独立）。
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    callbacks: list[BaseCallback] = [interrupt_callback]  # 默认先启用中断回调（保证可快停）。
    eval_env = None  # 评估关闭时保持 None，避免额外开销。
    if args.eval_freq > 0:  # 评估频率大于 0 时才创建评估环境与回调。
        eval_env = make_vec_env(  # 创建评估环境（与训练环境分离）。
            ENV_ID,  # 环境 id。
            n_envs=1,  # 评估只需 1 个环境。
            seed=args.seed + 1,  # 用不同种子。
            env_kwargs=_make_env(args, None),  # 评估阶段不启用渲染。
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # 评估只归一化观测。
        eval_env.obs_rms = env.obs_rms  # 共享训练环境的观测统计量。

        best_model_save_path = None  # 状态或中间变量：调试时多观察它的值如何流动
        if args.save_best_model:  # 条件分支：学习时先看触发条件，再看两边行为差异
            os.makedirs(paths["best_model_dir"], exist_ok=True)  # 创建最优模型目录。
            best_model_save_path = paths["best_model_dir"]  # 状态或中间变量：调试时多观察它的值如何流动
        eval_callback = EvalCallback(  # SB3 评估回调。
            eval_env,  # 评估环境。
            best_model_save_path=best_model_save_path,  # 可选：最优模型保存目录。
            log_path=paths["log_run_dir"],  # 评估日志目录（按算法独立）。
            eval_freq=max(args.eval_freq, 1),  # 防止评估频率为 0。
            n_eval_episodes=max(args.n_eval_episodes, 1),  # 每次评估回合数。
            deterministic=True,  # 评估用确定性策略。
            render=False,  # 评估时不渲染。
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        callbacks = [eval_callback, interrupt_callback]  # 启用评估与中断回调。
        if args.save_best_model:  # 条件分支：学习时先看触发条件，再看两边行为差异
            save_norm_callback = SaveVecNormalizeCallback(  # 最优模型刷新时同步保存 VecNormalize。
                eval_callback=eval_callback,  # 关联评估回调。
                save_path=paths["best_norm"],  # 归一化统计文件路径。
                verbose=1,  # 打印保存日志。
            )  # 收束上一段结构，阅读时回看上面的参数或元素
            callbacks = [eval_callback, save_norm_callback, interrupt_callback]  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            print("评估已开启，但已禁用 best_model 保存（--no-save-best-model）。")  # 代码执行语句：结合上下文理解它对后续流程的影响
    else:  # 兜底分支：当前面条件都不满足时走这里
        print("评估回调已关闭（--eval-freq 0），训练结束会更快退出。")  # 提示当前是快速退出模式。
    if args.render:  # 开启训练渲染时追加渲染回调。
        callbacks.append(RenderDuringTrainingCallback(render_freq=args.render_freq))  # 按频率刷新窗口。

    print("开始训练...")  # 训练启动提示。
    start_t = time.time()  # 记录开始时间。
    model.learn(  # 进入 SB3 训练主循环。
        total_timesteps=max(args.timesteps, 1),  # 防止传入 0 步。
        callback=callbacks,  # 回调列表。
        log_interval=max(args.log_interval, 1),  # 防止日志间隔为 0。
        progress_bar=True,  # 显示进度条。
        reset_num_timesteps=not args.resume,  # 继续训练时保留时间轴。
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    elapsed = time.time() - start_t  # 计算训练耗时。

    if interrupt_callback.interrupted:  # 若是中断退出，不写 final，避免与完整训练混淆。
        if hasattr(model, "save_replay_buffer"):  # SAC/TD3 中断时额外保存回放缓存。
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                os.makedirs(os.path.dirname(paths["interrupted_replay"]), exist_ok=True)  # 状态或中间变量：调试时多观察它的值如何流动
                model.save_replay_buffer(paths["interrupted_replay"])  # 代码执行语句：结合上下文理解它对后续流程的影响
            except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
                print(f"中断回放缓存保存失败（可忽略）: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
        env.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
        if eval_env is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            eval_env.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
        print(f"训练已中断，总耗时 {elapsed:.2f}s")  # 代码执行语句：结合上下文理解它对后续流程的影响
        print(f"中断模型路径: {paths['interrupted_model']}.zip")  # 代码执行语句：结合上下文理解它对后续流程的影响
        print(f"中断归一化路径: {paths['interrupted_norm']}")  # 代码执行语句：结合上下文理解它对后续流程的影响
        return  # 提前结束当前函数，回到上层流程

    os.makedirs(os.path.dirname(paths["final_model"]), exist_ok=True)  # 创建 final 目录。
    env.save(paths["final_norm"])  # 保存训练后归一化统计。
    model.save(paths["final_model"])  # 保存最终模型。
    if hasattr(model, "save_replay_buffer"):  # SAC/TD3 额外保存回放缓存，便于下次继续训练。
        try:  # 异常保护：把高风险调用包起来，避免主流程中断
            model.save_replay_buffer(paths["final_replay"])  # 代码执行语句：结合上下文理解它对后续流程的影响
        except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
            print(f"最终回放缓存保存失败（可忽略）: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
    env.close()  # 关闭训练环境。
    if eval_env is not None:  # 仅在创建了评估环境时关闭。
        eval_env.close()  # 关闭评估环境。

    print(f"训练完成，总耗时 {elapsed:.2f}s")  # 输出总耗时。
    print(f"最终模型路径: {paths['final_model']}.zip")  # 输出模型路径。
    print(f"最终归一化路径: {paths['final_norm']}")  # 输出归一化路径。
    if args.save_best_model:  # 条件分支：学习时先看触发条件，再看两边行为差异
        print(f"最优模型目录: {paths['best_model_dir']}")  # 输出最优模型目录。
    else:  # 兜底分支：当前面条件都不满足时走这里
        print("最优模型保存: 已关闭（仅保留评估日志）。")  # 代码执行语句：结合上下文理解它对后续流程的影响


def test(args: TrainArgs):  # 测试主流程：验收模型表现时会走这里
    """测试主流程（确定性策略推理）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    _apply_zero_original_preset(args)  # zero 测试也保持与训练同一套默认语义。
    register_env()  # 注册环境。
    paths = _build_run_paths(args)  # 按算法/运行名生成标准路径。
    _sync_legacy_run_artifacts(args, paths)  # 测试前也先把旧目录补齐到当前分层目录。
    model_path, norm_path = _resolve_test_artifact_paths(args, paths)  # 自动解析 final/interrupted 路径。

    device = _normalize_device_arg(args.device)  # 读取并规范化目标设备。
    if device.startswith("cuda") and not torch.cuda.is_available():  # CUDA 不可用时回退。
        print("请求使用 cuda，但当前不可用，自动回退到 cpu")  # 提示回退。
        device = "cpu"  # 使用 CPU。
    print(f"classic 物理后端: {args.physics_backend}")  # 打印物理后端设置。

    env = make_vec_env(  # 创建测试环境。
        ENV_ID,  # 环境 id。
        n_envs=1,  # 测试阶段 1 环境即可。
        seed=args.seed + 2,  # 与训练/评估使用不同种子。
        env_kwargs=_make_env(args, args.render_mode if args.render else None),  # 测试时按 render 开关决定是否创建渲染上下文。
    )  # 收束上一段结构，阅读时回看上面的参数或元素
    if os.path.exists(norm_path):  # 有归一化文件时加载。
        env = VecNormalize.load(norm_path, env)  # 载入归一化统计。
        env.training = False  # 推理时不再更新统计。
        env.norm_reward = False  # 推理时关闭奖励归一化。

    model = _load_model(args, model_path, env, device)  # 按算法类型加载模型。

    rewards = []  # 存放每回合总奖励。
    for ep in range(max(args.episodes, 1)):  # 按回合循环推理。
        obs = env.reset()  # VecEnv reset 返回 batched obs。
        done = np.array([False], dtype=bool)  # VecEnv done 也是批量布尔数组。
        total = 0.0  # 当前回合累计奖励。
        steps = 0  # 当前回合步数计数。
        for _ in range(max(args.max_steps, 1)):  # 每回合最多执行 max_steps。
            if bool(done[0]):  # 第 0 个环境 done 则结束本回合。
                break  # 跳出步循环。
            action, _states = model.predict(obs, deterministic=True)  # 用确定性策略输出动作。
            obs, reward, done, _info = env.step(action)  # VecEnv step 返回 4 元组。
            total += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)  # 兼容标量/数组奖励。
            if args.render:  # 条件分支：学习时先看触发条件，再看两边行为差异
                env.render()  # 刷新可视化窗口。
            if args.render and args.render_mode == "human":  # human 模式下稍微 sleep，避免画面过快。
                time.sleep(0.01)  # 10ms 间隔。
            steps += 1  # 步数 +1。
        print(f"第 {ep + 1} 回合: 步数={steps}, 奖励={total:.3f}")  # 打印单回合结果。
        rewards.append(total)  # 保存单回合总奖励。
    env.close()  # 关闭测试环境。
    if rewards:  # 非空时计算平均奖励。
        print(f"平均奖励: {float(np.mean(rewards)):.3f}")  # 打印平均奖励。


def parse_args() -> TrainArgs:  # 命令行解析入口：外部调参首先会影响这里
    """解析命令行参数并组装成 TrainArgs。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    p = argparse.ArgumentParser(description="纯 MuJoCo UR5 训练脚本（Gym + SB3）")  # 创建参数解析器。
    p.add_argument("--test", action="store_true")  # 进入测试模式。
    p.add_argument("--algo", choices=["sac", "ppo", "td3"], default="sac")  # 选择算法。
    p.add_argument("--timesteps", type=int, default=5_000_000)  # 训练步数。
    p.add_argument("--episodes", type=int, default=5000)  # 测试回合数。
    p.add_argument("--seed", type=int, default=42)  # 随机种子。
    p.add_argument("--n-envs", type=int, default=8)  # 并行环境数（UR5 默认优化值）。
    p.add_argument("--device", type=str, default="cuda")  # 设备。
    p.add_argument("--render", action="store_true")  # 显式打开渲染。
    p.add_argument("--no-render", action="store_false", dest="render")  # 显式关闭渲染。
    p.set_defaults(render=False)  # 默认训练不渲染。
    p.add_argument("--render-mode", choices=["human", "rgb_array"], default="human")  # 渲染模式。
    p.add_argument("--model-dir", type=str, default="models/classic")  # 模型目录。
    p.add_argument("--log-dir", type=str, default="logs/classic")  # 日志目录。
    p.add_argument("--run-name", type=str, default="ur5_mujoco")  # 运行名称。
    p.add_argument("--eval-freq", type=int, default=5000)  # 评估频率。
    p.add_argument("--n-eval-episodes", type=int, default=1)  # 每次评估回合数。
    p.add_argument("--save-best-model", action="store_true", dest="save_best_model")  # 开启最优模型保存。
    p.add_argument("--no-save-best-model", action="store_false", dest="save_best_model")  # 关闭最优模型保存。
    p.set_defaults(save_best_model=True)  # 状态或中间变量：调试时多观察它的值如何流动
    p.add_argument("--render-freq", type=int, default=1)  # 渲染频率。
    p.add_argument("--log-interval", type=int, default=1000)  # 日志间隔。
    p.add_argument("--batch-size", type=int, default=512)  # batch 大小（UR5 默认优化值）。
    p.add_argument("--buffer-size", type=int, default=200_000)  # SAC/TD3 回放池容量（16G 内存友好）。
    p.add_argument("--gradient-steps", type=int, default=4)  # 梯度步数（UR5 默认优化值）。
    p.add_argument("--learning-starts", type=int, default=20000)  # 开始学习步数。
    p.add_argument("--action-noise-sigma", type=float, default=2.5)  # TD3 动作噪声。
    p.add_argument("--policy-delay", type=int, default=4)  # TD3 actor 延迟更新。
    p.add_argument("--target-policy-noise", type=float, default=0.2)  # TD3 目标策略噪声。
    p.add_argument("--target-noise-clip", type=float, default=0.5)  # TD3 目标噪声裁剪。
    p.add_argument("--max-steps", type=int, default=3000)  # 单回合最大步数。
    p.add_argument("--success-threshold", type=float, default=0.01)  # 成功阈值。
    p.add_argument("--frame-skip", type=int, default=1)  # frame-skip。
    p.add_argument("--physics-backend", choices=["auto", "mujoco", "warp"], default="mujoco")  # 物理后端。
    p.add_argument("--legacy-zero-ee-velocity", action="store_true")  # 兼容 zero 原始 `cvel[:3]` 末端速度读取。
    p.add_argument("--robot", choices=["ur5_cxy", "zero_robotiq"], default="ur5_cxy")  # 机械臂模型选择。
    p.add_argument("--lock-camera", action="store_true", dest="lock_camera")  # 锁定到固定相机。
    p.add_argument("--free-camera", action="store_false", dest="lock_camera")  # 使用自由相机。
    p.set_defaults(lock_camera=False)  # 默认自由相机，支持鼠标拖动视角。
    p.add_argument("--ur5-target-x-min", type=float, default=-0.95)  # UR5 目标范围 x 最小值。
    p.add_argument("--ur5-target-x-max", type=float, default=-0.60)  # UR5 目标范围 x 最大值。
    p.add_argument("--ur5-target-y-min", type=float, default=0.15)  # UR5 目标范围 y 最小值。
    p.add_argument("--ur5-target-y-max", type=float, default=0.50)  # UR5 目标范围 y 最大值。
    p.add_argument("--ur5-target-z-min", type=float, default=0.12)  # UR5 目标范围 z 最小值。
    p.add_argument("--ur5-target-z-max", type=float, default=0.30)  # UR5 目标范围 z 最大值。
    p.add_argument("--zero-target-x-min", type=float, default=-1.00)  # ZERO 目标范围 x 最小值。
    p.add_argument("--zero-target-x-max", type=float, default=-0.62)  # ZERO 目标范围 x 最大值。
    p.add_argument("--zero-target-y-min", type=float, default=0.08)  # ZERO 目标范围 y 最小值。
    p.add_argument("--zero-target-y-max", type=float, default=0.48)  # ZERO 目标范围 y 最大值。
    p.add_argument("--zero-target-z-min", type=float, default=0.10)  # ZERO 目标范围 z 最小值。
    p.add_argument("--zero-target-z-max", type=float, default=0.35)  # ZERO 目标范围 z 最大值。
    p.add_argument("--curriculum-stage1-fixed-episodes", type=int, default=200)  # 阶段 1 回合数。
    p.add_argument("--curriculum-stage2-random-episodes", type=int, default=800)  # 阶段 2 回合数。
    p.add_argument("--curriculum-stage2-range-scale", type=float, default=0.35)  # 阶段 2 范围缩放。
    p.add_argument("--fixed-target-x", type=float, default=None)  # 固定目标 x。
    p.add_argument("--fixed-target-y", type=float, default=None)  # 固定目标 y。
    p.add_argument("--fixed-target-z", type=float, default=None)  # 固定目标 z。
    p.add_argument("--model-path", type=str, default="")  # 测试模型路径覆盖。
    p.add_argument("--normalize-path", type=str, default="")  # 测试归一化路径覆盖。
    p.add_argument("--resume", action="store_true")  # 从已有模型继续训练。
    p.add_argument("--resume-model-path", type=str, default="")  # 继续训练模型路径覆盖。
    p.add_argument("--resume-normalize-path", type=str, default="")  # 继续训练归一化路径覆盖。
    p.add_argument("--resume-replay-path", type=str, default="")  # 继续训练回放缓存路径覆盖。
    p.add_argument("--skip-replay-buffer", action="store_true")  # 继续训练时跳过旧回放缓存恢复。
    ns = p.parse_args()  # 解析命令行为 Namespace。
    return TrainArgs(  # 显式映射到 dataclass，避免隐藏字段错误。
        test=ns.test,  # 测试模式开关。
        algo=ns.algo,  # 算法名。
        timesteps=ns.timesteps,  # 训练步数。
        episodes=ns.episodes,  # 回合数。
        seed=ns.seed,  # 种子。
        n_envs=ns.n_envs,  # 并行环境数。
        device=ns.device,  # 设备。
        render=ns.render,  # 渲染开关。
        render_mode=ns.render_mode,  # 渲染模式。
        model_dir=ns.model_dir,  # 模型目录。
        log_dir=ns.log_dir,  # 日志目录。
        run_name=ns.run_name,  # 运行名。
        eval_freq=ns.eval_freq,  # 评估频率。
        n_eval_episodes=ns.n_eval_episodes,  # 每次评估回合数。
        save_best_model=ns.save_best_model,  # 评估时是否保存 best_model。
        render_freq=ns.render_freq,  # 渲染频率。
        log_interval=ns.log_interval,  # 日志间隔。
        batch_size=ns.batch_size,  # batch 大小。
        buffer_size=ns.buffer_size,  # 回放池容量。
        gradient_steps=ns.gradient_steps,  # 梯度步数。
        learning_starts=ns.learning_starts,  # 开始学习步数。
        action_noise_sigma=ns.action_noise_sigma,  # TD3 动作噪声。
        policy_delay=ns.policy_delay,  # TD3 actor 延迟更新。
        target_policy_noise=ns.target_policy_noise,  # TD3 目标策略噪声。
        target_noise_clip=ns.target_noise_clip,  # TD3 目标噪声裁剪。
        max_steps=ns.max_steps,  # 单回合最大步数。
        success_threshold=ns.success_threshold,  # 成功阈值。
        frame_skip=ns.frame_skip,  # frame-skip。
        physics_backend=ns.physics_backend,  # 物理后端。
        legacy_zero_ee_velocity=ns.legacy_zero_ee_velocity,  # 是否兼容 zero 原始末端速度读取。
        robot=ns.robot,  # 机械臂模型选择。
        lock_camera=ns.lock_camera,  # 是否锁定固定相机。
        ur5_target_x_min=ns.ur5_target_x_min,  # UR5 目标范围 x 最小值。
        ur5_target_x_max=ns.ur5_target_x_max,  # UR5 目标范围 x 最大值。
        ur5_target_y_min=ns.ur5_target_y_min,  # UR5 目标范围 y 最小值。
        ur5_target_y_max=ns.ur5_target_y_max,  # UR5 目标范围 y 最大值。
        ur5_target_z_min=ns.ur5_target_z_min,  # UR5 目标范围 z 最小值。
        ur5_target_z_max=ns.ur5_target_z_max,  # UR5 目标范围 z 最大值。
        zero_target_x_min=ns.zero_target_x_min,  # ZERO 目标范围 x 最小值。
        zero_target_x_max=ns.zero_target_x_max,  # ZERO 目标范围 x 最大值。
        zero_target_y_min=ns.zero_target_y_min,  # ZERO 目标范围 y 最小值。
        zero_target_y_max=ns.zero_target_y_max,  # ZERO 目标范围 y 最大值。
        zero_target_z_min=ns.zero_target_z_min,  # ZERO 目标范围 z 最小值。
        zero_target_z_max=ns.zero_target_z_max,  # ZERO 目标范围 z 最大值。
        curriculum_stage1_fixed_episodes=ns.curriculum_stage1_fixed_episodes,  # 阶段 1 回合数。
        curriculum_stage2_random_episodes=ns.curriculum_stage2_random_episodes,  # 阶段 2 回合数。
        curriculum_stage2_range_scale=ns.curriculum_stage2_range_scale,  # 阶段 2 范围缩放。
        fixed_target_x=ns.fixed_target_x,  # 固定目标 x。
        fixed_target_y=ns.fixed_target_y,  # 固定目标 y。
        fixed_target_z=ns.fixed_target_z,  # 固定目标 z。
        model_path=ns.model_path,  # 模型路径覆盖。
        normalize_path=ns.normalize_path,  # 归一化路径覆盖。
        resume=ns.resume,  # 是否继续训练。
        resume_model_path=ns.resume_model_path,  # 继续训练模型路径覆盖。
        resume_normalize_path=ns.resume_normalize_path,  # 继续训练归一化路径覆盖。
        resume_replay_path=ns.resume_replay_path,  # 继续训练回放缓存路径覆盖。
        skip_replay_buffer=ns.skip_replay_buffer,  # 是否跳过旧回放缓存恢复。
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def main():  # 脚本入口：先看它如何把各个步骤串起来
    """脚本入口函数。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    args = parse_args()  # 解析命令行参数。
    if args.test:  # 根据模式选择测试或训练。
        test(args)  # 运行测试流程。
    else:  # 兜底分支：当前面条件都不满足时走这里
        train(args)  # 运行训练流程。


if __name__ == "__main__":  # 条件分支：学习时先看触发条件，再看两边行为差异
    main()  # 直接运行脚本时进入主函数。
