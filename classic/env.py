"""纯 MuJoCo 版 UR5 + CXY Robotiq 到点环境。

该文件是完全独立实现：
1) 不依赖 ROS
2) 不依赖 Gazebo
3) 直接实现 Gymnasium 环境接口

学习入口：
- `MujocoEnvConfig`：实验参数集中管理
- `UR5MujocoEnv.reset`：每回合重置与目标采样
- `UR5MujocoEnv.step`：扭矩控制与奖励计算

学习建议（按阅读顺序）：
1) 先看 `__init__`：理解 MuJoCo 的 model/data 与动作观测定义。
2) 再看 `reset`：理解 Gymnasium 每回合初始化流程。
3) 最后看 `step`：理解控制、仿真推进、奖励、终止判定。
"""

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import math  # 依赖导入：先认清这个文件需要哪些外部能力
import threading  # 依赖导入：先认清这个文件需要哪些外部能力
from dataclasses import dataclass  # 依赖导入：先认清这个文件需要哪些外部能力
from pathlib import Path  # 依赖导入：先认清这个文件需要哪些外部能力
from typing import Optional  # 依赖导入：先认清这个文件需要哪些外部能力

import gymnasium as gym  # 依赖导入：先认清这个文件需要哪些外部能力
import mujoco  # 依赖导入：先认清这个文件需要哪些外部能力
import numpy as np  # 依赖导入：先认清这个文件需要哪些外部能力
from gymnasium import spaces  # 依赖导入：先认清这个文件需要哪些外部能力
try:  # 异常保护：把高风险调用包起来，避免主流程中断
    import mujoco_warp as mjwarp  # 依赖导入：先认清这个文件需要哪些外部能力
    import warp as wp  # 依赖导入：先认清这个文件需要哪些外部能力

    WARP_IMPORT_ERROR: Exception | None = None  # 状态或中间变量：调试时多观察它的值如何流动
except Exception as exc:  # pragma: no cover - 依赖本机运行环境
    mjwarp = None  # 状态或中间变量：调试时多观察它的值如何流动
    wp = None  # 状态或中间变量：调试时多观察它的值如何流动
    WARP_IMPORT_ERROR = exc  # 状态或中间变量：调试时多观察它的值如何流动

try:  # 异常保护：把高风险调用包起来，避免主流程中断
    # `mujoco.viewer` 需要显式导入，不能假设 `mujoco` 顶层一定挂载 viewer 子模块。
    import mujoco.viewer as mj_viewer  # 依赖导入：先认清这个文件需要哪些外部能力
except Exception:  # 异常分支：排查失败时先看这里如何兜底
    mj_viewer = None  # 状态或中间变量：调试时多观察它的值如何流动


ENV_ID = "UR5MujocoReach-v0"  # 状态或中间变量：调试时多观察它的值如何流动
_WARP_INITIALIZED = False  # 状态或中间变量：调试时多观察它的值如何流动


def _warp_available() -> bool:  # 函数定义：先看输入输出，再理解内部控制流
    return mjwarp is not None and wp is not None  # 把当前结果返回给上层调用方


def _ensure_warp_runtime() -> None:  # 函数定义：先看输入输出，再理解内部控制流
    global _WARP_INITIALIZED  # 代码执行语句：结合上下文理解它对后续流程的影响
    if not _warp_available():  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise RuntimeError(f"请求使用 MuJoCo Warp，但当前不可用: {WARP_IMPORT_ERROR!r}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
    if _WARP_INITIALIZED:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程
    wp.init()  # 代码执行语句：结合上下文理解它对后续流程的影响
    _WARP_INITIALIZED = True  # 状态或中间变量：调试时多观察它的值如何流动


def _resolve_physics_backend(requested: str) -> str:  # 函数定义：先看输入输出，再理解内部控制流
    choice = str(requested or "auto").lower()  # 状态或中间变量：调试时多观察它的值如何流动
    if choice not in {"auto", "mujoco", "warp"}:  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise ValueError(f"不支持的物理后端: {requested}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
    if choice == "mujoco":  # 条件分支：学习时先看触发条件，再看两边行为差异
        return "mujoco"  # 把当前结果返回给上层调用方
    if choice == "warp":  # 条件分支：学习时先看触发条件，再看两边行为差异
        if not _warp_available():  # 条件分支：学习时先看触发条件，再看两边行为差异
            raise RuntimeError(f"请求使用 MuJoCo Warp，但当前不可用: {WARP_IMPORT_ERROR!r}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
        return "warp"  # 把当前结果返回给上层调用方
    # auto：优先 warp，不可用则回退经典 MuJoCo
    return "warp" if _warp_available() else "mujoco"  # 把当前结果返回给上层调用方


@dataclass  # 用 dataclass 收拢配置，调参时优先回到这里
class MujocoEnvConfig:  # 类定义：先理解职责边界，再进入方法细节
    """到点任务参数配置。"""  # 代码执行语句：结合上下文理解它对后续流程的影响

    # 使用 cxy1997/Robotiq-UR5 的模型。
    model_xml: str = "assets/robotiq_cxy/lab_env.xml"  # 状态或中间变量：调试时多观察它的值如何流动
    frame_skip: int = 1  # 状态或中间变量：调试时多观察它的值如何流动
    max_steps: int = 3000  # 状态或中间变量：调试时多观察它的值如何流动
    success_threshold: float = 0.01  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    # 目标采样空间（按当前模型可达区设置）。
    target_x_min: float = -0.95  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    target_x_max: float = -0.60  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    target_y_min: float = 0.15  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    target_y_max: float = 0.50  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    target_z_min: float = 0.12  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    target_z_max: float = 0.30  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    # 课程学习（自动分阶段，不需要手动切换）：
    # 第 1 阶段：固定目标点，先学会“基本靠近”。
    curriculum_stage1_fixed_episodes: int = 200  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    # 第 2 阶段：小范围随机，逐步增加泛化难度。
    curriculum_stage2_random_episodes: int = 800  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    # 第 2 阶段随机范围缩放比例（相对全范围的一半宽度）。
    curriculum_stage2_range_scale: float = 0.35  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    # 第 1 阶段固定目标点（None 表示自动使用采样空间中心）。
    fixed_target_x: Optional[float] = None  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    fixed_target_y: Optional[float] = None  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    fixed_target_z: Optional[float] = None  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    # zero 原始项目兼容模式：启用原始目标采样与奖励公式。
    zero_original_mode: bool = False  # 状态或中间变量：调试时多观察它的值如何流动
    # 物理后端：mujoco（经典 CPU）/warp（MuJoCo Warp）/auto（优先 warp）。
    physics_backend: str = "mujoco"  # 运行配置：这类参数会直接改变执行路径或调试体验
    # 兼容 zero 原始项目：沿用旧版 `cvel[:3]` 作为末端速度。
    # 默认关闭，改为使用“两指中心点”的有限差分线速度。
    legacy_zero_ee_velocity: bool = False  # 状态或中间变量：调试时多观察它的值如何流动

    # 扭矩动作范围（与 zero-arm 一致）。
    torque_low: float = -15.0  # 状态或中间变量：调试时多观察它的值如何流动
    torque_high: float = 15.0  # 状态或中间变量：调试时多观察它的值如何流动
    # 到点任务默认保持夹爪打开。
    fixed_gripper_ctrl: float = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
    # 保持 cxy 模型中目标方块的重力电机设置。
    enable_gravity_motors: bool = True  # 状态或中间变量：调试时多观察它的值如何流动
    gravity_ctrl: float = -1.0  # 状态或中间变量：调试时多观察它的值如何流动

    # 初始姿态参考原仓库。
    home_pose_mode: str = "ur5_coupled"  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint1: float = math.radians(29.7)  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint2: float = math.radians(-85.0)  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint3: float = math.radians(115.0)  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint4: float = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint5: float = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
    home_joint6: float = 0.0  # 状态或中间变量：调试时多观察它的值如何流动

    # 奖励参数（按 zero-arm 原始风格）。
    step_penalty: float = 0.1  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    base_distance_weight: float = 0.8  # 状态或中间变量：调试时多观察它的值如何流动
    improvement_gain: float = 1.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    regress_gain: float = 0.8  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    speed_penalty_threshold: float = 0.5  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    speed_penalty_value: float = 0.2  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    direction_reward_gain: float = 1.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    joint_vel_change_penalty_gain: float = 0.03  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    phase_thresholds: tuple[float, ...] = (0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    phase_rewards: tuple[float, ...] = (100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    success_bonus: float = 10000.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    success_remaining_step_gain: float = 4.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    success_speed_bonus_very_slow: float = 2000.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    success_speed_bonus_slow: float = 1000.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    success_speed_bonus_medium: float = 500.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    collision_penalty_value: float = 5000.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
    # zero 风格显示：默认自由相机（可鼠标拖动）+ 可选固定相机。
    render_camera_name: str = "workbench_camera"  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_lock_camera: bool = False  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_hide_ui: bool = False  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_azimuth: float = 135.0  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_elevation: float = -22.0  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_distance: float = 1.8  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_lookat_x: float = -0.18  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_lookat_y: float = 0.25  # 状态或中间变量：调试时多观察它的值如何流动
    viewer_fallback_lookat_z: float = 0.28  # 状态或中间变量：调试时多观察它的值如何流动


class UR5MujocoEnv(gym.Env):  # 类定义：先理解职责边界，再进入方法细节
    """基于 CXY UR5+Robotiq 的 24 维状态到点环境。

    观测维度定义（24）：
    [0:3]   目标-末端 相对位置
    [3:9]   6 个关节角
    [9:15]  6 个关节角速度
    [15:21] 上一时刻扭矩动作
    [21:24] 末端线速度
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}  # 运行配置：这类参数会直接改变执行路径或调试体验

    def __init__(self, render_mode: Optional[str] = None, config: Optional[MujocoEnvConfig] = None) -> None:  # 构造函数：实例状态、依赖和默认值都从这里落地
        # Python 面向对象基础：
        # - 子类初始化时，先调用父类构造函数，确保 Gym 内部状态正确建立。
        super().__init__()  # 代码执行语句：结合上下文理解它对后续流程的影响
        # 保存统一配置，后续所有逻辑都从这里读取参数。
        self.config = config or MujocoEnvConfig()  # 状态或中间变量：调试时多观察它的值如何流动
        self.render_mode = render_mode  # 运行配置：这类参数会直接改变执行路径或调试体验
        # Gymnasium 规范：render_mode 在创建环境时确定，不建议中途动态切换。
        if self.render_mode not in (None, "human", "rgb_array"):  # 条件分支：学习时先看触发条件，再看两边行为差异
            raise ValueError(f"不支持的 render_mode={self.render_mode}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
        # viewer: 有头实时窗口；renderer: 离屏渲染（给 `rgb_array`）。
        self.viewer = None  # human 渲染窗口句柄。  
        self.renderer = None  # rgb_array 离屏渲染器句柄。  
        self._target_viz_added = False  # 是否已在 user_scn 中添加目标装饰球。  
        self._target_geom_index: Optional[int] = None  # 目标装饰球在 user_scn.geoms 中的索引。  
        self._render_camera_id = -1  # 状态或中间变量：调试时多观察它的值如何流动
        # 防止每帧重复打印同类渲染报错。
        self._render_error_logged = False  # 状态或中间变量：调试时多观察它的值如何流动

        # 相对当前文件定位模型路径。
        xml_path = Path(__file__).resolve().parents[1] / self.config.model_xml  # 状态或中间变量：调试时多观察它的值如何流动
        if not xml_path.exists():  # 条件分支：学习时先看触发条件，再看两边行为差异
            raise FileNotFoundError(f"未找到 MuJoCo 模型文件: {xml_path}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
        # MuJoCo 关键概念：
        # - MjModel: 静态结构（关节、刚体、几何体、执行器定义等）。
        # - MjData: 动态状态（qpos/qvel/ctrl/contact 等会随时间变化的数据）。
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))  # 状态或中间变量：调试时多观察它的值如何流动
        self.data = mujoco.MjData(self.model)  # 状态或中间变量：调试时多观察它的值如何流动
        self.physics_backend = _resolve_physics_backend(self.config.physics_backend)  # 运行配置：这类参数会直接改变执行路径或调试体验
        self.model_warp = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.data_warp = None  # 状态或中间变量：调试时多观察它的值如何流动
        if self.physics_backend == "warp":  # 条件分支：学习时先看触发条件，再看两边行为差异
            _ensure_warp_runtime()  # 代码执行语句：结合上下文理解它对后续流程的影响
            self.model_warp = mjwarp.put_model(self.model)  # 状态或中间变量：调试时多观察它的值如何流动
            self.data_warp = mjwarp.put_data(self.model, self.data, nworld=1)  # 状态或中间变量：调试时多观察它的值如何流动
        if self.config.render_camera_name:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self._render_camera_id = mujoco.mj_name2id(  # 状态或中间变量：调试时多观察它的值如何流动
                self.model,  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mjtObj.mjOBJ_CAMERA,  # 代码执行语句：结合上下文理解它对后续流程的影响
                self.config.render_camera_name,  # 代码执行语句：结合上下文理解它对后续流程的影响
            )  # 收束上一段结构，阅读时回看上面的参数或元素

        # 机械臂 6 自由度关节与执行器名称映射（来自 cxy 模型命名）。
        self.arm_joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")  # 状态或中间变量：调试时多观察它的值如何流动
        self.arm_actuator_names = (  # 状态或中间变量：调试时多观察它的值如何流动
            "joint1_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "joint2_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "joint3_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "joint4_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "joint5_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
            "joint6_motor",  # 代码执行语句：结合上下文理解它对后续流程的影响
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        self.arm_joint_ids = np.array(  # 状态或中间变量：调试时多观察它的值如何流动
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names],  # 代码执行语句：结合上下文理解它对后续流程的影响
            dtype=np.int32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        self.arm_actuator_ids = np.array(  # 状态或中间变量：调试时多观察它的值如何流动
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.arm_actuator_names],  # 代码执行语句：结合上下文理解它对后续流程的影响
            dtype=np.int32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=np.int32)  # 状态或中间变量：调试时多观察它的值如何流动
        self.arm_qvel_adr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=np.int32)  # 状态或中间变量：调试时多观察它的值如何流动
        # `qposadr`/`dofadr` 是 MuJoCo 中读取关节状态的核心索引。

        # 夹爪执行器 id。
        self.gripper_actuator_ids = np.array(  # 状态或中间变量：调试时多观察它的值如何流动
            [  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "close_1"),  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "close_2"),  # 代码执行语句：结合上下文理解它对后续流程的影响
            ],  # 收束上一段结构，阅读时回看上面的参数或元素
            dtype=np.int32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        # cxy 模型里额外定义的重力补偿执行器 id。
        self.gravity_actuator_ids = np.array(  # 状态或中间变量：调试时多观察它的值如何流动
            [  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_1"),  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_2"),  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_3"),  # 代码执行语句：结合上下文理解它对后续流程的影响
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gravity_4"),  # 代码执行语句：结合上下文理解它对后续流程的影响
            ],  # 收束上一段结构，阅读时回看上面的参数或元素
            dtype=np.int32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素

        # 末端与目标体 id。
        # 训练中我们使用“两指中心点”作为到点参考，而不是腕部中心。
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")  # 状态或中间变量：调试时多观察它的值如何流动
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_follower_link")  # 状态或中间变量：调试时多观察它的值如何流动
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_follower_link")  # 状态或中间变量：调试时多观察它的值如何流动
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body_1")  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.target_x_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_x_1")]  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.target_y_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_y_1")]  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.target_z_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_z_1")]  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.target_ball_qpos_adr = self.model.jnt_qposadr[  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_ball_1")  # 代码执行语句：结合上下文理解它对后续流程的影响
        ]  # 收束上一段结构，阅读时回看上面的参数或元素

        # Gymnasium action_space/observation_space 是算法接口契约：
        # - SB3 会据此检查动作范围、观测维度与数据类型。
        # - 这里动作是 6 维连续值，范围 [-15, 15]。
        self.action_space = spaces.Box(  # 状态或中间变量：调试时多观察它的值如何流动
            low=np.full((6,), self.config.torque_low, dtype=np.float32),  # 状态或中间变量：调试时多观察它的值如何流动
            high=np.full((6,), self.config.torque_high, dtype=np.float32),  # 状态或中间变量：调试时多观察它的值如何流动
            dtype=np.float32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素
        # 状态空间不做硬边界，训练时由 VecNormalize 处理归一化。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)  # 状态或中间变量：调试时多观察它的值如何流动

        # 保存初始快照，用于 reset 回到稳定状态。
        mujoco.mj_forward(self.model, self.data)  # 代码执行语句：结合上下文理解它对后续流程的影响
        self.home_qpos = self.data.qpos.copy()  # 状态或中间变量：调试时多观察它的值如何流动
        self.home_qvel = self.data.qvel.copy()  # 状态或中间变量：调试时多观察它的值如何流动

        # 运行时缓存：每回合/每步更新，避免重复分配数组并支持奖励差分项。
        self.target_pos = np.zeros(3, dtype=np.float32)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.prev_torque = np.zeros(6, dtype=np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_joint_vel = np.zeros(6, dtype=np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_qpos: Optional[np.ndarray] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_qvel: Optional[np.ndarray] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_xpos: Optional[np.ndarray] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_cvel: Optional[np.ndarray] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_ncon: int = 0  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_ee_pos: Optional[np.ndarray] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.current_ee_vel = np.zeros(3, dtype=np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_distance: Optional[float] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.min_distance: Optional[float] = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.step_count = 0  # 状态或中间变量：调试时多观察它的值如何流动
        # 课程学习计数器：记录当前环境已经重置过多少个 episode。
        self.episode_count = 0  # 状态或中间变量：调试时多观察它的值如何流动
        # 记录当前 episode 的课程阶段名称，便于日志观察。
        self.curriculum_stage = "stage1_fixed"  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self._phase_rewards_given: set[float] = set()  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    def _set_home_pose(self) -> None:  # 函数定义：先看输入输出，再理解内部控制流
        """设置稳定初始姿态（参考 cxy 原项目）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        self.data.qpos[:] = self.home_qpos  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.qvel[:] = self.home_qvel  # 状态或中间变量：调试时多观察它的值如何流动

        # 先设置前三个主关节角。
        self.data.qpos[self.arm_qpos_adr[0]] = float(self.config.home_joint1)  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.qpos[self.arm_qpos_adr[1]] = float(self.config.home_joint2)  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.qpos[self.arm_qpos_adr[2]] = float(self.config.home_joint3)  # 状态或中间变量：调试时多观察它的值如何流动

        if self.config.home_pose_mode == "direct6":  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.data.qpos[self.arm_qpos_adr[3]] = float(self.config.home_joint4)  # 状态或中间变量：调试时多观察它的值如何流动
            self.data.qpos[self.arm_qpos_adr[4]] = float(self.config.home_joint5)  # 状态或中间变量：调试时多观察它的值如何流动
            self.data.qpos[self.arm_qpos_adr[5]] = float(self.config.home_joint6)  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            # 用与参考实现一致的耦合关系计算后三轴，保证末端朝下并与桌边对齐。
            q1 = self.data.qpos[self.arm_qpos_adr[0]]  # 状态或中间变量：调试时多观察它的值如何流动
            q2 = self.data.qpos[self.arm_qpos_adr[1]]  # 状态或中间变量：调试时多观察它的值如何流动
            q3 = self.data.qpos[self.arm_qpos_adr[2]]  # 状态或中间变量：调试时多观察它的值如何流动
            self.data.qpos[self.arm_qpos_adr[3]] = 1.5 * math.pi - q2 - q3  # 状态或中间变量：调试时多观察它的值如何流动
            self.data.qpos[self.arm_qpos_adr[4]] = 1.5 * math.pi  # 状态或中间变量：调试时多观察它的值如何流动
            self.data.qpos[self.arm_qpos_adr[5]] = 1.25 * math.pi + q1  # 状态或中间变量：调试时多观察它的值如何流动

        # 到点任务中夹爪保持张开，减少抓取动力学干扰。
        self.data.qpos[6:14] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.qvel[:] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动

    def _set_target_xyz(self, x: float, y: float, z: float) -> None:  # 函数定义：先看输入输出，再理解内部控制流
        """把 target_body_1 放到采样坐标，姿态固定为单位四元数。"""  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.data.qpos[self.target_x_qpos_adr] = float(x)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.data.qpos[self.target_y_qpos_adr] = float(y)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.data.qpos[self.target_z_qpos_adr] = float(z)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.data.qpos[self.target_ball_qpos_adr:self.target_ball_qpos_adr + 4] = np.array([1.0, 0.0, 0.0, 0.0])  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

    def _sample_target_pos(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """在可达工作空间内均匀采样目标点。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        return np.array(  # 把当前结果返回给上层调用方
            [  # 代码执行语句：结合上下文理解它对后续流程的影响
                self.np_random.uniform(self.config.target_x_min, self.config.target_x_max),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                self.np_random.uniform(self.config.target_y_min, self.config.target_y_max),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                self.np_random.uniform(self.config.target_z_min, self.config.target_z_max),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            ],  # 收束上一段结构，阅读时回看上面的参数或元素
            dtype=np.float32,  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素

    def _sample_target_pos_zero_original(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """zero 原始项目风格目标采样。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        target = self.np_random.uniform(-0.3, 0.3, size=3).astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        if target[0] >= 0.0:  # 条件分支：学习时先看触发条件，再看两边行为差异
            target[0] = max(target[0], 0.1)  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            target[0] = min(target[0], -0.1)  # 状态或中间变量：调试时多观察它的值如何流动
        if target[1] >= 0.0:  # 条件分支：学习时先看触发条件，再看两边行为差异
            target[1] = max(target[1], 0.1)  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            target[1] = min(target[1], -0.1)  # 状态或中间变量：调试时多观察它的值如何流动
        target[2] = max(target[2], 0.05)  # 状态或中间变量：调试时多观察它的值如何流动
        return target  # 把当前结果返回给上层调用方

    def _sample_target_pos_curriculum(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """按课程学习阶段采样目标点（自动推进）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        # Python 语法说明：
        # - `episode_index` 是当前即将开始的 episode 序号（从 0 开始）。
        # - 这里用 `self.episode_count` 而不是 `self.step_count`，
        #   因为课程学习按“回合”推进更稳定。
        episode_index = int(self.episode_count)  # 状态或中间变量：调试时多观察它的值如何流动

        # 第 1 阶段持续回合数，转为 int 避免外部传 float 带来比较歧义。
        stage1_episodes = max(int(self.config.curriculum_stage1_fixed_episodes), 0)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        # 第 2 阶段持续回合数，转为 int 并限制为非负。
        stage2_episodes = max(int(self.config.curriculum_stage2_random_episodes), 0)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 第 1 阶段：固定目标点。
        if episode_index < stage1_episodes:  # 条件分支：学习时先看触发条件，再看两边行为差异
            # 如果用户没显式指定固定目标，就用采样空间中心点。
            x = (  # 状态或中间变量：调试时多观察它的值如何流动
                float(self.config.fixed_target_x)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                if self.config.fixed_target_x is not None  # 条件分支：学习时先看触发条件，再看两边行为差异
                else 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            )  # 收束上一段结构，阅读时回看上面的参数或元素
            # y 轴同理：优先用用户给定值，否则取采样范围中心。
            y = (  # 状态或中间变量：调试时多观察它的值如何流动
                float(self.config.fixed_target_y)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                if self.config.fixed_target_y is not None  # 条件分支：学习时先看触发条件，再看两边行为差异
                else 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            )  # 收束上一段结构，阅读时回看上面的参数或元素
            # z 轴同理：优先用用户给定值，否则取采样范围中心。
            z = (  # 状态或中间变量：调试时多观察它的值如何流动
                float(self.config.fixed_target_z)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                if self.config.fixed_target_z is not None  # 条件分支：学习时先看触发条件，再看两边行为差异
                else 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            )  # 收束上一段结构，阅读时回看上面的参数或元素
            # 记录阶段名字到实例属性，方便 reset 的 info 输出。
            self.curriculum_stage = "stage1_fixed"  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            # 返回 float32，和 observation/action 的 dtype 保持一致。
            return np.array([x, y, z], dtype=np.float32)  # 把当前结果返回给上层调用方

        # 第 2 阶段：小范围随机（围绕中心点随机，难度比全范围低）。
        if episode_index < stage1_episodes + stage2_episodes:  # 条件分支：学习时先看触发条件，再看两边行为差异
            # 把缩放比例裁剪到 [1e-3, 1.0]，避免传入异常值。
            # - 太小会几乎固定点（学习会过拟合）
            # - 大于 1 会超出全范围（与设计目标不一致）
            scale = float(np.clip(self.config.curriculum_stage2_range_scale, 1e-3, 1.0))  # 限幅操作：调大调小时要关注稳定性和探索范围

            # 先计算 x 的中心与半宽，再按 scale 缩小随机范围。
            x_center = 0.5 * (float(self.config.target_x_min) + float(self.config.target_x_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            x_half = 0.5 * (float(self.config.target_x_max) - float(self.config.target_x_min)) * scale  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            x = self.np_random.uniform(x_center - x_half, x_center + x_half)  # 状态或中间变量：调试时多观察它的值如何流动

            # y 轴同样使用中心 + 缩小半宽的采样方式。
            y_center = 0.5 * (float(self.config.target_y_min) + float(self.config.target_y_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            y_half = 0.5 * (float(self.config.target_y_max) - float(self.config.target_y_min)) * scale  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            y = self.np_random.uniform(y_center - y_half, y_center + y_half)  # 状态或中间变量：调试时多观察它的值如何流动

            # z 轴也做同样处理，保证三维随机策略一致。
            z_center = 0.5 * (float(self.config.target_z_min) + float(self.config.target_z_max))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            z_half = 0.5 * (float(self.config.target_z_max) - float(self.config.target_z_min)) * scale  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            z = self.np_random.uniform(z_center - z_half, z_center + z_half)  # 状态或中间变量：调试时多观察它的值如何流动

            # 记录阶段名称，便于训练日志中识别当前难度区间。
            self.curriculum_stage = "stage2_small_random"  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            # 返回阶段二采样结果。
            return np.array([x, y, z], dtype=np.float32)  # 把当前结果返回给上层调用方

        # 第 3 阶段：全范围随机（最终训练目标）。
        self.curriculum_stage = "stage3_full_random"  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        # 直接复用原有全范围采样函数，保持行为一致、减少重复代码。
        return self._sample_target_pos()  # 把当前结果返回给上层调用方

    def _get_ee_pos(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """读取夹爪两指中心点的位置。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.physics_backend == "warp" and self._warp_xpos is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            left_pos = self._warp_xpos[self.left_finger_body_id]  # 状态或中间变量：调试时多观察它的值如何流动
            right_pos = self._warp_xpos[self.right_finger_body_id]  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            left_pos = self.data.xpos[self.left_finger_body_id]  # 状态或中间变量：调试时多观察它的值如何流动
            right_pos = self.data.xpos[self.right_finger_body_id]  # 状态或中间变量：调试时多观察它的值如何流动
        center = 0.5 * (left_pos + right_pos)  # 状态或中间变量：调试时多观察它的值如何流动
        return center.copy().astype(np.float32)  # 把当前结果返回给上层调用方

    def _get_target_pos(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """读取目标体中心在世界坐标系下的位置。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.physics_backend == "warp" and self._warp_xpos is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return self._warp_xpos[self.target_body_id].copy().astype(np.float32)  # 把当前结果返回给上层调用方
        return self.data.xpos[self.target_body_id].copy().astype(np.float32)  # 把当前结果返回给上层调用方

    def _get_legacy_ee_vel(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """兼容 zero 原始项目的旧版末端速度读取。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.physics_backend == "warp" and self._warp_cvel is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return self._warp_cvel[self.ee_body_id][:3].copy().astype(np.float32)  # 把当前结果返回给上层调用方
        return self.data.cvel[self.ee_body_id][:3].copy().astype(np.float32)  # 把当前结果返回给上层调用方

    def _get_ee_vel(self, _ee_pos: np.ndarray) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """读取末端速度。

        默认使用“两指中心点”的有限差分线速度，避免把 `cvel[:3]`
        的角速度误当作末端线速度；兼容模式下可切回 zero 原始实现。
        """
        if self.config.legacy_zero_ee_velocity:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return self._get_legacy_ee_vel()  # 把当前结果返回给上层调用方
        return self.current_ee_vel.copy().astype(np.float32)  # 把当前结果返回给上层调用方

    def _get_obs(self) -> np.ndarray:  # 函数定义：先看输入输出，再理解内部控制流
        """组装 24 维观测向量。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        ee_pos = self._get_ee_pos()  # 状态或中间变量：调试时多观察它的值如何流动
        target_pos = self._get_target_pos()  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        relative_pos = target_pos - ee_pos  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        if self.physics_backend == "warp" and self._warp_qpos is not None and self._warp_qvel is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            joint_pos = self._warp_qpos[self.arm_qpos_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
            joint_vel = self._warp_qvel[self.arm_qvel_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            joint_pos = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
            joint_vel = self.data.qvel[self.arm_qvel_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        ee_vel = self._get_ee_vel(ee_pos)  # 状态或中间变量：调试时多观察它的值如何流动
        # Gym/SB3 约定观测必须是 numpy 数组；
        # 统一为 float32 可以减少训练时 dtype 转换开销。
        return np.concatenate([relative_pos, joint_pos, joint_vel, self.prev_torque, ee_vel]).astype(np.float32)  # 把当前结果返回给上层调用方

    def _sync_warp_cache(self) -> None:  # 函数定义：先看输入输出，再理解内部控制流
        """只拉取观测与奖励所需的小数组，减少 host<->device 往返。"""  # 运行配置：这类参数会直接改变执行路径或调试体验
        if self.physics_backend != "warp" or self.data_warp is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return  # 提前结束当前函数，回到上层流程
        self._warp_qpos = self.data_warp.qpos.numpy()[0]  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_qvel = self.data_warp.qvel.numpy()[0]  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_xpos = self.data_warp.xpos.numpy()[0]  # 状态或中间变量：调试时多观察它的值如何流动
        self._warp_cvel = self.data_warp.cvel.numpy()[0]  # 状态或中间变量：调试时多观察它的值如何流动
        if getattr(self.data_warp, "nacon", None) is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self._warp_ncon = int(self.data_warp.nacon.numpy()[0])  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            self._warp_ncon = 0  # 状态或中间变量：调试时多观察它的值如何流动

    def reset(self, *, seed: Optional[int] = None, options=None):  # 重置入口：每回合初始化和目标采样都在这里
        """Gymnasium reset：重置场景并采样新目标。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        # Gymnasium 规范：
        # 1) 必须调用 super().reset(seed=seed)
        # 2) 返回值必须是 (obs, info)
        # super().reset 会帮你设置 self.np_random（可复现实验随机源）。
        super().reset(seed=seed)  # 状态或中间变量：调试时多观察它的值如何流动
        # 先做 MuJoCo 硬重置。
        mujoco.mj_resetData(self.model, self.data)  # 代码执行语句：结合上下文理解它对后续流程的影响
        # 再回到确定性的初始机械臂姿态。
        self._set_home_pose()  # 代码执行语句：结合上下文理解它对后续流程的影响

        # zero 原始模式：使用原始采样；否则使用当前课程学习采样。
        if self.config.zero_original_mode:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.target_pos = self._sample_target_pos_zero_original()  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        else:  # 兜底分支：当前面条件都不满足时走这里
            self.target_pos = self._sample_target_pos_curriculum()  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self._set_target_xyz(*self.target_pos.tolist())  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        mujoco.mj_forward(self.model, self.data)  # 代码执行语句：结合上下文理解它对后续流程的影响

        # 清空回合缓存。
        self.prev_torque[:] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_joint_vel[:] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_distance = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.min_distance = None  # 状态或中间变量：调试时多观察它的值如何流动
        self.step_count = 0  # 状态或中间变量：调试时多观察它的值如何流动
        self._phase_rewards_given.clear()  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.current_ee_vel[:] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动

        # 维持参考模型中的重力补偿行为。
        if self.config.enable_gravity_motors:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.ctrl[self.arm_actuator_ids] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
        # 到点任务：夹爪固定为张开值。
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_torque[:] = self.data.ctrl[self.arm_actuator_ids].astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动

        # Gymnasium reset 返回 `(obs, info)`。
        # info 常用于调试信息、额外统计，不直接进入策略网络。
        if self.physics_backend == "warp":  # 条件分支：学习时先看触发条件，再看两边行为差异
            # reset 后把 host 状态同步到 Warp 批量数据容器（nworld=1）。
            self.data_warp = mjwarp.put_data(self.model, self.data, nworld=1)  # 状态或中间变量：调试时多观察它的值如何流动
            self._sync_warp_cache()  # 代码执行语句：结合上下文理解它对后续流程的影响
        self.prev_ee_pos = self._get_ee_pos()  # 状态或中间变量：调试时多观察它的值如何流动
        obs = self._get_obs()  # 状态或中间变量：调试时多观察它的值如何流动
        # Python 字典语法：用键值对传递额外调试信息。
        info = {  # 状态或中间变量：调试时多观察它的值如何流动
            "target_pos": self.target_pos.copy(),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "curriculum_stage": self.curriculum_stage,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "episode_index": int(self.episode_count),  # 代码执行语句：结合上下文理解它对后续流程的影响
            "physics_backend": self.physics_backend,  # 运行配置：这类参数会直接改变执行路径或调试体验
        }  # 收束上一段结构，阅读时回看上面的参数或元素
        # 课程计数器在 episode 初始化完成后 +1。
        self.episode_count += 1  # 状态或中间变量：调试时多观察它的值如何流动
        # 按 Gymnasium 规范返回 (obs, info)。
        return obs, info  # 把当前结果返回给上层调用方

    def step(self, action: np.ndarray):  # 步进入口：动作、仿真、奖励和终止判断都串在这里
        """Gymnasium step：输入 6 维关节扭矩动作。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        # zero-arm 同款：在写入新动作前，先记录“上一时刻控制量/关节速度”。
        self.prev_torque = self.data.ctrl[self.arm_actuator_ids].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        if self.physics_backend == "warp" and self._warp_qvel is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.prev_joint_vel = self._warp_qvel[self.arm_qvel_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            self.prev_joint_vel = self.data.qvel[self.arm_qvel_adr].copy().astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动

        # 1) 动作清洗与限幅。
        torque_cmd = np.asarray(action, dtype=np.float32).reshape(6)  # 状态或中间变量：调试时多观察它的值如何流动
        torque_cmd = np.clip(torque_cmd, float(self.config.torque_low), float(self.config.torque_high))  # 限幅操作：调大调小时要关注稳定性和探索范围

        # 2) 执行扭矩控制。
        # 注意：这一步是“策略输出 -> 物理引擎控制输入”的关键桥梁。
        self.data.ctrl[self.arm_actuator_ids] = torque_cmd  # 状态或中间变量：调试时多观察它的值如何流动
        self.data.ctrl[self.gripper_actuator_ids] = float(self.config.fixed_gripper_ctrl)  # 状态或中间变量：调试时多观察它的值如何流动
        if self.config.enable_gravity_motors:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.data.ctrl[self.gravity_actuator_ids] = float(self.config.gravity_ctrl)  # 状态或中间变量：调试时多观察它的值如何流动

        # 3) 前向仿真（按配置切后端）。
        # `frame_skip` 表示一个 RL step 对应多少个积分步。
        if self.physics_backend == "warp":  # 条件分支：学习时先看触发条件，再看两边行为差异
            ctrl_world = self.data.ctrl[None, :].astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
            wp.copy(self.data_warp.ctrl, wp.array(ctrl_world))  # 代码执行语句：结合上下文理解它对后续流程的影响
            for _ in range(int(self.config.frame_skip)):  # 循环逻辑：关注迭代对象、次数和循环体副作用
                mjwarp.step(self.model_warp, self.data_warp)  # 物理推进核心：状态变化都从这一步开始发生
            self._sync_warp_cache()  # 代码执行语句：结合上下文理解它对后续流程的影响
        else:  # 兜底分支：当前面条件都不满足时走这里
            for _ in range(int(self.config.frame_skip)):  # 循环逻辑：关注迭代对象、次数和循环体副作用
                mujoco.mj_step(self.model, self.data)  # 物理推进核心：状态变化都从这一步开始发生

        # 4) 读取新状态并计算奖励。
        self.step_count += 1  # 状态或中间变量：调试时多观察它的值如何流动
        ee_pos = self._get_ee_pos()  # 状态或中间变量：调试时多观察它的值如何流动
        if self.config.legacy_zero_ee_velocity:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.current_ee_vel = self._get_legacy_ee_vel()  # 状态或中间变量：调试时多观察它的值如何流动
        else:  # 兜底分支：当前面条件都不满足时走这里
            dt = max(float(self.model.opt.timestep) * max(int(self.config.frame_skip), 1), 1e-9)  # 状态或中间变量：调试时多观察它的值如何流动
            if self.prev_ee_pos is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
                self.current_ee_vel[:] = 0.0  # 状态或中间变量：调试时多观察它的值如何流动
            else:  # 兜底分支：当前面条件都不满足时走这里
                self.current_ee_vel = ((ee_pos - self.prev_ee_pos) / dt).astype(np.float32)  # 状态或中间变量：调试时多观察它的值如何流动
        self.prev_ee_pos = ee_pos.copy()  # 状态或中间变量：调试时多观察它的值如何流动
        obs = self._get_obs()  # 状态或中间变量：调试时多观察它的值如何流动
        relative_pos = obs[0:3]  # 状态或中间变量：调试时多观察它的值如何流动
        joint_vel = obs[9:15]  # 状态或中间变量：调试时多观察它的值如何流动
        ee_vel = obs[21:24]  # 状态或中间变量：调试时多观察它的值如何流动

        distance = float(np.linalg.norm(relative_pos))  # 范数计算：距离或速度奖励通常围绕它设计
        ee_speed = float(np.linalg.norm(ee_vel))  # 范数计算：距离或速度奖励通常围绕它设计
        reward = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        if self.config.zero_original_mode:  # 条件分支：学习时先看触发条件，再看两边行为差异
            ee_pos = self._get_ee_pos().astype(np.float64)  # 状态或中间变量：调试时多观察它的值如何流动
            distance_to_target = float(np.linalg.norm(ee_pos - self.target_pos.astype(np.float64)))  # 范数计算：距离或速度奖励通常围绕它设计

            # 1) 密度奖励：靠近目标时更密集。
            density_reward = 1.2 * (1.0 - math.exp(-2.0 * (0.6 - distance_to_target)))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            if distance_to_target < 0.001:  # 条件分支：学习时先看触发条件，再看两边行为差异
                density_reward += (0.001 - distance_to_target) * 300.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif distance_to_target < 0.005:  # 补充分支：用于细化前面条件没有覆盖的情况
                density_reward += (0.005 - distance_to_target) * 100.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif distance_to_target < 0.01:  # 补充分支：用于细化前面条件没有覆盖的情况
                density_reward += (0.01 - distance_to_target) * 50.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif distance_to_target < 0.05:  # 补充分支：用于细化前面条件没有覆盖的情况
                density_reward += (0.05 - distance_to_target) * 10.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif distance_to_target < 0.1:  # 补充分支：用于细化前面条件没有覆盖的情况
                density_reward += (0.1 - distance_to_target) * 2.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif distance_to_target < 0.3:  # 补充分支：用于细化前面条件没有覆盖的情况
                density_reward += (0.3 - distance_to_target) * 0.5  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            # 2) 分布奖励：速度与加速度分布。
            distribution_reward = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            if 0.01 <= ee_speed <= 0.1:  # 条件分支：学习时先看触发条件，再看两边行为差异
                distribution_reward += 0.5  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif 0.005 <= ee_speed < 0.01:  # 补充分支：用于细化前面条件没有覆盖的情况
                distribution_reward += 0.3  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif 0.1 < ee_speed <= 0.2:  # 补充分支：用于细化前面条件没有覆盖的情况
                distribution_reward += 0.2  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif ee_speed < 0.005:  # 补充分支：用于细化前面条件没有覆盖的情况
                distribution_reward += 0.1  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            joint_velocities = joint_vel.astype(np.float64)  # 状态或中间变量：调试时多观察它的值如何流动
            joint_speeds = np.abs(joint_velocities)  # 状态或中间变量：调试时多观察它的值如何流动
            avg_joint_speed = float(np.mean(joint_speeds))  # 状态或中间变量：调试时多观察它的值如何流动
            if joint_speeds.size > 1:  # 条件分支：学习时先看触发条件，再看两边行为差异
                speed_std = float(np.std(joint_speeds))  # 状态或中间变量：调试时多观察它的值如何流动
                if speed_std < avg_joint_speed * 0.5:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    distribution_reward += 0.3  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                elif speed_std < avg_joint_speed:  # 补充分支：用于细化前面条件没有覆盖的情况
                    distribution_reward += 0.1  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            dt = max(float(self.model.opt.timestep), 1e-9)  # 状态或中间变量：调试时多观察它的值如何流动
            joint_accelerations = (joint_velocities - self.prev_joint_vel.astype(np.float64)) / dt  # 状态或中间变量：调试时多观察它的值如何流动
            joint_acc_magnitude = float(np.linalg.norm(joint_accelerations))  # 范数计算：距离或速度奖励通常围绕它设计
            if joint_acc_magnitude < 10.0:  # 条件分支：学习时先看触发条件，再看两边行为差异
                distribution_reward += 0.2  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif joint_acc_magnitude < 20.0:  # 补充分支：用于细化前面条件没有覆盖的情况
                distribution_reward += 0.1  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            # 3) 平稳性奖励。
            smoothness_reward = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            action_change = float(np.linalg.norm(torque_cmd - self.prev_torque))  # 范数计算：距离或速度奖励通常围绕它设计
            if action_change < 2.0:  # 条件分支：学习时先看触发条件，再看两边行为差异
                smoothness_reward += 0.3  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif action_change < 5.0:  # 补充分支：用于细化前面条件没有覆盖的情况
                smoothness_reward += 0.1  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            # 4) 惩罚项与其他项。
            joint_speed_penalty = -0.005 * float(np.sum(np.square(joint_velocities)))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            action_penalty = -0.001 * float(np.sum(np.square(torque_cmd)))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            collision_detected = False  # 状态或中间变量：调试时多观察它的值如何流动
            collision_contacts = self._warp_ncon if self.physics_backend == "warp" else int(self.data.ncon)  # 运行配置：这类参数会直接改变执行路径或调试体验
            collision_penalty = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            if collision_contacts > 0:  # 条件分支：学习时先看触发条件，再看两边行为差异
                collision_detected = True  # 状态或中间变量：调试时多观察它的值如何流动
                collision_penalty -= 100.0 * float(collision_contacts)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            height_penalty = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            if float(ee_pos[2]) < 0.05:  # 条件分支：学习时先看触发条件，再看两边行为差异
                height_penalty = -5.0 * (0.05 - float(ee_pos[2]))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            vertical_reward = 0.1 if 0.05 <= float(ee_pos[2]) <= 0.4 else 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            step_penalty = -0.001 * float(self.step_count)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            reward = (  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                density_reward  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + distribution_reward  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + smoothness_reward  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + joint_speed_penalty  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + action_penalty  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + collision_penalty  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + height_penalty  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + vertical_reward  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                + step_penalty  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            )  # 收束上一段结构，阅读时回看上面的参数或元素

            success = distance_to_target < float(self.config.success_threshold)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            terminated = bool(success or collision_detected)  # 状态或中间变量：调试时多观察它的值如何流动
            truncated = self.step_count >= int(self.config.max_steps)  # 代码执行语句：结合上下文理解它对后续流程的影响
            if success:  # 条件分支：学习时先看触发条件，再看两边行为差异
                success_reward = 200.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                if ee_speed < 0.01:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    speed_reward_on_success = 100.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                elif ee_speed < 0.05:  # 补充分支：用于细化前面条件没有覆盖的情况
                    speed_reward_on_success = 50.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                elif ee_speed < 0.1:  # 补充分支：用于细化前面条件没有覆盖的情况
                    speed_reward_on_success = 20.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                else:  # 兜底分支：当前面条件都不满足时走这里
                    speed_reward_on_success = 0.0  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                reward += success_reward + speed_reward_on_success  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

            info = {  # 状态或中间变量：调试时多观察它的值如何流动
                "distance": distance_to_target,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "success": success,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "target_pos": self.target_pos.copy(),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "physics_backend": self.physics_backend,  # 运行配置：这类参数会直接改变执行路径或调试体验
                "ee_speed": ee_speed,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "collision": collision_detected,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "collision_contacts": collision_contacts,  # 代码执行语句：结合上下文理解它对后续流程的影响
                "curriculum_stage": self.curriculum_stage,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                "episode_index": int(self.episode_count),  # 代码执行语句：结合上下文理解它对后续流程的影响
                "reward_info": {  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "density_reward": density_reward,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "distribution_reward": distribution_reward,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "smoothness_reward": smoothness_reward,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "joint_speed_penalty": joint_speed_penalty,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "action_penalty": action_penalty,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "collision_penalty": collision_penalty,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "height_penalty": height_penalty,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "vertical_reward": vertical_reward,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                    "step_penalty": step_penalty,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                },  # 收束上一段结构，阅读时回看上面的参数或元素
            }  # 收束上一段结构，阅读时回看上面的参数或元素
            return obs, float(reward), terminated, truncated, info  # 把当前结果返回给上层调用方

        # ---- 以下奖励项按 zero-arm 逻辑对齐 ----
        # 时间惩罚：鼓励更快完成任务。
        reward -= float(self.config.step_penalty)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        # 基础距离惩罚：-0.8 * sqrt(distance)。
        reward += -float(self.config.base_distance_weight) * math.sqrt(max(distance, 1e-9))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 距离改进奖励 / 退步惩罚。
        if self.min_distance is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.min_distance = distance  # 状态或中间变量：调试时多观察它的值如何流动
        elif distance < self.min_distance:  # 补充分支：用于细化前面条件没有覆盖的情况
            reward += float(self.config.improvement_gain) * float(self.min_distance - distance)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            self.min_distance = distance  # 状态或中间变量：调试时多观察它的值如何流动
        elif self.prev_distance is not None and distance > self.prev_distance:  # 补充分支：用于细化前面条件没有覆盖的情况
            reward -= float(self.config.regress_gain) * float(distance - self.prev_distance)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        self.prev_distance = distance  # 状态或中间变量：调试时多观察它的值如何流动

        # 阶段奖励：首次进入阈值区间时一次性加分。
        for thresh, phase_reward in zip(self.config.phase_thresholds, self.config.phase_rewards):  # 循环逻辑：关注迭代对象、次数和循环体副作用
            if distance < float(thresh) and thresh not in self._phase_rewards_given:  # 条件分支：学习时先看触发条件，再看两边行为差异
                reward += float(phase_reward)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
                self._phase_rewards_given.add(thresh)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 速度惩罚：末端过快时扣分。
        if ee_speed > float(self.config.speed_penalty_threshold):  # 条件分支：学习时先看触发条件，再看两边行为差异
            reward -= float(self.config.speed_penalty_value)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 方向奖励：只奖励朝目标方向的运动（余弦 > 0）。
        if ee_speed > 1e-6 and distance > 1e-9:  # 条件分支：学习时先看触发条件，再看两边行为差异
            to_target = relative_pos / (distance + 1e-6)  # 状态或中间变量：调试时多观察它的值如何流动
            movement_dir = ee_vel / (ee_speed + 1e-6)  # 状态或中间变量：调试时多观察它的值如何流动
            direction_cos = float(np.dot(to_target, movement_dir))  # 状态或中间变量：调试时多观察它的值如何流动
            reward += max(0.0, direction_cos) ** 2 * float(self.config.direction_reward_gain)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 关节角速度变化惩罚：抑制抖动/抽动。
        joint_vel_change = np.abs(joint_vel - self.prev_joint_vel)  # 状态或中间变量：调试时多观察它的值如何流动
        reward -= float(self.config.joint_vel_change_penalty_gain) * float(np.sum(joint_vel_change))  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 碰撞惩罚（zero-arm 同款）：任意接触都算碰撞，且按接触点数累加惩罚。
        collision_detected = False  # 状态或中间变量：调试时多观察它的值如何流动
        collision_contacts = self._warp_ncon if self.physics_backend == "warp" else int(self.data.ncon)  # 运行配置：这类参数会直接改变执行路径或调试体验
        if collision_contacts > 0:  # 条件分支：学习时先看触发条件，再看两边行为差异
            collision_detected = True  # 状态或中间变量：调试时多观察它的值如何流动
            reward -= float(self.config.collision_penalty_value) * float(collision_contacts)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 终止判定（Gymnasium 新接口）：
        # - terminated: 任务语义上的结束（成功/失败）。
        # - truncated: 时间上限等外部截断。
        success = distance <= float(self.config.success_threshold)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        terminated = bool(success or collision_detected)  # 状态或中间变量：调试时多观察它的值如何流动
        truncated = self.step_count >= int(self.config.max_steps)  # 代码执行语句：结合上下文理解它对后续流程的影响
        if success:  # 条件分支：学习时先看触发条件，再看两边行为差异
            # 成功奖励 + 剩余步数奖励。
            reward += float(self.config.success_bonus)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            reward += float(self.config.success_remaining_step_gain) * float(self.config.max_steps - self.step_count)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            # 成功时的速度奖励。
            if ee_speed < 0.01:  # 条件分支：学习时先看触发条件，再看两边行为差异
                reward += float(self.config.success_speed_bonus_very_slow)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif ee_speed < 0.05:  # 补充分支：用于细化前面条件没有覆盖的情况
                reward += float(self.config.success_speed_bonus_slow)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            elif ee_speed < 0.1:  # 补充分支：用于细化前面条件没有覆盖的情况
                reward += float(self.config.success_speed_bonus_medium)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
        elif collision_detected:  # 补充分支：用于细化前面条件没有覆盖的情况
            # 扣掉剩余步时间惩罚，避免策略学会“自杀式终止”。
            reward += -float(self.config.step_penalty) * float(self.config.max_steps - self.step_count)  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度

        # 诊断信息：
        # 训练算法不直接用这些字段，但你可以在日志里观测学习状态。
        info = {  # 状态或中间变量：调试时多观察它的值如何流动
            "distance": distance,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "success": success,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "target_pos": self.target_pos.copy(),  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "physics_backend": self.physics_backend,  # 运行配置：这类参数会直接改变执行路径或调试体验
            "ee_speed": ee_speed,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "collision": collision_detected,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "collision_contacts": collision_contacts,  # 代码执行语句：结合上下文理解它对后续流程的影响
            "curriculum_stage": self.curriculum_stage,  # 关键调参点：改这里通常会明显改变训练稳定性或收敛速度
            "episode_index": int(self.episode_count),  # 代码执行语句：结合上下文理解它对后续流程的影响
        }  # 收束上一段结构，阅读时回看上面的参数或元素
        return obs, float(reward), terminated, truncated, info  # 把当前结果返回给上层调用方

    def render(self):  # 渲染入口：调试画面或录制时重点看这里
        """按 Gymnasium 的 render_mode 渲染。"""  # 运行配置：这类参数会直接改变执行路径或调试体验
        if self.render_mode is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return None  # 把当前结果返回给上层调用方
        if self.physics_backend == "warp" and self.data_warp is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            # 仅在渲染时同步 host 数据，避免训练阶段每步全量回传。
            mjwarp.get_data_into(self.data, self.model, self.data_warp)  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.render_mode == "rgb_array":  # 条件分支：学习时先看触发条件，再看两边行为差异
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                if self.renderer is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    # 延迟初始化：无渲染训练时不占用额外上下文资源。
                    self.renderer = mujoco.Renderer(self.model)  # 状态或中间变量：调试时多观察它的值如何流动
                self.renderer.update_scene(self.data)  # 代码执行语句：结合上下文理解它对后续流程的影响
                return self.renderer.render()  # 把当前结果返回给上层调用方
            except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
                if not self._render_error_logged:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    print(f"[渲染] rgb_array 模式失败: {type(e).__name__}: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
                    self._render_error_logged = True  # 状态或中间变量：调试时多观察它的值如何流动
                return None  # 把当前结果返回给上层调用方

        # 某些桌面/X11 场景下，窗口被关闭或 drawable 失效后继续 sync 可能触发 GLX 错误。
        # 这里先探测 viewer 是否还在运行，不在运行就重置句柄，避免继续向失效窗口提交帧。
        if self.viewer is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            is_running = getattr(self.viewer, "is_running", None)  # 状态或中间变量：调试时多观察它的值如何流动
            if callable(is_running):  # 条件分支：学习时先看触发条件，再看两边行为差异
                try:  # 异常保护：把高风险调用包起来，避免主流程中断
                    if not bool(is_running()):  # 条件分支：学习时先看触发条件，再看两边行为差异
                        try:  # 异常保护：把高风险调用包起来，避免主流程中断
                            self.viewer.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
                        except Exception:  # 异常分支：排查失败时先看这里如何兜底
                            pass  # 占位语句：说明这里当前没有额外逻辑
                        self.viewer = None  # 状态或中间变量：调试时多观察它的值如何流动
                except Exception:  # 异常分支：排查失败时先看这里如何兜底
                    try:  # 异常保护：把高风险调用包起来，避免主流程中断
                        self.viewer.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
                    except Exception:  # 异常分支：排查失败时先看这里如何兜底
                        pass  # 占位语句：说明这里当前没有额外逻辑
                    self.viewer = None  # 状态或中间变量：调试时多观察它的值如何流动

        if self.viewer is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            if mj_viewer is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
                if not self._render_error_logged:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    print("[渲染] mujoco.viewer 导入失败，请检查 GLFW/OpenGL 运行环境。")  # 代码执行语句：结合上下文理解它对后续流程的影响
                    self._render_error_logged = True  # 状态或中间变量：调试时多观察它的值如何流动
                return None  # 把当前结果返回给上层调用方
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                if self.config.viewer_hide_ui:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    try:  # 异常保护：把高风险调用包起来，避免主流程中断
                        self.viewer = mj_viewer.launch_passive(  # zero 风格：隐藏左右 UI 面板。  
                            self.model,  # 代码执行语句：结合上下文理解它对后续流程的影响
                            self.data,  # 代码执行语句：结合上下文理解它对后续流程的影响
                            show_left_ui=False,  # 状态或中间变量：调试时多观察它的值如何流动
                            show_right_ui=False,  # 状态或中间变量：调试时多观察它的值如何流动
                        )  # 收束上一段结构，阅读时回看上面的参数或元素
                    except TypeError:  # 异常分支：排查失败时先看这里如何兜底
                        self.viewer = mj_viewer.launch_passive(self.model, self.data)  # 状态或中间变量：调试时多观察它的值如何流动
                else:  # 兜底分支：当前面条件都不满足时走这里
                    self.viewer = mj_viewer.launch_passive(self.model, self.data)  # zero 同款：被动 viewer。  
                self._target_viz_added = False  # 新窗口需重新添加目标装饰球。  
                self._target_geom_index = None  # 索引重置，避免引用旧窗口对象。  
                self._apply_zero_style_camera()  # 代码执行语句：结合上下文理解它对后续流程的影响
            except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
                if not self._render_error_logged:  # 条件分支：学习时先看触发条件，再看两边行为差异
                    print(f"[渲染] launch_passive 失败: {type(e).__name__}: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
                    self._render_error_logged = True  # 状态或中间变量：调试时多观察它的值如何流动
                return None  # 把当前结果返回给上层调用方
        try:  # 异常保护：把高风险调用包起来，避免主流程中断
            is_running = getattr(self.viewer, "is_running", None)  # 状态或中间变量：调试时多观察它的值如何流动
            if callable(is_running) and not bool(is_running()):  # 条件分支：学习时先看触发条件，再看两边行为差异
                self.viewer = None  # 状态或中间变量：调试时多观察它的值如何流动
                return None  # 把当前结果返回给上层调用方
            self._add_target_visualization()  # 每帧更新目标装饰球位置（zero 同款视觉）。  
            self.viewer.sync()  # 代码执行语句：结合上下文理解它对后续流程的影响
        except Exception as e:  # 异常分支：排查失败时先看这里如何兜底
            if not self._render_error_logged:  # 条件分支：学习时先看触发条件，再看两边行为差异
                print(f"[渲染] viewer.sync 失败: {type(e).__name__}: {e}")  # 代码执行语句：结合上下文理解它对后续流程的影响
                self._render_error_logged = True  # 状态或中间变量：调试时多观察它的值如何流动
            self.viewer = None  # 状态或中间变量：调试时多观察它的值如何流动
        return None  # 把当前结果返回给上层调用方

    def _apply_zero_style_camera(self) -> None:  # 函数定义：先看输入输出，再理解内部控制流
        """应用 zero 风格相机：默认自由视角，可选锁定到固定相机。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.viewer is None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            return  # 提前结束当前函数，回到上层流程
        if self.config.viewer_lock_camera and self._render_camera_id >= 0:  # 条件分支：学习时先看触发条件，再看两边行为差异
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 状态或中间变量：调试时多观察它的值如何流动
            self.viewer.cam.fixedcamid = int(self._render_camera_id)  # 状态或中间变量：调试时多观察它的值如何流动
            return  # 提前结束当前函数，回到上层流程
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.azimuth = float(self.config.viewer_fallback_azimuth)  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.elevation = float(self.config.viewer_fallback_elevation)  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.distance = float(self.config.viewer_fallback_distance)  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.lookat[0] = float(self.config.viewer_fallback_lookat_x)  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.lookat[1] = float(self.config.viewer_fallback_lookat_y)  # 状态或中间变量：调试时多观察它的值如何流动
        self.viewer.cam.lookat[2] = float(self.config.viewer_fallback_lookat_z)  # 状态或中间变量：调试时多观察它的值如何流动

    def _add_target_visualization(self) -> None:  # 函数定义：先看输入输出，再理解内部控制流
        """在 viewer.user_scn 里动态维护目标装饰球。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.viewer is None:  # viewer 未初始化时直接返回。  
            return  # 避免空对象访问。  
        scn = self.viewer.user_scn  # 取 MuJoCo viewer 的用户场景对象。  
        if scn is None:  # 某些后端异常时 user_scn 可能为空。  
            return  # 保守退出，避免渲染异常中断训练。  

        if not self._target_viz_added:  # 首次渲染：创建装饰几何体。  
            if scn.ngeom >= scn.maxgeom:  # 无剩余几何槽位时无法添加。  
                return  # 不抛异常，保持训练可继续。  
            geom_index = int(scn.ngeom)  # 当前尾部索引就是新几何体位置。  
            geom = scn.geoms[geom_index]  # 取得待初始化的几何体引用。  
            scn.ngeom += 1  # 场景几何体计数 +1。  
            mujoco.mjv_initGeom(  # 初始化为可视化球体（不参与物理）。  
                geom,  # 目标几何体对象。  
                mujoco.mjtGeom.mjGEOM_SPHERE,  # 几何类型：球。  
                np.array([0.03, 0.03, 0.03], dtype=np.float64),  # 球半径（x/y/z 三轴一致）。  
                self.target_pos.astype(np.float64),  # 球心坐标：当前目标点。  
                np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # 单位旋转矩阵。  
                np.array([0.1, 1.0, 0.2, 0.8], dtype=np.float32),  # RGBA：绿色半透明。  
            )  # 收束上一段结构，阅读时回看上面的参数或元素
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR  # 仅装饰显示，不参与碰撞。  
            self._target_viz_added = True  # 标记已创建。  
            self._target_geom_index = geom_index  # 记录索引，后续只更新位置。  
            return  # 首次创建完成后返回即可。  

        if self._target_geom_index is None:  # 理论不该发生，但做保护。  
            return  # 防止无效索引访问。  
        if self._target_geom_index >= scn.ngeom:  # 场景变化导致索引失效时保护退出。  
            return  # 避免越界。  
        scn.geoms[self._target_geom_index].pos = self.target_pos.astype(np.float64)  # 每帧更新目标球位置。  

    def close(self):  # 函数定义：先看输入输出，再理解内部控制流
        """释放渲染资源。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
        if self.viewer is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            viewer = self.viewer  # 状态或中间变量：调试时多观察它的值如何流动
            self.viewer = None  # 状态或中间变量：调试时多观察它的值如何流动
            # 某些驱动/GLX 场景下 viewer.close 可能阻塞，这里异步关闭避免卡住主线程。
            def _close_viewer():  # 函数定义：先看输入输出，再理解内部控制流
                try:  # 异常保护：把高风险调用包起来，避免主流程中断
                    viewer.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
                except Exception:  # 异常分支：排查失败时先看这里如何兜底
                    pass  # 占位语句：说明这里当前没有额外逻辑

            t = threading.Thread(target=_close_viewer, daemon=True)  # 状态或中间变量：调试时多观察它的值如何流动
            t.start()  # 代码执行语句：结合上下文理解它对后续流程的影响
            t.join(timeout=0.5)  # 状态或中间变量：调试时多观察它的值如何流动
        if self.renderer is not None:  # 条件分支：学习时先看触发条件，再看两边行为差异
            try:  # 异常保护：把高风险调用包起来，避免主流程中断
                self.renderer.close()  # 代码执行语句：结合上下文理解它对后续流程的影响
            except Exception:  # 异常分支：排查失败时先看这里如何兜底
                pass  # 占位语句：说明这里当前没有额外逻辑
            self.renderer = None  # 状态或中间变量：调试时多观察它的值如何流动


def register_env() -> None:  # 环境注册入口：gym.make 和 SB3 创建环境都依赖这里
    """注册 Gymnasium 环境 id。

    使用方式：
    1) 在训练脚本中先调用 `register_env()`
    2) 再通过 `make_vec_env(ENV_ID, ...)` 创建向量化环境
    3) 交给 SB3 的 SAC/PPO 直接训练
    """
    # 注册是幂等的：重复调用不会重复注册。
    if ENV_ID not in gym.registry:  # 条件分支：学习时先看触发条件，再看两边行为差异
        gym.register(  # 代码执行语句：结合上下文理解它对后续流程的影响
            id=ENV_ID,  # 状态或中间变量：调试时多观察它的值如何流动
            entry_point="classic.env:UR5MujocoEnv",  # 状态或中间变量：调试时多观察它的值如何流动
        )  # 收束上一段结构，阅读时回看上面的参数或元素
