"""MuJoCo Playground + MJWarp backend helpers."""  # 代码执行语句：结合上下文理解它对后续流程的影响

from __future__ import annotations  # 依赖导入：先认清这个文件需要哪些外部能力

import importlib.util  # 依赖导入：先认清这个文件需要哪些外部能力
import os  # 依赖导入：先认清这个文件需要哪些外部能力
import shutil  # 依赖导入：先认清这个文件需要哪些外部能力
from pathlib import Path  # 依赖导入：先认清这个文件需要哪些外部能力

try:  # 异常保护：把高风险调用包起来，避免主流程中断
    import warp as wp  # 依赖导入：先认清这个文件需要哪些外部能力
except Exception:  # 异常分支：排查失败时先看这里如何兜底
    wp = None  # 状态或中间变量：调试时多观察它的值如何流动

try:  # 异常保护：把高风险调用包起来，避免主流程中断
    import mujoco_warp as mjwarp  # 依赖导入：先认清这个文件需要哪些外部能力
except Exception:  # 异常分支：排查失败时先看这里如何兜底
    mjwarp = None  # 状态或中间变量：调试时多观察它的值如何流动

_WARP_INITIALIZED = False  # 状态或中间变量：调试时多观察它的值如何流动


def warp_available() -> bool:  # 函数定义：先看输入输出，再理解内部控制流
    return wp is not None and mjwarp is not None  # 把当前结果返回给上层调用方


def ensure_warp_runtime() -> None:  # 函数定义：先看输入输出，再理解内部控制流
    global _WARP_INITIALIZED  # 代码执行语句：结合上下文理解它对后续流程的影响
    if not warp_available():  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise RuntimeError("MuJoCo Warp 不可用，请先安装 `mujoco-warp` 与 `warp-lang`。")  # 主动抛错：用来尽早暴露错误输入或不支持状态
    if _WARP_INITIALIZED:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return  # 提前结束当前函数，回到上层流程
    wp.init()  # 代码执行语句：结合上下文理解它对后续流程的影响
    _WARP_INITIALIZED = True  # 状态或中间变量：调试时多观察它的值如何流动


def describe_warp_runtime() -> str:  # 函数定义：先看输入输出，再理解内部控制流
    if not warp_available():  # 条件分支：学习时先看触发条件，再看两边行为差异
        return "warp unavailable"  # 把当前结果返回给上层调用方
    ensure_warp_runtime()  # 代码执行语句：结合上下文理解它对后续流程的影响
    try:  # 异常保护：把高风险调用包起来，避免主流程中断
        device = str(wp.get_device())  # 运行配置：这类参数会直接改变执行路径或调试体验
    except Exception:  # 异常分支：排查失败时先看这里如何兜底
        device = "unknown"  # 运行配置：这类参数会直接改变执行路径或调试体验
    try:  # 异常保护：把高风险调用包起来，避免主流程中断
        cuda_devices = list(wp.get_cuda_devices())  # 运行配置：这类参数会直接改变执行路径或调试体验
    except Exception:  # 异常分支：排查失败时先看这里如何兜底
        cuda_devices = []  # 运行配置：这类参数会直接改变执行路径或调试体验
    return f"warp_device={device}, cuda_devices={cuda_devices}"  # 把当前结果返回给上层调用方


def detect_playground_root(explicit_root: str = "") -> str:  # 函数定义：先看输入输出，再理解内部控制流
    """返回可用的 MuJoCo Playground 根目录（找不到则返回空字符串）。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if explicit_root:  # 条件分支：学习时先看触发条件，再看两边行为差异
        p = Path(explicit_root).expanduser().resolve()  # 状态或中间变量：调试时多观察它的值如何流动
        if (p / "learning").exists() and (p / "mujoco_playground").exists():  # 条件分支：学习时先看触发条件，再看两边行为差异
            return str(p)  # 把当前结果返回给上层调用方
    env_root = os.getenv("MUJOCO_PLAYGROUND_ROOT", "").strip()  # 状态或中间变量：调试时多观察它的值如何流动
    if env_root:  # 条件分支：学习时先看触发条件，再看两边行为差异
        p = Path(env_root).expanduser().resolve()  # 状态或中间变量：调试时多观察它的值如何流动
        if (p / "learning").exists() and (p / "mujoco_playground").exists():  # 条件分支：学习时先看触发条件，再看两边行为差异
            return str(p)  # 把当前结果返回给上层调用方
    # 常见路径兜底
    for cand in (Path.cwd(), Path.cwd().parent, Path.home() / "mujoco_playground"):  # 循环逻辑：关注迭代对象、次数和循环体副作用
        p = cand.resolve()  # 状态或中间变量：调试时多观察它的值如何流动
        if (p / "learning").exists() and (p / "mujoco_playground").exists():  # 条件分支：学习时先看触发条件，再看两边行为差异
            return str(p)  # 把当前结果返回给上层调用方
    return ""  # 把当前结果返回给上层调用方


def resolve_trainer_entry(trainer: str, playground_root: str = "") -> list[str]:  # 函数定义：先看输入输出，再理解内部控制流
    """解析训练入口命令。优先使用 PATH 里的 CLI，回退到 `python learning/*.py`。"""  # 代码执行语句：结合上下文理解它对后续流程的影响
    if trainer not in {"jax-ppo", "rsl-ppo"}:  # 条件分支：学习时先看触发条件，再看两边行为差异
        raise ValueError(f"不支持的 trainer: {trainer}")  # 主动抛错：用来尽早暴露错误输入或不支持状态
    bin_name = "train-jax-ppo" if trainer == "jax-ppo" else "train-rsl-ppo"  # 代码执行语句：结合上下文理解它对后续流程的影响
    bin_path = shutil.which(bin_name)  # 状态或中间变量：调试时多观察它的值如何流动
    if bin_path:  # 条件分支：学习时先看触发条件，再看两边行为差异
        return [bin_path]  # 把当前结果返回给上层调用方

    root = detect_playground_root(playground_root)  # 状态或中间变量：调试时多观察它的值如何流动
    if root:  # 条件分支：学习时先看触发条件，再看两边行为差异
        script = Path(root) / "learning" / ("train_jax_ppo.py" if trainer == "jax-ppo" else "train_rsl_ppo.py")  # 代码执行语句：结合上下文理解它对后续流程的影响
        if script.exists():  # 条件分支：学习时先看触发条件，再看两边行为差异
            return ["python", str(script)]  # 把当前结果返回给上层调用方
    raise RuntimeError(  # 主动抛错：用来尽早暴露错误输入或不支持状态
        f"未找到 `{bin_name}`。请先安装 MuJoCo Playground，或设置 MUJOCO_PLAYGROUND_ROOT。"  # 代码执行语句：结合上下文理解它对后续流程的影响
    )  # 收束上一段结构，阅读时回看上面的参数或元素


def playground_importable() -> bool:  # 函数定义：先看输入输出，再理解内部控制流
    return importlib.util.find_spec("mujoco_playground") is not None  # 把当前结果返回给上层调用方
