# 项目结构

仓库当前只保留一份总览文档和一份结构文档。

## 目录分工

- `classic/`：主训练线
- `warp_gpu/`：Warp GPU + MuJoCo Playground / Brax 训练线
- `assets/`：MuJoCo XML、网格和纹理资源
- `docs/`：补充说明
- `requirements.txt`：Python 依赖
- `Dockerfile`：容器化运行环境

## `classic/`

- `env.py`：Gymnasium 环境、奖励、课程学习、渲染
- `train.py`：训练、评估、中断保存、继续训练、阶段训练日志
- `test.py`：推理测试和随机策略冒烟

适用场景：

- 主实验
- 参数调优
- UR5 与 zero_robotiq 对比

## `warp_gpu/`

- `runtime.py`：Warp CUDA 运行时检测与设备初始化
- `smoke.py`：轻量自检与 smoke test
- `env.py`：Warp GPU reach 环境
- `train.py`：Playground / Brax 训练入口、进度条和阶段训练日志

适用场景：

- 纯 GPU 训练
- PPO / SAC 并行训练
- 跟 Brax 训练器对接
- 当前只使用 Brax 官方安装包自带的算法入口

## 资源目录

- `assets/robotiq_cxy/`：UR5 CXY 模型资源
- `assets/zero_arm/`：zero 机械臂与 zero+Robotiq 资源

## 输出目录约定

`classic/` 会写入：

```text
models/classic/{algo}/{robot}/{run_name}/
logs/classic/{algo}/{robot}/{run_name}/
```

`warp_gpu/` 会写入：

```text
models/warp_gpu/{algo}/{robot}/{run_name}/
```
