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

- `env.py`：Gymnasium 环境、奖励、课程学习、渲染、终止原因诊断
- `train.py`：训练、评估、中断保存、继续训练、阶段训练日志
- `test.py`：推理测试、固定目标调试、慢速渲染和随机策略冒烟

适用场景：

- 主实验
- 参数调优
- UR5 与 zero_robotiq 对比

## `warp_gpu/`

- `runtime.py`：Warp CUDA 运行时检测与设备初始化
- `smoke.py`：轻量自检与 smoke test
- `env.py`：Warp GPU reach 环境、课程采样和失败类型诊断
- `train.py`：Playground / Brax 训练入口、进度条和阶段训练日志
- `test.py`：Warp GPU 策略推理测试（final_policy 或 checkpoint）

适用场景：

- 纯 GPU 训练
- PPO / SAC 并行训练
- 跟 Brax 训练器对接
- 当前只使用 Brax 官方安装包自带的算法入口

## 资源目录

- `assets/robotiq_cxy/`：UR5 CXY 模型资源
- `assets/zero_arm/`：zero 机械臂与 zero+Robotiq 资源

补充说明：

- `assets/robotiq_cxy/lab_env.xml` 当前采用“视觉体和碰撞体分离”的写法
- 视觉 `mesh` 只负责显示，不参与碰撞
- `capsule / cylinder / box` 代理体负责训练时的碰撞检测
- 这样可以减少复杂网格接触带来的误判和求解开销

## 文档

- [TRAINING_IMPROVEMENTS.md](/home/zzyfan/mujoco_ur5_rl/docs/TRAINING_IMPROVEMENTS.md)：按迭代顺序整理训练改进思路、实现方法和当前推荐课程学习流程
- [UPDATE_LOG.md](/home/zzyfan/mujoco_ur5_rl/docs/UPDATE_LOG.md)：按时间顺序汇总已经完成的实现变化

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
