# 项目结构

仓库当前只保留一份总览文档和一份结构文档。

## 目录分工

- `classic/`：主训练线
- `mjx/`：Playground + MJWarp 实验线
- `assets/`：MuJoCo XML、网格和纹理资源
- `docs/`：补充说明
- `requirements.txt`：Python 依赖
- `Dockerfile`：容器化运行环境

## `classic/`

- `env.py`：Gymnasium 环境、奖励、课程学习、渲染
- `train.py`：训练、评估、中断保存、继续训练
- `test.py`：推理测试和随机策略冒烟

适用场景：

- 主实验
- 参数调优
- UR5 与 zero_robotiq 对比

## `mjx/`

- `backend.py`：Warp 和 MuJoCo Playground 检测
- `benchmark.py`：轻量自检与 smoke test
- `train.py`：Playground 训练入口适配

适用场景：

- Warp / Playground 实验
- 跟官方训练器对接

## 资源目录

- `assets/robotiq_cxy/`：UR5 CXY 模型资源
- `assets/zero_arm/`：zero 机械臂与 zero+Robotiq 资源

## 输出目录约定

`classic/` 会写入：

```text
models/classic/{algo}/{robot}/{run_name}/
logs/classic/{algo}/{robot}/{run_name}/
```

`mjx/` 输出目录由 Playground 侧决定。
