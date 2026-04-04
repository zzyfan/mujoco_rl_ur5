# 项目结构说明（已分组）

为了“经典训练线”和“MJX 训练线”清晰区分，项目入口分成两组：

## 1) classic（原始主线）

- `classic/train.py`：经典训练入口
- `classic/test.py`：经典测试入口
- `classic/env.py`：经典环境定义

这组保留完整约束与渲染能力，适合主实验。

## 2) mjx（独立新线）

- `mjx/train.py`：MJX 训练/测试入口
- `mjx/benchmark.py`：MJX 并行步进压测入口
- `mjx/env.py`：MJX / MuJoCo Warp 环境定义
- `mjx/backend.py`：物理后端检测与切换

这组用于 GPU 并行后端实验与迁移验证。

## 3) 兼容性说明

- 训练与测试入口已经统一收口到 `classic/` 与 `mjx/`
- 若仓库里还保留旧版 `models/{algo}/...`、`logs/{algo}/...` 结果，`classic/train.py` 与 `classic/test.py` 会自动把缺失文件同步到 `models/classic/...`、`logs/classic/...`

## 4) 输出目录层级（默认）

通过分组入口启动时，默认输出为：

```text
models/
  classic/{algo}/{robot}/{run_name}/...
  mjx/{algo}/{robot}/{run_name}/...

logs/
  classic/{algo}/{robot}/{run_name}/...
  mjx/{algo}/{robot}/{run_name}/...
```

## 4) 推荐命令

经典训练：

```bash
python classic/train.py --algo sac --robot ur5_cxy --run-name exp_sac_ur5 --timesteps 500000 --device cuda --no-render
```

MJX 训练：

```bash
python mjx/train.py --algo sac --robot zero_robotiq --run-name mjx_sac_zero --timesteps 500000 --n-envs 1 --device cuda --safe-disable-constraints
```

MuJoCo Warp 训练：

```bash
python mjx/train.py --algo sac --robot zero_robotiq --run-name mjx_sac_zero_warp --timesteps 500000 --n-envs 1 --device cuda --safe-disable-constraints --physics-backend warp
```
