# Portability Notes

本文档说明本项目的可移植性设计，重点是降低跨机器迁移时的改动成本。

## Path Design

- 所有代码都通过 [ur5_reach_config.py](/home/zzyfan/mujoco_ur5_rl/ur5_reach_config.py) 里的 `project_root()` 解析仓库根目录。
- 训练和测试入口都不依赖当前工作目录。
- 主线产物目录统一落到 `runs/main/{algo}/{run_name}/`。
- Warp 产物目录统一落到 `runs/warp/{algo}/{run_name}/`。

## Dependency Design

- 只保留 MuJoCo + Gymnasium + Stable-Baselines3 这条基础依赖链。
- 不再依赖 `warp`、`sbx`、额外实验分支。
- Notebook 依赖也直接写进 [requirements.txt](/home/zzyfan/mujoco_ur5_rl/requirements.txt)。

## Asset Design

- 只保留 `assets/robotiq_cxy/`。
- 不再依赖其他机器人线的 XML、mesh 或脚本。

## Runtime Notes

无图形界面环境下，先设置：

```bash
export MUJOCO_GL=egl
```

如果是本地桌面测试，需要 human 渲染时再加 `--render`。

如果需要在训练时启用主进程旁观窗口，也需要桌面图形环境；服务器纯命令行环境通常不适合开启 `--spectator-render`。

## Artifact Transfer

一个训练实验最少只需要拷贝：

```text
runs/main/{algo}/{run_name}/run_config.json
runs/main/{algo}/{run_name}/final/final_model.zip
runs/main/{algo}/{run_name}/final/vec_normalize.pkl
```

如需保留最佳模型，也建议同时复制 `best_model/`。

## Scope

当前仓库只保留 `UR5` 到点任务，因此：

- 没有 `zero_robotiq`
- 没有 `warp_gpu`
- 没有 `sbx_runner`
- 没有多后端兼容逻辑

迁移到新机器时，主要检查以下内容：

1. MuJoCo 是否正常安装
2. `assets/robotiq_cxy` 是否完整
3. Python 依赖是否安装成功

## Future Improvements

后续可以继续扩展以下能力：

- 补一个最小 Docker 工作流
- 补一个 `export_run.py`，把模型、配置和参数说明一起打包

当前目录结构已经为这两类扩展预留了位置。
