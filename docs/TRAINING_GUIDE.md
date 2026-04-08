# Training Guide

## Main Line

### Train

```bash
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --total-timesteps 1500000
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --total-timesteps 1500000
python train_ur5_reach.py --algo ppo --run-name ur5_ppo_main --total-timesteps 1500000
python train_ur5_reach.py --algo td3 --run-name ur5_td3_main --total-timesteps 1500000 --disable-gripper-end-effector
```

### Test

测试命令现在统一采用“代码内解析模型目录”的方式，不再要求手写模型路径。

```bash
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model best --episodes 1 --render-mode human
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model final --episodes 1 --render-mode human
python train_ur5_reach.py --algo sac --run-name ur5_sac_main --test --model best --episodes 1 --render-mode human --disable-gripper-end-effector
```

### Common Flags

- `--algo`：`td3` / `sac` / `ppo`
- `--run-name`：实验目录名
- `--model`：`best` / `final`
- `--render-mode`：`none` / `human`
- `--n-envs`：并行环境数
- `--device`：`auto` / `cpu` / `cuda`
- `--disable-gripper-end-effector`：切换到不带夹爪的简化末端模型

说明：

- 默认使用带夹爪的原始模型
- 打开 `--disable-gripper-end-effector` 后，会改用简化末端 XML
- 两种末端模型的训练产物会自动分开保存，不会覆盖彼此

## Warp Line

### Train

```bash
python train_ur5_reach_warp.py --algo sac --run-name ur5_warp_sac --num-envs 256 --num-timesteps 5000000
python train_ur5_reach_warp.py --algo ppo --run-name ur5_warp_ppo --num-envs 256 --num-timesteps 5000000
```

### Common Flags

- `--algo`：`sac` / `ppo`
- `--run-name`：实验目录名
- `--num-timesteps`：总训练步数
- `--num-envs`：并行训练环境数
- `--num-eval-envs`：并行评估环境数
- `--learning-rate`：学习率
- `--target-sampling-mode`：`full_random` / `small_random` / `fixed`
- `--controller-mode`：`torque` / `joint_position_delta`

说明：

- Warp 测试入口暂时留到服务器环境验证
- 本地仓库当前主要保证 Warp 训练链路和产物目录规则

## Artifact Layout

主线产物：

```text
runs/{local|server}/main/{algo}/{run_name}/
  best_model/
  final_model/
  interrupted/
  tensorboard/
  final_eval.json
```

若使用 `--disable-gripper-end-effector`，主线产物会自动写到：

```text
runs/{local|server}/main/{algo}/{run_name}__no_gripper/
  best_model/
  final_model/
  interrupted/
  tensorboard/
  final_eval.json
```

Warp 产物：

```text
runs/{local|server}/warp/{algo}/{run_name}/
  checkpoints/
  best_model/
  final_model/
  final_eval.json
```

## Final Eval

训练结束时两条线都会输出：

- `min_distance`
- `max_return`
- `successes`
- `success_rate`

主线来自显式测试回合，Warp 线来自训练评估流汇总。

补充说明：

- 主线最终评估和测试加载都会根据是否传入 `--disable-gripper-end-effector` 自动选择对应模型目录
- 带夹爪模型默认使用两指中点作为参考点
- 不带夹爪模型默认使用 `ee_link` 原点作为参考点

## Design Notes

如果你想先理解“为什么代码要这样组织”，推荐配合下面这份 notebook 一起读：

- `notebooks/03_main_and_warp_design_notes.ipynb`

它把两条线分开写：

- Main class 线为什么用 `gymnasium.Env + SB3`
- Warp 线为什么只保留本地训练入口
- wrist 奖惩为什么改成“近目标微调奖励 + 异常旋转惩罚”
- 两条线为什么统一模型目录和最终评估摘要
