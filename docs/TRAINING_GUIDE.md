# Training Guide

这份文档面向代码阅读和复现，结构尽量接近常见开源仓库风格。

## Main Line (SB3)

### Train

```bash
python train_ur5_reach.py --algo sac --total-timesteps 1500000
python train_ur5_reach.py --algo td3 --total-timesteps 1500000
python train_ur5_reach.py --algo ppo --total-timesteps 1500000
```

### Test

```bash
python train_ur5_reach.py --algo sac --test --episodes 1 --render
```

### Common Flags

- `--algo`：`td3` / `sac` / `ppo`
- `--total-timesteps`：训练总步数
- `--n-envs`：并行环境数
- `--render`：测试时打开窗口
- `--render-every`：训练时渲染频率（需 `--render`）
- `--device`：`auto` / `cpu` / `cuda`

## Warp Line (Brax + MJX/Warp)

### Train

```bash
python train_ur5_reach_warp.py --algo sac --num-envs 256 --num-timesteps 5000000
python train_ur5_reach_warp.py --algo ppo --num-envs 256 --num-timesteps 5000000
```

### Common Flags

- `--algo`：`sac` / `ppo`
- `--num-timesteps`：训练总步数
- `--num-envs`：并行环境数
- `--num-eval-envs`：并行评估环境数
- `--learning-rate`：学习率

## Output

主线训练产物：

- `./logs/best_model/`
- `./models/`

Warp 线训练产物：

- `./warp_runs/{algo}/{run_name}/`

