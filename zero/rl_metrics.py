import csv
import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def compute_inference_metrics(
    model,
    env,
    num_episodes,
    render=True,
    max_steps_per_episode=5000,
    sleep_seconds=0.01,
):
    episode_rows = []
    for episode_idx in range(int(num_episodes)):
        obs = env.reset()
        total_reward = 0.0
        episode_length = 0
        success = False
        collision = False
        distance_trace = []

        action_change_sum = 0.0
        action_change_count = 0
        prev_action = None

        for _ in range(int(max_steps_per_episode)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_scalar = float(reward[0] if isinstance(reward, np.ndarray) else reward)
            total_reward += reward_scalar
            episode_length += 1

            info0 = info[0] if isinstance(info, (list, tuple)) else info
            if isinstance(info0, dict):
                if "distance" in info0:
                    distance_trace.append(float(info0["distance"]))
                success = success or bool(info0.get("success", False))
                collision = collision or bool(info0.get("collision", False))

            current_action = np.asarray(action, dtype=np.float64).reshape(-1)
            if prev_action is not None and prev_action.shape == current_action.shape:
                action_change_sum += float(np.linalg.norm(current_action - prev_action))
                action_change_count += 1
            prev_action = current_action

            if render:
                env.render()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            done_flag = bool(done[0] if isinstance(done, np.ndarray) else done)
            if done_flag:
                break

        final_distance = float(distance_trace[-1]) if distance_trace else float("nan")
        mean_distance = float(np.mean(distance_trace)) if distance_trace else float("nan")
        smoothness = action_change_sum / max(1, action_change_count)

        row = {
            "episode": episode_idx + 1,
            "reward": total_reward,
            "episode_length": episode_length,
            "success": int(success),
            "collision": int(collision),
            "final_distance": final_distance,
            "mean_distance": mean_distance,
            "smoothness": smoothness,
        }
        episode_rows.append(row)

    success_rate = float(np.mean([r["success"] for r in episode_rows])) if episode_rows else 0.0
    collision_rate = float(np.mean([r["collision"] for r in episode_rows])) if episode_rows else 0.0
    mean_distance = float(np.nanmean([r["final_distance"] for r in episode_rows])) if episode_rows else float("nan")
    mean_episode_length = float(np.mean([r["episode_length"] for r in episode_rows])) if episode_rows else 0.0
    mean_smoothness = float(np.mean([r["smoothness"] for r in episode_rows])) if episode_rows else 0.0
    mean_reward = float(np.mean([r["reward"] for r in episode_rows])) if episode_rows else 0.0

    summary = {
        "num_episodes": int(num_episodes),
        "success_rate": success_rate,
        "mean_final_distance": mean_distance,
        "mean_episode_length": mean_episode_length,
        "collision_rate": collision_rate,
        "mean_smoothness": mean_smoothness,
        "mean_reward": mean_reward,
    }
    return summary, episode_rows


class InferenceMetricsEvalCallback(BaseCallback):
    """
    训练期间周期性评估推理指标，并写入 SB3 logger。
    """

    def __init__(
        self,
        eval_env,
        eval_freq=5000,
        n_eval_episodes=10,
        save_dir=None,
        deterministic=True,
        render=False,
        max_steps_per_episode=5000,
        verbose=0,
    ):
        super(InferenceMetricsEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.save_dir = save_dir
        self.deterministic = deterministic
        self.render = render
        self.max_steps_per_episode = max(1, int(max_steps_per_episode))
        self.history = []

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass

        summary, _ = compute_inference_metrics(
            model=self.model,
            env=self.eval_env,
            num_episodes=self.n_eval_episodes,
            render=self.render,
            max_steps_per_episode=self.max_steps_per_episode,
            sleep_seconds=0.0,
        )

        self.logger.record("eval_custom/success_rate", summary["success_rate"])
        self.logger.record("eval_custom/mean_distance", summary["mean_final_distance"])
        self.logger.record("eval_custom/episode_length", summary["mean_episode_length"])
        self.logger.record("eval_custom/collision_rate", summary["collision_rate"])
        self.logger.record("eval_custom/smoothness", summary["mean_smoothness"])
        self.logger.record("eval_custom/mean_reward", summary["mean_reward"])

        row = {"timesteps": int(self.num_timesteps), **summary}
        self.history.append(row)
        if self.verbose > 0:
            print(
                f"[eval_custom] step={self.num_timesteps} "
                f"success={summary['success_rate'] * 100:.2f}% "
                f"dist={summary['mean_final_distance']:.6f} "
                f"len={summary['mean_episode_length']:.2f}"
            )
        return True

    def _on_training_end(self) -> None:
        if not self.save_dir or not self.history:
            return

        with open(os.path.join(self.save_dir, "inference_callback_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        keys = [
            "timesteps",
            "num_episodes",
            "success_rate",
            "mean_final_distance",
            "mean_episode_length",
            "collision_rate",
            "mean_smoothness",
            "mean_reward",
        ]
        with open(os.path.join(self.save_dir, "inference_callback_history.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

        _plot_inference_callback_history(self.history, self.save_dir)


class TrainingMetricsCallback(BaseCallback):
    """
    采集训练过程指标，并在训练结束后导出数据与曲线图。
    """

    def __init__(
        self,
        save_dir,
        loss_log_freq=1000,
        episode_print_freq=1,
        record_to_sb3_logger=True,
        verbose=0,
    ):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.loss_log_freq = max(1, int(loss_log_freq))
        self.episode_print_freq = max(1, int(episode_print_freq))
        self.record_to_sb3_logger = bool(record_to_sb3_logger)

        self.episode_timesteps = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_rows = []

        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")

        if infos is not None and dones is not None:
            for info, done in zip(infos, dones):
                if not bool(done):
                    continue
                if not isinstance(info, dict):
                    continue
                ep_info = info.get("episode")
                if not isinstance(ep_info, dict):
                    continue

                reward = _safe_float(ep_info.get("r"))
                length = _safe_float(ep_info.get("l"))
                if reward is None or length is None:
                    continue

                self.episode_timesteps.append(int(self.num_timesteps))
                self.episode_rewards.append(reward)
                self.episode_lengths.append(int(length))

                # 通过 SB3 logger 输出训练阶段核心指标
                if self.record_to_sb3_logger:
                    self.logger.record("train_custom/episode_reward", reward)
                    self.logger.record("train_custom/episode_length", length)
                window = min(100, len(self.episode_rewards))
                if window > 0 and self.record_to_sb3_logger:
                    self.logger.record(
                        "train_custom/reward_mean_100",
                        float(np.mean(self.episode_rewards[-window:])),
                    )
                    self.logger.record(
                        "train_custom/length_mean_100",
                        float(np.mean(self.episode_lengths[-window:])),
                    )

                if self.verbose > 0 and len(self.episode_rewards) % self.episode_print_freq == 0:
                    reward_mean = float(np.mean(self.episode_rewards[-window:])) if window > 0 else reward
                    length_mean = float(np.mean(self.episode_lengths[-window:])) if window > 0 else length
                    print(
                        f"[train_custom] step={self.num_timesteps} "
                        f"episode={len(self.episode_rewards)} "
                        f"reward={reward:.3f} length={int(length)} "
                        f"mean_reward_100={reward_mean:.3f} "
                        f"mean_length_100={length_mean:.2f}"
                    )

        if self.num_timesteps % self.loss_log_freq == 0:
            logger_values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
            if isinstance(logger_values, dict):
                row = {"timesteps": int(self.num_timesteps)}
                has_loss = False
                for key, value in logger_values.items():
                    if not isinstance(key, str):
                        continue
                    if not key.startswith("train/"):
                        continue
                    if "loss" not in key and "entropy" not in key:
                        continue
                    number = _safe_float(value)
                    if number is None:
                        continue
                    row[key] = number
                    if self.record_to_sb3_logger:
                        self.logger.record(f"train_custom/{key.replace('/', '_')}", number)
                    has_loss = True
                if has_loss:
                    self.loss_rows.append(row)
                    if self.verbose > 0:
                        loss_parts = [
                            f"{k.split('/', 1)[1]}={row[k]:.6f}"
                            for k in sorted(row.keys())
                            if k != "timesteps"
                        ]
                        if loss_parts:
                            print(f"[train_loss] step={self.num_timesteps} " + " ".join(loss_parts))

        return True

    def save_and_plot(self):
        os.makedirs(self.save_dir, exist_ok=True)

        metrics_json = {
            "episode_timesteps": self.episode_timesteps,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "loss_rows": self.loss_rows,
        }
        with open(os.path.join(self.save_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)

        episodes_csv = os.path.join(self.save_dir, "training_episodes.csv")
        with open(episodes_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "episode_reward", "episode_length"])
            for t, r, l in zip(self.episode_timesteps, self.episode_rewards, self.episode_lengths):
                writer.writerow([t, r, l])

        if self.loss_rows:
            all_keys = sorted({k for row in self.loss_rows for k in row.keys() if k != "timesteps"})
            losses_csv = os.path.join(self.save_dir, "training_losses.csv")
            with open(losses_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps"] + all_keys)
                for row in self.loss_rows:
                    writer.writerow([row["timesteps"]] + [row.get(k, "") for k in all_keys])

        self._plot_curves()

    def _plot_curves(self):
        plt = _try_import_matplotlib()
        if plt is None:
            if self.verbose > 0:
                print("未安装 matplotlib，已跳过训练曲线绘图。")
            return

        if self.episode_rewards:
            x = np.array(self.episode_timesteps, dtype=np.float64)
            y = np.array(self.episode_rewards, dtype=np.float64)
            plt.figure(figsize=(8, 4))
            plt.plot(x, y, alpha=0.5, label="Episode Reward")
            if len(y) >= 5:
                window = min(20, len(y))
                y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                x_smooth = x[window - 1 :]
                plt.plot(x_smooth, y_smooth, linewidth=2.0, label=f"Moving Avg ({window})")
            plt.title("Training Reward Curve")
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "training_reward_curve.png"), dpi=150)
            plt.close()

        if self.episode_lengths:
            x = np.array(self.episode_timesteps, dtype=np.float64)
            y = np.array(self.episode_lengths, dtype=np.float64)
            plt.figure(figsize=(8, 4))
            plt.plot(x, y, alpha=0.7, label="Episode Length")
            if len(y) >= 5:
                window = min(20, len(y))
                y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                x_smooth = x[window - 1 :]
                plt.plot(x_smooth, y_smooth, linewidth=2.0, label=f"Moving Avg ({window})")
            plt.title("Training Episode Length Curve")
            plt.xlabel("Timesteps")
            plt.ylabel("Episode Length")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "training_episode_length_curve.png"), dpi=150)
            plt.close()

        if self.loss_rows:
            loss_keys = sorted({k for row in self.loss_rows for k in row.keys() if k != "timesteps"})
            plt.figure(figsize=(8, 4))
            for key in loss_keys:
                xs = []
                ys = []
                for row in self.loss_rows:
                    if key in row:
                        xs.append(row["timesteps"])
                        ys.append(row[key])
                if xs:
                    plt.plot(xs, ys, label=key)
            plt.title("Training Loss Curve")
            plt.xlabel("Timesteps")
            plt.ylabel("Loss / Entropy")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "training_loss_curve.png"), dpi=150)
            plt.close()


def evaluate_inference_metrics(
    model,
    env,
    num_episodes,
    save_dir,
    render=True,
    max_steps_per_episode=5000,
    sleep_seconds=0.01,
):
    """
    评估推理阶段指标并导出报告。
    """
    os.makedirs(save_dir, exist_ok=True)

    summary, episode_rows = compute_inference_metrics(
        model=model,
        env=env,
        num_episodes=num_episodes,
        render=render,
        max_steps_per_episode=max_steps_per_episode,
        sleep_seconds=sleep_seconds,
    )

    with open(os.path.join(save_dir, "inference_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "inference_per_episode.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "reward",
                "episode_length",
                "success",
                "collision",
                "final_distance",
                "mean_distance",
                "smoothness",
            ],
        )
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)

    _plot_inference_metrics(episode_rows, save_dir)
    return summary, episode_rows


def _plot_inference_callback_history(history_rows, save_dir):
    if not history_rows:
        return

    plt = _try_import_matplotlib()
    if plt is None:
        return

    steps = [r["timesteps"] for r in history_rows]
    success = [r["success_rate"] for r in history_rows]
    distance = [r["mean_final_distance"] for r in history_rows]
    length = [r["mean_episode_length"] for r in history_rows]
    collision = [r["collision_rate"] for r in history_rows]
    smoothness = [r["mean_smoothness"] for r in history_rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(steps, success, marker="o")
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, distance, marker="o")
    axes[0, 1].set_title("Mean Final Distance")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("Distance")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, length, marker="o")
    axes[1, 0].set_title("Mean Episode Length")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, smoothness, marker="o", label="smoothness")
    axes[1, 1].plot(steps, collision, marker="x", label="collision_rate")
    axes[1, 1].set_title("Smoothness / Collision")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inference_callback_curve.png"), dpi=150)
    plt.close()


def _plot_inference_metrics(episode_rows, save_dir):
    if not episode_rows:
        return

    plt = _try_import_matplotlib()
    if plt is None:
        return

    episodes = [r["episode"] for r in episode_rows]
    final_distances = [r["final_distance"] for r in episode_rows]
    lengths = [r["episode_length"] for r in episode_rows]
    successes = [r["success"] for r in episode_rows]
    smoothness = [r["smoothness"] for r in episode_rows]
    collisions = [r["collision"] for r in episode_rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(episodes, successes, marker="o")
    axes[0, 0].set_title("Success (per episode)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Success(0/1)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, final_distances, marker="o")
    axes[0, 1].set_title("Final Distance")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Distance")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, lengths, marker="o")
    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, smoothness, marker="o", label="smoothness")
    axes[1, 1].plot(episodes, collisions, marker="x", label="collision(0/1)")
    axes[1, 1].set_title("Smoothness / Collision")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inference_metrics.png"), dpi=150)
    plt.close()
