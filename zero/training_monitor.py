import math
from stable_baselines3.common.callbacks import BaseCallback


class ParallelTrainingStatsCallback(BaseCallback):
    """
    统计并打印并行环境中的回合信息。
    """

    def __init__(self, summary_every_steps=5000, verbose=1):
        super().__init__(verbose)
        self.summary_every_steps = max(1, int(summary_every_steps))
        self.num_envs = 0
        self.episode_counts = []
        self.success_counts = []
        self.first_success_episode = []
        self.latest_distance = []
        self.latest_step_count = []

    def _on_training_start(self) -> None:
        self.num_envs = int(self.training_env.num_envs)
        self.episode_counts = [0 for _ in range(self.num_envs)]
        self.success_counts = [0 for _ in range(self.num_envs)]
        self.first_success_episode = [None for _ in range(self.num_envs)]
        self.latest_distance = [math.nan for _ in range(self.num_envs)]
        self.latest_step_count = [0 for _ in range(self.num_envs)]
        print(f"训练统计回调已启用，并行环境数: {self.num_envs}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            if "distance_to_target" in info:
                self.latest_distance[env_idx] = float(info["distance_to_target"])
            if "step_count" in info:
                self.latest_step_count[env_idx] = int(info["step_count"])

            if env_idx < len(dones) and bool(dones[env_idx]):
                self.episode_counts[env_idx] += 1
                episode_id = self.episode_counts[env_idx]
                success = bool(info.get("episode_success", info.get("is_success", False)))
                if success:
                    self.success_counts[env_idx] += 1
                    if self.first_success_episode[env_idx] is None:
                        self.first_success_episode[env_idx] = episode_id

                first_success = self.first_success_episode[env_idx]
                first_success_text = "-" if first_success is None else str(first_success)
                print(
                    "[训练统计]"
                    f"[env {env_idx:03d}]"
                    f" 回合={episode_id}"
                    f" 回报={float(info.get('episode_reward', 0.0)):.2f}"
                    f" 步数={int(info.get('episode_length', info.get('step_count', 0)))}"
                    f" 成功={success}"
                    f" 总成功数={self.success_counts[env_idx]}"
                    f" 首次成功回合={first_success_text}"
                    f" 最终距离={float(info.get('distance_to_target', math.nan)):.4f}"
                    f" 碰撞={bool(info.get('collision_detected', False))}"
                )

        if self.n_calls % self.summary_every_steps == 0:
            total_episodes = sum(self.episode_counts)
            total_successes = sum(self.success_counts)
            success_rate = 0.0 if total_episodes == 0 else total_successes / total_episodes
            env_summaries = []
            for env_idx in range(self.num_envs):
                first_success = self.first_success_episode[env_idx]
                first_success_text = "-" if first_success is None else str(first_success)
                distance = self.latest_distance[env_idx]
                distance_text = "nan" if math.isnan(distance) else f"{distance:.4f}"
                env_summaries.append(
                    f"env{env_idx}:eps={self.episode_counts[env_idx]},"
                    f"succ={self.success_counts[env_idx]},"
                    f"first={first_success_text},"
                    f"step={self.latest_step_count[env_idx]},"
                    f"dist={distance_text}"
                )

            print(
                "[训练统计][汇总]"
                f" 调用步={self.n_calls}"
                f" 总回合数={total_episodes}"
                f" 总成功数={total_successes}"
                f" 成功率={success_rate:.2%}"
            )
            for env_summary in env_summaries:
                print(f"[训练统计][环境] {env_summary}")

        return True
