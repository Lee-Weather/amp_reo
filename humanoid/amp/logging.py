"""Original runner's completed-episode statistics and display, extended for AMP."""
from collections import deque
import statistics

import torch


class TrainingLog:
    def __init__(self, num_envs, device):
        self.sums = torch.zeros(num_envs, 3, device=device)
        self.lengths = torch.zeros(num_envs, device=device)
        self.returns = [deque(maxlen=100) for _ in range(3)]
        self.completed_lengths = deque(maxlen=100)
        self.episode_infos = []
        self.total_steps = 0
        self.total_time = 0.
        self.call_start_time = 0.

    def begin_rollout(self, obs):
        self.episode_infos.clear()
        self.metric_sums = {}
        self.metric_counts = {}
        self.obs_std, self.obs_mean = torch.std_mean(obs, dim=0, unbiased=False)

    def record(self, mixed, task, style, dones, info):
        for key, value in info.get("training_metrics", {}).items():
            if value.numel():
                total = value.detach().float().sum()
                if key not in self.metric_sums:
                    self.metric_sums[key] = total
                    self.metric_counts[key] = value.numel()
                else:
                    self.metric_sums[key] += total
                    self.metric_counts[key] += value.numel()
        self.sums += torch.stack((mixed, task, style), dim=-1)
        self.lengths += 1
        ids = dones.nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        for index, buffer in enumerate(self.returns):
            buffer.extend(self.sums[ids, index].cpu().tolist())
        self.completed_lengths.extend(self.lengths[ids].cpu().tolist())
        self.sums[ids] = 0
        self.lengths[ids] = 0
        # LeggedRobot retains extras['episode'] on non-reset steps; count once.
        if "episode" in info:
            self.episode_infos.append({key: torch.as_tensor(value).detach().clone()
                                       for key, value in info["episode"].items()})

    def log(self, runner, metrics, collection_time, learn_time, end, first_iteration):
        iteration = runner.current_learning_iteration
        duration = collection_time + learn_time
        self.total_steps += runner.num_steps_per_env * runner.env.num_envs
        self.total_time += duration
        scalar = {
            "Loss/value_function": metrics["value_loss"], "Loss/surrogate": metrics["surrogate_loss"],
            "Loss/learning_rate": runner.optimizer.param_groups[0]["lr"],
            "Policy/mean_noise_std": runner.actor_critic.noise_std().mean().item(),
            "Perf/total_fps": metrics["steps_per_second"],
            "Perf/collection time": collection_time, "Perf/learning_time": learn_time,
            "Perf/total_timesteps": self.total_steps, "Perf/total_time": self.total_time,
        }
        # Preserve existing AMP curves. episode_length is the historical current
        # age metric; the unambiguous name below should be used for new charts.
        scalar.update(("AMP/" + key, value) for key, value in metrics.items())
        if "state_estimator_loss" in metrics:
            scalar["Loss/state_estimator"] = metrics["state_estimator_loss"]
        scalar["AMP/current_episode_age"] = metrics["episode_length"]
        scalar.update({key: (total/self.metric_counts[key]).item() for key, total in self.metric_sums.items()})
        episode_values = {}
        for key in sorted({key for item in self.episode_infos for key in item}):
            values = [item[key].float().reshape(-1).cpu() for item in self.episode_infos if key in item]
            episode_values[key] = torch.cat(values).mean().item()
            scalar["Episode/" + key] = episode_values[key]
        if self.completed_lengths:
            scalar["Train/mean_reward"] = statistics.mean(self.returns[0])
            scalar["Train/mean_episode_length"] = statistics.mean(self.completed_lengths)
            scalar["AMP/mean_episode_task_reward"] = statistics.mean(self.returns[1])
            scalar["AMP/mean_episode_style_reward"] = statistics.mean(self.returns[2])
        if runner.writer:
            for key, value in scalar.items():
                runner.writer.add_scalar(key, value, iteration)
            if self.completed_lengths:
                for name in ("mean_reward", "mean_episode_length"):
                    runner.writer.add_scalar("Train/" + name + "/time", scalar["Train/" + name], self.total_time)
            mean, std = self.obs_mean, self.obs_std
            prefix = "Observation/obs_"
            if getattr(runner, "state_estimator_obs_index", None) is not None:
                # Historical copies would create 12,672 per-feature curves for
                # DH. Keep the same statistics for the newest 96-D frame only.
                width = runner.env.cfg.env.num_single_obs
                mean, std = mean[-width:], std[-width:]
                prefix = "Observation/current_frame_"
            for index, (mean_value, std_value) in enumerate(zip(mean.tolist(), std.tolist())):
                runner.writer.add_scalar(prefix+"mean_"+str(index), mean_value, iteration)
                runner.writer.add_scalar(prefix+"std_"+str(index), std_value, iteration)
        lines = ["#" * 80, " Learning iteration {}/{} ".format(iteration, end).center(80), ""]
        def line(label, value):
            lines.append("{:>35} {}".format(label + ":", value))
        line("Computation", "{:.0f} steps/s (collection: {:.3f}s, learning {:.3f}s)".format(metrics["steps_per_second"], collection_time, learn_time))
        line("Value function loss", "{:.4f}".format(metrics["value_loss"]))
        line("Surrogate loss", "{:.4f}".format(metrics["surrogate_loss"]))
        line("Mean action noise std", "{:.2f}".format(scalar["Policy/mean_noise_std"]))
        if self.completed_lengths:
            line("Mean reward", "{:.2f}".format(scalar["Train/mean_reward"]))
            line("Mean episode length", "{:.2f}".format(scalar["Train/mean_episode_length"]))
        for key, value in episode_values.items():
            line("Mean episode " + key, "{:.4f}".format(value))
        lines.append("-" * 80)
        if self.completed_lengths:
            line("Mean episode task reward", "{:.4f}".format(scalar["AMP/mean_episode_task_reward"]))
            line("Mean episode style reward", "{:.4f}".format(scalar["AMP/mean_episode_style_reward"]))
        for key in ("task_reward", "style_reward", "disc_loss", "grad_penalty", "expert_score", "policy_score", "entropy", "kl", "done_fraction"):
            line("AMP " + key, "{:.6f}".format(metrics[key]))
        for key in ("style_score", "kl_max", "kl_early_stop", "policy_updates", "state_estimator_loss"):
            if key in metrics:
                line("AMP " + key, "{:.6f}".format(metrics[key]))
        line("PPO learning rate", "{:.8f}".format(runner.optimizer.param_groups[0]["lr"]))
        line("Current episode age (steps)", "{:.2f}".format(metrics["episode_length"]))
        for key in sorted(self.metric_sums):
            if not key.startswith("Control/raw_target_clip_rate/"):
                line(key, "{:.6f}".format(scalar[key]))
        lines.append("-" * 80)
        line("Total timesteps", self.total_steps)
        line("Iteration time", "{:.2f}s".format(duration))
        line("Total time", "{:.2f}s".format(self.total_time))
        # Local training-call progress, so resuming at a high iteration does not
        # make ETA artificially small. Counters describe this runner session.
        completed = iteration - first_iteration
        elapsed = self.total_time - self.call_start_time
        line("ETA", "{:.1f}s".format(elapsed / completed * (end - iteration)))
        print("\n".join(lines), flush=True)
