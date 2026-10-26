"""On-policy PPO with AMP rewards, replay, gradient penalty and resumable state."""
import json
import os
import time

import torch
from torch import nn

from .models import ActorCritic, Discriminator, ReplayBuffer
from .logging import TrainingLog


class AMPOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device, self.log_dir = env, device, log_dir
        self.config = train_cfg
        self.cfg, self.alg_cfg, self.disc_cfg = [train_cfg[k] for k in ("runner", "algorithm", "discriminator")]
        self.actor_critic = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_actions,
                                        **train_cfg["policy"]).to(device)
        lib = env.motion_library
        self.discriminator = Discriminator(lib.feature_dim, lib.history_steps,
                                            self.disc_cfg["hidden_dims"]).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.alg_cfg["learning_rate"])
        self.disc_optimizer = torch.optim.Adam(self.discriminator.network.parameters(),
                                               lr=self.disc_cfg["learning_rate"],
                                               weight_decay=self.disc_cfg["weight_decay"])
        self.replay = ReplayBuffer(self.disc_cfg["replay_size"], (lib.history_steps, lib.feature_dim), device)
        if not 0 <= self.disc_cfg["task_reward_weight"] <= 1:
            raise ValueError("task_reward_weight must be in [0, 1]")
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.current_learning_iteration = 0
        self.training_log = TrainingLog(env.num_envs, self.device)
        self.writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(train_cfg, f, ensure_ascii=False, indent=2)
        self.env.reset()

    def collect_rollout(self):
        env, model = self.env, self.actor_critic
        steps, n = self.num_steps_per_env, env.num_envs
        data = {"obs": torch.empty(steps, n, env.num_obs, device=self.device),
                "critic_obs": torch.empty(steps, n, env.num_privileged_obs, device=self.device),
                "actions": torch.empty(steps, n, env.num_actions, device=self.device)}
        for name in ("log_prob", "values", "rewards", "dones", "task_rewards", "style_rewards"):
            data[name] = torch.empty(steps, n, device=self.device)
        data["amp"] = torch.empty(steps, n, env.motion_library.history_steps,
                                    env.motion_library.feature_dim, device=self.device)
        obs = env.get_observations().to(self.device)
        critic = env.get_privileged_observations().to(self.device)
        self.training_log.begin_rollout(obs)
        with torch.no_grad():
            for step in range(steps):
                # Copy before env.step mutates its state and returns auto-reset observations.
                data["obs"][step].copy_(obs)
                data["critic_obs"][step].copy_(critic)
                distribution = model.distribution(obs)
                actions = distribution.sample()
                data["actions"][step].copy_(actions)
                data["log_prob"][step].copy_(distribution.log_prob(actions).sum(-1))
                data["values"][step].copy_(model.evaluate(critic))
                obs, critic, task_reward, dones, info = env.step(actions.to(env.device))
                obs, critic = obs.to(self.device), critic.to(self.device)
                amp = info["amp_obs"].to(self.device)
                task_reward, dones = task_reward.to(self.device), dones.to(self.device)
                style = self.discriminator.style_reward(amp) * self.disc_cfg["style_reward_scale"] * env.dt
                weight = self.disc_cfg["task_reward_weight"]
                reward = weight*task_reward + (1-weight)*style
                # Report actual returns, never critic bootstrap corrections.
                self.training_log.record(reward, task_reward, style, dones, info)
                # Bootstrap time-limit truncation from the pre-reset critic observation.
                if "time_outs" in info:
                    timeout = info["time_outs"].to(self.device).bool()
                    terminal = info["terminal_critic_obs"].to(self.device)
                    reward = reward + self.alg_cfg["gamma"]*model.evaluate(terminal)*timeout
                data["rewards"][step].copy_(reward)
                data["dones"][step].copy_(dones)
                data["amp"][step].copy_(amp)
                data["task_rewards"][step].copy_(task_reward)
                data["style_rewards"][step].copy_(style)
            last_value = model.evaluate(critic)
            advantage = torch.zeros(n, device=self.device)
            data["returns"] = torch.empty_like(data["values"])
            for step in reversed(range(steps)):
                next_value = last_value if step == steps-1 else data["values"][step+1]
                mask = 1-data["dones"][step]
                delta = data["rewards"][step] + self.alg_cfg["gamma"]*next_value*mask-data["values"][step]
                advantage = delta + self.alg_cfg["gamma"]*self.alg_cfg["lam"]*mask*advantage
                data["returns"][step] = advantage+data["values"][step]
            advantages = data["returns"]-data["values"]
            data["advantages"] = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        return {name: x.flatten(0, 1) for name, x in data.items()}

    def update_policy(self, batch):
        cfg, model = self.alg_cfg, self.actor_critic
        n = len(batch["obs"])
        if n < cfg["num_mini_batches"]:
            raise ValueError("Rollout smaller than num_mini_batches")
        totals, count = {"value_loss": 0., "surrogate_loss": 0., "entropy": 0., "kl": 0.}, 0
        for _ in range(cfg["num_learning_epochs"]):
            for ids in torch.tensor_split(torch.randperm(n, device=self.device), cfg["num_mini_batches"]):
                distribution = model.distribution(batch["obs"][ids])
                log_prob = distribution.log_prob(batch["actions"][ids]).sum(-1)
                log_ratio = log_prob-batch["log_prob"][ids]
                ratio = log_ratio.exp()
                advantage = batch["advantages"][ids]
                surrogate = torch.maximum(-advantage*ratio,
                                           -advantage*ratio.clamp(1-cfg["clip_param"], 1+cfg["clip_param"])).mean()
                value = model.evaluate(batch["critic_obs"][ids])
                clipped = batch["values"][ids]+(value-batch["values"][ids]).clamp(-cfg["clip_param"], cfg["clip_param"])
                value_loss = torch.maximum((value-batch["returns"][ids]).square(),
                                             (clipped-batch["returns"][ids]).square()).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = surrogate+cfg["value_loss_coef"]*value_loss-cfg["entropy_coef"]*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                kl = ((ratio-1)-log_ratio).mean().detach().clamp(min=0)
                for key, x in zip(totals, (value_loss, surrogate, entropy, kl)):
                    totals[key] += x.item()
                count += 1
            if cfg.get("desired_kl") and kl > 1.5*cfg["desired_kl"]:
                break
        return {key: x/count for key, x in totals.items()}

    def update_discriminator(self, amp):
        cfg, disc = self.disc_cfg, self.discriminator
        self.replay.insert(amp)
        totals = {"disc_loss": 0., "grad_penalty": 0., "expert_score": 0., "policy_score": 0.}
        for _ in range(cfg["num_updates"]):
            policy = self.replay.sample(cfg["batch_size"])
            expert = self.env.motion_library.sample(cfg["batch_size"]).to(self.device)
            with torch.no_grad():
                disc.normalizer.update(torch.cat((policy, expert)))
            # Penalty is with respect to normalized expert input, as in the reference.
            normalized = disc.normalizer(expert).flatten(1).detach().requires_grad_(True)
            expert_score = disc.network(normalized).squeeze(-1)
            policy_score = disc(policy)
            loss = 0.5*((expert_score-1).square().mean()+(policy_score+1).square().mean())
            gradient = torch.autograd.grad(expert_score.sum(), normalized, create_graph=True)[0]
            penalty = cfg["grad_penalty"]*gradient.square().sum(-1).mean()
            if not torch.isfinite(loss+penalty):
                raise FloatingPointError("Non-finite AMP loss")
            self.disc_optimizer.zero_grad()
            (loss+penalty).backward()
            nn.utils.clip_grad_norm_(disc.network.parameters(), 1.)
            self.disc_optimizer.step()
            for key, x in zip(totals, (loss, penalty, expert_score.mean(), policy_score.mean())):
                totals[key] += x.item()
        return {key: x/cfg["num_updates"] for key, x in totals.items()}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf.random_(int(self.env.max_episode_length))
        end = self.current_learning_iteration+num_learning_iterations
        first_iteration = self.current_learning_iteration
        self.training_log.call_start_time = self.training_log.total_time
        while self.current_learning_iteration < end:
            start = time.time()
            batch = self.collect_rollout()
            collection_time = time.time()-start
            learn_start = time.time()
            metrics = self.update_policy(batch)
            metrics.update(self.update_discriminator(batch["amp"]))
            metrics.update(task_reward=batch["task_rewards"].mean().item(),
                           style_reward=batch["style_rewards"].mean().item(),
                           done_fraction=batch["dones"].mean().item(),
                           episode_length=self.env.episode_length_buf.float().mean().item())
            self.current_learning_iteration += 1
            learn_time = time.time()-learn_start
            metrics["steps_per_second"] = len(batch["obs"])/(collection_time+learn_time)
            self.training_log.log(self, metrics, collection_time, learn_time, end, first_iteration)
            if self.log_dir and self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        if self.log_dir:
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        if self.writer:
            self.writer.flush()

    def _metadata(self):
        metadata = {"schema_version": 1, "dof_names": list(self.env.dof_names), "num_obs": self.env.num_obs,
                "key_body_names": list(self.env.motion_library.key_body_names),
                "history_steps": self.env.motion_library.history_steps,
                "policy_dt": self.env.dt,
                "default_dof_pos": self.env.default_dof_pos.flatten().tolist(),
                "action_scale": self.env.cfg.control.action_scale}
        if hasattr(self.env, "checkpoint_control_metadata"):
            metadata.update(schema_version=2, control=self.env.checkpoint_control_metadata())
        return metadata

    def save(self, path, infos=None):
        torch.save({"model_state_dict": self.actor_critic.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
                    "iter": self.current_learning_iteration, "metadata": self._metadata(),
                    "infos": infos, "config": self.config}, path)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device)
        if state.get("metadata") != self._metadata():
            raise ValueError("Checkpoint robot/observation conventions do not match x1_amp")
        self.actor_critic.load_state_dict(state["model_state_dict"])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.disc_optimizer.load_state_dict(state["disc_optimizer_state_dict"])
        self.current_learning_iteration = state["iter"]
        return state.get("infos")

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
