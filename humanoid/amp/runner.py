"""On-policy PPO with AMP rewards, replay, gradient penalty and resumable state."""
import json
import hashlib
import os
import time

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from .models import ActorCritic, Discriminator, ReplayBuffer
from .dh_models import ActorCriticDHAMP
from .logging import TrainingLog


class AMPOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device, self.log_dir = env, device, log_dir
        self.config = train_cfg
        self.cfg, self.alg_cfg, self.disc_cfg = [train_cfg[k] for k in ("runner", "algorithm", "discriminator")]
        policy_cfg = dict(train_cfg["policy"])
        if policy_cfg.get("motion_exploration") is not None:
            settings = dict(policy_cfg["motion_exploration"])
            names = settings.pop("joint_names")
            settings["indices"] = [env.dof_names.index(name) for name in names]
            settings["command_index"] = env.num_obs-env.cfg.env.num_single_obs
            settings["command_threshold"] = settings.pop("speed_threshold")*env.obs_scales.lin_vel
            policy_cfg["motion_exploration"] = settings
        policy_name = self.cfg.get("policy_class_name", "ActorCritic")
        self.state_estimator_obs_index = None
        if policy_name == "ActorCriticDHAMP":
            policy_cfg.update(num_short_obs=env.cfg.env.short_frame_stack*env.cfg.env.num_single_obs,
                              num_proprio_obs=env.cfg.env.num_single_obs)
            if policy_cfg["in_channels"] != env.cfg.env.frame_stack:
                raise ValueError("DH CNN channels must match the environment frame_stack")
            self.state_estimator_obs_index = self.alg_cfg.get("state_estimator_obs_index")
            if (self.state_estimator_obs_index != env.cfg.env.num_single_obs
                    or self.state_estimator_obs_index+3 != env.num_privileged_obs):
                raise ValueError("DH velocity supervision must use the critic's appended scaled body velocity")
            coefficient = self.alg_cfg.get("state_estimator_loss_coef", 1.)
            if not 0 < coefficient < float("inf"):
                raise ValueError("DH state_estimator_loss_coef must be finite and positive")
            policy_class = ActorCriticDHAMP
        elif policy_name == "ActorCritic":
            policy_class = ActorCritic
        else:
            raise ValueError("Unknown AMP policy: " + policy_name)
        self.actor_critic = policy_class(env.num_obs, env.num_privileged_obs, env.num_actions,
                                         **policy_cfg).to(device)
        lib = env.motion_library
        self.discriminator = Discriminator(lib.feature_dim, lib.history_steps,
                                            self.disc_cfg["hidden_dims"]).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.alg_cfg["learning_rate"])
        if self.alg_cfg.get("kl_check") == "minibatch":
            minimum = self.alg_cfg.get("min_learning_rate", 1e-5)
            maximum = self.alg_cfg.get("max_learning_rate", self.alg_cfg["learning_rate"])
            if not 0 < minimum <= self.alg_cfg["learning_rate"] <= maximum:
                raise ValueError("Invalid adaptive PPO learning-rate bounds")
            if self.alg_cfg.get("desired_kl", 0) <= 0:
                raise ValueError("Minibatch KL checks require positive desired_kl")
        warmup = self.alg_cfg.get("policy_transfer_warmup_iterations", 0)
        warmup_lr = self.alg_cfg.get("policy_transfer_warmup_max_lr", self.alg_cfg["learning_rate"])
        if (any(key in self.alg_cfg for key in ("policy_transfer_warmup_iterations", "policy_transfer_warmup_max_lr"))
                and (not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0
                or not self.alg_cfg.get("min_learning_rate", 1e-5) <= warmup_lr
                <= self.alg_cfg.get("max_learning_rate", self.alg_cfg["learning_rate"]))):
            raise ValueError("Invalid policy-transfer learning-rate warmup")
        self.disc_optimizer = torch.optim.Adam(self.discriminator.network.parameters(),
                                               lr=self.disc_cfg["learning_rate"],
                                               weight_decay=self.disc_cfg["weight_decay"])
        self.replay = ReplayBuffer(self.disc_cfg["replay_size"], (lib.history_steps, lib.feature_dim), device)
        if not 0 <= self.disc_cfg["task_reward_weight"] <= 1:
            raise ValueError("task_reward_weight must be in [0, 1]")
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.current_learning_iteration = 0
        self.training_origin = None
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
        if hasattr(env, "set_training_iteration"):
            # Absolute completed-update count preserves curricula after resume.
            env.set_training_iteration(self.current_learning_iteration)
        steps, n = self.num_steps_per_env, env.num_envs
        data = {"obs": torch.empty(steps, n, env.num_obs, device=self.device),
                "critic_obs": torch.empty(steps, n, env.num_privileged_obs, device=self.device),
                "actions": torch.empty(steps, n, env.num_actions, device=self.device)}
        if self.alg_cfg.get("kl_check") == "minibatch":
            for name in ("old_mean", "old_std"):
                data[name] = torch.empty_like(data["actions"])
        for name in ("log_prob", "values", "rewards", "dones", "task_rewards", "style_rewards", "style_scores"):
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
                if "old_mean" in data:
                    data["old_mean"][step].copy_(distribution.loc)
                    data["old_std"][step].copy_(distribution.scale)
                actions = distribution.sample()
                data["actions"][step].copy_(actions)
                data["log_prob"][step].copy_(distribution.log_prob(actions).sum(-1))
                data["values"][step].copy_(model.evaluate(critic))
                obs, critic, task_reward, dones, info = env.step(actions.to(env.device))
                if model.motion_std_floor > 0:
                    # Use the pre-step command and actual sampling distribution.
                    # These are nominal target noise scales before clipping.
                    command = data["obs"][step, :, model.motion_command_index]
                    moving = command.abs() > model.motion_command_threshold
                    zero = command.abs() <= 1e-6
                    selected = distribution.scale[:, model.motion_std_mask]
                    degrees = torch.rad2deg(selected*env.cfg.control.action_scale)
                    extra = info.setdefault("training_metrics", {})
                    extra["Policy/moving_selected_raw_target_noise_std_deg"] = degrees[moving].flatten()
                    extra["Policy/zero_selected_raw_target_noise_std_deg"] = degrees[zero].flatten()
                obs, critic = obs.to(self.device), critic.to(self.device)
                amp = info["amp_obs"].to(self.device)
                task_reward, dones = task_reward.to(self.device), dones.to(self.device)
                style_score = self.discriminator.style_reward(amp)
                style = style_score * self.disc_cfg["style_reward_scale"] * env.dt
                if "amp_style_multiplier" in info:
                    style = style * info["amp_style_multiplier"].to(self.device)
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
                data["style_scores"][step].copy_(style_score)
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

    def _policy_learning_rate_cap(self):
        cap = self.alg_cfg.get("max_learning_rate", self.alg_cfg["learning_rate"])
        origin = self.training_origin or {}
        if ((origin.get("mode") == "policy_transfer_fresh_critic_amp_optimizers"
                or self.alg_cfg.get("policy_transfer_warmup_for_scratch", False))
                and self.current_learning_iteration < self.alg_cfg.get("policy_transfer_warmup_iterations", 0)):
            cap = min(cap, self.alg_cfg["policy_transfer_warmup_max_lr"])
        return cap

    def update_policy(self, batch):
        if self.alg_cfg.get("policy_transfer_warmup_iterations", 0):
            for group in self.optimizer.param_groups:
                group["lr"] = min(group["lr"], self._policy_learning_rate_cap())
        cfg, model = self.alg_cfg, self.actor_critic
        if cfg.get("kl_check") == "minibatch":
            return self._update_policy_minibatch(batch)
        n = len(batch["obs"])
        if n < cfg["num_mini_batches"]:
            raise ValueError("Rollout smaller than num_mini_batches")
        totals, count = {"value_loss": 0., "surrogate_loss": 0., "entropy": 0., "kl": 0.}, 0
        if self.state_estimator_obs_index is not None:
            totals["state_estimator_loss"] = 0.
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
                estimator_loss = self._state_estimator_loss(batch["obs"][ids], batch["critic_obs"][ids])
                if estimator_loss is not None:
                    loss = loss+cfg.get("state_estimator_loss_coef", 1.)*estimator_loss
                    totals["state_estimator_loss"] += estimator_loss.item()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                model.constrain_noise_std()
                kl = ((ratio-1)-log_ratio).mean().detach().clamp(min=0)
                for key, x in zip(("value_loss", "surrogate_loss", "entropy", "kl"), (value_loss, surrogate, entropy, kl)):
                    totals[key] += x.item()
                count += 1
            if cfg.get("desired_kl") and kl > 1.5*cfg["desired_kl"]:
                break
        return {key: x/count for key, x in totals.items()}

    def _state_estimator_loss(self, obs, critic_obs):
        if self.state_estimator_obs_index is None:
            return None
        index = self.state_estimator_obs_index
        target = critic_obs[:, index:index+3].detach()
        return (self.actor_critic.estimate_velocity(obs)-target).square().mean()

    def _update_policy_minibatch(self, batch):
        """Check exact KL(old || current) before each update; adapt LR once.

        A step can still overshoot. The check prevents subsequent steps from
        continuing after that overshoot; final KL is measured over the rollout.
        """
        cfg, model = self.alg_cfg, self.actor_critic
        n = len(batch["obs"])
        if n < cfg["num_mini_batches"]:
            raise ValueError("Rollout smaller than num_mini_batches")
        target = cfg["desired_kl"]
        totals = {"value_loss": 0., "surrogate_loss": 0., "entropy": 0.}
        if self.state_estimator_obs_index is not None:
            totals["state_estimator_loss"] = 0.
        count, maximum_kl, stopped = 0, 0., False
        for _ in range(cfg["num_learning_epochs"]):
            for ids in torch.tensor_split(torch.randperm(n, device=self.device), cfg["num_mini_batches"]):
                distribution = model.distribution(batch["obs"][ids])
                with torch.no_grad():
                    old = Normal(batch["old_mean"][ids], batch["old_std"][ids])
                    kl = kl_divergence(old, distribution).sum(-1).mean().clamp(min=0)
                    if not torch.isfinite(kl):
                        raise FloatingPointError("Non-finite PPO KL")
                    maximum_kl = max(maximum_kl, kl.item())
                if kl.item() > 1.5*target:
                    stopped = True
                    break
                log_prob = distribution.log_prob(batch["actions"][ids]).sum(-1)
                ratio = (log_prob-batch["log_prob"][ids]).exp()
                advantage = batch["advantages"][ids]
                surrogate = torch.maximum(-advantage*ratio,
                                           -advantage*ratio.clamp(1-cfg["clip_param"], 1+cfg["clip_param"])).mean()
                value = model.evaluate(batch["critic_obs"][ids])
                clipped = batch["values"][ids]+(value-batch["values"][ids]).clamp(-cfg["clip_param"], cfg["clip_param"])
                value_loss = torch.maximum((value-batch["returns"][ids]).square(),
                                             (clipped-batch["returns"][ids]).square()).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = surrogate+cfg["value_loss_coef"]*value_loss-cfg["entropy_coef"]*entropy
                estimator_loss = self._state_estimator_loss(batch["obs"][ids], batch["critic_obs"][ids])
                if estimator_loss is not None:
                    loss = loss+cfg.get("state_estimator_loss_coef", 1.)*estimator_loss
                    totals["state_estimator_loss"] += estimator_loss.item()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                model.constrain_noise_std()
                for key, value in zip(("value_loss", "surrogate_loss", "entropy"), (value_loss, surrogate, entropy)):
                    totals[key] += value.item()
                count += 1
            if stopped:
                break
        # Includes the final parameter update, even if it was the last minibatch.
        final_kl = 0.
        with torch.no_grad():
            for ids in torch.tensor_split(torch.arange(n, device=self.device), cfg["num_mini_batches"]):
                current = model.distribution(batch["obs"][ids])
                old = Normal(batch["old_mean"][ids], batch["old_std"][ids])
                kl = kl_divergence(old, current).sum(-1).mean().clamp(min=0)
                if not torch.isfinite(kl):
                    raise FloatingPointError("Non-finite final PPO KL")
                final_kl += kl.item()*len(ids)/n
                maximum_kl = max(maximum_kl, kl.item())
        if cfg.get("schedule") == "adaptive":
            for group in self.optimizer.param_groups:
                lr = group["lr"]
                if maximum_kl > 1.5*target:
                    lr /= 1.5
                elif maximum_kl < .5*target:
                    lr *= 1.5
                group["lr"] = min(self._policy_learning_rate_cap(),
                                  max(cfg.get("min_learning_rate", 1e-5), lr))
        result = {key: value/max(count, 1) for key, value in totals.items()}
        result.update(kl=final_kl, kl_max=maximum_kl, kl_early_stop=float(stopped), policy_updates=count)
        return result

    def update_discriminator(self, amp):
        cfg, disc = self.disc_cfg, self.discriminator
        self.replay.insert(amp)
        totals = {"disc_loss": 0., "grad_penalty": 0., "expert_score": 0., "policy_score": 0.}
        for _ in range(cfg["num_updates"]):
            policy = self.replay.sample(cfg["batch_size"])
            sampler = getattr(self.env, "sample_amp_expert", self.env.motion_library.sample)
            expert = sampler(cfg["batch_size"]).to(self.device)
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
        if self.cfg.get("require_fresh_initialization", False):
            if self.training_origin is not None:
                raise ValueError("This scratch control must not inherit a transferred policy")
            if self.current_learning_iteration == 0:
                if self.optimizer.state or self.disc_optimizer.state or self.replay.size:
                    raise ValueError("Scratch initialization must have fresh optimizers and replay")
                print("FRESH_START_OK "+json.dumps({"iteration": 0, "training_origin": None,
                      "source_checkpoint": None, "optimizer_state_count": len(self.optimizer.state),
                      "disc_optimizer_state_count": len(self.disc_optimizer.state), "replay_size": self.replay.size,
                      "learning_rate": self.optimizer.param_groups[0]["lr"],
                      "mean_noise_std": self.actor_critic.noise_std().mean().item()}), flush=True)
        if (self.cfg.get("require_policy_transfer", False)
                and (self.training_origin or {}).get("mode") != "policy_transfer_fresh_critic_amp_optimizers"):
            raise ValueError("This task requires --warm_start with its source policy checkpoint, or an exact resume of that transferred run")
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
                           style_score=batch["style_scores"].mean().item(),
                           done_fraction=batch["dones"].mean().item(),
                           episode_length=self.env.episode_length_buf.float().mean().item())
            self.current_learning_iteration += 1
            learn_time = time.time()-learn_start
            metrics["steps_per_second"] = len(batch["obs"])/(collection_time+learn_time)
            self.training_log.log(self, metrics, collection_time, learn_time, end, first_iteration)
            # The next collect_rollout() allocates a new batch before assignment
            # releases the old one. Drop it now to avoid two full history buffers
            # coexisting (each DH rollout is 9.28 GiB at 4096x96x6336).
            del batch
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
        if self.state_estimator_obs_index is not None:
            metadata.update(schema_version=3, policy={
                "architecture": "ActorCriticDHAMP", "frame_stack": self.env.cfg.env.frame_stack,
                "short_frame_stack": self.env.cfg.env.short_frame_stack,
                "num_single_obs": self.env.cfg.env.num_single_obs,
                "history_order": "oldest_to_newest",
                "critic_velocity_index": self.state_estimator_obs_index,
                "estimated_velocity_units": "body_linear_velocity_times_obs_scale_lin_vel"})
        if self.actor_critic.max_noise_std is not None:
            metadata.update(schema_version=4, max_noise_std=self.actor_critic.max_noise_std)
        if hasattr(self.env, "checkpoint_training_metadata"):
            metadata.update(schema_version=4, training=self.env.checkpoint_training_metadata())
        return metadata

    def save(self, path, infos=None):
        torch.save({"model_state_dict": self.actor_critic.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
                    "iter": self.current_learning_iteration, "metadata": self._metadata(),
                    "infos": infos, "config": self.config, "training_origin": self.training_origin}, path)
        # Local persistence and SDK upload are separate outcomes. Log proof of
        # the former so a missing platform artifact can be traced precisely.
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        print("CHECKPOINT_SAVED "+json.dumps({"iteration": self.current_learning_iteration,
              "path": os.path.abspath(path), "bytes": os.path.getsize(path), "sha256": digest}), flush=True)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device)
        if state.get("metadata") != self._metadata():
            raise ValueError("Checkpoint robot/observation/policy conventions do not match this AMP task; MLP and DH checkpoints cannot be resumed interchangeably")
        self.actor_critic.load_state_dict(state["model_state_dict"])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.disc_optimizer.load_state_dict(state["disc_optimizer_state_dict"])
        self.current_learning_iteration = state["iter"]
        self.training_origin = state.get("training_origin")
        if load_optimizer:
            optimizer_steps = {}
            for name, optimizer in (("actor_critic", self.optimizer), ("discriminator", self.disc_optimizer)):
                steps = [float(value["step"]) for value in optimizer.state.values() if "step" in value]
                optimizer_steps[name] = {"state_count": len(optimizer.state),
                                         "minimum_step": min(steps) if steps else None,
                                         "maximum_step": max(steps) if steps else None}
            print("CHECKPOINT_RESTORED "+json.dumps({"iteration": self.current_learning_iteration,
                  "path": os.path.abspath(path), "optimizers": optimizer_steps,
                  "learning_rate": self.optimizer.param_groups[0]["lr"]}), flush=True)
        return state.get("infos")

    def warm_start(self, path, expected_sha256=None):
        """Apply the task's explicit transfer mode; legacy tasks retain all weights."""
        mode = self.cfg.get("warm_start_mode", "all_weights")
        if mode == "scratch":
            raise ValueError("This scratch control does not accept --warm_start")
        if mode == "policy_only":
            return self._warm_start_policy(path, expected_sha256)
        if mode != "all_weights":
            raise ValueError("Unknown warm-start mode: " + str(mode))
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Warm-start checkpoint SHA256 mismatch")
        self.load(path, load_optimizer=False)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.alg_cfg["learning_rate"])
        self.disc_optimizer = torch.optim.Adam(self.discriminator.network.parameters(),
                                               lr=self.disc_cfg["learning_rate"],
                                               weight_decay=self.disc_cfg["weight_decay"])
        self.training_origin = {"mode": "warm_start_fresh_optimizers", "source_file": os.path.basename(path),
                                "source_sha256": digest, "source_iteration": self.current_learning_iteration}
        print("WARM_START_OK "+json.dumps(dict(self.training_origin, learning_rate=self.alg_cfg["learning_rate"])), flush=True)

    def _warm_start_policy(self, path, expected_sha256=None):
        """Transfer DH inference weights/noise into a fresh training experiment.

        Training-reward metadata may change only through this explicit mode.
        Robot, PD, history, observation and policy conventions remain exact.
        All compatibility checks finish before any model state is changed.
        """
        if (not isinstance(self.actor_critic, ActorCriticDHAMP)
                or self.current_learning_iteration != 0 or self.training_origin is not None
                or self.optimizer.state or self.disc_optimizer.state or self.replay.size):
            raise ValueError("Policy transfer requires a fresh DH runner")
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        for expected in (expected_sha256, self.cfg.get("policy_transfer_source_sha256")):
            if expected is not None and digest != expected:
                raise ValueError("Policy-transfer checkpoint SHA256 mismatch")
        state = torch.load(path, map_location=self.device)
        source_metadata = dict(state.get("metadata", {}))
        target_metadata = dict(self._metadata())
        source_training = source_metadata.pop("training", {})
        target_metadata.pop("training", None)
        if source_metadata != target_metadata:
            raise ValueError("Policy-transfer robot/observation/policy/control conventions do not match")
        expected_curriculum = self.cfg.get("policy_transfer_source_curriculum")
        if expected_curriculum is not None and source_training.get("curriculum") != expected_curriculum:
            raise ValueError("Policy-transfer source training metadata does not match")
        source_iteration = state.get("iter")
        if not isinstance(source_iteration, int) or source_iteration < 0:
            raise ValueError("Invalid policy-transfer source iteration")
        fresh = self.actor_critic.state_dict()
        source = state["model_state_dict"]
        if source.keys() != fresh.keys():
            raise ValueError("Policy-transfer model components do not match")
        for key, value in source.items():
            if (not isinstance(value, torch.Tensor) or value.shape != fresh[key].shape
                    or value.dtype != fresh[key].dtype or not torch.isfinite(value).all()):
                raise ValueError("Invalid policy-transfer tensor: " + key)
        retained = ("actor.", "long_history.", "state_estimator.")
        transfer = {key: value for key, value in source.items()
                    if key == "log_std" or key.startswith(retained)}
        expected_keys = {key for key in fresh if not key.startswith("critic.")}
        if transfer.keys() != expected_keys:
            raise ValueError("Policy-transfer inference components do not match")
        fresh.update(transfer)
        self.actor_critic.load_state_dict(fresh, strict=True)
        self.actor_critic.constrain_noise_std()
        # Critic, D/normalizer and replay retain their fresh constructor state.
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.alg_cfg["learning_rate"])
        self.disc_optimizer = torch.optim.Adam(self.discriminator.network.parameters(),
                                               lr=self.disc_cfg["learning_rate"],
                                               weight_decay=self.disc_cfg["weight_decay"])
        self.training_origin = {
            "mode": "policy_transfer_fresh_critic_amp_optimizers",
            "source_file": os.path.basename(path), "source_sha256": digest,
            "source_iteration": source_iteration, "new_iteration": 0,
            "retained": ["actor", "long_history", "state_estimator", "log_std"],
            "reset": ["critic", "discriminator", "normalizer", "optimizers", "replay"]}
        print("POLICY_TRANSFER_OK "+json.dumps(dict(self.training_origin,
              learning_rate=self.optimizer.param_groups[0]["lr"])), flush=True)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
