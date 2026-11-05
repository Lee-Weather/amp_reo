"""exp2.13 environment: heading feedback, clearance and load symmetry.

The 98-D frame is unchanged. The otherwise unused vy command column carries
the wrapped yaw error relative to the reset heading, both in the policy frame
and in every stored history frame. Rewards still use vy=0.
"""
import torch

from humanoid.envs.base.legged_robot import get_euler_xyz_tensor
from humanoid.amp.exp20_rewards import huber
from humanoid.amp.heading import heading_error, heading_observation, clearance_shortfall, contact_share
from .x1_amp_env import X1AMPFinalEnv


class X1AMPExp213Env(X1AMPFinalEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.heading_reference = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.contact_share_ema = torch.full((self.num_envs, 2), 0.5, device=self.device)

    def reset_idx(self, ids):
        super().reset_idx(ids)
        env_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()
        if len(env_ids):
            self.heading_reference[env_ids] = get_euler_xyz_tensor(
                self.root_states[env_ids, 3:7])[:, 2]
            self.heading_error[env_ids] = 0.
            self.contact_share_ema[env_ids] = 0.5

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        yaw = get_euler_xyz_tensor(self.base_quat)[:, 2]
        self.heading_error[:] = heading_error(yaw, self.heading_reference)

    def _heading_observation(self):
        return heading_observation(self.heading_error, self.cfg.rewards.heading_clip,
                                   self.cfg.rewards.heading_obs_scale)

    def _policy_frame(self):
        frame = super()._policy_frame()
        # Index 1 is the unused vy command in this straight-line task.
        frame[..., 1] = self._heading_observation()
        return frame

    def _apply_heading_observation(self):
        value = self._heading_observation()
        self.obs_history[-1][:, 1] = value
        width = self.cfg.env.num_single_obs
        self.obs_buf[:, -width + 1] = value
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:, 1] = value

    def step(self, actions):
        result = super().step(actions)
        # sync_policy_commands rewrites the newest vy slot with zero; restore
        # the heading observation after the simulator-side update.
        self._apply_heading_observation()
        return result

    def _reference_weight(self):
        return self.cfg.rewards.guide_weight_override

    def _raw_weight(self):
        return self.cfg.rewards.raw_weight_override

    def _heading_valid(self):
        upright = (self.projected_gravity[:, 2] < -.7) & (
            (self.root_states[:, 2] - self.env_origins[:, 2]) > .45)
        return upright & (~self.reset_buf.bool() | self.time_out_buf)

    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        dt = self.dt
        valid = self._heading_valid()
        error = self.heading_error
        tolerance = max(float(cfg.heading_tolerance), 1e-6)
        heading_cost = huber(error / tolerance)
        self.rew_buf += float(cfg.heading_scale) * heading_cost * dt * valid

        details = getattr(self, "dense_swing_details", None)
        clearance = torch.zeros(self.num_envs, device=self.device)
        if details is not None:
            clearance = clearance_shortfall(details["height"], details["phase"],
                                            cfg.swing_clearance_threshold)
            self.rew_buf += float(cfg.swing_clearance_penalty) * clearance * dt * self._reference_valid()

        load = self.contact_forces[:, self.feet_indices, 2]
        share = contact_share(load)
        decay = float(cfg.contact_balance_decay)
        self.contact_share_ema.mul_(decay).add_(share * (1. - decay))
        imbalance = (self.contact_share_ema[:, 0] - self.contact_share_ema[:, 1]).abs()
        self.rew_buf += float(cfg.contact_balance_scale) * imbalance.square() * dt * valid

        metrics = self.extras.setdefault("training_metrics", {})
        metrics["Heading/error_rad"] = error
        metrics["Heading/cost"] = heading_cost
        metrics["Heading/valid"] = valid.float()
        metrics["Gait/clearance_shortfall"] = clearance
        metrics["Symmetry/contact_share_difference"] = imbalance

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        cfg = self.cfg.rewards
        result.update(
            heading_observation="vy_command_slot_v1",
            heading_tolerance=float(cfg.heading_tolerance),
            heading_clip=float(cfg.heading_clip),
            heading_obs_scale=float(cfg.heading_obs_scale),
            heading_scale=float(cfg.heading_scale),
            swing_clearance_penalty=float(cfg.swing_clearance_penalty),
            contact_balance_scale=float(cfg.contact_balance_scale),
            guide_weight_override=float(cfg.guide_weight_override),
            raw_weight_override=float(cfg.raw_weight_override),
        )
        return result