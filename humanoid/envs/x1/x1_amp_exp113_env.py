"""Keep standing control; teach motion without conditioning AMP credit on speed."""
import torch

from humanoid.amp.gait_rewards import motion_velocity_reward
from .x1_amp_exp112_env import X1AMPExp112Env


class X1AMPExp113Env(X1AMPExp112Env):
    def _reward_backward_progress(self):
        # Retired in this experiment; do not double-count reverse progress.
        return torch.zeros_like(self.commands[:, 0])

    def _reward_motion_progress(self):
        upright = ((self.projected_gravity[:, 2] < -.7)
                   & ((self.root_states[:, 2]-self.env_origins[:, 2]) > .45))
        valid = upright & (~self.reset_buf.bool() | self.time_out_buf)
        return motion_velocity_reward(self.commands[:, 0], self.base_lin_vel[:, 0],
                                      self.cfg.rewards.motion_velocity_scale)*valid

    def compute_reward(self):
        super().compute_reward()
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        vx = self.commands[:, 0]
        # Only the zero-command stand ramp reduces AMP. Neither actual speed
        # nor training iteration gates imitation credit while walking.
        standing = 1-(1-self.cfg.amp.standing_style_multiplier)*self.stand_gate
        multiplier = torch.where(zero, standing, torch.ones_like(vx))
        self.extras["amp_style_multiplier"] = multiplier.clone()
        metrics = self.extras["training_metrics"]
        metrics["AMP/standing_style_multiplier"] = multiplier[zero]
        metrics["AMP/moving_style_multiplier"] = multiplier[vx.abs() > .05]
        # The inherited speed diagnostic is useful, but is no longer an AMP gate.
        metrics["Tracking/moving_speed_quality"] = metrics.pop("AMP/moving_speed_quality")
        metrics.pop("Tracking/backward_progress_bonus", None)
        metrics.pop("Tracking/backward_velocity_reward", None)
        reward = self._reward_motion_progress()
        metrics["Tracking/forward_progress_reward"] = reward[vx > .05]
        metrics["Tracking/backward_progress_reward"] = reward[vx < -.05]

    def checkpoint_training_metadata(self):
        cfg = self.cfg.rewards
        return {"curriculum": "policy_transfer_standing_v1",
                "start_iteration": cfg.curriculum_start_iteration,
                "end_iteration": cfg.curriculum_end_iteration,
                "initial_stand_weight": cfg.curriculum_initial_stand_weight,
                "moving_style": "ungated_v1",
                "motion_reward": "normalized_huber_progress_v1",
                "motion_velocity_scale": cfg.motion_velocity_scale,
                "motion_progress_weight": cfg.scales.motion_progress,
                "backward_progress_weight": cfg.scales.backward_progress}
