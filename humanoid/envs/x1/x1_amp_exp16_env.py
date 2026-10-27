"""Keep exp1.5 control and standing rewards; refine moving reward/exploration diagnostics."""
import torch

from humanoid.amp.gait_rewards import moving_speed_quality
from .x1_amp_exp15_env import X1AMPExp15Env


class X1AMPExp16Env(X1AMPExp15Env):
    def compute_reward(self):
        super().compute_reward()
        vx, actual = self.commands[:, 0], self.base_lin_vel[:, 0]
        moving = vx.abs() > .05
        quality = moving_speed_quality(vx, actual, self.cfg.amp.motion_style_velocity_sigma)
        multiplier = self.extras["amp_style_multiplier"]*quality
        self.extras["amp_style_multiplier"] = multiplier.clone()
        metrics = self.extras["training_metrics"]
        metrics["AMP/moving_speed_quality"] = quality[moving]
        metrics["AMP/moving_style_multiplier"] = multiplier[moving]
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        groups = {"backward": vx < -.05, "slow_forward": (vx > .05) & (vx <= .2),
                  "forward": vx > .2, "zero": zero}
        # Applied commands and episode events are captured before auto-reset.
        # Signed velocity alone is not a walking-success metric (falls can be fast).
        for name, mask in groups.items():
            metrics["Commands/"+name+"_applied_fraction"] = mask.float()
            metrics["Tracking/"+name+"_vx"] = actual[mask]
            metrics["Tracking/"+name+"_abs_vx_error"] = (actual-vx)[mask].abs()
            ended = self.reset_buf.bool() & mask
            metrics["Episode/"+name+"_completed_length"] = self.episode_length_buf[ended].float()
            metrics["Episode/"+name+"_non_timeout_fraction"] = (~self.time_out_buf[ended]).float()
