"""exp3.0 MLP environment: flat-ground step command plus heading feedback."""
import torch

from humanoid.envs.base.legged_robot import get_euler_xyz_tensor
from humanoid.amp.exp20_rewards import huber
from humanoid.amp.heading import heading_error, heading_observation
from .x1_amp_exp18_env import X1AMPExp18Env


class X1AMPExp30Env(X1AMPExp18Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.heading_reference = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if len(ids):
            self.heading_reference[ids] = get_euler_xyz_tensor(self.root_states[ids, 3:7])[:, 2]
            self.heading_error[ids] = 0.

    def _heading_command(self):
        cfg = self.cfg.rewards
        moving = ((self.commands[:, 0].abs() > cfg.heading_command_threshold)
                  | (self.standing_commands.remaining > 0))
        correction = (-cfg.heading_command_gain * self.heading_error).clamp(
            -cfg.heading_command_clip, cfg.heading_command_clip)
        return torch.where(moving, correction, torch.zeros_like(correction))

    def _heading_observation(self):
        return heading_observation(self.heading_error, self.cfg.rewards.heading_clip,
                                   self.cfg.rewards.heading_obs_scale)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        yaw = get_euler_xyz_tensor(self.base_quat)[:, 2]
        self.heading_error[:] = heading_error(yaw, self.heading_reference)
        self.commands[:, 2] = self._heading_command()

    def _policy_frame(self):
        frame = super()._policy_frame()
        frame[..., 1] = self._heading_observation()
        return frame

    def _apply_heading_observation(self):
        value = self._heading_observation()
        self.obs_history[-1][:, 1] = value
        width = self.cfg.env.num_single_obs
        self.obs_buf[:, -width + 1] = value
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:, 1] = value

    def _apply_heading_command(self):
        value = self._heading_command()
        self.commands[:, 2] = value
        scaled = value * self.commands_scale[2]
        self.obs_history[-1][:, 2] = scaled
        width = self.cfg.env.num_single_obs
        self.obs_buf[:, -width + 2] = scaled
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:, 2] = scaled

    def step(self, actions):
        result = super().step(actions)
        # standing_commands.applied() and sync_policy_commands rewrite the
        # newest vy/wz slots with zero; restore heading after the physics step.
        self._apply_heading_observation()
        self._apply_heading_command()
        return result

    def _heading_valid(self):
        upright = (self.projected_gravity[:, 2] < -.7) & (
            (self.root_states[:, 2] - self.env_origins[:, 2]) > .45)
        return upright & (~self.reset_buf.bool() | self.time_out_buf)

    def compute_reward(self):
        super().compute_reward()
        cfg = self.cfg.rewards
        error = self.heading_error
        valid = self._heading_valid()
        tolerance = max(float(cfg.heading_tolerance), 1e-6)
        cost = huber(error / tolerance)
        self.rew_buf += float(cfg.heading_scale) * cost * self.dt * valid
        metrics = self.extras.setdefault("training_metrics", {})
        metrics["Heading/error_rad"] = error
        metrics["Heading/cost"] = cost
        metrics["Heading/valid"] = valid.float()