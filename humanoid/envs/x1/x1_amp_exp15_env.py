"""The control and observation conventions remain compatible with exp1.4."""
import torch

from humanoid.amp.gait_rewards import tracking_denominator, sole_heights, scuff_cost, style_multiplier
from .x1_amp_exp14_env import X1AMPExp14Env


class X1AMPExp15Env(X1AMPExp14Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.leg_indices = [i for i, n in enumerate(self.dof_names)
                            if any(j in n for j in ("hip_", "knee_", "ankle_"))]
        if len(self.leg_indices) != 12:
            raise ValueError("Expected 12 leg joints")
        floor = self.cfg.amp.standing_style_multiplier
        if not 0 <= floor <= 1:
            raise ValueError("Standing AMP multiplier must be in [0,1]")
        lib = self.motion_library
        meshes = {lib.key_body_names[i]: mesh for i, mesh in lib.grounding.feet}
        bodies = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.sole_vertices = [meshes[bodies[i]] for i in self.feet_indices.tolist()]

    def compute_reward(self):
        feet = self.rigid_state[:, self.feet_indices]
        self.sole_height = sole_heights(feet, self.sole_vertices, self.env_origins[:, 2])
        self.foot_horizontal_speed = feet[:, :, 7:9].norm(dim=-1)
        super().compute_reward()
        # Clone before auto-reset changes velocities/stand_gate. The signed
        # speed fraction prevents a stationary policy earning full moving AMP.
        multiplier = style_multiplier(self.commands[:, 0], self.base_lin_vel[:, 0],
                                      self.stand_gate, self.cfg.amp.standing_style_multiplier)
        self.extras["amp_style_multiplier"] = multiplier.clone()
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        metrics = self.extras["training_metrics"]
        metrics["AMP/standing_style_multiplier"] = multiplier[zero]
        metrics["AMP/moving_style_multiplier"] = multiplier[self.commands[:, 0].abs() > .05]
        metrics["Standing/leg_joint_speed_rms"] = self.dof_vel[zero][:, self.leg_indices].square().mean(-1).sqrt()
        metrics["Gait/scuff_cost"] = self._reward_feet_scuff()[~zero]
        metrics["Tracking/linear_tolerance_m_s"] = self._tracking_denominator().sqrt()

    def _tracking_denominator(self):
        cfg = self.cfg.rewards
        return tracking_denominator(self.commands[:, 0], cfg.tracking_min_std,
                                    cfg.tracking_max_std, cfg.tracking_max_speed)

    def _reward_tracking_lin_vel(self):
        error = (self.commands[:, :2]-self.base_lin_vel[:, :2]).square().sum(-1)
        return torch.exp(-error/self._tracking_denominator())

    def _reward_stand_dof_vel(self):
        cost = self.dof_vel[:, self.leg_indices].square().mean(-1)
        return self.stand_gate*cost/self.cfg.rewards.stand_joint_speed_scale**2

    def _reward_stand_pose(self):
        error = (self.dof_pos[:, self.leg_indices]-self.standing_state[13:42][self.leg_indices])
        return self.stand_gate*error.square().mean(-1)/self.cfg.rewards.stand_pose_scale**2

    def _reward_feet_scuff(self):
        return scuff_cost(self.sole_height, self.foot_horizontal_speed,
                          self.commands[:, 0], self.cfg.rewards.scuff_clearance)
