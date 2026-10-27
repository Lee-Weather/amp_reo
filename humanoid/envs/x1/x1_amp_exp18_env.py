"""Keep exp1.7 controls and gait constraints; remove the reverse return-stroke reward loophole."""
import torch

from humanoid.amp.gait_rewards import backward_velocity_reward, braking_leg_gate
from .x1_amp_exp17_env import X1AMPExp17Env


class X1AMPExp18Env(X1AMPExp17Env):
    def _reward_backward_progress(self):
        upright = (self.projected_gravity[:,2] < -.7) & ((self.root_states[:,2]-self.env_origins[:,2]) > .45)
        valid = upright & (~self.reset_buf.bool() | self.time_out_buf)
        return backward_velocity_reward(self.commands[:,0], self.base_lin_vel[:,0],
                                        self.cfg.rewards.backward_speed_scale)*valid

    def _braking_leg_gate(self):
        cfg = self.cfg.rewards
        return braking_leg_gate(self.base_lin_vel[:,:2], self.base_ang_vel[:,2], self.stop_pending,
                                cfg.brake_leg_linear_scale, cfg.brake_leg_angular_scale)

    def _reward_stand_dof_vel(self):
        return super()._reward_stand_dof_vel()*self._braking_leg_gate()

    def _reward_stand_pose(self):
        return super()._reward_stand_pose()*self._braking_leg_gate()

    def compute_reward(self):
        super().compute_reward()
        backward = self.commands[:,0] < -.05
        zero = self.commands[:,:3].abs().amax(-1) <= 1e-6
        metrics = self.extras['training_metrics']
        metrics['Tracking/backward_velocity_reward'] = self._reward_backward_progress()[backward]
        metrics['Tracking/backward_abs_vx'] = self.base_lin_vel[backward,0].abs()
        metrics['Tracking/backward_wrong_direction_fraction'] = (self.base_lin_vel[backward,0] > .05).float()
        metrics['Standing/braking_leg_gate'] = self._braking_leg_gate()[zero]
