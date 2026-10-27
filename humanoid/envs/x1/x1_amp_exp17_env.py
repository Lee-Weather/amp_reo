"""Retain exp1.6 interfaces while correcting motion incentives and scuff semantics."""
import torch

from humanoid.amp.gait_rewards import backward_progress, lowest_sole_kinematics, scuff_cost
from humanoid.amp.stationary import CommandMotionSampler, sample_directional_commands
from .x1_amp_exp16_env import X1AMPExp16Env


class X1AMPExp17Env(X1AMPExp16Env):
    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.domain_rand.randomize_com or self.cfg.domain_rand.randomize_link_com:
            raise ValueError("exp1.7 nominal COM cache requires COM randomization disabled")
        props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        self.foot_com_local = torch.tensor([[props[i].com.x,props[i].com.y,props[i].com.z]
                                            for i in self.feet_indices.tolist()], device=self.device)
        a = self.cfg.amp
        self.command_motion = CommandMotionSampler(self.motion_library,a.rsi_speed_tolerance,
            a.expert_standing_probability,a.quiet_root_speed,a.quiet_angular_speed,a.quiet_joint_speed,
            exclude_quiet_for_moving=True,relative_speed_tolerance=a.rsi_relative_speed_tolerance)

    def _draw_commands(self,count):
        return sample_directional_commands(self.command_ranges,count,self.device,
            self.cfg.commands.zero_probability,self.cfg.commands.backward_probability)

    def _reward_tracking_lin_vel(self):
        error = self.commands[:,:2]-self.base_lin_vel[:,:2]
        return torch.exp(-error[:,0].square()/self._tracking_denominator()
                         -error[:,1].square()/self.cfg.rewards.tracking_lateral_std**2)

    def _reward_backward_progress(self):
        upright = (self.projected_gravity[:,2] < -.7) & ((self.root_states[:,2]-self.env_origins[:,2]) > .45)
        valid = upright & (~self.reset_buf.bool() | self.time_out_buf)
        return backward_progress(self.commands[:,0],self.base_lin_vel[:,0])*valid

    def _reward_feet_scuff(self):
        return scuff_cost(self.scuff_point_height,self.scuff_point_speed,
                          self.commands[:,0],self.cfg.rewards.scuff_clearance)

    def compute_reward(self):
        self.scuff_point_height,self.scuff_point_speed = lowest_sole_kinematics(
            self.rigid_state[:,self.feet_indices],self.sole_vertices,self.foot_com_local,self.env_origins[:,2])
        super().compute_reward()
        moving = self.commands[:,0].abs() > .05
        m = self.extras["training_metrics"]
        m["Gait/lowest_point_speed"] = self.scuff_point_speed[moving].flatten()
        m["Gait/com_speed"] = self.foot_horizontal_speed[moving].flatten()
        m["Tracking/lateral_abs_velocity"] = self.base_lin_vel[moving,1].abs()
        m["Tracking/moving_abs_yaw_rate"] = self.base_ang_vel[moving,2].abs()
        m["Tracking/backward_progress_bonus"] = self._reward_backward_progress()[self.commands[:,0] < -.05]
