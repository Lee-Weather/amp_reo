"""Reward actual steps and displacement; a static pose cannot collect guidance."""
import torch

from humanoid.amp.gait_guidance import GaitWindow, MotionScales, normalized_score, span_improvement
from humanoid.amp.gait_rewards import motion_velocity_reward
from humanoid.amp.motion import quat_inverse_rotate
from .x1_amp_exp114_env import X1AMPExp114Env


class X1AMPExp115Env(X1AMPExp114Env):
    _style_iteration = 1500  # Playback uses the final definition.

    def _init_buffers(self):
        super()._init_buffers()
        self.guidance_window = GaitWindow(self.num_envs, round(self.cfg.rewards.guidance_window_seconds/self.dt)+1, self.device)
        self.motion_scales = MotionScales(self.reference_guide, self.reference_foot_fk,
                                         self.reference_guide.pitch_indices, [.1, .3, .4, .6, .65, -.2])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.guidance_window.reset(env_ids)

    def _draw_commands(self, count):
        result = torch.zeros(count, 3, device=self.device)
        # Explicit evaluation commands must survive automatic episode resets.
        if getattr(self, '_guidance_evaluation', False) or getattr(self.cfg.commands, 'guidance_evaluation', False):
            for i, key in enumerate(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')):
                result[:, i] = self.command_ranges[key][0]
            return result
        result[:, 0] = .3
        result[torch.rand(count, device=self.device) < self.cfg.commands.zero_probability] = 0.
        return result

    def set_evaluation_command(self, vx, vy=0., wz=0.):
        self._guidance_evaluation = True
        return super().set_evaluation_command(vx, vy, wz)

    def set_training_iteration(self, iteration):
        super().set_training_iteration(iteration)
        self._style_iteration = iteration

    def _style_weight(self):
        a, b = self.cfg.amp.style_start_iteration, self.cfg.amp.style_full_iteration
        if not 0 <= a < b:
            raise ValueError('Invalid AMP curriculum')
        return min(1., max(0., (self._style_iteration-a)/(b-a)))

    def compute_reward(self):
        # Read the current physical feet. AMP history is refreshed later by
        # the ancestor compute_reward, and would otherwise lag by one step.
        relative = self.rigid_state[:, self.feet_indices, :3]-self.root_states[:, None, :3]
        feet = quat_inverse_rotate(self.base_quat[:, None].expand(-1, 2, -1), relative)
        quat = self.root_states[:, 3:7]
        x, y, z, w = quat.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        heading = torch.stack((yaw.cos(), yaw.sin()), -1)
        self.guide_scales = self.motion_scales.sample(self.commands[:, 0])
        spans, displacement, ready = self.guidance_window.update(
            self.dof_pos[:, self.reference_guide.pitch_indices], feet[:, :, 0],
            self.root_states[:, :2], heading, self.commands[:, 0])
        self.guide_ready = ready
        self.guide_activity = (spans/self.guide_scales['span']).clamp(0, 1).mean(-1)*ready
        self.guide_amplitude = span_improvement(spans, self.guide_scales['span'], self.cfg.rewards.guidance_tolerance)*ready
        self.guide_window_vx = displacement/((len(self.guidance_window.data)-1)*self.dt)
        super().compute_reward()
        weight = self._style_weight()
        self.extras['amp_style_multiplier'] *= weight
        m = self.extras['training_metrics']
        moving = self.commands[:, 0].abs() > .05
        for key, value in dict(activity=self.guide_activity, amplitude=self.guide_amplitude,
                               window_vx=self.guide_window_vx, window_ready=ready.float()).items():
            m['Guidance/'+key] = value[moving]
        m['AMP/curriculum_weight'] = torch.full_like(self.commands[:, 0], weight)
        m['AMP/moving_style_multiplier'] = self.extras['amp_style_multiplier'][moving]
        zero = ~moving
        m['AMP/standing_style_multiplier'] = self.extras['amp_style_multiplier'][zero]

    def _reward_reference_pose(self):
        ids = self.reference_guide.pitch_indices
        self.reference_pose_score = normalized_score(self.dof_pos[:, ids], self.reference_q[:, ids],
                                                     self.guide_scales['q'], self.cfg.rewards.guidance_tolerance)
        return self.reference_pose_score*self.guide_activity*self._reference_weight()*self._reference_valid()

    def _reward_reference_velocity(self):
        ids = self.reference_guide.pitch_indices
        score = normalized_score(self.dof_vel[:, ids], self.reference_dq[:, ids], self.guide_scales['dq'], self.cfg.rewards.guidance_tolerance)
        held = normalized_score(torch.zeros_like(self.dof_vel[:, ids]), self.reference_dq[:, ids], self.guide_scales['dq'], self.cfg.rewards.guidance_tolerance)
        self.reference_velocity_score = score-held
        return self.reference_velocity_score*self._reference_weight()*self._reference_valid()

    def _reward_reference_feet(self):
        actual = self.amp_history[:, -1, self.reference_foot_columns]
        self.reference_feet_score = normalized_score(actual, self.reference_feet, self.guide_scales['feet'], self.cfg.rewards.guidance_tolerance)
        return self.reference_feet_score*self.guide_activity*self._reference_weight()*self._reference_valid()

    def _reward_reference_amplitude(self):
        return self.guide_amplitude*self._reference_weight()*self._reference_valid()

    def _reward_reference_support(self):
        lift = self.reference_feet.reshape(-1, 2, 3)[:, :, 2]-self.guide_scales['feet_min_z']
        swing = lift > .015
        clearance = (self.sole_height/lift.clamp(min=.015)).clamp(0, 1)
        support = (self.contact_forces[:, self.feet_indices, 2] > 20).float().flip(-1)
        score = (clearance*support*swing).sum(-1)/swing.sum(-1).clamp(min=1)
        return score*self._reference_weight()*self._reference_valid()

    def _reward_net_displacement(self):
        score = motion_velocity_reward(self.commands[:, 0], self.guide_window_vx, self.cfg.rewards.motion_velocity_scale)
        return score*self.guide_ready*self._reference_weight()*self._reference_valid()

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        result.update(curriculum='normalized_forward_guidance_v1', reference_route='normalized_phase_temporal_v1',
                      command_events={'forward_vx': .3, 'forward_probability': .8, 'zero_probability': .2},
                      guidance_window_seconds=self.cfg.rewards.guidance_window_seconds,
                      guidance_tolerance=self.cfg.rewards.guidance_tolerance,
                      amp_style_curriculum=[self.cfg.amp.style_start_iteration, self.cfg.amp.style_full_iteration],
                      temporal_weights={k:getattr(self.cfg.rewards.scales, k) for k in
                                        ('reference_amplitude', 'reference_support', 'net_displacement')})
        return result
