"""Immediate phase feedback, with fixed poses receiving no guidance credit."""
import torch

from humanoid.amp.motion import quat_inverse_rotate
from humanoid.amp.phase_rewards import phase_quality, phase_progress, phase_velocity_credit, phase_support_credit
from .x1_amp_exp115_env import X1AMPExp115Env


class X1AMPExp116Env(X1AMPExp115Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.previous_guidance_q = self.dof_pos.clone()
        self.previous_guidance_feet = self._guidance_feet().clone()
        self.previous_guidance_command = self.commands[:, 0].clone()
        self.guidance_transition_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _guidance_feet(self):
        relative = self.rigid_state[:, self.feet_indices, :3]-self.root_states[:, None, :3]
        return quat_inverse_rotate(self.base_quat[:, None].expand(-1, 2, -1), relative).flatten(-2)

    def step(self, actions):
        # Capture before physics, not after the auto-reset writes a new pose.
        self.previous_guidance_q.copy_(self.dof_pos)
        self.previous_guidance_feet.copy_(self._guidance_feet())
        self.previous_guidance_command.copy_(self.commands[:, 0])
        self.guidance_transition_valid.copy_(self.episode_length_buf > 0)
        return super().step(actions)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, 'guidance_transition_valid'):
            self.guidance_transition_valid[env_ids] = False

    def _transition_valid(self):
        return (self.guidance_transition_valid
                & ((self.commands[:, 0]-self.previous_guidance_command).abs() <= 1e-6))

    def _style_weight(self):
        weight = self.cfg.amp.fixed_style_weight
        if not 0 <= weight <= 1:
            raise ValueError('Invalid fixed AMP style weight')
        return weight

    def _reward_reference_pose(self):
        ids = self.reference_guide.pitch_indices
        q, target = self.dof_pos[:, ids], self.reference_q[:, ids]
        self.reference_pose_score = phase_quality(q, target, self.guide_scales['q'])
        progress = phase_progress(self.previous_guidance_q[:, ids], q, target, self.guide_scales['q'], self.dt)
        self.phase_pose_progress = progress
        return progress*self._transition_valid()*self._reference_valid()*self._reference_weight()

    def _reward_reference_velocity(self):
        ids = self.reference_guide.pitch_indices
        self.reference_velocity_score = phase_velocity_credit(
            self.dof_vel[:, ids], self.reference_dq[:, ids], self.guide_scales['dq'])
        return self.reference_velocity_score*self._reference_valid()*self._reference_weight()

    def _reward_reference_feet(self):
        actual = self._guidance_feet()
        self.reference_feet_score = phase_quality(actual, self.reference_feet, self.guide_scales['feet'])
        progress = phase_progress(self.previous_guidance_feet, actual, self.reference_feet, self.guide_scales['feet'], self.dt)
        self.phase_feet_progress = progress
        return progress*self._transition_valid()*self._reference_valid()*self._reference_weight()

    def _reward_reference_support(self):
        lift = self.reference_feet.reshape(-1, 2, 3)[:, :, 2]-self.guide_scales['feet_min_z']
        credit = phase_velocity_credit(self.dof_vel[:, self.reference_guide.pitch_indices],
                                       self.reference_dq[:, self.reference_guide.pitch_indices], self.guide_scales['dq'])
        support = phase_support_credit(self.sole_height, self.contact_forces[:, self.feet_indices, 2], lift, credit)
        return support*self._reference_valid()*self._reference_weight()

    def compute_reward(self):
        super().compute_reward()
        moving = self.commands[:, 0].abs() > .05
        metrics = self.extras['training_metrics']
        metrics['Guidance/pose_progress'] = self.phase_pose_progress[moving]
        metrics['Guidance/feet_progress'] = self.phase_feet_progress[moving]
        metrics['Guidance/transition_valid'] = self._transition_valid()[moving].float()
        metrics['Reference/phase'] = self.gait_phase[moving]
        if getattr(self.cfg.env, 'capture_replay_state', False):
            self.extras['replay_state'].update(reference_phase=self.gait_phase.clone(),
                reference_q=self.reference_q.clone(), reference_dq=self.reference_dq.clone())

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        result.update(curriculum='dense_phase_refinement_v1', reference_route='signed_per_coordinate_phase_progress_v1',
                      fixed_amp_style_weight=self.cfg.amp.fixed_style_weight,
                      phase_progress_gate='same_episode_same_command_one_step; no two-second gate',
                      phase_quality_metric='mean reciprocal per-coordinate Huber error; diagnostic, not static reward',
                      support_rule='soft phase lift/opposite stance/swing unloading times positive velocity credit')
        return result
