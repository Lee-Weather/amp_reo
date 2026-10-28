"""Ease scratch learning, then restore full standing costs and AMP speed gates."""
import torch

from .x1_amp_exp18_env import X1AMPExp18Env


class X1AMPExp112Env(X1AMPExp18Env):
    # Playback defaults to full requirements; only rollout collection enables
    # the training curriculum. Episode resets never rewind its progress.
    _curriculum_progress = 1.

    def set_training_iteration(self, iteration):
        cfg = self.cfg.rewards
        start, end = cfg.curriculum_start_iteration, cfg.curriculum_end_iteration
        if iteration < 0 or not 0 <= start < end:
            raise ValueError("Invalid training curriculum iteration range")
        self._curriculum_progress = min(1., max(0., (iteration-start)/(end-start)))

    def _standing_curriculum_weight(self):
        initial = self.cfg.rewards.curriculum_initial_stand_weight
        if not 0 <= initial <= 1:
            raise ValueError("Initial standing curriculum weight must be in [0, 1]")
        return initial+(1-initial)*self._curriculum_progress

    def _reward_stand_lin_vel(self):
        return super()._reward_stand_lin_vel()*self._standing_curriculum_weight()

    def _reward_stand_ang_vel(self):
        return super()._reward_stand_ang_vel()*self._standing_curriculum_weight()

    def _reward_stand_dof_vel(self):
        return super()._reward_stand_dof_vel()*self._standing_curriculum_weight()

    def _reward_stand_pose(self):
        return super()._reward_stand_pose()*self._standing_curriculum_weight()

    def _reward_stand_orientation(self):
        return super()._reward_stand_orientation()*self._standing_curriculum_weight()

    def compute_reward(self):
        super().compute_reward()
        progress = self._curriculum_progress
        original = self.extras["amp_style_multiplier"]
        multiplier = (1-progress)+progress*original
        self.extras["amp_style_multiplier"] = multiplier
        metrics = self.extras["training_metrics"]
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        moving = self.commands[:, 0].abs() > .05
        metrics["AMP/standing_style_multiplier"] = multiplier[zero]
        metrics["AMP/moving_style_multiplier"] = multiplier[moving]
        values = torch.ones_like(self.commands[:, 0])
        metrics["Curriculum/progress"] = values*progress
        metrics["Curriculum/standing_penalty_weight"] = values*self._standing_curriculum_weight()

    def checkpoint_training_metadata(self):
        cfg = self.cfg.rewards
        return {"curriculum": "scratch_standing_and_style_v1",
                "start_iteration": cfg.curriculum_start_iteration,
                "end_iteration": cfg.curriculum_end_iteration,
                "initial_stand_weight": cfg.curriculum_initial_stand_weight}
