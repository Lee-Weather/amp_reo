"""Keep phase-free DH inputs; compare actual leg motion with real reference histories."""
from humanoid.amp.reference_distance import ReferenceTransition
from .x1_amp_exp114_common_env import X1AMPExp114CommonEnv


class X1AMPExp114tEnv(X1AMPExp114CommonEnv):
    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        self.reference_guide = ReferenceTransition(self.command_motion, cfg.reference_pose_sigma,
                                                   cfg.reference_velocity_sigma, cfg.reference_feet_sigma)
        self._reference_cache_step = -1

    def _reward_reference_transition(self):
        if self._reference_cache_step != self.common_step_counter:
            self.reference_bonus, self.reference_score, self.reference_baseline = self.reference_guide.reward(
                self.amp_history, self.commands[:, 0])
            self._reference_cache_step = self.common_step_counter
        return self.reference_bonus*self._reference_weight()*self._reference_valid()

    def compute_reward(self):
        super().compute_reward()
        moving = self.commands[:, 0].abs() > .05
        metrics = self.extras["training_metrics"]
        metrics["Reference/transition_bonus"] = self.reference_bonus[moving]
        metrics["Reference/current_similarity"] = self.reference_score[moving]
        metrics["Reference/static_similarity"] = self.reference_baseline[moving]

    def checkpoint_training_metadata(self):
        metadata = super().checkpoint_training_metadata()
        metadata.update(reference_route="phase_free_transition_contrast_v1",
                        reference_transition_weight=self.cfg.rewards.scales.reference_transition,
                        reference_distance_sigmas=[self.cfg.rewards.reference_pose_sigma,
                                                    self.cfg.rewards.reference_velocity_sigma,
                                                    self.cfg.rewards.reference_feet_sigma])
        return metadata
