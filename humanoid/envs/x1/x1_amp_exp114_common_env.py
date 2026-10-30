"""Shared guidance schedule and diagnostics without changing nominal control."""
import hashlib
from pathlib import Path

import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from .x1_amp_exp113_env import X1AMPExp113Env


class X1AMPExp114CommonEnv(X1AMPExp113Env):
    _reference_progress = 1.  # Playback uses the full final reward definition.

    def _init_buffers(self):
        super()._init_buffers()
        directory = Path(self.cfg.amp.motion_dir.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.reference_source_hashes = {name: hashlib.sha256((directory/(name+".pkl")).read_bytes()).hexdigest()
                                        for name in self.motion_library.names}

    def set_training_iteration(self, iteration):
        super().set_training_iteration(iteration)
        duration = self.cfg.rewards.reference_warmup_iterations
        if duration <= 0:
            raise ValueError("Reference warmup must have positive duration")
        self._reference_progress = min(1., iteration/duration)

    def _reference_weight(self):
        initial = self.cfg.rewards.reference_initial_weight
        if not 0 <= initial <= 1:
            raise ValueError("Invalid initial reference guidance weight")
        return initial+(1-initial)*self._reference_progress

    def _reference_valid(self):
        return ((self.commands[:, 0].abs() > .05) & (self.projected_gravity[:, 2] < -.7)
                & ((self.root_states[:, 2]-self.env_origins[:, 2]) > .45)
                & (~self.reset_buf.bool() | self.time_out_buf))

    def compute_reward(self):
        super().compute_reward()
        metrics = self.extras["training_metrics"]
        metrics["Reference/guide_weight"] = torch.full_like(self.commands[:, 0], self._reference_weight())
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        for side in ("left", "right"):
            index = self.dof_names.index(side+"_knee_pitch_joint")
            error = (self.dof_pos[zero, index]-self.standing_state[13+index]).abs()
            metrics["Standing/"+side+"_knee_reference_error_deg"] = torch.rad2deg(error)

    def checkpoint_training_metadata(self):
        metadata = super().checkpoint_training_metadata()
        metadata.update(curriculum="reference_guidance_v1",
                        reference_initial_weight=self.cfg.rewards.reference_initial_weight,
                        reference_warmup_iterations=self.cfg.rewards.reference_warmup_iterations,
                        stand_pose_weight=self.cfg.rewards.scales.stand_pose,
                        raw_target_violation_weight=self.cfg.rewards.scales.raw_target_violation,
                        reference_distance_sigmas=[self.cfg.rewards.reference_pose_sigma,
                                                    self.cfg.rewards.reference_velocity_sigma,
                                                    self.cfg.rewards.reference_feet_sigma],
                        reference_source_sha256=self.reference_source_hashes)
        return metadata
