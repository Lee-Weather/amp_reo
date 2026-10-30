"""A state-aligned phase clock supplies dense motion guidance, without reference actions."""
import math
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.amp.motion import URDFKinematics
from humanoid.amp.phase_reference import PhaseReference
from .x1_amp_exp114_common_env import X1AMPExp114CommonEnv


class X1AMPExp114Env(X1AMPExp114CommonEnv):
    def _init_buffers(self):
        super()._init_buffers()
        self.reference_guide = PhaseReference(self.motion_library, self.standing_state[13:42], self.cfg.amp.reference_cycles)
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.previous_phase_command = torch.zeros_like(self.gait_phase)
        feet = [side+"_ankle_roll_link" for side in ("left", "right")]
        self.reference_foot_fk = URDFKinematics(
            self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), self.dof_names, feet, self.device)
        start = 10+2*len(self.dof_names)
        self.reference_foot_columns = [start+3*self.motion_library.key_body_names.index(name)+axis
                                       for name in feet for axis in range(3)]

    def _phase_features(self):
        angle = 2*math.pi*self.gait_phase
        return torch.stack((angle.sin(), angle.cos()), -1)*(self.commands[:, 0].abs() > .05)[:, None]

    def _policy_frame(self):
        return torch.cat((super()._policy_frame(), self._phase_features()), -1)

    def _sync_phase_observation(self):
        command = self.commands[:, 0]
        changed = (command-self.previous_phase_command).abs() > 1e-6
        ids = changed.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.gait_phase[ids] = self.reference_guide.nearest_phase(self.dof_pos[ids], self.dof_vel[ids], command[ids])
        self.previous_phase_command.copy_(command)
        phase = self._phase_features()
        self.obs_history[-1][:, -2:] = phase
        self.obs_buf[:, -2:] = phase
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf[:, 96:98] = phase
        return self.obs_buf

    def step(self, actions):
        result = super().step(actions)
        # The inherited standing buffer can apply a newly requested command
        # after observations were built. Synchronize its phase in that frame.
        self._sync_phase_observation()
        return result

    def set_evaluation_command(self, vx, vy=0., wz=0.):
        super().set_evaluation_command(vx, vy, wz)
        return self._sync_phase_observation()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        moving = self.commands[:, 0].abs() > .05
        increment = self.reference_guide.phase_rate(self.commands[:, 0])*self.dt
        self.gait_phase = torch.where(moving, (self.gait_phase+increment).remainder(1), self.gait_phase)

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        super().reset_idx(env_ids)
        self.gait_phase[env_ids] = self.reference_guide.nearest_phase(
            self.dof_pos[env_ids], self.dof_vel[env_ids], self.standing_commands.requested[env_ids, 0])
        self.previous_phase_command[env_ids] = self.commands[env_ids, 0]

    def compute_reward(self):
        self.reference_q, self.reference_dq = self.reference_guide.sample(self.commands[:, 0], self.gait_phase)
        self.reference_feet = self.reference_foot_fk.key_positions(self.reference_q).flatten(-2)
        super().compute_reward()
        moving = self.commands[:, 0].abs() > .05
        metrics = self.extras["training_metrics"]
        metrics["Reference/pose_score"] = self.reference_pose_score[moving]
        metrics["Reference/velocity_score"] = self.reference_velocity_score[moving]
        metrics["Reference/feet_score"] = self.reference_feet_score[moving]
        metrics["Reference/phase_rate_hz"] = self.reference_guide.phase_rate(self.commands[:, 0])[moving]

    def _reward_reference_pose(self):
        ids = self.reference_guide.pitch_indices
        difference = (self.dof_pos[:, ids]-self.reference_q[:, ids])/self.cfg.rewards.reference_pose_sigma
        self.reference_pose_score = (-.5*difference.square().mean(-1)).exp()
        return self.reference_pose_score*self._reference_weight()*self._reference_valid()

    def _reward_reference_velocity(self):
        ids = self.reference_guide.pitch_indices
        difference = (self.dof_vel[:, ids]-self.reference_dq[:, ids])/self.cfg.rewards.reference_velocity_sigma
        self.reference_velocity_score = (-.5*difference.square().mean(-1)).exp()
        return self.reference_velocity_score*self._reference_weight()*self._reference_valid()

    def _reward_reference_feet(self):
        actual = self.amp_history[:, -1, self.reference_foot_columns]
        difference = (actual-self.reference_feet)/self.cfg.rewards.reference_feet_sigma
        self.reference_feet_score = (-.5*difference.square().mean(-1)).exp()
        return self.reference_feet_score*self._reference_weight()*self._reference_valid()

    def checkpoint_training_metadata(self):
        metadata = super().checkpoint_training_metadata()
        metadata.update(reference_route="phase_tracking_v1", phase_input_columns=[96, 97],
                        phase_input_order=["sin", "cos"], cycles=self.reference_guide.metadata,
                        reference_weights={"pose": self.cfg.rewards.scales.reference_pose,
                                           "velocity": self.cfg.rewards.scales.reference_velocity,
                                           "feet": self.cfg.rewards.scales.reference_feet})
        return metadata
