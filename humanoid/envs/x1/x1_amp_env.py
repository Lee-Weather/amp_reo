"""Full-body X1 AMP task for Isaac Gym Preview 4."""
import torch
from isaacgym import gymtorch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot import LeggedRobot
from humanoid.amp.motion import MotionLibrary, amp_features, quat_inverse_rotate


class X1AMPEnv(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        if self.num_dof != 29 or self.num_actions != self.num_dof:
            raise ValueError("x1_amp requires exactly 29 actuated joints")
        if (self.p_gains <= 0).any() or (self.d_gains <= 0).any():
            raise ValueError("Every AMP joint needs positive PD gains")
        key_ids = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n)
                   for n in self.cfg.amp.key_body_names]
        if min(key_ids) < 0 or len(self.feet_indices) != 2:
            raise ValueError("Missing AMP key bodies or feet in imported URDF")
        self.key_body_indices = torch.tensor(key_ids, device=self.device)
        self.motion_library = MotionLibrary(
            self.cfg.amp.motion_dir.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            self.dof_names, self.cfg.amp.key_body_names, self.dt, self.cfg.amp.history_steps,
            self.device, self.cfg.amp.motion_weights, self.cfg.amp.limit_tolerance)
        self.amp_history = torch.zeros(self.num_envs, self.cfg.amp.history_steps,
                                       self.motion_library.feature_dim, device=self.device)
        self._reset_reference = None

    def _get_noise_scale_vec(self, cfg):
        scale = torch.zeros(cfg.env.num_single_obs, device=self.device)
        s, o = cfg.noise.noise_scales, self.obs_scales
        scale[3:6] = s.ang_vel * o.ang_vel
        scale[6:9] = s.gravity
        scale[9:38] = s.dof_pos * o.dof_pos
        scale[38:67] = s.dof_vel * o.dof_vel
        return scale * cfg.noise.noise_level

    def _policy_frame(self):
        return torch.cat((self.commands[:, :3] * self.commands_scale,
                          self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity,
                          (self.dof_pos-self.default_dof_pos) * self.obs_scales.dof_pos,
                          self.dof_vel * self.obs_scales.dof_vel, self.actions), -1)

    def _critic_obs(self):
        return torch.cat((self._policy_frame(), self.base_lin_vel * self.obs_scales.lin_vel), -1)

    def compute_observations(self):
        frame = self._policy_frame()
        self.privileged_obs_buf = self._critic_obs()
        if self.cfg.noise.add_noise:
            frame = frame + (2*torch.rand_like(frame)-1)*self.noise_scale_vec
        self.obs_history.append(frame)
        self.obs_buf = torch.cat(list(self.obs_history), -1)

    def get_amp_observations(self):
        return self.amp_history

    def compute_reward(self):
        # Capture the true final state BEFORE LeggedRobot automatically resets.
        root = self.root_states
        relative = self.rigid_state[:, self.key_body_indices, :3] - root[:, None, :3]
        quat = self.base_quat[:, None, :].expand(-1, len(self.key_body_indices), -1)
        key_pos = quat_inverse_rotate(quat, relative)
        feature = amp_features(root[:, 2:3]-self.env_origins[:, 2:3], self.base_quat,
                               root[:, 7:10], root[:, 10:13], self.dof_pos, self.dof_vel, key_pos)
        self.amp_history[:, :-1] = self.amp_history[:, 1:].clone()
        self.amp_history[:, -1] = feature
        self.extras["amp_obs"] = self.amp_history.clone()
        self.extras["terminal_critic_obs"] = self._critic_obs().clone()
        # Base reset_idx only writes this on reset steps; refresh it every step
        # so a previous timeout mask cannot leak into a subsequent transition.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()
        super().compute_reward()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.amp.reference_state_initialization:
            ids, times = self.motion_library.sample_ids_times(len(env_ids))
            self._reset_reference = self.motion_library.state(ids, times)
            history = self.motion_library.history(ids, times)
        else:
            self._reset_reference = None
            history = None
        super().reset_idx(env_ids)
        for frame in self.obs_history:
            frame[env_ids] = 0
        if history is not None:
            self.amp_history[env_ids] = history
        else:
            states = torch.cat((self.root_states[env_ids], self.dof_pos[env_ids], self.dof_vel[env_ids]), -1)
            self.amp_history[env_ids] = self.motion_library.features(states)[:, None, :]
        self._reset_reference = None

    def _reset_dofs(self, env_ids):
        if self._reset_reference is None:
            return super()._reset_dofs(env_ids)
        self.dof_pos[env_ids] = self._reset_reference[:, 13:42]
        self.dof_vel[env_ids] = self._reset_reference[:, 42:71]
        indices = env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(indices), len(indices))

    def _reset_root_states(self, env_ids):
        if self._reset_reference is None:
            return super()._reset_root_states(env_ids)
        self.root_states[env_ids] = self._reset_reference[:, :13]
        self.root_states[env_ids, :2] = self.env_origins[env_ids, :2]
        self.root_states[env_ids, 2] += self.env_origins[env_ids, 2]
        indices = env_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(indices), len(indices))

    def _resample_commands(self, env_ids):
        # Keep small walking commands supported by the slow reference dataset.
        for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
            lo, hi = self.command_ranges[name]
            self.commands[env_ids, i] = lo+(hi-lo)*torch.rand(len(env_ids), device=self.device)

    def _reward_tracking_lin_vel(self):
        return torch.exp(-(self.commands[:, :2]-self.base_lin_vel[:, :2]).square().sum(-1)/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        return torch.exp(-(self.commands[:, 2]-self.base_ang_vel[:, 2]).square()/self.cfg.rewards.tracking_sigma)

    def _reward_alive(self):
        return (~self.reset_buf.bool()).float()

    def _reward_lin_vel_z(self):
        return self.base_lin_vel[:, 2].square()

    def _reward_ang_vel_xy(self):
        return self.base_ang_vel[:, :2].square().sum(-1)

    def _reward_orientation(self):
        return self.projected_gravity[:, :2].square().sum(-1)

    def _reward_dof_vel(self):
        return self.dof_vel.square().sum(-1)

    def _reward_dof_acc(self):
        return ((self.dof_vel-self.last_dof_vel)/self.dt).square().sum(-1)

    def _reward_action_rate(self):
        return (self.actions-self.last_actions).square().sum(-1)

    def _reward_torques(self):
        return self.torques.square().sum(-1)

    def _reward_dof_pos_limits(self):
        return ((self.dof_pos_limits[:, 0]-self.dof_pos).clamp(min=0)
                + (self.dof_pos-self.dof_pos_limits[:, 1]).clamp(min=0)).sum(-1)

    def _reward_collision(self):
        return (self.contact_forces[:, self.penalised_contact_indices].norm(dim=-1) > 1.).float().sum(-1)
