"""Stationary rewards and consistent command/reference resets for exp1.4."""
import torch

from humanoid.amp.motion import quat_inverse_rotate
from humanoid.amp.stationary import CommandMotionSampler, sample_commands
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp3_env import X1AMPExp3Env


class X1AMPExp14Env(X1AMPExp3Env):
    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.amp
        self.command_motion = CommandMotionSampler(
            self.motion_library, cfg.rsi_speed_tolerance, cfg.expert_standing_probability,
            cfg.quiet_root_speed, cfg.quiet_angular_speed, cfg.quiet_joint_speed)
        self.zero_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stop_hold_steps = torch.zeros_like(self.zero_steps)
        self.previous_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stop_pending = torch.zeros_like(self.previous_moving)
        self.stand_gate = torch.zeros(self.num_envs, device=self.device)
        gravity = self.standing_state.new_tensor([0., 0., -1.])
        self.standing_gravity = quat_inverse_rotate(self.standing_state[3:7], gravity)
        self._reset_commands = None
        print("STANDING_REFERENCE_WINDOWS {} / {}".format(
            int(self.command_motion.quiet.sum()), len(self.command_motion.times)), flush=True)

    def sample_amp_expert(self, count):
        return self.command_motion.sample_expert(count)

    def _draw_commands(self, count):
        return sample_commands(self.command_ranges, count, self.device, self.cfg.commands.zero_probability)

    def _resample_commands(self, env_ids):
        previous = self.commands[:, :3].clone()
        requested = self._reset_commands if self._reset_commands is not None else self._draw_commands(len(env_ids))
        self.standing_commands.requested[env_ids] = requested
        if self._inside_step and self._reset_reference is None:
            self.commands[:, :3] = previous
        else:
            self.commands[:, :3] = self.standing_commands.applied()

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        lib = self.motion_library
        requested = self._draw_commands(len(env_ids))
        ids, times = self.command_motion.sample_matching(requested)
        state, history = lib.state(ids, times), lib.history(ids, times)
        probability = self.cfg.amp.rsi_probability if self.cfg.amp.reference_state_initialization else 0.
        if not 0 <= probability <= 1:
            raise ValueError("rsi_probability must be in [0,1]")
        standing = torch.rand(len(env_ids), device=self.device) >= probability
        state[standing] = self.standing_state
        noise = self.cfg.amp.standing_joint_noise
        state[standing, 13:42] += (2*torch.rand(int(standing.sum()), 29, device=self.device)-1)*noise
        state[:, 13:42] = state[:, 13:42].maximum(self.hard_dof_limits[:, 0]).minimum(self.hard_dof_limits[:, 1])
        state[standing, :3] = lib.grounding.align(state[standing, :3], state[standing, 3:7], state[standing, 13:42])
        history[standing] = lib.features(state[standing])[:, None, :]
        self._reset_reference, self._reset_commands = state, requested
        self.reset_standing[env_ids] = standing
        self.standing_commands.reset(env_ids, standing)
        LeggedRobot.reset_idx(self, env_ids)
        for frame in self.obs_history:
            frame[env_ids] = 0
        self.amp_history[env_ids] = history
        self._reset_reference, self._reset_commands = None, None
        self.zero_steps[env_ids] = 0
        self.stop_hold_steps[env_ids] = 0
        self.previous_moving[env_ids] = False
        self.stop_pending[env_ids] = False
        self.stand_gate[env_ids] = 0

    def compute_reward(self):
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        self.zero_steps = torch.where(zero, self.zero_steps+1, torch.zeros_like(self.zero_steps))
        self.stand_gate = (self.zero_steps*self.dt/self.cfg.rewards.stand_ramp_seconds).clamp(max=1.)
        self.stop_pending |= zero & self.previous_moving
        self.stop_pending &= zero
        self.previous_moving.copy_(~zero)
        slow = (self.base_lin_vel[:, :2].norm(dim=-1) <= .02) & (self.base_ang_vel[:, 2].abs() <= .03)
        self.stop_hold_steps = torch.where(zero & slow, self.stop_hold_steps+1, torch.zeros_like(self.stop_hold_steps))
        hold = max(1, round(.2/self.dt))
        reached = self.stop_pending & (self.stop_hold_steps >= hold)
        braking = (self.zero_steps[reached]-hold+1)*self.dt
        self.stop_pending[reached] = False
        super().compute_reward()
        metrics = self.extras["training_metrics"]
        metrics.update({
            "Standing/command_fraction": zero.float(),
            "Standing/linear_speed": self.base_lin_vel[zero, :2].norm(dim=-1),
            "Standing/angular_speed": self.base_ang_vel[zero].norm(dim=-1),
            "Standing/joint_speed_rms": self.dof_vel[zero].square().mean(-1).sqrt(),
            "Standing/pose_rms": (self.dof_pos[zero]-self.standing_state[13:42]).square().mean(-1).sqrt(),
            "Standing/both_feet_contact": (self.contact_forces[zero][:, self.feet_indices, 2] > 20).all(-1).float(),
            "Stop/braking_time_s": braking,
            "Stop/pending_fraction": self.stop_pending.float(),
        })

    def _reward_stand_lin_vel(self):
        cost = self.base_lin_vel[:, :2].square().sum(-1)/self.cfg.rewards.stand_linear_scale**2
        return self.stand_gate*cost

    def _reward_stand_ang_vel(self):
        return self.stand_gate*self.base_ang_vel.square().sum(-1)/self.cfg.rewards.stand_angular_scale**2

    def _reward_stand_dof_vel(self):
        return self.stand_gate*self.dof_vel.square().mean(-1)/self.cfg.rewards.stand_joint_speed_scale**2

    def _reward_stand_pose(self):
        error = (self.dof_pos-self.standing_state[13:42])/self.cfg.rewards.stand_pose_scale
        return self.stand_gate*error.square().mean(-1)

    def _reward_stand_orientation(self):
        error = (self.projected_gravity-self.standing_gravity)/self.cfg.rewards.stand_gravity_scale
        return self.stand_gate*error.square().sum(-1)
