"""Isolated exp_1.1 implementation; the original x1_amp task stays unchanged."""
import torch

from humanoid.amp.control import bounded_pd, StandingCommands, sync_policy_commands
from humanoid.amp.motion import quat_mul, quat_rotate
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_env import X1AMPEnv


class X1AMPExp11Env(X1AMPEnv):
    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            # Preserve importer hard bounds BEFORE the base class shrinks its
            # separate dof_pos_limits buffer to the 0.95 soft penalty interval.
            self.hard_dof_limits = torch.tensor(
                [[float(p["lower"]), float(p["upper"])] for p in props], device=self.device)
        return super()._process_dof_props(props, env_id)

    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.control.control_type != "P" or not self.cfg.control.clip_pd_targets:
            raise ValueError("exp_1.1 requires bounded position PD")
        for flag in ("add_lag", "randomize_gains", "randomize_coulomb_friction", "randomize_torque"):
            if getattr(self.cfg.domain_rand, flag, False):
                raise ValueError("exp_1.1 nominal control does not support " + flag)
        self.standing_commands = StandingCommands(self.num_envs, self.device,
                                                  round(self.cfg.amp.stand_buffer_seconds/self.dt))
        self.reset_standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pd_target_raw = torch.zeros_like(self.dof_pos)
        self.pd_target_effective = torch.zeros_like(self.dof_pos)
        self.raw_target_error_sum = torch.zeros_like(self.dof_pos)
        self.target_clip_counts = torch.zeros_like(self.dof_pos)
        self.torque_clip_counts = torch.zeros_like(self.dof_pos)
        self.posture_groups = {
            "waist": [i for i, n in enumerate(self.dof_names) if n.startswith("lumbar_")],
            "upper": [i for i, n in enumerate(self.dof_names) if any(x in n for x in ("shoulder", "elbow", "wrist"))],
            "leg_lateral": [i for i, n in enumerate(self.dof_names) if any(x in n for x in ("hip_roll", "hip_yaw", "ankle_roll"))],
        }
        if [len(self.posture_groups[k]) for k in ("waist", "upper", "leg_lateral")] != [3, 14, 6]:
            raise ValueError("Unexpected posture joint groups")
        lib = self.motion_library
        clip = lib.names.index(self.cfg.amp.standing_clip)
        time = self.cfg.amp.standing_frame / float(lib.fps[clip])
        if not 0 <= time <= float(lib.durations[clip]):
            raise ValueError("Standing frame is outside the clip")
        state = lib.state(torch.tensor([clip], device=self.device), torch.tensor([time], device=self.device))[0]
        if state[7:10].norm() > .05 or state[42:].square().mean().sqrt() > .15:
            raise ValueError("Standing candidate must have low root and joint speeds")
        # Canonicalize heading, while retaining the demonstrated roll/pitch.
        qx, qy, qz, qw = state[3:7]
        yaw = torch.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        heading_inverse = state.new_tensor([0., 0., 0., 1.])
        heading_inverse[2], heading_inverse[3] = torch.sin(-yaw/2), torch.cos(-yaw/2)
        state[3:7] = quat_mul(heading_inverse, state[3:7])
        state[:2] = 0
        state[7:13] = 0
        state[42:] = 0
        feet = lib.fk.key_positions(state[13:42])[0:2]
        feet_world = quat_rotate(state[3:7].expand(2, -1), feet) + state[:3]
        if (feet_world[:, 2].max()-feet_world[:, 2].min()) > .03:
            raise ValueError("Standing candidate feet are not near the same ground plane")
        self.standing_state = state
        self._inside_step = False

    def step(self, actions):
        self.raw_target_error_sum.zero_()
        self.target_clip_counts.zero_()
        self.torque_clip_counts.zero_()
        self._inside_step = True
        try:
            result = super().step(actions)
        finally:
            self._inside_step = False
        self.standing_commands.advance(result[3])
        self.commands[:, :3] = self.standing_commands.applied()
        sync_policy_commands(self)
        return result

    def reset(self):
        # Construct an observation at the true reset state. The base reset()
        # takes an implicit zero-action step toward the *mean walking pose*,
        # which would kick an otherwise stationary standing candidate.
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        clip = self.cfg.normalization.clip_observations
        self.obs_buf = self.obs_buf.clamp(-clip, clip)
        self.privileged_obs_buf = self.privileged_obs_buf.clamp(-clip, clip)
        return self.obs_buf, self.privileged_obs_buf

    def _compute_torques(self, actions):
        self.lagged_actions_scaled = actions*self.cfg.control.action_scale
        raw = self.default_dof_pos+self.lagged_actions_scaled+self.motor_offsets
        target, torques, error = bounded_pd(raw, self.dof_pos, self.dof_vel,
                                          self.hard_dof_limits[:, 0], self.hard_dof_limits[:, 1],
                                          self.p_gains, self.d_gains, self.torque_limits)
        self.pd_target_raw.copy_(raw)
        self.pd_target_effective.copy_(target)
        self.raw_target_error_sum.add_(error)
        self.target_clip_counts.add_((raw-target).abs() > 1e-7)
        saturated = torques.abs() >= self.torque_limits-1e-6
        self.torque_clip_counts.add_(saturated)
        if getattr(self.cfg.env, "capture_replay_state", False):
            self._replay_torque_clips += saturated.sum(-1)
        return torques

    def _resample_commands(self, env_ids):
        previous = self.commands[:, :3].clone()
        super()._resample_commands(env_ids)
        self.standing_commands.requested[env_ids] = self.commands[env_ids, :3]
        if self._inside_step and self._reset_reference is None:
            # A scheduled resample is for the NEXT policy decision, not the
            # transition just simulated with the previous observed command.
            self.commands[:, :3] = previous
        else:
            self.commands[:, :3] = self.standing_commands.applied()

    def set_evaluation_command(self, vx, vy=0., wz=0.):
        for i, (name, value) in enumerate(zip(("lin_vel_x", "lin_vel_y", "ang_vel_yaw"), (vx, vy, wz))):
            self.command_ranges[name] = [value, value]
            self.standing_commands.requested[:, i] = value
        self.commands[:, :3] = self.standing_commands.applied()
        return sync_policy_commands(self)

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        lib = self.motion_library
        probability = self.cfg.amp.rsi_probability if self.cfg.amp.reference_state_initialization else 0.
        if not 0 <= probability <= 1:
            raise ValueError("rsi_probability must be in [0,1]")
        standing = torch.rand(len(env_ids), device=self.device) >= probability
        ids, times = lib.sample_ids_times(len(env_ids))
        state = lib.state(ids, times)
        history = lib.history(ids, times)
        state[standing] = self.standing_state
        noise = self.cfg.amp.standing_joint_noise
        state[standing, 13:42] += (2*torch.rand(int(standing.sum()), 29, device=self.device)-1)*noise
        state[:, 13:42] = state[:, 13:42].maximum(self.hard_dof_limits[:, 0]).minimum(self.hard_dof_limits[:, 1])
        state[standing, :3] = lib.grounding.align(state[standing, :3], state[standing, 3:7], state[standing, 13:42])
        history[standing] = lib.features(state[standing])[:, None, :]
        self._reset_reference = state
        self.reset_standing[env_ids] = standing
        self.standing_commands.reset(env_ids, standing)
        LeggedRobot.reset_idx(self, env_ids)
        for frame in self.obs_history:
            frame[env_ids] = 0
        self.amp_history[env_ids] = history
        self._reset_reference = None

    def compute_reward(self):
        # Save branch and length BEFORE automatic reset, including terminal rows.
        super().compute_reward()
        scale = self.cfg.control.decimation
        metrics = {
            "Control/raw_target_clip_rate": self.target_clip_counts.mean(-1)/scale,
            "Control/effective_target_violation_rate": ((self.pd_target_effective < self.hard_dof_limits[:, 0]-1e-6)
                                                        | (self.pd_target_effective > self.hard_dof_limits[:, 1]+1e-6)).float().mean(-1),
            "Control/torque_saturation_rate": self.torque_clip_counts.mean(-1)/scale,
        }
        for i, name in enumerate(self.dof_names):
            metrics["Control/raw_target_clip_rate/"+name] = self.target_clip_counts[:, i]/scale
        for name, indices in self.posture_groups.items():
            metrics["Posture/"+name+"_rms_rad"] = (self.dof_pos[:, indices]-self.default_dof_pos[:, indices]).square().mean(-1).sqrt()
        for name, mask in (("standing", self.reset_standing), ("rsi", ~self.reset_standing)):
            done = mask & self.reset_buf.bool()
            metrics["Init/"+name+"_completed_length"] = self.episode_length_buf[done].float()
            metrics["Init/"+name+"_non_timeout_rate"] = (~self.time_out_buf[done]).float()
            metrics["Init/"+name+"_early_failure_rate"] = ((self.episode_length_buf[done]*self.dt <= 3.) & ~self.time_out_buf[done]).float()
        self.extras["training_metrics"] = metrics
        if getattr(self.cfg.env, "capture_replay_state", False):
            self.extras["replay_state"].update({
                "raw_target": self.pd_target_raw.clone(), "effective_target": self.pd_target_effective.clone(),
                "target_clip_counts": self.target_clip_counts.clone(), "standing_reset": self.reset_standing.clone(),
                "command_requested": self.standing_commands.requested.clone(),
                "stand_steps_left": self.standing_commands.remaining.clone(),
            })

    def _reward_raw_target_violation(self):
        return self.raw_target_error_sum.sum(-1)/self.cfg.control.decimation

    def _posture_cost(self, group):
        ids = self.posture_groups[group]
        return (self.dof_pos[:, ids]-self.default_dof_pos[:, ids]).square().sum(-1)

    def _reward_waist_deviation(self):
        return self._posture_cost("waist")

    def _reward_upper_deviation(self):
        return self._posture_cost("upper")

    def _reward_leg_lateral_deviation(self):
        return self._posture_cost("leg_lateral")

    def checkpoint_control_metadata(self):
        return {"controller": "exp11_nominal_bounded_pd_v1", "hard_limits": self.hard_dof_limits.tolist(),
                "standing_clip": self.cfg.amp.standing_clip, "standing_frame": self.cfg.amp.standing_frame,
                "reference_ground_clearance": self.cfg.amp.ground_clearance,
                "standing_state": [round(x, 6) for x in self.standing_state.tolist()],
                "stiffness": self.p_gains.tolist(), "damping": self.d_gains.tolist(),
                "torque_limits": self.torque_limits.tolist()}
