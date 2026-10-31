"""Absolute phase credit, collision-sole support and phase-aligned forward resets."""
import json

import torch

from humanoid.amp.gait_rewards import sole_heights
from humanoid.amp.phase_initialization import ForwardPhaseBank
from humanoid.amp.phase_rewards import (absolute_phase_credit, foot_support_score,
                                       static_cycle_baseline, static_velocity_baseline)
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp116_env import X1AMPExp116Env


class X1AMPExp117Env(X1AMPExp116Env):
    """Exp1.16 policy retained; dense credit and reset distribution are rebuilt."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        if cfg.only_positive_rewards:
            raise ValueError("exp1.17 requires negative reward components")
        if cfg.phase_credit_negative_floor > 0 or cfg.phase_credit_positive_ceiling <= 0:
            raise ValueError("Phase credit bounds must straddle zero")
        if not 0 <= cfg.reference_coupling_floor <= 1:
            raise ValueError("Reference coupling floor must be in [0, 1]")
        if cfg.reference_warmup_iterations <= 0 or cfg.forward_rsi_state_frames < 16:
            raise ValueError("Invalid reference warmup or phase bank size")
        self.forward_speed = float(self.cfg.commands.ranges.lin_vel_x[1])
        self._iteration = int(cfg.forward_rsi_source_iteration)
        names = self.dof_names
        self.pitch_indices = list(self.reference_guide.pitch_indices)
        self.lateral_indices = [i for i, n in enumerate(names)
                                if any(j in n for j in ("hip_roll", "hip_yaw", "ankle_roll"))]
        self.waist_indices = [i for i, n in enumerate(names) if n.startswith("lumbar_")]
        if [len(self.pitch_indices), len(self.lateral_indices), len(self.waist_indices)] != [6, 6, 3]:
            raise ValueError("Unexpected phase guidance joint groups")
        self.phase_bank = None
        if self.forward_speed > 0:
            clip, first, last = self.cfg.amp.reference_cycles[0]
            self.phase_bank = ForwardPhaseBank(
                self.reference_guide, self.motion_library, clip, first, last, self.forward_speed,
                self.hard_dof_limits[:, 0], self.hard_dof_limits[:, 1],
                phases=cfg.forward_rsi_state_frames, history_steps=self.cfg.amp.history_steps,
                dt=self.dt, clearance=self.cfg.amp.ground_clearance)
            print("EXP117_PHASE_BANK " + json.dumps(self.phase_bank.report, sort_keys=True), flush=True)
            if self.phase_bank.report["stance_slip_exceeds_0_10_m_s"]:
                print("EXP117_PHASE_BANK_WARNING reference stance foot slip exceeds 0.10 m/s; "
                      "the kinematic target is not contact-consistent", flush=True)
        self._build_static_baseline()

    def _build_static_baseline(self):
        cfg = self.cfg.rewards
        self.static_count = 256 if self.forward_speed > 0 else 1
        phases = (torch.arange(self.static_count, device=self.device, dtype=torch.float32)+.5)/self.static_count
        command = torch.full_like(phases, max(self.forward_speed, 0.))
        q, dq = self.reference_guide.sample(command, phases)
        feet = self.reference_foot_fk.key_positions(q).flatten(-2)
        self.static_pose = static_cycle_baseline(q[:, self.pitch_indices], q.new_full((1,), cfg.reference_pose_scale))
        self.static_velocity = static_velocity_baseline(dq[:, self.pitch_indices], q.new_full((1,), cfg.reference_velocity_scale))
        self.static_feet = static_cycle_baseline(feet, q.new_full((1,), cfg.reference_feet_scale))
        self.static_lateral = static_cycle_baseline(q[:, self.lateral_indices], q.new_full((1,), cfg.reference_lateral_scale))
        self.static_waist = static_cycle_baseline(q[:, self.waist_indices], q.new_full((1,), cfg.reference_waist_scale))

    def _baseline(self, table):
        index = (self.gait_phase*self.static_count).floor().long().clamp(max=self.static_count-1)
        return table[index]

    def set_training_iteration(self, iteration):
        super().set_training_iteration(iteration)
        self._iteration = int(iteration)
        cfg = self.cfg.rewards
        additional = max(0., float(iteration-cfg.forward_rsi_source_iteration))
        self._reference_progress = min(1., additional/cfg.reference_warmup_iterations)

    def _rsi_probability(self):
        if getattr(self, "_guidance_evaluation", False) or getattr(self.cfg.commands, "guidance_evaluation", False):
            return 1. if self.cfg.amp.reference_state_initialization else 0.
        if not self.cfg.amp.reference_state_initialization or self.phase_bank is None:
            return 0.
        cfg = self.cfg.rewards
        relative = max(0., float(self._iteration-cfg.forward_rsi_source_iteration))
        schedule = sorted(cfg.forward_rsi_schedule)
        if relative <= schedule[0][0]:
            return float(schedule[0][1])
        for (start, first), (end, last) in zip(schedule[:-1], schedule[1:]):
            if start <= relative <= end:
                if end == start:
                    return float(last)
                fraction = (relative-start)/(end-start)
                return float(first+(last-first)*fraction)
        return float(schedule[-1][1])

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        requested = self._draw_commands(len(env_ids))
        forward = requested[:, 0].abs() > .05
        rsi = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        probability = self._rsi_probability()
        if probability > 0 and bool(forward.any()):
            rsi[forward] = torch.rand(int(forward.sum()), device=self.device) < probability
        standing = ~rsi
        state = self.standing_state.expand(len(env_ids), -1).clone()
        history = self.motion_library.features(state)[:, None, :].expand(
            -1, self.cfg.amp.history_steps, -1).clone()
        phase = torch.zeros(len(env_ids), device=self.device)
        if rsi.any():
            states, histories, phases = self.phase_bank.sample(int(rsi.sum()))
            state[rsi], history[rsi], phase[rsi] = states, histories, phases
        if standing.any():
            noise = self.cfg.amp.standing_joint_noise
            state[standing, 13:42] += (2*torch.rand(int(standing.sum()), 29, device=self.device)-1)*noise
        state[:, 13:42] = state[:, 13:42].maximum(self.hard_dof_limits[:, 0]).minimum(self.hard_dof_limits[:, 1])
        if standing.any():
            state[standing, :3] = self.motion_library.grounding.align(
                state[standing, :3], state[standing, 3:7], state[standing, 13:42])
            history[standing] = self.motion_library.features(state[standing])[:, None, :]
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
        self.gait_phase[env_ids] = phase
        # A forward command that starts standing still needs its nearest phase.
        restart = (standing & forward).nonzero(as_tuple=False).flatten()
        if len(restart):
            absolute = env_ids[restart]
            self.gait_phase[absolute] = self.reference_guide.nearest_phase(
                self.dof_pos[absolute], self.dof_vel[absolute], requested[restart, 0])
        self.previous_phase_command[env_ids] = self.commands[env_ids, 0]

    def _phase_coupling(self):
        moving = self.commands[:, 0].abs() > .05
        command = self.commands[:, 0].abs().clamp(min=1e-6)
        aligned = (self.base_lin_vel[:, 0]*self.commands[:, 0] >= 0).float()
        progress = (self.base_lin_vel[:, 0].abs()/command).clamp(0, 1)*aligned
        floor = float(self.cfg.rewards.reference_coupling_floor)
        return torch.where(moving, floor+(1-floor)*progress, torch.ones_like(progress))

    def _phase_credit(self, actual, reference, baseline, scale):
        if baseline.shape != reference.shape:
            raise ValueError("Static baseline does not match current reference")
        cfg = self.cfg.rewards
        return absolute_phase_credit(actual, reference, baseline, reference.new_full(reference.shape, scale),
                                     cfg.phase_credit_negative_floor, cfg.phase_credit_positive_ceiling)

    def _reward_reference_pose(self):
        credit = self._phase_credit(self.dof_pos[:, self.pitch_indices],
                                    self.reference_q[:, self.pitch_indices],
                                    self._baseline(self.static_pose), self.cfg.rewards.reference_pose_scale)
        self.reference_pose_score, self.reference_pose_credit = credit, credit
        self.phase_pose_progress = credit
        return credit*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reward_reference_velocity(self):
        credit = self._phase_credit(self.dof_vel[:, self.pitch_indices],
                                    self.reference_dq[:, self.pitch_indices],
                                    self._baseline(self.static_velocity), self.cfg.rewards.reference_velocity_scale)
        self.reference_velocity_score, self.reference_velocity_credit = credit, credit
        return credit*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reward_reference_feet(self):
        credit = self._phase_credit(self._guidance_feet(), self.reference_feet,
                                    self._baseline(self.static_feet), self.cfg.rewards.reference_feet_scale)
        self.reference_feet_score, self.reference_feet_credit = credit, credit
        self.phase_feet_progress = credit
        return credit*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reward_reference_lateral(self):
        credit = self._phase_credit(self.dof_pos[:, self.lateral_indices],
                                    self.reference_q[:, self.lateral_indices],
                                    self._baseline(self.static_lateral), self.cfg.rewards.reference_lateral_scale)
        self.reference_lateral_credit = credit
        return credit*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reward_reference_waist(self):
        credit = self._phase_credit(self.dof_pos[:, self.waist_indices],
                                    self.reference_q[:, self.waist_indices],
                                    self._baseline(self.static_waist), self.cfg.rewards.reference_waist_scale)
        self.reference_waist_credit = credit
        return credit*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reference_sole_heights(self):
        local, rotations = self.reference_foot_fk.key_transforms(self.reference_q)
        zeros = torch.zeros_like(local)
        feet = torch.cat((local, rotations, zeros, zeros), -1)
        heights = sole_heights(feet, self.sole_vertices, 0.)
        return heights-heights.min(-1, keepdim=True).values+self.cfg.amp.ground_clearance

    def _reward_reference_support(self):
        cfg = self.cfg.rewards
        score = foot_support_score(self.sole_height, self._reference_sole_heights(),
                                   self.contact_forces[:, self.feet_indices, 2],
                                   self.total_mass[:, 0]*9.81, cfg.support_height_scale,
                                   cfg.support_load_scale, cfg.support_load_fraction)
        self.reference_support_score = score
        return score*self._phase_coupling()*self._reference_valid()*self._reference_weight()

    def _reward_raw_target_violation(self):
        excess = ((self.hard_dof_limits[:, 0]-self.pd_target_raw).clamp(min=0)
                  + (self.pd_target_raw-self.hard_dof_limits[:, 1]).clamp(min=0))
        normalized = excess/self.cfg.rewards.raw_target_scale
        huber = torch.where(normalized <= 1., .5*normalized.square(), normalized-.5)
        return huber.sum(-1).clamp(max=self.cfg.rewards.raw_target_max_sum)

    def compute_reward(self):
        super().compute_reward()
        failure = (self.reset_buf.bool() & ~self.time_out_buf).float()
        penalty = failure*self.cfg.rewards.non_timeout_termination_penalty
        self.rew_buf += penalty
        self.non_timeout_penalty = penalty
        moving = self.commands[:, 0].abs() > .05
        metrics = self.extras["training_metrics"]
        metrics["Guidance/pose_credit_dense"] = self.reference_pose_credit[moving]
        metrics["Guidance/velocity_credit_dense"] = self.reference_velocity_credit[moving]
        metrics["Guidance/feet_credit_dense"] = self.reference_feet_credit[moving]
        metrics["Guidance/lateral_credit_dense"] = self.reference_lateral_credit[moving]
        metrics["Guidance/waist_credit_dense"] = self.reference_waist_credit[moving]
        metrics["Guidance/support_score"] = self.reference_support_score[moving]
        metrics["Guidance/coupling"] = self._phase_coupling()[moving]
        metrics["Control/raw_target_huber"] = self._reward_raw_target_violation()[moving]
        metrics["Init/forward_rsi_probability"] = torch.full_like(self.commands[:, 0], self._rsi_probability())
        metrics["Episode/non_timeout_penalty"] = penalty[failure.bool()]

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        cfg = self.cfg.rewards
        result.update(
            curriculum="absolute_phase_support_v1",
            reference_route="absolute_huber_cycle_baseline_v1",
            reference_pose_scale=cfg.reference_pose_scale,
            reference_velocity_scale=cfg.reference_velocity_scale,
            reference_feet_scale=cfg.reference_feet_scale,
            reference_lateral_scale=cfg.reference_lateral_scale,
            reference_waist_scale=cfg.reference_waist_scale,
            phase_credit_bounds=[cfg.phase_credit_negative_floor, cfg.phase_credit_positive_ceiling],
            phase_progress_gate="absolute per-coordinate Huber credit; no improvement-rate gate",
            static_baseline="per-coordinate cycle median under the same Huber scale",
            reference_coupling_floor=cfg.reference_coupling_floor,
            support_rule="minimum collision-sole height and independent per-foot load error with total-load gate",
            support_scales=[cfg.support_height_scale, cfg.support_load_scale, cfg.support_load_fraction],
            forward_rsi_schedule=[list(item) for item in cfg.forward_rsi_schedule],
            forward_rsi_source_iteration=cfg.forward_rsi_source_iteration,
            phase_bank=self.phase_bank.report if self.phase_bank is not None else None,
            non_timeout_termination_penalty=cfg.non_timeout_termination_penalty)
        return result
