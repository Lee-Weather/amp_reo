"""X1 29-DOF AMP tasks.

X1AMPEnv is the original exp1.0 baseline and remains the base of the exp1.x/exp2.x
chain. X1AMPFinalEnv is the consolidated exp2.12 behaviour flattened into one class:
every experiment override in the chain is inlined in MRO order, so `x1_amp` no longer
depends on x1_amp_exp*.py files. The two classes share no inheritance on purpose.
"""
import hashlib
import json
from pathlib import Path

import torch
from isaacgym import gymtorch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot import LeggedRobot
from humanoid.amp.motion import MotionLibrary, amp_features, quat_inverse_rotate, quat_mul, quat_rotate
from humanoid.amp.control import bounded_pd, StandingCommands, sync_policy_commands
from humanoid.amp.stationary import CommandMotionSampler
from humanoid.amp.gait_rewards import (tracking_denominator, sole_heights, style_multiplier,
                                      moving_speed_quality, lowest_sole_kinematics, braking_leg_gate,
                                      motion_velocity_reward)
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.exp20_commands import SmoothCommands
from humanoid.amp.exp20_rewards import similarity, raw_violation, margin_cost, support_terms
from humanoid.amp.gait_curriculum import GaitCurriculum
from humanoid.amp.exp23_curriculum import Exp23Curriculum
from humanoid.amp.exp25_curriculum import Exp25Curriculum
from humanoid.amp.exp26_curriculum import Exp26Curriculum
from humanoid.amp.exp26_swing import dense_swing, ClearanceRuns
from humanoid.amp.exp27_curriculum import Exp27Curriculum
from humanoid.amp.exp27_balance import corridor_costs
from humanoid.amp.exp28_curriculum import Exp28Curriculum
from humanoid.amp.exp28_tracking import forward_cost
from humanoid.amp.exp29_curriculum import Exp29Curriculum
from humanoid.amp.exp29_phase import phase_speed_scale, advance_phase
from humanoid.amp.contact_domain import sample_offsets, set_contact_offsets
from humanoid.amp.nominal_startup import cohort_mask, training_metadata


# --- original exp1.0 baseline (base class of the exp1.x/exp2.x chain) ---------

class X1AMPEnv(LeggedRobot):
    def step(self, actions):
        if getattr(self.cfg.env, "capture_replay_state", False):
            self._replay_torque_clips = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return super().step(actions)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        if getattr(self.cfg.env, "capture_replay_state", False):
            self._replay_torque_clips += (torques.abs() >= self.torque_limits - 1e-6).sum(-1)
        return torques

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
            self.device, self.cfg.amp.motion_weights, self.cfg.amp.limit_tolerance,
            ground_clearance=getattr(self.cfg.amp, "ground_clearance", None))
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
        if getattr(self.cfg.env, "capture_replay_state", False):
            # Snapshot terminal states before the base class resets the actor.
            self.extras["replay_state"] = {
                name: value.clone() for name, value in {
                    "root": self.root_states, "velocity": self.base_lin_vel,
                    "omega": self.base_ang_vel, "gravity": self.projected_gravity,
                    "q": self.dof_pos, "dq": self.dof_vel, "torque": self.torques,
                    "action": self.actions, "contact": self.contact_forces[:, self.feet_indices],
                    "feet": self.rigid_state[:, self.feet_indices],
                    "timeout": self.time_out_buf, "episode_step": self.episode_length_buf,
                    "command": self.commands[:, :3],
                    "command_observed": self.obs_history[-1][:, :3] / self.commands_scale,
                    "target": self.default_dof_pos + self.lagged_actions_scaled + self.motor_offsets,
                    "torque_clip_step": self._replay_torque_clips,
                }.items()
            }

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

# --- shared helpers used by the consolidated class ---------------------------

def validate_policy_bank(bank,names,dt):
    if bank['dof_names']!=list(names):raise ValueError('Policy-state bank joint order mismatch')
    # Isaac Gym exposes SimParams.dt through a C float; .001*20 is not
    # bitwise equal to the Python literal .02 stored in the replay manifest.
    if abs(bank['control_dt']-dt)>1e-7:raise ValueError('Policy-state bank control period mismatch')

class PolicyStateCurriculum(GaitCurriculum):
    @property
    def weights(self):
        result=super().weights
        result['rsi_probability']=min(.5,result['rsi_probability'])
        return result

# --- consolidated exp2.12 behaviour (flattened, no experiment inheritance) ---

class X1AMPFinalEnv(LeggedRobot):
    _curriculum_progress = 1.0
    _iteration = 3000

    def step(self,actions):
        self.raw_violation_sum.zero_()
        self.raw_target_error_sum.zero_()
        self.target_clip_counts.zero_()
        self.torque_clip_counts.zero_()
        self._inside_step = True
        try:
            if getattr(self.cfg.env, "capture_replay_state", False):
                self._replay_torque_clips = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            _super_step_8 = super().step(actions)
            result = _super_step_8
        finally:
            self._inside_step = False
        self.standing_commands.advance(result[3])
        self.commands[:, :3] = self.standing_commands.applied()
        sync_policy_commands(self)
        _super_step_4 = result
        result = _super_step_4
        self._sync_phase()
        return result

    def _compute_torques(self,actions):
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
        _super__compute_torques_3 = torques
        torques = _super__compute_torques_3
        self.raw_violation_sum.add_(raw_violation(self.pd_target_raw,
            self.hard_dof_limits[:,0],self.hard_dof_limits[:,1],self.cfg.rewards.raw_target_scale))
        return torques

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
            self.device, self.cfg.amp.motion_weights, self.cfg.amp.limit_tolerance,
            ground_clearance=getattr(self.cfg.amp, "ground_clearance", None))
        self.amp_history = torch.zeros(self.num_envs, self.cfg.amp.history_steps,
                                       self.motion_library.feature_dim, device=self.device)
        self._reset_reference = None
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
        self.leg_indices = [i for i, n in enumerate(self.dof_names)
                            if any(j in n for j in ("hip_", "knee_", "ankle_"))]
        if len(self.leg_indices) != 12:
            raise ValueError("Expected 12 leg joints")
        floor = self.cfg.amp.standing_style_multiplier
        if not 0 <= floor <= 1:
            raise ValueError("Standing AMP multiplier must be in [0,1]")
        lib = self.motion_library
        meshes = {lib.key_body_names[i]: mesh for i, mesh in lib.grounding.feet}
        bodies = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        self.sole_vertices = [meshes[bodies[i]] for i in self.feet_indices.tolist()]
        if self.cfg.domain_rand.randomize_com or self.cfg.domain_rand.randomize_link_com:
            raise ValueError("exp1.7 nominal COM cache requires COM randomization disabled")
        props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
        self.foot_com_local = torch.tensor([[props[i].com.x,props[i].com.y,props[i].com.z]
                                            for i in self.feet_indices.tolist()], device=self.device)
        a = self.cfg.amp
        self.command_motion = CommandMotionSampler(self.motion_library,a.rsi_speed_tolerance,
            a.expert_standing_probability,a.quiet_root_speed,a.quiet_angular_speed,a.quiet_joint_speed,
            exclude_quiet_for_moving=True,relative_speed_tolerance=a.rsi_relative_speed_tolerance)
        cfg = self.cfg
        if cfg.amp.reference_state_initialization:
            raise ValueError('exp2.0 initialization is standing-only')
        bodies = self.gym.get_actor_rigid_body_names(self.envs[0],self.actor_handles[0])
        if [bodies[i] for i in self.feet_indices.tolist()]!=['left_ankle_roll_link','right_ankle_roll_link']:
            raise ValueError('exp2.0 requires explicit left/right foot ordering')
        correction = None
        self.correction_sha256 = None
        if cfg.amp.native_correction_file:
            contents = Path(cfg.amp.native_correction_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)).read_bytes()
            correction = json.loads(contents)
            self.correction_sha256 = hashlib.sha256(json.dumps(correction,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.reference_guide = NativeCycle(self.motion_library,cfg.amp.native_clip,
            cfg.amp.native_first_frame,cfg.amp.native_last_frame,contact_correction=correction)
        self.sole_vertices = [v for i,v in self.reference_guide.feet]
        self.reference_report = self.reference_guide.audit()
        self.transition_report = self.reference_guide.audit_transitions(self.standing_state)
        print('EXP20_REFERENCE_AUDIT '+json.dumps(self.reference_report,sort_keys=True),flush=True)
        if cfg.amp.require_reference_acceptance and not self.reference_report['passed']:
            raise ValueError('exp2.0 reference failed contact/derivative acceptance; run CPU preparation first')
        if cfg.amp.require_reference_acceptance and not self.transition_report['passed']:
            raise ValueError('exp2.0 start/stop pose derivative or geometry validation failed')
        if abs(self.reference_guide.nominal_speed-cfg.commands.native_speed)>.002:
            raise ValueError('exp2.0 command speed does not match its native reference')
        # Quiet histories remain real, complete, oldest-first reference windows.
        sampler = self.command_motion
        self.reference_guide.quiet_histories = self.motion_library.history(sampler.ids[sampler.quiet],sampler.times[sampler.quiet])
        self.standing_commands = SmoothCommands(self.num_envs,self.device,
            round(cfg.amp.stand_buffer_seconds/self.dt),self.dt,cfg.commands.transition_seconds,cfg.commands.native_speed)
        self.gait_phase = torch.zeros(self.num_envs,device=self.device)
        self.pitch_indices = self.reference_guide.pitch_indices
        self.lateral_indices = self.posture_groups['leg_lateral']
        self.waist_indices = self.posture_groups['waist']
        self.upper_indices = self.posture_groups['upper']
        self.native_foot_ids = [self.motion_library.key_body_names.index(side+'_ankle_roll_link') for side in ('left','right')]
        self.raw_violation_sum = torch.zeros_like(self.dof_pos)
        self.amp_valid_history = torch.zeros(self.num_envs,cfg.amp.history_steps,dtype=torch.bool,device=self.device)
        directory = Path(cfg.amp.motion_dir.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.reference_source_hashes = {name:hashlib.sha256((directory/(name+'.pkl')).read_bytes()).hexdigest()
            for name in self.motion_library.names}
        self.gait_curriculum=GaitCurriculum()
        self.branch_statistics=torch.zeros(2,4,device=self.device)
        self.coverage_statistics=torch.zeros(2,device=self.device)
        self.phase_statistics=torch.zeros(5,2,device=self.device)
        self.completed_cycles=torch.zeros(self.num_envs,device=self.device)
        self.evaluation_reset_mode=None
        self.gait_curriculum=PolicyStateCurriculum()
        path=Path(LEGGED_GYM_ROOT_DIR)/'humanoid/amp/reference_data/exp22_policy_state_bank.json'
        raw=path.read_bytes();bank=json.loads(raw)
        validate_policy_bank(bank,self.dof_names,self.dt)
        self.policy_bank_sha256=hashlib.sha256(raw).hexdigest()
        assert self.policy_bank_sha256==self.cfg.amp.policy_state_bank_sha256,'Policy-state bank SHA mismatch'
        self.bank={k:torch.tensor(bank[k],device=self.device,dtype=torch.float) for k in ('states','phases','histories','amp_states')}
        anchors=torch.cat((self.bank['histories'],torch.tensor(bank['quiet_histories'],device=self.device,dtype=torch.float)))
        self.source_anchor_observations=anchors.flatten(1)
        self._bank_pending=None
        self.gait_curriculum=Exp23Curriculum()
        self.gait_curriculum=Exp25Curriculum()
        self.gait_curriculum=Exp26Curriculum()
        self.swing_air_tracker=ClearanceRuns(self.num_envs,self.device,self.dt)
        self.gait_curriculum=Exp27Curriculum()
        self.gait_curriculum = Exp28Curriculum()
        self.gait_curriculum = Exp29Curriculum()
        self.contact_offsets=self.contact_offset_values.to(self.device)
        checked=[]
        for index in sorted({0,self.num_envs//2,3*self.num_envs//4}):
            props=self.gym.get_actor_rigid_shape_properties(self.envs[index],self.actor_handles[index])
            expected=float(self.contact_offset_values[index])
            if not props or any(abs(p.contact_offset-expected)>1e-7 or abs(p.rest_offset)>1e-7 for p in props):
                raise ValueError('Actor did not retain requested contact offset')
            checked.append(dict(env_id=index,contact_offset_m=expected,shapes=len(props)))
        print('CONTACT_RANDOMIZATION_READY '+json.dumps(dict(verified=True,
            assigned_min_m=float(self.contact_offsets.min()),assigned_max_m=float(self.contact_offsets.max()),
            near_fraction=float((self.contact_offsets<=.00100001).float().mean()),
            original_fraction=float((self.contact_offsets==.01).float().mean()),
            sampled_actor_checks=checked)),flush=True)
        self.nominal_startup_mask=self._nominal(torch.arange(self.num_envs,device=self.device))
        count=int(self.nominal_startup_mask.sum())
        checks=self._verify_nominal_startup_shapes(count)
        print('NOMINAL_STARTUP_READY '+json.dumps(dict(verified=True,nominal_envs=count,
            total_envs=self.num_envs,fraction=count/self.num_envs,sampled_actor_checks=checks)),flush=True)

    def _verify_nominal_startup_shapes(self,count):
        checks=[]
        for index in (sorted({0,count-1}) if count else []):
            props=self.gym.get_actor_rigid_shape_properties(self.envs[index],self.actor_handles[index])
            if not props or any(abs(p.friction-.8)>1e-6 or abs(p.contact_offset-.001)>1e-7 or abs(p.rest_offset)>1e-7 for p in props):
                raise ValueError('Nominal cohort shape properties did not persist')
            checks.append(dict(env_id=index,shapes=len(props),friction=float(props[0].friction),
                contact_offset_m=float(props[0].contact_offset)))
        return checks

    def _get_noise_scale_vec(self, cfg):
        scale = torch.zeros(cfg.env.num_single_obs, device=self.device)
        s, o = cfg.noise.noise_scales, self.obs_scales
        scale[3:6] = s.ang_vel * o.ang_vel
        scale[6:9] = s.gravity
        scale[9:38] = s.dof_pos * o.dof_pos
        scale[38:67] = s.dof_vel * o.dof_vel
        return scale * cfg.noise.noise_level

    def _policy_frame(self):
        _super__policy_frame_3 = torch.cat((self.commands[:, :3] * self.commands_scale,
                          self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity,
                          (self.dof_pos-self.default_dof_pos) * self.obs_scales.dof_pos,
                          self.dof_vel * self.obs_scales.dof_vel, self.actions), -1)
        return torch.cat((_super__policy_frame_3,self._phase_features()),-1)

    def _critic_obs(self):
        return torch.cat((self._policy_frame(), self.base_lin_vel * self.obs_scales.lin_vel), -1)

    def compute_observations(self):
        frame = self._policy_frame()
        self.privileged_obs_buf = self._critic_obs()
        if self.cfg.noise.add_noise:
            frame = frame + (2*torch.rand_like(frame)-1)*self.noise_scale_vec
        self.obs_history.append(frame)
        self.obs_buf = torch.cat(list(self.obs_history), -1)
        if self._bank_pending is not None:
            ids,history=self._bank_pending
            for i,frame in enumerate(self.obs_history):frame[ids]=history[:,i]
            self.obs_buf[ids]=history.flatten(1)
            self._bank_pending=None
        mask=self._nominal(torch.arange(self.num_envs,device=self.device))
        frame=self._policy_frame()
        self.obs_history[-1][mask]=frame[mask]
        self.obs_buf[mask,-frame.shape[-1]:]=frame[mask]
        if not getattr(self,'_nominal_obs_verified',False) and mask.any():
            assert torch.equal(self.obs_buf[mask,-frame.shape[-1]:],frame[mask])
            self._nominal_obs_verified=True
            print('NOMINAL_STARTUP_OBS_VERIFIED '+json.dumps(dict(verified=True,
                nominal_envs=int(mask.sum()),frame_dimension=frame.shape[-1],noise_max_error=0.)),flush=True)

    def get_amp_observations(self):
        return self.amp_history

    def compute_reward(self):
        blend,derivative = self.standing_commands.blend()
        reference_state = self._blended_reference(blend,derivative)
        self.reference_q,self.reference_dq = reference_state[:,13:42],reference_state[:,42:]
        self.reference_feet = self.motion_library.fk.key_positions(self.reference_q)[:,self.native_foot_ids].flatten(-2)
        target_height = self.reference_guide.sole_height(reference_state).clamp(min=0)
        height,speed = lowest_sole_kinematics(self.rigid_state[:,self.feet_indices],self.sole_vertices,self.foot_com_local,self.env_origins[:,2])
        self.swing_score,self.support_score,self.slip_cost = self._support_terms(height,target_height,
            self.contact_forces[:,self.feet_indices,2],self.total_mass[:,0]*9.81,speed)
        self.scuff_point_height,self.scuff_point_speed = lowest_sole_kinematics(
            self.rigid_state[:,self.feet_indices],self.sole_vertices,self.foot_com_local,self.env_origins[:,2])
        feet = self.rigid_state[:, self.feet_indices]
        self.sole_height = sole_heights(feet, self.sole_vertices, self.env_origins[:, 2])
        self.foot_horizontal_speed = feet[:, :, 7:9].norm(dim=-1)
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
        # Save branch and length BEFORE automatic reset, including terminal rows.
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
        if getattr(self.cfg.env, "capture_replay_state", False):
            # Snapshot terminal states before the base class resets the actor.
            self.extras["replay_state"] = {
                name: value.clone() for name, value in {
                    "root": self.root_states, "velocity": self.base_lin_vel,
                    "omega": self.base_ang_vel, "gravity": self.projected_gravity,
                    "q": self.dof_pos, "dq": self.dof_vel, "torque": self.torques,
                    "action": self.actions, "contact": self.contact_forces[:, self.feet_indices],
                    "feet": self.rigid_state[:, self.feet_indices],
                    "timeout": self.time_out_buf, "episode_step": self.episode_length_buf,
                    "command": self.commands[:, :3],
                    "command_observed": self.obs_history[-1][:, :3] / self.commands_scale,
                    "target": self.default_dof_pos + self.lagged_actions_scaled + self.motor_offsets,
                    "torque_clip_step": self._replay_torque_clips,
                }.items()
            }
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
        # Clone before auto-reset changes velocities/stand_gate. The signed
        # speed fraction prevents a stationary policy earning full moving AMP.
        multiplier = style_multiplier(self.commands[:, 0], self.base_lin_vel[:, 0],
                                      self.stand_gate, self.cfg.amp.standing_style_multiplier)
        self.extras["amp_style_multiplier"] = multiplier.clone()
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        metrics = self.extras["training_metrics"]
        metrics["AMP/standing_style_multiplier"] = multiplier[zero]
        metrics["AMP/moving_style_multiplier"] = multiplier[self.commands[:, 0].abs() > .05]
        metrics["Standing/leg_joint_speed_rms"] = self.dof_vel[zero][:, self.leg_indices].square().mean(-1).sqrt()
        metrics["Gait/scuff_cost"] = self._reward_feet_scuff()[~zero]
        metrics["Tracking/linear_tolerance_m_s"] = self._tracking_denominator().sqrt()
        vx, actual = self.commands[:, 0], self.base_lin_vel[:, 0]
        moving = vx.abs() > .05
        quality = moving_speed_quality(vx, actual, self.cfg.amp.motion_style_velocity_sigma)
        multiplier = self.extras["amp_style_multiplier"]*quality
        self.extras["amp_style_multiplier"] = multiplier.clone()
        metrics = self.extras["training_metrics"]
        metrics["AMP/moving_speed_quality"] = quality[moving]
        metrics["AMP/moving_style_multiplier"] = multiplier[moving]
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        groups = {"backward": vx < -.05, "slow_forward": (vx > .05) & (vx <= .2),
                  "forward": vx > .2, "zero": zero}
        # Applied commands and episode events are captured before auto-reset.
        # Signed velocity alone is not a walking-success metric (falls can be fast).
        for name, mask in groups.items():
            metrics["Commands/"+name+"_applied_fraction"] = mask.float()
            metrics["Tracking/"+name+"_vx"] = actual[mask]
            metrics["Tracking/"+name+"_abs_vx_error"] = (actual-vx)[mask].abs()
            ended = self.reset_buf.bool() & mask
            metrics["Episode/"+name+"_completed_length"] = self.episode_length_buf[ended].float()
            metrics["Episode/"+name+"_non_timeout_fraction"] = (~self.time_out_buf[ended]).float()
        moving = self.commands[:,0].abs() > .05
        m = self.extras["training_metrics"]
        m["Gait/lowest_point_speed"] = self.scuff_point_speed[moving].flatten()
        m["Gait/com_speed"] = self.foot_horizontal_speed[moving].flatten()
        m["Tracking/lateral_abs_velocity"] = self.base_lin_vel[moving,1].abs()
        m["Tracking/moving_abs_yaw_rate"] = self.base_ang_vel[moving,2].abs()
        m["Tracking/backward_progress_bonus"] = self._reward_backward_progress()[self.commands[:,0] < -.05]
        backward = self.commands[:,0] < -.05
        zero = self.commands[:,:3].abs().amax(-1) <= 1e-6
        metrics = self.extras['training_metrics']
        metrics['Tracking/backward_velocity_reward'] = self._reward_backward_progress()[backward]
        metrics['Tracking/backward_abs_vx'] = self.base_lin_vel[backward,0].abs()
        metrics['Tracking/backward_wrong_direction_fraction'] = (self.base_lin_vel[backward,0] > .05).float()
        metrics['Standing/braking_leg_gate'] = self._braking_leg_gate()[zero]
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
        zero = self.commands[:, :3].abs().amax(-1) <= 1e-6
        vx = self.commands[:, 0]
        # Only the zero-command stand ramp reduces AMP. Neither actual speed
        # nor training iteration gates imitation credit while walking.
        standing = 1-(1-self.cfg.amp.standing_style_multiplier)*self.stand_gate
        multiplier = torch.where(zero, standing, torch.ones_like(vx))
        self.extras["amp_style_multiplier"] = multiplier.clone()
        metrics = self.extras["training_metrics"]
        metrics["AMP/standing_style_multiplier"] = multiplier[zero]
        metrics["AMP/moving_style_multiplier"] = multiplier[vx.abs() > .05]
        # The inherited speed diagnostic is useful, but is no longer an AMP gate.
        metrics["Tracking/moving_speed_quality"] = metrics.pop("AMP/moving_speed_quality")
        metrics.pop("Tracking/backward_progress_bonus", None)
        metrics.pop("Tracking/backward_velocity_reward", None)
        reward = self._reward_motion_progress()
        metrics["Tracking/forward_progress_reward"] = reward[vx > .05]
        metrics["Tracking/backward_progress_reward"] = reward[vx < -.05]
        failure = self.reset_buf.bool()&~self.time_out_buf
        self.rew_buf += failure*self.cfg.rewards.non_timeout_termination_penalty
        eligible = ~self.standing_commands.transition
        self.amp_valid_history[:,:-1] = self.amp_valid_history[:,1:].clone()
        self.amp_valid_history[:,-1] = eligible
        valid = self.amp_valid_history.all(-1)
        moving = self.standing_commands.value>.05
        alpha = .5*min(1.,max(0.,(self._iteration-200)/600.))
        multiplier = torch.where(moving,torch.ones_like(blend),torch.full_like(blend,self.cfg.amp.standing_style_multiplier))*alpha*valid
        self.extras['amp_style_multiplier'] = multiplier.clone()
        self.extras['amp_valid'] = valid.clone()
        self.extras['amp_moving'] = moving.clone()
        metrics = self.extras['training_metrics']
        metrics.update({'Reference/guide_weight':blend.new_full(blend.shape,self._reference_weight()),
            'Control/raw_target_weight':blend.new_full(blend.shape,self._raw_weight()),
            'Control/raw_target_huber':self.raw_violation_sum.mean(-1)/self.cfg.control.decimation,
            'Commands/transition_fraction':self.standing_commands.transition.float(),
            'Commands/goal_vx':self.standing_commands.requested[:,0].clone(),
            'Commands/applied_vx':self.commands[:,0].clone(),
            'AMP/eligible_fraction':valid.float(),'AMP/moving_style_multiplier':multiplier[moving],
            'AMP/standing_style_multiplier':multiplier[~moving],
            'AMP/curriculum_weight':blend.new_full(blend.shape,alpha),
            'Episode/non_timeout_penalty':failure[failure]*self.cfg.rewards.non_timeout_termination_penalty,
            'Reference/phase_rate_hz':blend.new_full(blend.shape,1/self.reference_guide.period)[moving],
            'Gait/swing_score':self.swing_score[moving],'Gait/support_score':self.support_score[moving],
            'Gait/stance_slip_cost':self.slip_cost[moving]})
        for name,indices,scale in (('pose',self.pitch_indices,self.cfg.rewards.reference_pose_scale),
            ('lateral',self.lateral_indices,self.cfg.rewards.reference_lateral_scale),
            ('waist',self.waist_indices,self.cfg.rewards.reference_waist_scale),
            ('upper',self.upper_indices,self.cfg.rewards.reference_upper_scale)):
            metrics['Reference/'+name+'_score'] = similarity(self.dof_pos[:,indices],self.reference_q[:,indices],scale)[moving]
            metrics['Reference/'+name+'_rmse_rad'] = (self.dof_pos[:,indices]-self.reference_q[:,indices]).square().mean(-1).sqrt()[moving]
        distance = torch.minimum(self.dof_pos-self.hard_dof_limits[:,0],self.hard_dof_limits[:,1]-self.dof_pos)
        for i,name in enumerate(self.dof_names):
            metrics['Control/raw_target_huber/'+name] = self.raw_violation_sum[:,i]/self.cfg.control.decimation
            metrics['Posture/limit_distance_rad/'+name] = distance[:,i]
        if getattr(self.cfg.env,'capture_replay_state',False):
            self.extras['replay_state'].update(reference_phase=self.gait_phase.clone(),reference_q=self.reference_q.clone(),
                reference_dq=self.reference_dq.clone(),command_goal=self.standing_commands.requested.clone(),
                command_transition=self.standing_commands.transition.clone())
        valid=self.extras['amp_valid']
        moving=self.extras['amp_moving']
        alpha=self.gait_curriculum.weights['amp']
        multiplier=torch.where(moving,torch.ones_like(self.commands[:,0]),
            torch.full_like(self.commands[:,0],self.cfg.amp.standing_style_multiplier))*alpha*valid
        self.extras['amp_style_multiplier']=multiplier.clone()
        done=self.reset_buf.bool()
        fail=done&~self.time_out_buf
        seconds=self.episode_length_buf*self.dt
        for i,branch in enumerate((self.reset_standing,~self.reset_standing)):
            ended=done&branch
            self.branch_statistics[i]+=torch.stack((ended.sum(),seconds[ended].sum(),fail[branch].sum(),
                ((seconds<=3)&fail&branch).sum()))
        full=(self.commands[:,0]>.99*self.cfg.commands.native_speed)&~self.standing_commands.transition&self._reference_valid()
        self.coverage_statistics+=self.commands.new_tensor([self.num_envs,0.])
        self.coverage_statistics[1]+=full.sum()
        self.completed_cycles+=full.float()*self.dt/self.reference_guide.period
        hold=self.standing_commands.remaining>0
        start=self.standing_commands.transition&(self.standing_commands.active_goal>0)
        stop=self.standing_commands.transition&~start
        steady=(self.standing_commands.value>.05)&~self.standing_commands.transition&~hold
        quiet=~(hold|start|stop|steady)
        metrics=self.extras['training_metrics']
        for i,(name,mask) in enumerate(zip(('buffer','start','steady','stop','quiet'),(hold,start,steady,stop,quiet))):
            self.phase_statistics[i,0]+=mask.sum()
            self.phase_statistics[i,1]+=(fail&mask).sum()
            metrics['Phase/'+name+'_fraction']=mask.float()
            metrics['Phase/'+name+'_failure_count']=(fail&mask).float()
        metrics.update({'AMP/curriculum_weight':torch.full_like(multiplier,alpha),
            'AMP/moving_style_multiplier':multiplier[moving],
            'AMP/standing_style_multiplier':multiplier[~moving],
            'Curriculum/stage':torch.full_like(multiplier,float(self.gait_curriculum.stage)),
            'Curriculum/rsi_probability':torch.full_like(multiplier,self.gait_curriculum.weights['rsi_probability']),
            'Curriculum/full_speed_fraction':full.float(),
            'Gait/completed_reference_cycles':self.completed_cycles[done].clone()})
        info=self.dense_swing_details
        valid=self._reference_valid()
        events=self.swing_air_tracker.update(info['height'],valid)
        metrics=self.extras['training_metrics']
        factor=self._reference_weight()*valid*self.gait_curriculum.swing_multiplier*self.reward_scales['reference_swing']
        for i,side in enumerate(('left','right')):
            mask=valid&(info['phase'][:,i]>0)
            prefix='Swing/'+side+'/'
            metrics[prefix+'height_error_m']=(info['height'][:,i]-info['target'][:,i]).abs()[mask]
            metrics[prefix+'load_bodyweight']=(info['force'][:,i]/info['body_weight'][:,0])[mask]
            metrics[prefix+'below_5mm_fraction']=(info['height'][:,i]<.005).float()[mask]
            metrics[prefix+'clearance_runs_per10s']=(events[:,i].float()*10/self.dt)[valid]
            metrics[prefix+'positive_reward_fraction']=(info['contributions'][:,i].sum(-1)>0).float()[mask]
            for j,term in enumerate(('height','unload','other_support')):
                metrics[prefix+term+'_reward']=(info['contributions'][:,i,j]*factor)[valid]
        metrics['Swing/multiplier']=self.commands[:,0].new_full((self.num_envs,),self.gait_curriculum.swing_multiplier)
        lateral,yaw=self._corridor_components();valid=self._reference_valid()
        metrics=self.extras['training_metrics'];gain=self.gait_curriculum.balance_multiplier
        metrics['Balance/lateral_cost']=lateral[valid]
        metrics['Balance/yaw_cost']=yaw[valid]
        metrics['Balance/lateral_reward']=(lateral*gain*self.reward_scales['corridor_lateral'])[valid]
        metrics['Balance/yaw_reward']=(yaw*gain*self.reward_scales['corridor_yaw'])[valid]
        metrics['Balance/multiplier']=self.commands[:,0].new_full((self.num_envs,),gain)
        metrics['Balance/contact_friction']=self.env_frictions.reshape(self.num_envs,-1)[:,0]
        cost = self._forward_cost()
        valid = self._reference_valid()
        gain = self.gait_curriculum.forward_multiplier
        metrics = self.extras['training_metrics']
        metrics['Forward/cost'] = cost[valid]
        metrics['Forward/reward'] = (cost*gain*self.reward_scales['corridor_forward'])[valid]
        metrics['Forward/multiplier'] = self.commands[:,0].new_full((self.num_envs,), gain)
        metrics['Forward/overspeed_fraction'] = (self.base_lin_vel[:,0]-self.commands[:,0] > .2)[valid].float()
        moving = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        rate = phase_speed_scale(self.standing_commands.value,self.cfg.commands.native_speed)/self.reference_guide.period
        self.extras['training_metrics']['Reference/phase_rate_hz'] = rate[moving]
        self.extras['training_metrics']['Contact/offset_m']=self.contact_offsets
        mask=self.nominal_startup_mask
        moving=self.standing_commands.value>1e-7
        done=self.reset_buf.bool()&mask
        metrics=self.extras['training_metrics']
        metrics.update({'Nominal/fraction':mask.float(),
            'Nominal/episode_seconds':(self.episode_length_buf*self.dt)[done],
            'Nominal/moving_abs_yaw_rate':self.base_ang_vel[mask&moving,2].abs(),
            'Nominal/forward_vx':self.base_lin_vel[mask&moving,0],
            'Nominal/non_timeout_rate':(~self.time_out_buf[done]).float()})

    def reset_idx(self,ids):
        _arg_env_ids = ids
        if not len(_arg_env_ids):return
        count=len(_arg_env_ids)
        probability=self.gait_curriculum.weights['rsi_probability']
        if self.evaluation_reset_mode is not None:probability=float(self.evaluation_reset_mode=='rsi')
        standing=torch.rand(count,device=self.device)>=probability
        samples=torch.randint(len(self.bank['states']),(count,),device=self.device)
        state=self.bank['states'][samples].clone()
        phase=self.bank['phases'][samples].clone()
        state[standing]=self.standing_state
        state[standing,13:42]+=(2*torch.rand(int(standing.sum()),29,device=self.device)-1)*self.cfg.amp.standing_joint_noise
        standing,phase,state=self._adjust_reset_states(_arg_env_ids,standing,phase,state)
        state[:,13:42]=state[:,13:42].maximum(self.hard_dof_limits[:,0]).minimum(self.hard_dof_limits[:,1])
        if standing.any():
            state[standing,2]+=self.cfg.amp.ground_clearance-self.reference_guide.sole_height(state[standing]).amin(-1)
            phase[standing]=self.reference_guide.nearest_phase(state[standing,13:42],state[standing,42:])
        requested=self._draw_commands(count);requested[~standing,0]=self.cfg.commands.native_speed
        self._reset_reference,self._reset_commands=state,requested
        self.reset_standing[_arg_env_ids]=standing;self.standing_commands.reset(_arg_env_ids,standing)
        LeggedRobot.reset_idx(self,_arg_env_ids)
        ids=_arg_env_ids[~standing];bank_ids=samples[~standing]
        for value in (self.standing_commands.active_goal,self.standing_commands.start,self.standing_commands.value):value[ids]=self.cfg.commands.native_speed
        self.standing_commands.requested[_arg_env_ids]=requested
        self.commands[_arg_env_ids,:3]=self.standing_commands.applied()[_arg_env_ids]
        for frame in self.obs_history:frame[_arg_env_ids]=0
        history=self.bank['histories'][bank_ids]
        self.actions[ids]=history[:,-1,67:96]
        self.last_actions[ids]=self.actions[ids]
        self.last_last_actions[ids]=history[:,-2,67:96]
        self.last_dof_vel[ids]=self.dof_vel[ids]
        self._bank_pending=(ids,history)
        amp=self.motion_library.features(state)[:,None].expand(-1,self.cfg.amp.history_steps,-1).clone()
        if len(ids):
            amp[~standing]=self.motion_library.features(self.bank['amp_states'][bank_ids].reshape(-1,71)).reshape(len(ids),3,-1)
        self.amp_history[_arg_env_ids]=amp;self.amp_valid_history[_arg_env_ids]=True
        self._reset_reference,self._reset_commands=None,None
        for value in (self.zero_steps,self.stop_hold_steps,self.previous_moving,self.stop_pending,self.stand_gate):value[_arg_env_ids]=0
        self.gait_phase[_arg_env_ids]=phase;self.completed_cycles[_arg_env_ids]=0
        if hasattr(self,'swing_air_tracker'):self.swing_air_tracker.reset(_arg_env_ids)

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

    def _resample_commands(self,env_ids):
        nominal=self._nominal(env_ids)
        if self._reset_commands is not None:
            self._reset_commands[nominal]=self.commands.new_tensor([self.cfg.commands.native_speed,0.,0.])
        requested = self._reset_commands if self._reset_commands is not None else self._draw_commands(len(env_ids))
        self.standing_commands.requested[env_ids] = requested
        # Scheduling changes the goal, never the command of the transition being rewarded.
        if self._reset_reference is not None or not self._inside_step:
            self.commands[:,:3] = self.standing_commands.applied()
        self.standing_commands.requested[env_ids[nominal]]=self.commands.new_tensor([self.cfg.commands.native_speed,0.,0.])

    def _reward_tracking_lin_vel(self):
        error = self.commands[:,:2]-self.base_lin_vel[:,:2]
        return torch.exp(-error[:,0].square()/self._tracking_denominator()
                         -error[:,1].square()/self.cfg.rewards.tracking_lateral_std**2)

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

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            # Preserve importer hard bounds BEFORE the base class shrinks its
            # separate dof_pos_limits buffer to the 0.95 soft penalty interval.
            self.hard_dof_limits = torch.tensor(
                [[float(p["lower"]), float(p["upper"])] for p in props], device=self.device)
        return super()._process_dof_props(props, env_id)

    def reset(self):
        # The base reset takes a zero-action physics step before inference.
        # A moving source snapshot must receive its own policy action first.
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        self.compute_observations()
        return self.obs_buf,self.privileged_obs_buf

    def set_evaluation_command(self,vx,vy=0.,wz=0.):
        if vy!=0 or wz!=0 or (abs(vx)>1e-7 and abs(vx-self.cfg.commands.native_speed)>.002):
            raise ValueError('exp2.0 supports only zero and its native forward target')
        if abs(vx)>1e-7:
            vx = self.cfg.commands.native_speed
        self._guidance_evaluation = True
        self.command_ranges['lin_vel_x'] = [vx,vx]
        self.standing_commands.requested[:] = self.commands.new_tensor([vx,0.,0.])
        self.commands[:,:3] = self.standing_commands.applied()
        sync_policy_commands(self)
        return self.obs_buf

    def _reward_raw_target_violation(self):
        return self.raw_violation_sum.mean(-1)/self.cfg.control.decimation*self._raw_weight()

    def _posture_cost(self,group):
        quiet = (self.standing_commands.value<=1e-7)&~self.standing_commands.transition
        ids = self.posture_groups[group]
        _super__posture_cost_4 = (self.dof_pos[:, ids]-self.default_dof_pos[:, ids]).square().sum(-1)
        return _super__posture_cost_4*quiet

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

    def sample_amp_expert(self,count):
        raise ValueError('exp2.0 requires sample_amp_expert_by_category with paired policy categories')

    def _draw_commands(self,count):
        result = torch.zeros(count,3,device=self.device)
        if getattr(self,'_guidance_evaluation',False) or getattr(self.cfg.commands,'guidance_evaluation',False):
            result[:,0] = self.command_ranges['lin_vel_x'][0]
        else:
            result[:,0] = self.cfg.commands.native_speed
            result[torch.rand(count,device=self.device)<self.cfg.commands.zero_probability,0] = 0
        return result

    def _reward_stand_lin_vel(self):
        cost = self.base_lin_vel[:, :2].square().sum(-1)/self.cfg.rewards.stand_linear_scale**2
        _super__reward_stand_lin_vel_3 = self.stand_gate*cost
        return _super__reward_stand_lin_vel_3*self._standing_curriculum_weight()

    def _reward_stand_ang_vel(self):
        return (self.stand_gate*self.base_ang_vel.square().sum(-1)/self.cfg.rewards.stand_angular_scale**2)*self._standing_curriculum_weight()

    def _reward_stand_dof_vel(self):
        cost = self.dof_vel[:, self.leg_indices].square().mean(-1)
        _super__reward_stand_dof_vel_3 = self.stand_gate*cost/self.cfg.rewards.stand_joint_speed_scale**2
        _super__reward_stand_dof_vel_3 = _super__reward_stand_dof_vel_3*self._braking_leg_gate()
        return _super__reward_stand_dof_vel_3*self._standing_curriculum_weight()

    def _reward_stand_pose(self):
        error = (self.dof_pos[:, self.leg_indices]-self.standing_state[13:42][self.leg_indices])
        _super__reward_stand_pose_3 = self.stand_gate*error.square().mean(-1)/self.cfg.rewards.stand_pose_scale**2
        _super__reward_stand_pose_3 = _super__reward_stand_pose_3*self._braking_leg_gate()
        return _super__reward_stand_pose_3*self._standing_curriculum_weight()

    def _reward_stand_orientation(self):
        error = (self.projected_gravity-self.standing_gravity)/self.cfg.rewards.stand_gravity_scale
        _super__reward_stand_orientation_3 = self.stand_gate*error.square().sum(-1)
        return _super__reward_stand_orientation_3*self._standing_curriculum_weight()

    def _tracking_denominator(self):
        cfg = self.cfg.rewards
        return tracking_denominator(self.commands[:, 0], cfg.tracking_min_std,
                                    cfg.tracking_max_std, cfg.tracking_max_speed)

    def _reward_feet_scuff(self):
        return torch.zeros(self.num_envs,device=self.device)

    def _reward_backward_progress(self):
        # Retired in this experiment; do not double-count reverse progress.
        return torch.zeros_like(self.commands[:, 0])

    def _braking_leg_gate(self):
        cfg = self.cfg.rewards
        return braking_leg_gate(self.base_lin_vel[:,:2], self.base_ang_vel[:,2], self.stop_pending,
                                cfg.brake_leg_linear_scale, cfg.brake_leg_angular_scale)

    def set_training_iteration(self,iteration):
        if iteration<0: raise ValueError('Negative exp2.0 iteration')
        self._iteration = int(iteration)

    def _standing_curriculum_weight(self):
        initial = self.cfg.rewards.curriculum_initial_stand_weight
        if not 0 <= initial <= 1:
            raise ValueError("Initial standing curriculum weight must be in [0, 1]")
        return initial+(1-initial)*self._curriculum_progress

    def checkpoint_training_metadata(self):
        # Audit floating point results are logged separately. Checkpoint identity
        # must not depend on CPU/GPU rounding in an order statistic or FK result.
        native_identity = dict(clip=self.cfg.amp.native_clip,first_frame=self.cfg.amp.native_first_frame,
            last_frame=self.cfg.amp.native_last_frame,period_s=self.reference_guide.period,
            command_speed=self.cfg.commands.native_speed,correction_sha256=self.correction_sha256,
            sole_support_cache_sha256=self.reference_guide.sole_cache_sha256,
            hash_definition='canonical_json_sorted_keys_compact_utf8',
            builder='native_curve_margin_v1',time_scale=1.,amplitude_scale=1.)
        _super_checkpoint_training_metadata_3 = dict(curriculum='native_full_body_v1',reference_route='native_cycle_shared_with_amp_v1',
            reference_source_sha256=self.reference_source_hashes,native_reference=native_identity,
            phase_input_columns=[96,97],command_events=dict(forward_vx=self.cfg.commands.native_speed,
                forward_probability=1-self.cfg.commands.zero_probability,zero_probability=self.cfg.commands.zero_probability),
            expert_sampling=dict(moving='native_cycle',quiet='real_complete_quiet_histories',
                ratio='sampled_replay_categories',transition_excluded=True),transition_seconds=self.cfg.commands.transition_seconds,
            initialization='standing_only',conditional_experts=True,iteration_origin='exp2.0_new_updates')
        result=_super_checkpoint_training_metadata_3
        result.update(curriculum='performance_gated_native_v1',initialization='mixed_native_rsi_and_standing',
            curriculum_state_location='infos.exp21.learning_state',critic_only_iterations=50,
            freeze_std_iterations=200,initial_rsi_probability=.7,
            phase_coverage='buffer_start_steady_stop_quiet')
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        result.update(curriculum='anchored_policy_state_native_v1',initialization='mixed_successful_policy_state_and_standing',
            policy_bank_sha256=self.policy_bank_sha256,initial_rsi_probability=.5,
            exploration_gain=.5,freeze_std_iterations=1000000000,freeze_features_min_iterations=500,
            source_anchor_coef=10.,source_max_raw_pd_rms_rad=.04)
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        result.update(curriculum='relaxed_anchor_low_amp_v1',source_anchor_coef=1.,
            critic_only_iterations=0,freeze_features_min_iterations=1000000000,
            amp_entry_min_new_updates=200,amp_entry_min_seconds=14.,
            amp_entry_max_non_timeout=.5,amp_entry_good_windows=2,amp_entry_weight=.01,
            amp_entry_baseline_mean_ratio=.95,amp_entry_baseline_failure_tolerance=.05)
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        for key in list(result):
            if key.startswith('amp_entry_'):result.pop(key)
        result.update(curriculum='pooled100_low_amp_v1',amp_max_weight=.01,amp_ramp_updates=100,
            amp_entry_window_updates=100,amp_entry_min_episodes_per_branch=200,
            amp_entry_min_seconds=15.,amp_entry_max_non_timeout=.55,
            amp_entry_baseline_mean_ratio=.95,amp_entry_baseline_failure_tolerance=.05,
            amp_exit_min_seconds=13.,amp_exit_max_non_timeout=.6,amp_exit_bad_windows=2,
            swing_multiplier=1.,stagnation_stop_enabled=False,max_wall_seconds=19800)
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        for key in list(result):
            if key.startswith(('amp_entry_','amp_exit_')):result.pop(key)
        result.update(curriculum='dense_swing_low_amp_v1',amp_max_weight=.01,amp_ramp_updates=0,
            swing_formula='reference_phase_weighted_additive_v1',swing_component_weights=[.6,.2,.2],
            swing_height_scale_m=.01,swing_multiplier_ramp=[1.,2.,200],
            regression_window_updates=100,regression_bad_windows=2,
            regression_mean_drop_seconds=2.,regression_failure_increase=.1,
            swing_multiplier='linear_from_session_start',
            clearance_run_definition='lowest_sole_over_1cm_for80ms_during_valid_moving_steps;resets_clear_state')
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        result.update(curriculum='contact_balance_low_amp_v1',swing_multiplier=2.,swing_multiplier_ramp=[2.,2.,0],
            balance_cost='huber_lateral_vy_over0.15_and_yaw_error_over0.5',balance_cost_ramp_updates=200,
            balance_reward_scales=[-.15,-.10],friction_randomization=[.65,.95],
            robustness_scope='friction only; no changes to PD, gains, masses, armature or observation layout')
        _super_checkpoint_training_metadata_3 = result
        result = _super_checkpoint_training_metadata_3
        result.update(curriculum='forward_correction_low_amp_v1',
            balance_cost_ramp_updates=0, forward_cost='huber((body_vx-applied_vx_command)/0.2)',
            forward_reward_scale=-.30, forward_cost_ramp_updates=200)
        _super_checkpoint_training_metadata_3 = result
        result = _super_checkpoint_training_metadata_3
        result.update(curriculum='command_scaled_phase_low_amp_v1',phase_clock='command_scaled_v1',
            phase_rate='clamp(applied_vx/native_vx,0,1)/native_period',
            transition_dq='blend*phase_scale*native_dq+blend_derivative*(native_q-standing_q)',
            forward_cost_ramp_updates=0)
        _super_checkpoint_training_metadata_3 = result
        result=_super_checkpoint_training_metadata_3
        result.update(curriculum='contact_margin_robustness_v1',
            robustness_scope='contact_offset coverage plus unchanged source friction; original PD and rigid-body properties',
            contact_offset_randomization=dict(range_m=[.001,.020],near_fraction=.5,original_fraction=.25,
                random_fraction=.25,original_m=.01,rest_offset_m=0.,resample='static_per_environment'))
        _super_checkpoint_training_metadata_3 = result
        return training_metadata(_super_checkpoint_training_metadata_3)

    def _reward_motion_progress(self):
        upright = ((self.projected_gravity[:, 2] < -.7)
                   & ((self.root_states[:, 2]-self.env_origins[:, 2]) > .45))
        valid = upright & (~self.reset_buf.bool() | self.time_out_buf)
        return motion_velocity_reward(self.commands[:, 0], self.base_lin_vel[:, 0],
                                      self.cfg.rewards.motion_velocity_scale)*valid

    def _reference_weight(self):
        return self.gait_curriculum.weights['guide'] if hasattr(self,'gait_curriculum') else .1

    def _raw_weight(self):
        return self.gait_curriculum.weights['raw'] if hasattr(self,'gait_curriculum') else .1

    def _phase_features(self):
        active = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        angle = 2*torch.pi*self.gait_phase
        return torch.stack((angle.sin(),angle.cos()),-1)*active[:,None]

    def _sync_phase(self):
        # Stopping must retain phase: re-aligning at blend=1 would jump q_ref.
        restart = self.standing_commands.started & (self.standing_commands.start<=1e-7) & (self.standing_commands.active_goal>0)
        ids = restart.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.gait_phase[ids] = self.reference_guide.nearest_phase(self.dof_pos[ids],self.dof_vel[ids])
        value = self._phase_features()
        self.obs_history[-1][:,-2:] = value
        self.obs_buf[:,-2:] = value
        self.privileged_obs_buf[:,96:98] = value

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        active = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        self._advance_reference_phase(active)

    def _advance_reference_phase(self,active):
        self.gait_phase[active] = advance_phase(self.gait_phase[active],
            self.standing_commands.value[active],self.dt,self.reference_guide.period,self.cfg.commands.native_speed)

    def _blended_reference(self,blend,derivative):
        return self.reference_guide.blended_pose(self.gait_phase,self.standing_state,blend,derivative,
            phase_rate_scale=phase_speed_scale(self.standing_commands.value,self.cfg.commands.native_speed))

    def _reference_valid(self):
        moving = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        return moving & (self.projected_gravity[:,2]<-.7) & (self.root_states[:,2]-self.env_origins[:,2]>.45) & (~self.reset_buf.bool()|self.time_out_buf)

    def _score(self,actual,target,scale):
        return similarity(actual,target,scale)*self._reference_weight()*self._reference_valid()

    def _reward_reference_pose(self):
        return self._score(self.dof_pos[:,self.pitch_indices],self.reference_q[:,self.pitch_indices],self.cfg.rewards.reference_pose_scale)

    def _reward_reference_lateral(self):
        return self._score(self.dof_pos[:,self.lateral_indices],self.reference_q[:,self.lateral_indices],self.cfg.rewards.reference_lateral_scale)

    def _reward_reference_waist(self):
        return self._score(self.dof_pos[:,self.waist_indices],self.reference_q[:,self.waist_indices],self.cfg.rewards.reference_waist_scale)

    def _reward_reference_upper(self):
        return self._score(self.dof_pos[:,self.upper_indices],self.reference_q[:,self.upper_indices],self.cfg.rewards.reference_upper_scale)

    def _reward_reference_velocity(self):
        return self._score(self.dof_vel,self.reference_dq,self.cfg.rewards.reference_velocity_scale)

    def _reward_reference_feet(self):
        local = self.rigid_state[:,self.feet_indices,:3]-self.root_states[:,None,:3]
        actual = quat_inverse_rotate(self.base_quat[:,None].expand(-1,2,-1),local).flatten(-2)
        return self._score(actual,self.reference_feet,self.cfg.rewards.reference_feet_scale)

    def _reward_reference_swing(self):
        return (self.swing_score*self._reference_weight()*self._reference_valid())*self.gait_curriculum.swing_multiplier

    def _reward_reference_support(self):
        return self.support_score*self._reference_weight()*self._reference_valid()

    def _reward_stance_slip(self):
        weight=self.gait_curriculum.weights['slip'] if hasattr(self,'gait_curriculum') else 0.
        return self.slip_cost*weight*self._reference_valid()

    def _reward_joint_margin(self):
        return margin_cost(self.dof_pos,self.hard_dof_limits[:,0],self.hard_dof_limits[:,1])*self._reference_weight()

    def sample_amp_expert_by_category(self,moving):
        return self.reference_guide.expert(moving)

    def _support_terms(self,height,target,force,weight,speed):
        _,support,slip=(support_terms(height,target,force,weight,speed))
        swing,self.dense_swing_details=dense_swing(height,target,force,weight)
        return swing,support,slip

    def _adjust_reset_states(self,env_ids,standing,phase,state):
        standing,phase,state=(standing,phase,state)
        if self.evaluation_reset_mode is None and not hasattr(self,'diagnostic_reset_phases'):
            nominal=self._nominal(env_ids)
            standing[nominal]=True
            state[nominal]=self.standing_state
            if not getattr(self,'_nominal_reset_verified',False) and nominal.any():
                assert torch.equal(state[nominal],self.standing_state.expand(int(nominal.sum()),-1))
                self._nominal_reset_verified=True
                print('NOMINAL_STARTUP_RESET_VERIFIED '+json.dumps(dict(verified=True,
                    exact_standing_state=True,reset_count=int(nominal.sum()),joint_noise=0.)),flush=True)
        return standing,phase,state

    def complete_training_rollout(self,iteration):
        self.gait_curriculum.add(self.branch_statistics.cpu().tolist(),*map(int,self.coverage_statistics.cpu().tolist()))
        self.branch_statistics.zero_();self.coverage_statistics.zero_()
        report=self.gait_curriculum.finish(iteration)
        if report is not None:
            report['phase_statistics']=self.phase_statistics.cpu().tolist()
            self.phase_statistics.zero_()
            print('EXP21_PERFORMANCE_WINDOW '+json.dumps(report,sort_keys=True),flush=True)
        return report

    def discriminator_training_enabled(self):
        window=self.gait_curriculum.last_window
        return self.gait_curriculum.weights['amp']>0 and window is not None and window['full_speed_fraction']>=.2

    def checkpoint_learning_state(self):
        return dict(curriculum=self.gait_curriculum.state_dict(),branch_statistics=self.branch_statistics.cpu().tolist(),
            coverage_statistics=self.coverage_statistics.cpu().tolist(),phase_statistics=self.phase_statistics.cpu().tolist())

    def restore_learning_state(self,state):
        self.gait_curriculum.load_state_dict(state['curriculum'])
        for name in ('branch_statistics','coverage_statistics','phase_statistics'):
            value=getattr(self,name);value.copy_(torch.as_tensor(state[name],device=self.device,dtype=value.dtype))

    def _corridor_components(self):
        return corridor_costs(self.base_lin_vel[:,1],self.base_ang_vel[:,2],self.commands[:,2])

    def _reward_corridor_lateral(self):
        return self._corridor_components()[0]*self._reference_valid()*self.gait_curriculum.balance_multiplier

    def _reward_corridor_yaw(self):
        return self._corridor_components()[1]*self._reference_valid()*self.gait_curriculum.balance_multiplier

    def _forward_cost(self):
        return forward_cost(self.base_lin_vel[:,0], self.commands[:,0])

    def _reward_corridor_forward(self):
        return self._forward_cost()*self._reference_valid()*self.gait_curriculum.forward_multiplier

    def _process_rigid_shape_props(self,props,env_id):
        props=super()._process_rigid_shape_props(props,env_id)
        if env_id==0:
            if self.cfg.domain_rand.randomize_contact_offset:
                self.contact_offset_values=sample_offsets(self.num_envs,self.cfg.domain_rand.contact_offset_range)
            else:
                self.contact_offset_values=torch.full((self.num_envs,),float(self.cfg.sim.physx.contact_offset))
        _super__process_rigid_shape_props_3 = set_contact_offsets(props,self.contact_offset_values[env_id])
        props=_super__process_rigid_shape_props_3
        if env_id<int(self.num_envs*self.cfg.amp.nominal_startup_fraction):
            for p in props:p.friction=.8
            self.friction_coeffs[env_id]=.8
            self.env_frictions[env_id]=.8
        return props

    def _nominal(self,env_ids):
        return cohort_mask(env_ids,self.num_envs,self.cfg.amp.nominal_startup_fraction)
