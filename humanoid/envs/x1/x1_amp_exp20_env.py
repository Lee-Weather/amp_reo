"""Native-tempo full-body phase imitation, isolated from exp1 behavior."""
import hashlib
import json
from pathlib import Path

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.amp.native_reference import NativeCycle
from humanoid.amp.exp20_commands import SmoothCommands
from humanoid.amp.exp20_rewards import similarity,raw_violation,margin_cost,support_terms
from humanoid.amp.gait_rewards import lowest_sole_kinematics
from humanoid.amp.motion import quat_inverse_rotate
from humanoid.amp.control import sync_policy_commands
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp113_env import X1AMPExp113Env


class X1AMPExp20Env(X1AMPExp113Env):
    _iteration = 3000  # Inference uses full weights; runner sets relative train count.

    def _init_buffers(self):
        super()._init_buffers()
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

    def set_training_iteration(self,iteration):
        if iteration<0: raise ValueError('Negative exp2.0 iteration')
        self._iteration = int(iteration)

    def _reference_weight(self):
        return .25+.75*min(1.,self._iteration/500.)

    def _raw_weight(self):
        return .2+.8*min(1.,self._iteration/500.)

    def _phase_features(self):
        active = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        angle = 2*torch.pi*self.gait_phase
        return torch.stack((angle.sin(),angle.cos()),-1)*active[:,None]

    def _policy_frame(self):
        return torch.cat((super()._policy_frame(),self._phase_features()),-1)

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

    def step(self,actions):
        self.raw_violation_sum.zero_()
        result = super().step(actions)
        self._sync_phase()
        return result

    def _compute_torques(self,actions):
        torques = super()._compute_torques(actions)
        self.raw_violation_sum.add_(raw_violation(self.pd_target_raw,
            self.hard_dof_limits[:,0],self.hard_dof_limits[:,1],self.cfg.rewards.raw_target_scale))
        return torques

    def _draw_commands(self,count):
        result = torch.zeros(count,3,device=self.device)
        if getattr(self,'_guidance_evaluation',False) or getattr(self.cfg.commands,'guidance_evaluation',False):
            result[:,0] = self.command_ranges['lin_vel_x'][0]
        else:
            result[:,0] = self.cfg.commands.native_speed
            result[torch.rand(count,device=self.device)<self.cfg.commands.zero_probability,0] = 0
        return result

    def _resample_commands(self,env_ids):
        requested = self._reset_commands if self._reset_commands is not None else self._draw_commands(len(env_ids))
        self.standing_commands.requested[env_ids] = requested
        # Scheduling changes the goal, never the command of the transition being rewarded.
        if self._reset_reference is not None or not self._inside_step:
            self.commands[:,:3] = self.standing_commands.applied()

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

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        active = (self.standing_commands.value>1e-7)|self.standing_commands.transition
        self._advance_reference_phase(active)

    def _advance_reference_phase(self,active):
        self.gait_phase[active] = (self.gait_phase[active]+self.dt/self.reference_guide.period).remainder(1)

    def _blended_reference(self,blend,derivative):
        return self.reference_guide.blended_pose(self.gait_phase,self.standing_state,blend,derivative)

    def reset_idx(self,env_ids):
        if not len(env_ids): return
        state = self.standing_state.expand(len(env_ids),-1).clone()
        state[:,13:42] += (2*torch.rand(len(env_ids),29,device=self.device)-1)*self.cfg.amp.standing_joint_noise
        state[:,13:42] = state[:,13:42].maximum(self.hard_dof_limits[:,0]).minimum(self.hard_dof_limits[:,1])
        state[:,2] += self.cfg.amp.ground_clearance-self.reference_guide.sole_height(state).amin(-1)
        self._reset_reference,self._reset_commands = state,self._draw_commands(len(env_ids))
        self.reset_standing[env_ids] = True
        self.standing_commands.reset(env_ids,torch.ones(len(env_ids),dtype=torch.bool,device=self.device))
        LeggedRobot.reset_idx(self,env_ids)
        for frame in self.obs_history: frame[env_ids] = 0
        self.amp_history[env_ids] = self.motion_library.features(state)[:,None]
        self.amp_valid_history[env_ids] = False
        self._reset_reference,self._reset_commands = None,None
        for value in (self.zero_steps,self.stop_hold_steps,self.previous_moving,self.stop_pending,self.stand_gate): value[env_ids] = 0
        self.gait_phase[env_ids] = self.reference_guide.nearest_phase(state[:,13:42],state[:,42:])

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
        return self.swing_score*self._reference_weight()*self._reference_valid()

    def _reward_reference_support(self):
        return self.support_score*self._reference_weight()*self._reference_valid()

    def _reward_stance_slip(self):
        return self.slip_cost*self._reference_weight()*self._reference_valid()

    def _reward_raw_target_violation(self):
        return self.raw_violation_sum.mean(-1)/self.cfg.control.decimation*self._raw_weight()

    def _reward_joint_margin(self):
        return margin_cost(self.dof_pos,self.hard_dof_limits[:,0],self.hard_dof_limits[:,1])*self._reference_weight()

    def _posture_cost(self,group):
        quiet = (self.standing_commands.value<=1e-7)&~self.standing_commands.transition
        return super()._posture_cost(group)*quiet

    def _reward_feet_scuff(self):
        return torch.zeros(self.num_envs,device=self.device)

    def sample_amp_expert_by_category(self,moving):
        return self.reference_guide.expert(moving)

    def sample_amp_expert(self,count):
        raise ValueError('exp2.0 requires sample_amp_expert_by_category with paired policy categories')

    def _support_terms(self,height,target,force,weight,speed):
        return support_terms(height,target,force,weight,speed)

    def compute_reward(self):
        blend,derivative = self.standing_commands.blend()
        reference_state = self._blended_reference(blend,derivative)
        self.reference_q,self.reference_dq = reference_state[:,13:42],reference_state[:,42:]
        self.reference_feet = self.motion_library.fk.key_positions(self.reference_q)[:,self.native_foot_ids].flatten(-2)
        target_height = self.reference_guide.sole_height(reference_state).clamp(min=0)
        height,speed = lowest_sole_kinematics(self.rigid_state[:,self.feet_indices],self.sole_vertices,self.foot_com_local,self.env_origins[:,2])
        self.swing_score,self.support_score,self.slip_cost = self._support_terms(height,target_height,
            self.contact_forces[:,self.feet_indices,2],self.total_mass[:,0]*9.81,speed)
        super().compute_reward()
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

    def checkpoint_training_metadata(self):
        # Audit floating point results are logged separately. Checkpoint identity
        # must not depend on CPU/GPU rounding in an order statistic or FK result.
        native_identity = dict(clip=self.cfg.amp.native_clip,first_frame=self.cfg.amp.native_first_frame,
            last_frame=self.cfg.amp.native_last_frame,period_s=self.reference_guide.period,
            command_speed=self.cfg.commands.native_speed,correction_sha256=self.correction_sha256,
            sole_support_cache_sha256=self.reference_guide.sole_cache_sha256,
            hash_definition='canonical_json_sorted_keys_compact_utf8',
            builder='native_curve_margin_v1',time_scale=1.,amplitude_scale=1.)
        return dict(curriculum='native_full_body_v1',reference_route='native_cycle_shared_with_amp_v1',
            reference_source_sha256=self.reference_source_hashes,native_reference=native_identity,
            phase_input_columns=[96,97],command_events=dict(forward_vx=self.cfg.commands.native_speed,
                forward_probability=1-self.cfg.commands.zero_probability,zero_probability=self.cfg.commands.zero_probability),
            expert_sampling=dict(moving='native_cycle',quiet='real_complete_quiet_histories',
                ratio='sampled_replay_categories',transition_excluded=True),transition_seconds=self.cfg.commands.transition_seconds,
            initialization='standing_only',conditional_experts=True,iteration_origin='exp2.0_new_updates')
