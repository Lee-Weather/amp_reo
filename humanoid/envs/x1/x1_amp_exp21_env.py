"""Native RSI and disjoint completed-episode performance gates."""
import json
import torch

from humanoid.amp.gait_curriculum import GaitCurriculum
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp20_env import X1AMPExp20Env


class X1AMPExp21Env(X1AMPExp20Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=GaitCurriculum()
        self.branch_statistics=torch.zeros(2,4,device=self.device)
        self.coverage_statistics=torch.zeros(2,device=self.device)
        self.phase_statistics=torch.zeros(5,2,device=self.device)
        self.completed_cycles=torch.zeros(self.num_envs,device=self.device)
        self.evaluation_reset_mode=None

    def _reference_weight(self):
        return self.gait_curriculum.weights['guide'] if hasattr(self,'gait_curriculum') else .1

    def _raw_weight(self):
        return self.gait_curriculum.weights['raw'] if hasattr(self,'gait_curriculum') else .1

    def _reward_stance_slip(self):
        weight=self.gait_curriculum.weights['slip'] if hasattr(self,'gait_curriculum') else 0.
        return self.slip_cost*weight*self._reference_valid()

    def _adjust_reset_states(self,env_ids,standing,phase,state):
        return standing,phase,state

    def reset_idx(self,env_ids):
        if not len(env_ids): return
        count=len(env_ids)
        probability=self.gait_curriculum.weights['rsi_probability']
        if self.evaluation_reset_mode is not None: probability=float(self.evaluation_reset_mode=='rsi')
        standing=torch.rand(count,device=self.device)>=probability
        phase=torch.rand(count,device=self.device)
        if hasattr(self,'diagnostic_reset_phases'):
            phase=self.diagnostic_reset_phases[env_ids].clone()
            standing[:]=False
        state=self.reference_guide.state(phase)
        state[standing]=self.standing_state
        state[standing,13:42]+=(2*torch.rand(int(standing.sum()),29,device=self.device)-1)*self.cfg.amp.standing_joint_noise
        standing,phase,state=self._adjust_reset_states(env_ids,standing,phase,state)
        state[:,13:42]=state[:,13:42].maximum(self.hard_dof_limits[:,0]).minimum(self.hard_dof_limits[:,1])
        if standing.any():
            state[standing,2]+=self.cfg.amp.ground_clearance-self.reference_guide.sole_height(state[standing]).amin(-1)
            phase[standing]=self.reference_guide.nearest_phase(state[standing,13:42],state[standing,42:])
        requested=self._draw_commands(count)
        requested[~standing,0]=self.cfg.commands.native_speed
        self._reset_reference,self._reset_commands=state,requested
        self.reset_standing[env_ids]=standing
        self.standing_commands.reset(env_ids,standing)
        LeggedRobot.reset_idx(self,env_ids)
        # RSI begins in a moving state at its exact native phase, without a
        # stationary buffer or a second acceleration ramp.
        ids=env_ids[~standing]
        speed=self.cfg.commands.native_speed
        for value in (self.standing_commands.active_goal,self.standing_commands.start,self.standing_commands.value): value[ids]=speed
        self.standing_commands.requested[env_ids]=requested
        self.commands[env_ids,:3]=self.standing_commands.applied()[env_ids]
        for frame in self.obs_history: frame[env_ids]=0
        histories=self.reference_guide.history(phase)
        histories[standing]=self.motion_library.features(state[standing])[:,None]
        self.amp_history[env_ids]=histories
        self.amp_valid_history[env_ids]=True
        self._reset_reference,self._reset_commands=None,None
        for value in (self.zero_steps,self.stop_hold_steps,self.previous_moving,self.stop_pending,self.stand_gate): value[env_ids]=0
        self.gait_phase[env_ids]=phase
        self.completed_cycles[env_ids]=0

    def compute_reward(self):
        super().compute_reward()
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

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        result.update(curriculum='performance_gated_native_v1',initialization='mixed_native_rsi_and_standing',
            curriculum_state_location='infos.exp21.learning_state',critic_only_iterations=50,
            freeze_std_iterations=200,initial_rsi_probability=.7,
            phase_coverage='buffer_start_steady_stop_quiet')
        return result
