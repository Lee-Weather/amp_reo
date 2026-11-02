"""Recoverable source-policy states replace native-expert RSI in exp2.2."""
import hashlib
import json
from pathlib import Path
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.amp.gait_curriculum import GaitCurriculum
from humanoid.envs.base.legged_robot import LeggedRobot
from .x1_amp_exp21_env import X1AMPExp21Env

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

class X1AMPExp22Env(X1AMPExp21Env):
    def _init_buffers(self):
        super()._init_buffers()
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

    def reset(self):
        # The base reset takes a zero-action physics step before inference.
        # A moving source snapshot must receive its own policy action first.
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        self.compute_observations()
        return self.obs_buf,self.privileged_obs_buf

    def reset_idx(self,env_ids):
        if not len(env_ids):return
        count=len(env_ids)
        probability=self.gait_curriculum.weights['rsi_probability']
        if self.evaluation_reset_mode is not None:probability=float(self.evaluation_reset_mode=='rsi')
        standing=torch.rand(count,device=self.device)>=probability
        samples=torch.randint(len(self.bank['states']),(count,),device=self.device)
        state=self.bank['states'][samples].clone()
        phase=self.bank['phases'][samples].clone()
        state[standing]=self.standing_state
        state[standing,13:42]+=(2*torch.rand(int(standing.sum()),29,device=self.device)-1)*self.cfg.amp.standing_joint_noise
        state[:,13:42]=state[:,13:42].maximum(self.hard_dof_limits[:,0]).minimum(self.hard_dof_limits[:,1])
        if standing.any():
            state[standing,2]+=self.cfg.amp.ground_clearance-self.reference_guide.sole_height(state[standing]).amin(-1)
            phase[standing]=self.reference_guide.nearest_phase(state[standing,13:42],state[standing,42:])
        requested=self._draw_commands(count);requested[~standing,0]=self.cfg.commands.native_speed
        self._reset_reference,self._reset_commands=state,requested
        self.reset_standing[env_ids]=standing;self.standing_commands.reset(env_ids,standing)
        LeggedRobot.reset_idx(self,env_ids)
        ids=env_ids[~standing];bank_ids=samples[~standing]
        for value in (self.standing_commands.active_goal,self.standing_commands.start,self.standing_commands.value):value[ids]=self.cfg.commands.native_speed
        self.standing_commands.requested[env_ids]=requested
        self.commands[env_ids,:3]=self.standing_commands.applied()[env_ids]
        for frame in self.obs_history:frame[env_ids]=0
        history=self.bank['histories'][bank_ids]
        self.actions[ids]=history[:,-1,67:96]
        self.last_actions[ids]=self.actions[ids]
        self.last_last_actions[ids]=history[:,-2,67:96]
        self.last_dof_vel[ids]=self.dof_vel[ids]
        self._bank_pending=(ids,history)
        amp=self.motion_library.features(state)[:,None].expand(-1,self.cfg.amp.history_steps,-1).clone()
        if len(ids):
            amp[~standing]=self.motion_library.features(self.bank['amp_states'][bank_ids].reshape(-1,71)).reshape(len(ids),3,-1)
        self.amp_history[env_ids]=amp;self.amp_valid_history[env_ids]=True
        self._reset_reference,self._reset_commands=None,None
        for value in (self.zero_steps,self.stop_hold_steps,self.previous_moving,self.stop_pending,self.stand_gate):value[env_ids]=0
        self.gait_phase[env_ids]=phase;self.completed_cycles[env_ids]=0

    def compute_observations(self):
        super().compute_observations()
        if self._bank_pending is not None:
            ids,history=self._bank_pending
            for i,frame in enumerate(self.obs_history):frame[ids]=history[:,i]
            self.obs_buf[ids]=history.flatten(1)
            self._bank_pending=None

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        result.update(curriculum='anchored_policy_state_native_v1',initialization='mixed_successful_policy_state_and_standing',
            policy_bank_sha256=self.policy_bank_sha256,initial_rsi_probability=.5,
            exploration_gain=.5,freeze_std_iterations=1000000000,freeze_features_min_iterations=500,
            source_anchor_coef=10.,source_max_raw_pd_rms_rad=.04)
        return result
