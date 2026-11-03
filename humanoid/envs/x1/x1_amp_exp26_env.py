"""Only swing shaping changes; original support, slip and control are retained."""
from humanoid.amp.exp26_curriculum import Exp26Curriculum
from humanoid.amp.exp26_swing import dense_swing,ClearanceRuns
from .x1_amp_exp25_env import X1AMPExp25Env

class X1AMPExp26Env(X1AMPExp25Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=Exp26Curriculum()
        self.swing_air_tracker=ClearanceRuns(self.num_envs,self.device,self.dt)

    def reset_idx(self,ids):
        super().reset_idx(ids)
        if hasattr(self,'swing_air_tracker'):self.swing_air_tracker.reset(ids)

    def _support_terms(self,height,target,force,weight,speed):
        _,support,slip=super()._support_terms(height,target,force,weight,speed)
        swing,self.dense_swing_details=dense_swing(height,target,force,weight)
        return swing,support,slip

    def _reward_reference_swing(self):
        return super()._reward_reference_swing()*self.gait_curriculum.swing_multiplier

    def compute_reward(self):
        super().compute_reward()
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

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        for key in list(result):
            if key.startswith(('amp_entry_','amp_exit_')):result.pop(key)
        result.update(curriculum='dense_swing_low_amp_v1',amp_max_weight=.01,amp_ramp_updates=0,
            swing_formula='reference_phase_weighted_additive_v1',swing_component_weights=[.6,.2,.2],
            swing_height_scale_m=.01,swing_multiplier_ramp=[1.,2.,200],
            regression_window_updates=100,regression_bad_windows=2,
            regression_mean_drop_seconds=2.,regression_failure_increase=.1,
            swing_multiplier='linear_from_session_start',
            clearance_run_definition='lowest_sole_over_1cm_for80ms_during_valid_moving_steps;resets_clear_state')
        return result
