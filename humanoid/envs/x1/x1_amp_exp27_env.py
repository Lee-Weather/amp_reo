"""Preserve native swing; penalize lateral/yaw excursions in valid motion."""
from humanoid.amp.exp27_curriculum import Exp27Curriculum
from humanoid.amp.exp27_balance import corridor_costs
from .x1_amp_exp26_env import X1AMPExp26Env

class X1AMPExp27Env(X1AMPExp26Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=Exp27Curriculum()

    def _corridor_components(self):
        return corridor_costs(self.base_lin_vel[:,1],self.base_ang_vel[:,2],self.commands[:,2])

    def _reward_corridor_lateral(self):
        return self._corridor_components()[0]*self._reference_valid()*self.gait_curriculum.balance_multiplier

    def _reward_corridor_yaw(self):
        return self._corridor_components()[1]*self._reference_valid()*self.gait_curriculum.balance_multiplier

    def compute_reward(self):
        super().compute_reward()
        lateral,yaw=self._corridor_components();valid=self._reference_valid()
        metrics=self.extras['training_metrics'];gain=self.gait_curriculum.balance_multiplier
        metrics['Balance/lateral_cost']=lateral[valid]
        metrics['Balance/yaw_cost']=yaw[valid]
        metrics['Balance/lateral_reward']=(lateral*gain*self.reward_scales['corridor_lateral'])[valid]
        metrics['Balance/yaw_reward']=(yaw*gain*self.reward_scales['corridor_yaw'])[valid]
        metrics['Balance/multiplier']=self.commands[:,0].new_full((self.num_envs,),gain)
        metrics['Balance/contact_friction']=self.env_frictions.reshape(self.num_envs,-1)[:,0]

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        result.update(curriculum='contact_balance_low_amp_v1',swing_multiplier=2.,swing_multiplier_ramp=[2.,2.,0],
            balance_cost='huber_lateral_vy_over0.15_and_yaw_error_over0.5',balance_cost_ramp_updates=200,
            balance_reward_scales=[-.15,-.10],friction_randomization=[.65,.95],
            robustness_scope='friction only; no changes to PD, gains, masses, armature or observation layout')
        return result
