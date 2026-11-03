"""Keep exp2.7 intact and add a ramped forward error cost in valid motion."""
from humanoid.amp.exp28_curriculum import Exp28Curriculum
from humanoid.amp.exp28_tracking import forward_cost
from .x1_amp_exp27_env import X1AMPExp27Env


class X1AMPExp28Env(X1AMPExp27Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum = Exp28Curriculum()

    def _forward_cost(self):
        return forward_cost(self.base_lin_vel[:,0], self.commands[:,0])

    def _reward_corridor_forward(self):
        return self._forward_cost()*self._reference_valid()*self.gait_curriculum.forward_multiplier

    def compute_reward(self):
        super().compute_reward()
        cost = self._forward_cost()
        valid = self._reference_valid()
        gain = self.gait_curriculum.forward_multiplier
        metrics = self.extras['training_metrics']
        metrics['Forward/cost'] = cost[valid]
        metrics['Forward/reward'] = (cost*gain*self.reward_scales['corridor_forward'])[valid]
        metrics['Forward/multiplier'] = self.commands[:,0].new_full((self.num_envs,), gain)
        metrics['Forward/overspeed_fraction'] = (self.base_lin_vel[:,0]-self.commands[:,0] > .2)[valid].float()

    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        result.update(curriculum='forward_correction_low_amp_v1',
            balance_cost_ramp_updates=0, forward_cost='huber((body_vx-applied_vx_command)/0.2)',
            forward_reward_scale=-.30, forward_cost_ramp_updates=200)
        return result
