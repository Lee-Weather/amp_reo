"""Keep control/negative costs fixed; strengthen phase swing at stable stage2."""
from humanoid.amp.exp24_curriculum import Exp24Curriculum
from .x1_amp_exp23_env import X1AMPExp23Env

class X1AMPExp24Env(X1AMPExp23Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=Exp24Curriculum()

    def _reward_reference_swing(self):
        return super()._reward_reference_swing()*(2. if self.gait_curriculum.stage>=2 else 1.)

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        for key in list(result):
            if key.startswith('amp_entry_'):result.pop(key)
        result.update(curriculum='overnight_preheated_amp_v1',discriminator_preheat_updates=100,
            amp_stage_weights=[0.,.01,.03,.05],promotion_seconds=[14.,16.,18.],
            promotion_max_non_timeout=[.5,.35,.2],minimum_higher_stage_hold=200,
            swing_multiplier_stages=[1.,1.,2.,2.],max_wall_seconds=19800)
        return result
