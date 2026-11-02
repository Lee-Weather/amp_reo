"""Same state bank and control; explicit low-AMP entry for exp2.3."""
from humanoid.amp.exp23_curriculum import Exp23Curriculum
from .x1_amp_exp22_env import X1AMPExp22Env

class X1AMPExp23Env(X1AMPExp22Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=Exp23Curriculum()

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        result.update(curriculum='relaxed_anchor_low_amp_v1',source_anchor_coef=1.,
            critic_only_iterations=0,freeze_features_min_iterations=1000000000,
            amp_entry_min_new_updates=200,amp_entry_min_seconds=14.,
            amp_entry_max_non_timeout=.5,amp_entry_good_windows=2,amp_entry_weight=.01,
            amp_entry_baseline_mean_ratio=.95,amp_entry_baseline_failure_tolerance=.05)
        return result
