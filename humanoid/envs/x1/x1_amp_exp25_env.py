"""Keep original swing rewards; use pooled entry into a single low AMP stage."""
from humanoid.amp.exp25_curriculum import Exp25Curriculum
from .x1_amp_exp23_env import X1AMPExp23Env

class X1AMPExp25Env(X1AMPExp23Env):
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_curriculum=Exp25Curriculum()

    def checkpoint_training_metadata(self):
        result=super().checkpoint_training_metadata()
        for key in list(result):
            if key.startswith('amp_entry_'):result.pop(key)
        result.update(curriculum='pooled100_low_amp_v1',amp_max_weight=.01,amp_ramp_updates=100,
            amp_entry_window_updates=100,amp_entry_min_episodes_per_branch=200,
            amp_entry_min_seconds=15.,amp_entry_max_non_timeout=.55,
            amp_entry_baseline_mean_ratio=.95,amp_entry_baseline_failure_tolerance=.05,
            amp_exit_min_seconds=13.,amp_exit_max_non_timeout=.6,amp_exit_bad_windows=2,
            swing_multiplier=1.,stagnation_stop_enabled=False,max_wall_seconds=19800)
        return result
