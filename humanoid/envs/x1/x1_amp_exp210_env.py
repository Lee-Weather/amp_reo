"""Same native dynamics and phase; declare the changed yaw reward accurately."""
from .x1_amp_exp29_env import X1AMPExp29Env


class X1AMPExp210Env(X1AMPExp29Env):
    def checkpoint_training_metadata(self):
        result = super().checkpoint_training_metadata()
        result.update(curriculum='yaw_correction_low_amp_v1',
                      balance_reward_scales=[self.cfg.rewards.scales.corridor_lateral,
                                             self.cfg.rewards.scales.corridor_yaw])
        return result
