"""Identical exp1.13 rewards/control, with a distinct scratch checkpoint identity."""
from .x1_amp_exp113_env import X1AMPExp113Env


class X1AMPExp113tEnv(X1AMPExp113Env):
    def checkpoint_training_metadata(self):
        metadata = super().checkpoint_training_metadata()
        metadata["initialization"] = "scratch_exp113t_v1"
        return metadata
