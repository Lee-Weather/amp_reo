"""exp1.8: signed reverse tracking and leg settling after body braking."""
from .x1_amp_exp17_config import X1AMPExp17Cfg, X1AMPExp17CfgPPO


class X1AMPExp18Cfg(X1AMPExp17Cfg):
    class rewards(X1AMPExp17Cfg.rewards):
        backward_speed_scale = .2
        brake_leg_linear_scale = .08
        brake_leg_angular_scale = .15

        class scales(X1AMPExp17Cfg.rewards.scales):
            backward_progress = 1.5


class X1AMPExp18CfgPPO(X1AMPExp17CfgPPO):
    class runner(X1AMPExp17CfgPPO.runner):
        experiment_name = "x1_amp_exp18"
        run_name = "exp1.8"
        max_iterations = 1000
