"""exp1.7: backward progress, straight tracking and physically correct foot scuff."""
from .x1_amp_exp16_config import X1AMPExp16Cfg, X1AMPExp16CfgPPO


class X1AMPExp17Cfg(X1AMPExp16Cfg):
    class commands(X1AMPExp16Cfg.commands):
        backward_probability = .3

    class amp(X1AMPExp16Cfg.amp):
        motion_weights = {"F1_B1_stand_to_walk":1., "F1_B2_walk_to_stand":1.,
                          "F1_B3_continuous_walk":1., "F1_B4_walk_backwards":2.}
        rsi_relative_speed_tolerance = .5

    class rewards(X1AMPExp16Cfg.rewards):
        tracking_lateral_std = .10
        tracking_sigma = .09  # Angular exponent denominator: (0.3 rad/s)^2.

        class scales(X1AMPExp16Cfg.rewards.scales):
            backward_progress = 1.
            stand_dof_vel = -.9


class X1AMPExp17CfgPPO(X1AMPExp16CfgPPO):
    class runner(X1AMPExp16CfgPPO.runner):
        experiment_name = "x1_amp_exp17"
        run_name = "exp1.7"
        max_iterations = 1000
