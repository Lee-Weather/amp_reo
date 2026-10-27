"""exp1.5 candidate: low-speed tracking, quiet legs and clearance while moving."""
from .x1_amp_exp14_config import X1AMPExp14Cfg, X1AMPExp14CfgPPO


class X1AMPExp15Cfg(X1AMPExp14Cfg):
    class amp(X1AMPExp14Cfg.amp):
        standing_style_multiplier = .2

    class rewards(X1AMPExp14Cfg.rewards):
        tracking_min_std = .08
        tracking_max_std = .30
        tracking_max_speed = .65
        scuff_clearance = .015

        class scales(X1AMPExp14Cfg.rewards.scales):
            stand_dof_vel = -.6
            stand_pose = -.6
            feet_scuff = -.4


class X1AMPExp15CfgPPO(X1AMPExp14CfgPPO):
    class runner(X1AMPExp14CfgPPO.runner):
        experiment_name = "x1_amp_exp15"
        run_name = "exp1.5"
        max_iterations = 1000
