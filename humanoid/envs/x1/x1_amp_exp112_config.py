"""exp1.12: balanced AMP, bounded exploration and scratch-training curriculum."""
from .x1_amp_exp111_config import X1AMPExp111Cfg, X1AMPExp111CfgPPO


class X1AMPExp112Cfg(X1AMPExp111Cfg):
    class rewards(X1AMPExp111Cfg.rewards):
        curriculum_start_iteration = 200
        curriculum_end_iteration = 1500
        curriculum_initial_stand_weight = .1


class X1AMPExp112CfgPPO(X1AMPExp111CfgPPO):
    class policy(X1AMPExp111CfgPPO.policy):
        init_noise_std = .3
        max_noise_std = .5

    class algorithm(X1AMPExp111CfgPPO.algorithm):
        entropy_coef = .001

    class discriminator(X1AMPExp111CfgPPO.discriminator):
        num_updates = 4

    class runner(X1AMPExp111CfgPPO.runner):
        experiment_name = "x1_amp_exp112"
        run_name = "exp1.12"

