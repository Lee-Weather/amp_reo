"""exp3.1: KL anchor zero-command stability + friction randomization."""
from .x1_amp_exp30_config import X1AMPExp30Cfg, X1AMPExp30CfgPPO


class X1AMPExp31Cfg(X1AMPExp30Cfg):
    class commands(X1AMPExp30Cfg.commands):
        """提高零指令比例从 20% → 50%"""
        zero_probability = 0.5
        resampling_time = 6.

    class domain_rand(X1AMPExp30Cfg.domain_rand):
        """新增：域随机化"""
        randomize_friction = True
        friction_range = [0.6, 1.2]

    class rewards(X1AMPExp30Cfg.rewards):
        """新增：站姿稳定性奖赏"""
        standing_stability_angular_scale = -0.03
        standing_stability_max_angular = 0.15


class X1AMPExp31CfgPPO(X1AMPExp30CfgPPO):
    runner_class_name = "X1Exp31AMPOnPolicyRunner"

    class algorithm(X1AMPExp30CfgPPO.algorithm):
        anchor_loss_coef = 0.5
        anchor_zero_threshold = 0.01

    class runner(X1AMPExp30CfgPPO.runner):
        experiment_name = "x1_amp_exp31"
        run_name = "exp3.1"
        max_iterations = 500
        save_interval = 50
