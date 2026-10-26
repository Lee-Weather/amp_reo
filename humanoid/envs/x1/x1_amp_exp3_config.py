"""exp_3: x-only commands, stronger AMP feedback and controlled PPO updates."""
from .x1_amp_exp11_config import X1AMPExp11Cfg, X1AMPExp11CfgPPO


class X1AMPExp3Cfg(X1AMPExp11Cfg):
    class commands(X1AMPExp11Cfg.commands):
        class ranges(X1AMPExp11Cfg.commands.ranges):
            lin_vel_x = [-0.2, 0.65]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

    class rewards(X1AMPExp11Cfg.rewards):
        tracking_lin_sigma = 0.09
        # Angular tracking continues to use inherited tracking_sigma=0.25.


class X1AMPExp3CfgPPO(X1AMPExp11CfgPPO):
    class algorithm(X1AMPExp11CfgPPO.algorithm):
        learning_rate = 1e-4
        schedule = "adaptive"
        kl_check = "minibatch"
        desired_kl = 0.02
        min_learning_rate = 1e-5
        max_learning_rate = 1e-4

    class discriminator(X1AMPExp11CfgPPO.discriminator):
        style_reward_scale = 10.

    class runner(X1AMPExp11CfgPPO.runner):
        experiment_name = "x1_amp_exp3"
        run_name = "exp_3"
        max_iterations = 6000
        save_interval = 100
