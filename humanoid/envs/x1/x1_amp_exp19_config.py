"""exp1.9: credit across gait cycles and tighter straight-line tracking."""
from .x1_amp_exp18_config import X1AMPExp18Cfg, X1AMPExp18CfgPPO


class X1AMPExp19Cfg(X1AMPExp18Cfg):
    class rewards(X1AMPExp18Cfg.rewards):
        tracking_sigma = .04  # Angular exponent denominator: (0.2 rad/s)^2.


class X1AMPExp19CfgPPO(X1AMPExp18CfgPPO):
    class algorithm(X1AMPExp18CfgPPO.algorithm):
        lam = .98
        # Keep 24,576 transitions/minibatch as the rollout grows fourfold.
        num_mini_batches = 16

    class discriminator(X1AMPExp18CfgPPO.discriminator):
        # Preserve replay capacity and update count per environment transition.
        replay_size = 400000
        num_updates = 16

    class runner(X1AMPExp18CfgPPO.runner):
        experiment_name = "x1_amp_exp19"
        run_name = "exp1.9"
        num_steps_per_env = 96
        max_iterations = 250  # 98,304,000 transitions, equal to exp1.8's 1000.
        save_interval = 25
