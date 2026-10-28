"""exp1.11: original DH rollout and PPO update cadence with 29-DOF AMP."""
from .x1_amp_exp110_config import X1AMPExp110Cfg, X1AMPExp110CfgPPO


class X1AMPExp111Cfg(X1AMPExp110Cfg):
    pass


class X1AMPExp111CfgPPO(X1AMPExp110CfgPPO):
    class algorithm(X1AMPExp110CfgPPO.algorithm):
        num_learning_epochs = 2
        num_mini_batches = 4

    class runner(X1AMPExp110CfgPPO.runner):
        num_steps_per_env = 24
        experiment_name = "x1_amp_exp111"
        run_name = "exp1.11"
