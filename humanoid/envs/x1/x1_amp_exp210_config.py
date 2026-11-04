"""exp2.10: increase only the existing yaw corridor penalty."""
from .x1_amp_exp29_config import X1AMPExp29Cfg, X1AMPExp29CfgPPO


class X1AMPExp210Cfg(X1AMPExp29Cfg):
    class rewards(X1AMPExp29Cfg.rewards):
        class scales(X1AMPExp29Cfg.rewards.scales):
            corridor_yaw = -.30


class X1AMPExp210CfgPPO(X1AMPExp29CfgPPO):
    class runner(X1AMPExp29CfgPPO.runner):
        experiment_name = 'x1_amp_exp210'
        run_name = 'exp2.10'
        max_iterations = 1000
