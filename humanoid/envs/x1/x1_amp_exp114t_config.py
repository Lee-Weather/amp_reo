"""exp1.14t: dense real-transition guidance without changing observation dimensions."""
from .x1_amp_exp114_common_config import X1AMPExp114CommonCfg, X1AMPExp114CommonCfgPPO


class X1AMPExp114tCfg(X1AMPExp114CommonCfg):
    class rewards(X1AMPExp114CommonCfg.rewards):
        class scales(X1AMPExp114CommonCfg.rewards.scales):
            reference_transition = 4.


class X1AMPExp114tCfgPPO(X1AMPExp114CommonCfgPPO):
    class runner(X1AMPExp114CommonCfgPPO.runner):
        experiment_name = "x1_amp_exp114t"
        run_name = "exp1.14t"
