"""exp1.13t: scratch initialization control for the exp1.13 policy transfer."""
from .x1_amp_exp113_config import X1AMPExp113Cfg, X1AMPExp113CfgPPO


class X1AMPExp113tCfg(X1AMPExp113Cfg):
    pass


class X1AMPExp113tCfgPPO(X1AMPExp113CfgPPO):
    class algorithm(X1AMPExp113CfgPPO.algorithm):
        # Match the transferred run's first-200-update LR cap even with no origin.
        policy_transfer_warmup_for_scratch = True

    class runner(X1AMPExp113CfgPPO.runner):
        experiment_name = "x1_amp_exp113t"
        run_name = "exp1.13t"
        resume = False
        warm_start_mode = "scratch"
        require_policy_transfer = False
        require_fresh_initialization = True
        policy_transfer_source_curriculum = None
        policy_transfer_source_sha256 = None
