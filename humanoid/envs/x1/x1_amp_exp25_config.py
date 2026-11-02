"""2000 additional low-AMP updates from exp2.4 best1100."""
from .x1_amp_exp24_config import X1AMPExp24Cfg,X1AMPExp24CfgPPO

class X1AMPExp25Cfg(X1AMPExp24Cfg):
    pass

class X1AMPExp25CfgPPO(X1AMPExp24CfgPPO):
    runner_class_name='Exp25AMPOnPolicyRunner'
    class runner(X1AMPExp24CfgPPO.runner):
        experiment_name='x1_amp_exp25'
        run_name='exp2.5'
        max_iterations=2000
        policy_transfer_source_curriculum='overnight_preheated_amp_v1'
        policy_transfer_source_iteration=1100
        policy_transfer_source_task='x1_amp_exp24'
        policy_transfer_source_sha256='85f61e6aebdfad6e179e30423d28e9b7e351c6e22e6e14e3ca6aec357ee2f795'
