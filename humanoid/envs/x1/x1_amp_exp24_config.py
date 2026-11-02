"""4000 additional updates, bounded wall time, frozen DH features."""
from .x1_amp_exp23_config import X1AMPExp23Cfg,X1AMPExp23CfgPPO

class X1AMPExp24Cfg(X1AMPExp23Cfg):
    pass

class X1AMPExp24CfgPPO(X1AMPExp23CfgPPO):
    runner_class_name='Exp24AMPOnPolicyRunner'
    class runner(X1AMPExp23CfgPPO.runner):
        experiment_name='x1_amp_exp24'
        run_name='exp2.4'
        max_iterations=4000
        save_interval=100
        max_wall_seconds=19800
        policy_transfer_source_curriculum='relaxed_anchor_low_amp_v1'
        policy_transfer_source_iteration=200
        policy_transfer_source_task='x1_amp_exp23'
        policy_transfer_source_sha256='e453506b82725d470745993bb703475099be4411b02eb57df182ab9eaf3bf2fa'
