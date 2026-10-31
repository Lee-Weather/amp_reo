"""exp1.18: continue exp1.17 for 5000 more updates without changing rewards."""
from .x1_amp_exp117_config import X1AMPExp117Cfg, X1AMPExp117CfgPPO


class X1AMPExp118Cfg(X1AMPExp117Cfg):
    pass


class X1AMPExp118CfgPPO(X1AMPExp117CfgPPO):
    class runner(X1AMPExp117CfgPPO.runner):
        experiment_name = 'x1_amp_exp118'
        run_name = 'exp1.18'
        max_iterations = 5000  # Additional updates: parent7000 -> final12000.
        warm_start_mode = 'policy_amp_fresh_critic'
        require_policy_transfer = False
        require_reward_transfer = True
        policy_transfer_source_curriculum = 'absolute_phase_support_v1'
        policy_transfer_source_iteration = 7000
        policy_transfer_source_task = 'x1_amp_exp117'
        policy_transfer_source_sha256 = '4edb874efe478baea76af598db86e923c24d62e1653542bf144111a453971ac8'
