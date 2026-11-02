"""exp2.2: successful-policy resets and explicit source constraints."""
from .x1_amp_exp21_config import X1AMPExp21Cfg,X1AMPExp21CfgPPO

class X1AMPExp22Cfg(X1AMPExp21Cfg):
    class amp(X1AMPExp21Cfg.amp):
        policy_state_bank_sha256='dd3484954a5689a5936a5b30fcfbc8687cdcf7bfdfa15ca55edcafde6be56153'

class X1AMPExp22CfgPPO(X1AMPExp21CfgPPO):
    runner_class_name='AnchoredAMPOnPolicyRunner'
    class algorithm(X1AMPExp21CfgPPO.algorithm):
        exploration_gain=.5
        source_anchor_coef=10.
        source_max_pd_rms=.04
        freeze_features_iterations=500
        freeze_std_iterations=1000000000
    class runner(X1AMPExp21CfgPPO.runner):
        experiment_name='x1_amp_exp22'
        run_name='exp2.2'
        max_iterations=2000
