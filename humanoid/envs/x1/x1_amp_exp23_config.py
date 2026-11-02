"""exp2.3:200-update controlled continuation with a weaker source anchor."""
from .x1_amp_exp22_config import X1AMPExp22Cfg,X1AMPExp22CfgPPO

class X1AMPExp23Cfg(X1AMPExp22Cfg):
    pass

class X1AMPExp23CfgPPO(X1AMPExp22CfgPPO):
    runner_class_name='Exp23AMPOnPolicyRunner'
    class algorithm(X1AMPExp22CfgPPO.algorithm):
        source_anchor_coef=1.
        # Retained critic/Adam: continue Actor immediately, no fresh preheat.
        critic_only_iterations=0
        freeze_features_iterations=1000000000
    class runner(X1AMPExp22CfgPPO.runner):
        experiment_name='x1_amp_exp23'
        run_name='exp2.3'
        max_iterations=200
        # The explicit continuation runner validates its preserved-state mode.
        require_policy_transfer=False
        policy_transfer_source_curriculum='anchored_policy_state_native_v1'
        policy_transfer_source_iteration=950
        policy_transfer_source_task='x1_amp_exp22'
        policy_transfer_source_sha256='b284d5558826b781de1b1baa263b7e472207cf4e59dd94b74a97f49dabe89b51'
