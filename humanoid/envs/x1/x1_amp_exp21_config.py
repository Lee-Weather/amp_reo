"""exp2.1 validation: preserve inference weights, expose native gait, gate difficulty."""
from .x1_amp_exp20_config import X1AMPExp20Cfg,X1AMPExp20CfgPPO


class X1AMPExp21Cfg(X1AMPExp20Cfg):
    class rewards(X1AMPExp20Cfg.rewards):
        class scales(X1AMPExp20Cfg.rewards.scales):
            stance_slip=-.5
    class amp(X1AMPExp20Cfg.amp):
        mixed_native_initialization=True


class X1AMPExp21CfgPPO(X1AMPExp20CfgPPO):
    runner_class_name='ConservativeAMPOnPolicyRunner'
    class algorithm(X1AMPExp20CfgPPO.algorithm):
        learning_rate=1e-5
        min_learning_rate=1e-5
        max_learning_rate=1e-5
        critic_only_iterations=50
        freeze_std_iterations=200
    class discriminator(X1AMPExp20CfgPPO.discriminator):
        learning_rate=1e-5
    class runner(X1AMPExp20CfgPPO.runner):
        experiment_name='x1_amp_exp21'
        run_name='exp2.1'
        max_iterations=200
        save_interval=50
