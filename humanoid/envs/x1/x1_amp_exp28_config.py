"""Forward velocity correction on top of the exp2.7 continuation."""
from .x1_amp_exp27_config import X1AMPExp27Cfg, X1AMPExp27CfgPPO


class X1AMPExp28Cfg(X1AMPExp27Cfg):
    class rewards(X1AMPExp27Cfg.rewards):
        class scales(X1AMPExp27Cfg.rewards.scales):
            corridor_forward = -.30


class X1AMPExp28CfgPPO(X1AMPExp27CfgPPO):
    class runner(X1AMPExp27CfgPPO.runner):
        experiment_name = 'x1_amp_exp28'
        run_name = 'exp2.8'
        policy_transfer_source_curriculum = 'contact_balance_low_amp_v1'
        policy_transfer_source_iteration = 6300
        policy_transfer_source_task = 'x1_amp_exp27'
        policy_transfer_source_sha256 = '55c804b0e747455dd9d69ff2fa0f8188006d6e7ba9828d36b0fe3fee9f1dc6e8'
        policy_transfer_source_discriminator_updates = 6100
        curriculum_migration = 'exp27_to_forward_correction_exp28'
