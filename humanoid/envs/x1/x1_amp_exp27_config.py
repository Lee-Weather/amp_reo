"""Contact-friction robustness and lateral/yaw correction, unchanged policy/PD."""
from .x1_amp_exp26_config import X1AMPExp26Cfg,X1AMPExp26CfgPPO

class X1AMPExp27Cfg(X1AMPExp26Cfg):
    class domain_rand(X1AMPExp26Cfg.domain_rand):
        randomize_friction=True
        friction_range=[.65,.95]
        restitution_range=[0.,0.]

    class rewards(X1AMPExp26Cfg.rewards):
        class scales(X1AMPExp26Cfg.rewards.scales):
            corridor_lateral=-.15
            corridor_yaw=-.10

class X1AMPExp27CfgPPO(X1AMPExp26CfgPPO):
    class runner(X1AMPExp26CfgPPO.runner):
        experiment_name='x1_amp_exp27'
        run_name='exp2.7'
        policy_transfer_source_curriculum='dense_swing_low_amp_v1'
        policy_transfer_source_iteration=5100
        policy_transfer_source_task='x1_amp_exp26'
        policy_transfer_source_sha256='6d0b27bbeb45ece110cf3abece4475c2ea9d61c10669455a874ce587aecf9cd5'
        policy_transfer_source_discriminator_updates=4900
        curriculum_migration='exp26_to_contact_balance_exp27'
