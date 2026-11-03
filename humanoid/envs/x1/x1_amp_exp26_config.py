"""Dense swing only, 2000 further updates from audited exp2.5/model3100."""
from .x1_amp_exp25_config import X1AMPExp25Cfg,X1AMPExp25CfgPPO

class X1AMPExp26Cfg(X1AMPExp25Cfg):
    pass

class X1AMPExp26CfgPPO(X1AMPExp25CfgPPO):
    class runner(X1AMPExp25CfgPPO.runner):
        experiment_name='x1_amp_exp26'
        run_name='exp2.6'
        policy_transfer_source_curriculum='pooled100_low_amp_v1'
        policy_transfer_source_iteration=3100
        policy_transfer_source_task='x1_amp_exp25'
        policy_transfer_source_sha256='f7e542783590fb56ad3c9f10e94ae6993abc890e9b9b345ae7ba0d0af0d8d440'
        policy_transfer_source_discriminator_updates=2900
        curriculum_migration='exp25_to_dense_swing_exp26'
