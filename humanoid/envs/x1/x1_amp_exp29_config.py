"""Same reward/control setup; change only the start/stop phase time law."""
from .x1_amp_exp28_config import X1AMPExp28Cfg, X1AMPExp28CfgPPO


class X1AMPExp29Cfg(X1AMPExp28Cfg):
    pass


class X1AMPExp29CfgPPO(X1AMPExp28CfgPPO):
    class runner(X1AMPExp28CfgPPO.runner):
        experiment_name = 'x1_amp_exp29'
        run_name = 'exp2.9'
        max_iterations = 1000
        policy_transfer_source_curriculum = 'forward_correction_low_amp_v1'
        policy_transfer_source_iteration = 7500
        policy_transfer_source_task = 'x1_amp_exp28'
        policy_transfer_source_sha256 = 'b22a01d00919680e75064537aa1c8ba0b19a15f0d3d0d11a4b24913c50b3e540'
        policy_transfer_source_discriminator_updates = 7300
        curriculum_migration = 'exp28_to_command_scaled_phase_exp29'
