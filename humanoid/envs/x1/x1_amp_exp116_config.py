"""exp1.16: repair phase/clearance feedback while retaining exp1.15t skills."""
from .x1_amp_exp115_config import X1AMPExp115Cfg, X1AMPExp115CfgPPO


class X1AMPExp116Cfg(X1AMPExp115Cfg):
    class amp(X1AMPExp115Cfg.amp):
        fixed_style_weight = .5

    class rewards(X1AMPExp115Cfg.rewards):
        reference_initial_weight = 1.
        class scales(X1AMPExp115Cfg.rewards.scales):
            reference_amplitude = 0.
            net_displacement = 0.


class X1AMPExp116CfgPPO(X1AMPExp115CfgPPO):
    class algorithm(X1AMPExp115CfgPPO.algorithm):
        policy_transfer_warmup_iterations = 100  # Additional updates from parent1000.
        policy_transfer_warmup_max_lr = 1e-5

    class runner(X1AMPExp115CfgPPO.runner):
        experiment_name = 'x1_amp_exp116'
        run_name = 'exp1.16'
        max_iterations = 3000  # Additional updates: parent1000 -> final4000.
        warm_start_mode = 'policy_amp_fresh_critic'
        require_policy_transfer = False
        require_reward_transfer = True
        policy_transfer_source_curriculum = 'normalized_forward_guidance_v1'
        policy_transfer_source_iteration = 1000
        policy_transfer_source_task = 'x1_amp_exp115t'
        policy_transfer_source_sha256 = '7a720d52ee138a4a319100fef684a2719ee6ccb616690f7053892a6212527571'
