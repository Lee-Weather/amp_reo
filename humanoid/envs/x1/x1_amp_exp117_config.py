"""exp1.17: absolute phase credit, rebuilt support and phase-aligned forward RSI."""
from .x1_amp_exp116_config import X1AMPExp116Cfg, X1AMPExp116CfgPPO


class X1AMPExp117Cfg(X1AMPExp116Cfg):
    class rewards(X1AMPExp116Cfg.rewards):
        reference_initial_weight = .25
        reference_warmup_iterations = 300
        phase_credit_negative_floor = -.03
        phase_credit_positive_ceiling = 1.
        reference_coupling_floor = .4
        reference_pose_scale = .10
        reference_velocity_scale = .20
        reference_feet_scale = .02
        reference_lateral_scale = .10
        reference_waist_scale = .15
        support_height_scale = .02
        support_load_scale = .25
        support_load_fraction = .5
        raw_target_scale = .10
        raw_target_max_sum = 1.
        forward_rsi_source_iteration = 4000
        forward_rsi_schedule = ((0, .7), (200, .7), (400, .5), (700, .5), (900, .2), (1000, 0.))
        forward_rsi_state_frames = 128
        non_timeout_termination_penalty = -6.

        class scales(X1AMPExp116Cfg.rewards.scales):
            reference_pose = 1.
            reference_velocity = .5
            reference_feet = 1.
            reference_support = 1.
            reference_lateral = .5
            reference_waist = .3
            reference_amplitude = 0.
            net_displacement = 0.
            raw_target_violation = -.1


class X1AMPExp117CfgPPO(X1AMPExp116CfgPPO):
    class algorithm(X1AMPExp116CfgPPO.algorithm):
        # Parent4000 -> first output4100. Warmup counts new updates only.
        learning_rate = 1e-5
        min_learning_rate = 1e-5
        max_learning_rate = 3e-5
        policy_transfer_warmup_iterations = 200
        policy_transfer_warmup_max_lr = 1e-5

    class runner(X1AMPExp116CfgPPO.runner):
        experiment_name = 'x1_amp_exp117'
        run_name = 'exp1.17'
        max_iterations = 3000  # Additional updates: parent4000 -> final7000.
        warm_start_mode = 'policy_amp_fresh_critic'
        require_policy_transfer = False
        require_reward_transfer = True
        policy_transfer_source_curriculum = 'dense_phase_refinement_v1'
        policy_transfer_source_iteration = 4000
        policy_transfer_source_task = 'x1_amp_exp116'
        policy_transfer_source_sha256 = 'fa382e21547cd33fdc30904d7a1f978e50503541bd8532c999b86d72e3b97e84'
