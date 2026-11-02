"""exp2.0: native B3 speed, full-body phase tracking and fresh AMP."""
from .x1_amp_exp117_config import X1AMPExp117Cfg,X1AMPExp117CfgPPO


class X1AMPExp20Cfg(X1AMPExp117Cfg):
    class commands(X1AMPExp117Cfg.commands):
        native_speed = .5953696966
        transition_seconds = 2.
        class ranges(X1AMPExp117Cfg.commands.ranges):
            lin_vel_x = [.5953696966,.5953696966]
            lin_vel_y = [0.,0.]
            ang_vel_yaw = [0.,0.]

    class amp(X1AMPExp117Cfg.amp):
        reference_state_initialization = False
        rsi_probability = 0.
        native_clip = 'F1_B3_continuous_walk'
        expert_sampler = 'native_cycle_by_replay_category_v1'
        native_first_frame = 27
        native_last_frame = 90
        require_reference_acceptance = True
        native_correction_file = '{LEGGED_GYM_ROOT_DIR}/humanoid/amp/reference_data/exp20_native_correction.json'

    class rewards(X1AMPExp117Cfg.rewards):
        reference_pose_scale = .15
        reference_lateral_scale = .12
        reference_waist_scale = .12
        reference_upper_scale = .25
        reference_velocity_scale = 1.
        reference_feet_scale = .025
        reference_warmup_iterations = 500
        class scales(X1AMPExp117Cfg.rewards.scales):
            reference_pose = 1.5
            reference_lateral = .75
            reference_waist = 1.
            reference_upper = .75
            reference_velocity = .5
            reference_feet = 1.
            reference_swing = 1.
            reference_support = .5
            stance_slip = -.5
            joint_margin = -.5
            raw_target_violation = -1.
            feet_scuff = 0.


class X1AMPExp20CfgPPO(X1AMPExp117CfgPPO):
    class discriminator(X1AMPExp117CfgPPO.discriminator):
        num_updates = 1
        learning_rate = 5e-5
        conditional_experts = True

    class runner(X1AMPExp117CfgPPO.runner):
        experiment_name = 'x1_amp_exp20'
        run_name = 'exp2.0'
        max_iterations = 3000
        warm_start_mode = 'policy_only'
        require_policy_transfer = True
        require_reward_transfer = False
        policy_transfer_source_curriculum = 'absolute_phase_support_v1'
        policy_transfer_source_iteration = 7000
        policy_transfer_source_task = 'x1_amp_exp117'
        policy_transfer_source_sha256 = '4edb874efe478baea76af598db86e923c24d62e1653542bf144111a453971ac8'
        policy_transfer_protected_sections = ('policy','control','env','asset','sim','normalization','noise')
        policy_transfer_require_raw_reference_match = True
        policy_transfer_runtime_env_fields = ('num_envs','capture_replay_state','enable_camera_sensors','camera_width','camera_height')
        policy_transfer_diagnostic_mode = False
