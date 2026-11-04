"""X1 29-DOF AMP configurations.

X1AMPCfg / X1AMPCfgPPO are the original exp1.0 baseline and remain the root of the
exp1.x/exp2.x chain. X1AMPFinalCfg / X1AMPFinalCfgPPO are the consolidated exp2.12
values, written out explicitly with no experiment-to-experiment inheritance.
"""
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


DOF_NAMES = (["lumbar_" + x + "_joint" for x in ("yaw", "roll", "pitch")]
             + [side + "_" + x + "_joint" for side in ("left", "right")
                for x in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
                          "elbow_yaw", "wrist_pitch", "wrist_roll")]
             + [side + "_" + x + "_joint" for side in ("left", "right")
                for x in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")])
KEY_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_knee_pitch_link",
                  "right_knee_pitch_link", "left_elbow_yaw_link", "right_elbow_yaw_link"]


class X1AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 29
        num_commands = 3
        frame_stack = 5
        short_frame_stack = 5
        c_frame_stack = 1
        num_single_obs = 3 + 3 + 3 + 29 * 3  # commands, omega, gravity, q, dq, action
        num_observations = frame_stack * num_single_obs
        single_num_privileged_obs = num_single_obs + 3
        num_privileged_obs = single_num_privileged_obs
        episode_length_s = 20.

    class amp:
        motion_dir = "{LEGGED_GYM_ROOT_DIR}/resources/motions/slow_walk_amp_v1.2/normalized"
        history_steps = 3
        key_body_names = KEY_BODY_NAMES
        # Equal clip probabilities, not equal frame probabilities.
        motion_weights = {"F1_B1_stand_to_walk": 1., "F1_B2_walk_to_stand": 1.,
                          "F1_B3_continuous_walk": 1., "F1_B4_walk_backwards": 1.}
        limit_tolerance = 0.05
        reference_state_initialization = True

    class safety:
        pos_limit = 1.
        vel_limit = 1.
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf"
        name = "x1_amp"
        foot_name = "ankle_roll_link"
        knee_name = "knee_pitch_link"
        terminate_after_contacts_on = ["base_link", "lumbar", "shoulder", "elbow", "wrist", "hip", "knee"]
        penalize_contacts_on = ["base_link", "lumbar", "shoulder", "elbow", "wrist", "hip", "knee"]
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        collapse_fixed_joints = True
        armature = 0.01

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 0.8
        dynamic_friction = 0.8

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 3
        heading_command = False
        resampling_time = 6.
        class ranges:
            lin_vel_x = [-0.35, 0.65]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.3, 0.3]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.60]
        # Rounded mean of all supplied normalized frames in DOF_NAMES order.
        # The action centre must use this model's axes, including the mirrored hips.
        default_joint_angles = dict(zip(DOF_NAMES, [
            -0.0114, -0.0210, 0.3162,
            -0.0494, -0.1600, -0.0988, 0.5442, 0.1529, 0.2649, 0.2223,
            -0.0808, -0.2659, -0.2642, 0.6477, -0.0512, -0.1349, -0.0339,
            -0.0441, 0.0610, 0.0198, 0.2917, -0.1627, -0.0514,
            -0.0034, -0.0291, 0.0022, 0.3494, -0.1630, 0.0985]))

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"lumbar_yaw": 120., "lumbar_roll": 120., "lumbar_pitch": 150.,
                     "hip_pitch": 120., "hip_roll": 100., "hip_yaw": 100., "knee_pitch": 150.,
                     "ankle_pitch": 50., "ankle_roll": 50., "shoulder": 40.,
                     "elbow_pitch": 30., "elbow_yaw": 20., "wrist": 15.}
        damping = {"lumbar_yaw": 4., "lumbar_roll": 4., "lumbar_pitch": 5.,
                   "hip_pitch": 4., "hip_roll": 3.3, "hip_yaw": 3.3, "knee_pitch": 5.,
                   "ankle_pitch": 2.5, "ankle_roll": 2.5, "shoulder": 2.,
                   "elbow_pitch": 1.5, "elbow_yaw": 1., "wrist": 1.}
        action_scale = 0.25
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # policy 50 Hz, matching the reference motion fps
        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Start with a clean AMP baseline; add randomization after convergence.
        push_robots = False
        randomize_friction = False

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 5.
        clip_observations = 100.

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.95
        class scales:
            tracking_lin_vel = 1.25
            tracking_ang_vel = 1.25
            alive = 0.15
            lin_vel_z = -0.2
            ang_vel_xy = -0.1
            orientation = -1.2
            dof_vel = -2e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            torques = -1e-5
            dof_pos_limits = -1.
            collision = -1.


class X1AMPCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "AMPOnPolicyRunner"
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
    class algorithm:
        learning_rate = 3e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.
        entropy_coef = 0.005
        max_grad_norm = 1.
        desired_kl = 0.02
    class discriminator:
        hidden_dims = [512, 256]
        learning_rate = 1e-4
        grad_penalty = 10.
        weight_decay = 1e-4
        batch_size = 1024
        num_updates = 4
        replay_size = 100000
        style_reward_scale = 2.
        task_reward_weight = 0.5
    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 100
        experiment_name = "x1_amp"

# --- consolidated exp2.12 values (flat; no exp1.x/exp2.x inheritance) ---------

class X1AMPFinalCfg(LeggedRobotCfg):
    class amp:
        expert_sampler = 'native_cycle_by_replay_category_v1'
        expert_standing_probability = 0.2
        fixed_style_weight = 0.5
        ground_clearance = 0.002
        history_steps = 3
        key_body_names = ['left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_pitch_link', 'right_knee_pitch_link', 'left_elbow_yaw_link', 'right_elbow_yaw_link']
        limit_tolerance = 0.05
        mixed_native_initialization = True
        motion_dir = '{LEGGED_GYM_ROOT_DIR}/resources/motions/slow_walk_amp_v1.2/normalized'
        motion_style_velocity_sigma = 0.12
        motion_weights = {'F1_B1_stand_to_walk': 1.0, 'F1_B2_walk_to_stand': 1.0, 'F1_B3_continuous_walk': 1.0, 'F1_B4_walk_backwards': 2.0}
        native_clip = 'F1_B3_continuous_walk'
        native_correction_file = '{LEGGED_GYM_ROOT_DIR}/humanoid/amp/reference_data/exp20_native_correction.json'
        native_first_frame = 27
        native_last_frame = 90
        nominal_startup_fraction = 0.25
        policy_state_bank_sha256 = 'dd3484954a5689a5936a5b30fcfbc8687cdcf7bfdfa15ca55edcafde6be56153'
        quiet_angular_speed = 0.3
        quiet_joint_speed = 0.2
        quiet_root_speed = 0.05
        reference_cycles = [('F1_B3_continuous_walk', 27, 90), ('F1_B4_walk_backwards', 235, 305)]
        reference_state_initialization = False
        require_reference_acceptance = True
        rsi_probability = 0.
        rsi_relative_speed_tolerance = 0.5
        rsi_speed_tolerance = 0.15
        stand_buffer_seconds = 1.5
        standing_clip = 'F1_B1_stand_to_walk'
        standing_frame = 11
        standing_joint_noise = 0.03
        standing_style_multiplier = 0.2
        style_full_iteration = 1500
        style_start_iteration = 500

    class asset(LeggedRobotCfg.asset):
        armature = 0.01
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf'
        flip_visual_attachments = False
        foot_name = 'ankle_roll_link'
        knee_name = 'knee_pitch_link'
        name = 'x1_amp'
        penalize_contacts_on = ['base_link', 'lumbar', 'shoulder', 'elbow', 'wrist', 'hip', 'knee']
        replace_cylinder_with_capsule = False
        terminate_after_contacts_on = ['base_link', 'lumbar', 'shoulder', 'elbow', 'wrist', 'hip', 'knee']

    class commands(LeggedRobotCfg.commands):
        backward_probability = 0.
        curriculum = False
        heading_command = False
        native_speed = 0.5953696966
        num_commands = 3
        class ranges:
            ang_vel_yaw = [0.0, 0.0]
            lin_vel_x = [0.5953696966, 0.5953696966]
            lin_vel_y = [0.0, 0.0]

        resampling_time = 6.
        transition_seconds = 2.
        zero_probability = 0.2

    class control(LeggedRobotCfg.control):
        action_scale = 0.25
        clip_pd_targets = True
        damping = {'lumbar_yaw': 4.0, 'lumbar_roll': 4.0, 'lumbar_pitch': 5.0, 'hip_pitch': 4.0, 'hip_roll': 3.3, 'hip_yaw': 3.3, 'knee_pitch': 5.0, 'ankle_pitch': 2.5, 'ankle_roll': 2.5, 'shoulder': 2.0, 'elbow_pitch': 1.5, 'elbow_yaw': 1.0, 'wrist': 1.0}
        decimation = 10
        stiffness = {'lumbar_yaw': 120.0, 'lumbar_roll': 120.0, 'lumbar_pitch': 150.0, 'hip_pitch': 120.0, 'hip_roll': 100.0, 'hip_yaw': 100.0, 'knee_pitch': 150.0, 'ankle_pitch': 50.0, 'ankle_roll': 50.0, 'shoulder': 40.0, 'elbow_pitch': 30.0, 'elbow_yaw': 20.0, 'wrist': 15.0}

    class domain_rand(LeggedRobotCfg.domain_rand):
        contact_offset_range = [0.001, 0.02]
        friction_range = [0.65, 0.95]
        randomize_contact_offset = True
        randomize_friction = True
        restitution_range = [0.0, 0.0]

    class env(LeggedRobotCfg.env):
        append_gait_phase = True
        c_frame_stack = 1
        frame_stack = 66
        num_actions = 29
        num_commands = 3
        num_observations = 6468
        num_privileged_obs = 101
        num_single_obs = 98
        short_frame_stack = 5
        single_num_privileged_obs = 101

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {'lumbar_yaw_joint': -0.0114, 'lumbar_roll_joint': -0.021, 'lumbar_pitch_joint': 0.3162, 'left_shoulder_pitch_joint': -0.0494, 'left_shoulder_roll_joint': -0.16, 'left_shoulder_yaw_joint': -0.0988, 'left_elbow_pitch_joint': 0.5442, 'left_elbow_yaw_joint': 0.1529, 'left_wrist_pitch_joint': 0.2649, 'left_wrist_roll_joint': 0.2223, 'right_shoulder_pitch_joint': -0.0808, 'right_shoulder_roll_joint': -0.2659, 'right_shoulder_yaw_joint': -0.2642, 'right_elbow_pitch_joint': 0.6477, 'right_elbow_yaw_joint': -0.0512, 'right_wrist_pitch_joint': -0.1349, 'right_wrist_roll_joint': -0.0339, 'left_hip_pitch_joint': -0.0441, 'left_hip_roll_joint': 0.061, 'left_hip_yaw_joint': 0.0198, 'left_knee_pitch_joint': 0.2917, 'left_ankle_pitch_joint': -0.1627, 'left_ankle_roll_joint': -0.0514, 'right_hip_pitch_joint': -0.0034, 'right_hip_roll_joint': -0.0291, 'right_hip_yaw_joint': 0.0022, 'right_knee_pitch_joint': 0.3494, 'right_ankle_pitch_joint': -0.163, 'right_ankle_roll_joint': 0.0985}
        pos = [0.0, 0.0, 0.6]

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 5.

    class rewards(LeggedRobotCfg.rewards):
        backward_speed_scale = 0.2
        brake_leg_angular_scale = 0.15
        brake_leg_linear_scale = 0.08
        curriculum_end_iteration = 1500
        curriculum_initial_stand_weight = 1.
        curriculum_start_iteration = 200
        forward_rsi_schedule = ((0, 0.7), (200, 0.7), (400, 0.5), (700, 0.5), (900, 0.2), (1000, 0.0))
        forward_rsi_source_iteration = 4000
        forward_rsi_state_frames = 128
        guidance_tolerance = 0.5
        guidance_window_seconds = 2.
        motion_velocity_scale = 0.2
        non_timeout_termination_penalty = -6.
        only_positive_rewards = False
        phase_credit_negative_floor = -0.03
        phase_credit_positive_ceiling = 1.
        raw_target_max_sum = 1.
        raw_target_scale = 0.1
        reference_coupling_floor = 0.4
        reference_feet_scale = 0.025
        reference_feet_sigma = 0.06
        reference_initial_weight = 0.25
        reference_lateral_scale = 0.12
        reference_pose_scale = 0.15
        reference_pose_sigma = 0.35
        reference_upper_scale = 0.25
        reference_velocity_scale = 1.
        reference_velocity_sigma = 1.5
        reference_waist_scale = 0.12
        reference_warmup_iterations = 500
        class scales:
            action_rate = -0.01
            alive = 0.15
            ang_vel_xy = -0.1
            backward_progress = 0.
            collision = -1.
            corridor_forward = -0.3
            corridor_lateral = -0.15
            corridor_yaw = -0.1
            dof_acc = -2.5e-07
            dof_pos_limits = -1.
            dof_vel = -0.0002
            feet_scuff = 0.
            joint_margin = -0.5
            leg_lateral_deviation = -0.25
            lin_vel_z = -0.2
            motion_progress = 1.5
            net_displacement = 0.
            orientation = -1.2
            raw_target_violation = -1.
            reference_amplitude = 0.
            reference_feet = 1.
            reference_lateral = 0.75
            reference_pose = 1.5
            reference_support = 0.5
            reference_swing = 1.
            reference_upper = 0.75
            reference_velocity = 0.5
            reference_waist = 1.
            stance_slip = -0.5
            stand_ang_vel = -0.1
            stand_dof_vel = -0.9
            stand_lin_vel = -0.3
            stand_orientation = -0.2
            stand_pose = -1.2
            torques = -1e-05
            tracking_ang_vel = 1.25
            tracking_lin_vel = 1.25
            upper_deviation = -0.1
            waist_deviation = -0.5

        scuff_clearance = 0.015
        soft_dof_pos_limit = 0.95
        stand_angular_scale = 0.3
        stand_gravity_scale = 0.1
        stand_joint_speed_scale = 0.5
        stand_linear_scale = 0.1
        stand_pose_scale = 0.2
        stand_ramp_seconds = 0.5
        support_height_scale = 0.02
        support_load_fraction = 0.5
        support_load_scale = 0.25
        tracking_lateral_std = 0.1
        tracking_lin_sigma = 0.09
        tracking_max_speed = 0.65
        tracking_max_std = 0.3
        tracking_min_std = 0.08
        tracking_sigma = 0.04

    class safety:
        pos_limit = 1.
        torque_limit = 0.85
        vel_limit = 1.

    class sim(LeggedRobotCfg.sim):
        dt = 0.002
        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 2


    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        dynamic_friction = 0.8
        mesh_type = 'plane'
        static_friction = 0.8


class X1AMPFinalCfgPPO(LeggedRobotCfgPPO):
    class algorithm:
        clip_param = 0.2
        critic_only_iterations = 0
        desired_kl = 0.02
        entropy_coef = 0.001
        exploration_gain = 0.5
        freeze_features_iterations = 1000000000
        freeze_std_iterations = 1000000000
        gamma = 0.99
        kl_check = 'minibatch'
        lam = 0.98
        learning_rate = 1e-05
        max_grad_norm = 1.
        max_learning_rate = 1e-05
        min_learning_rate = 1e-05
        num_learning_epochs = 2
        num_mini_batches = 4
        policy_transfer_warmup_iterations = 200
        policy_transfer_warmup_max_lr = 1e-05
        schedule = 'adaptive'
        source_anchor_coef = 1.
        source_max_pd_rms = 0.04
        state_estimator_loss_coef = 1.
        state_estimator_obs_index = 98
        value_loss_coef = 1.

    class discriminator:
        batch_size = 1024
        conditional_experts = True
        grad_penalty = 10.
        hidden_dims = [512, 256]
        learning_rate = 1e-05
        num_updates = 1
        replay_size = 400000
        style_reward_scale = 10.
        task_reward_weight = 0.5
        weight_decay = 0.0001

    class policy(LeggedRobotCfgPPO.policy):
        critic_hidden_dims = [768, 256, 128]
        filter_size = [32, 16]
        in_channels = 66
        init_noise_std = 0.3
        kernel_size = [6, 4]
        lh_output_dim = 64
        max_noise_std = 0.5
        motion_exploration = {'joint_names': ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_knee_pitch_joint', 'right_knee_pitch_joint'], 'std_minimum': 0.25, 'speed_threshold': 0.05}
        state_estimator_hidden_dims = [256, 128, 64]
        stride_size = [3, 2]

    class runner(LeggedRobotCfgPPO.runner):
        curriculum_migration = 'exp28_to_command_scaled_phase_exp29'
        experiment_name = 'x1_amp'
        max_iterations = 1000
        max_wall_seconds = 19800
        policy_class_name = 'ActorCriticDHAMP'
        policy_transfer_diagnostic_mode = False
        policy_transfer_protected_sections = ('policy', 'control', 'env', 'asset', 'sim', 'normalization', 'noise')
        policy_transfer_require_raw_reference_match = True
        policy_transfer_runtime_env_fields = ('num_envs', 'capture_replay_state', 'enable_camera_sensors', 'camera_width', 'camera_height')
        policy_transfer_source_curriculum = 'forward_correction_low_amp_v1'
        policy_transfer_source_discriminator_updates = 7300
        policy_transfer_source_iteration = 7500
        policy_transfer_source_sha256 = 'b22a01d00919680e75064537aa1c8ba0b19a15f0d3d0d11a4b24913c50b3e540'
        policy_transfer_source_task = 'x1_amp_exp28'
        require_policy_transfer = False
        require_reward_transfer = False
        run_name = 'amp_final'
        warm_start_mode = 'policy_only'

    runner_class_name = 'Exp25AMPOnPolicyRunner'
    seed = 5
