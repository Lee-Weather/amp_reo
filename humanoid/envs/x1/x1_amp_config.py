"""29-DOF physically mirrored X1, trained with PPO and an AMP discriminator."""
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
        motion_dir = "{LEGGED_GYM_ROOT_DIR}/czy/slow_walk_amp_v1.2/normalized"
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
        file = "{LEGGED_GYM_ROOT_DIR}/czy/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf"
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
