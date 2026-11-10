"""exp3.3: standard standing pose for zero-command, same anchor/hard-rejection as exp3.2.

Key difference from exp3.2:
  - default_joint_angles is set to the robot's deployment standing pose (pd_stand)
  - standing_state is overridden to the pd_stand upright pose instead of the
    lean-forward "stand_to_walk" motion-clip frame.
  - The overall anchor and hard-rejection machinery from exp3.2 is retained.
"""
from .x1_amp_exp32_config import X1AMPExp32Cfg, X1AMPExp32CfgPPO


class X1AMPExp33Cfg(X1AMPExp32Cfg):
    """Config: deployment standard-upright standing for zero-command."""

    class commands(X1AMPExp32Cfg.commands):
        """Same 50% zero-command as prior experiments."""
        native_speed = 0.5953696966
        zero_probability = 0.5
        resampling_time = 6.

    class rewards(X1AMPExp32Cfg.rewards):
        standing_stability_angular_scale = -0.03
        standing_stability_angular_max = 0.15

    class init_state(X1AMPExp32Cfg.init_state):
        """default_joint_angles set to the deployment pd_stand upright pose.

        DOF order (from DOF_NAMES):
          [0:3]   lumbar yaw/roll/pitch
          [3:10]  left shoulder pitch/roll/yaw, elbow pitch/yaw, wrist pitch/roll
          [10:17] right shoulder pitch/roll/yaw, elbow pitch/yaw, wrist pitch/roll
          [17:23] left hip pitch/roll/yaw, knee pitch, ankle pitch/roll
          [23:29] right hip pitch/roll/yaw, knee pitch, ankle pitch/roll
        """
        pos = [0., 0., 0.64]
        default_joint_angles = dict(zip(
            ["lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
             "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
             "left_elbow_yaw_joint", "left_wrist_pitch_joint",
             "left_wrist_roll_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
             "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
             "right_elbow_yaw_joint", "right_wrist_pitch_joint",
             "right_wrist_roll_joint",
             "left_hip_pitch_joint", "left_hip_roll_joint",
             "left_hip_yaw_joint", "left_knee_pitch_joint",
             "left_ankle_pitch_joint", "left_ankle_roll_joint",
             "right_hip_pitch_joint", "right_hip_roll_joint",
             "right_hip_yaw_joint", "right_knee_pitch_joint",
             "right_ankle_pitch_joint", "right_ankle_roll_joint"],
            [
                # lumbar – neutral (0 = straight spine)
                0.00, 0.00, 0.00,
                # left arm – deployment pd_stand
                0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
                # right arm – deployment pd_stand
                0.15, -0.10, 0.00, 0.30, 0.00, 0.00, 0.00,
                # left leg – deployment pd_stand
                0.40, 0.05, -0.31, 0.49, -0.21, 0.00,
                # right leg – deployment pd_stand
                -0.40, -0.05, 0.31, 0.49, -0.21, 0.00,
            ]
        ))


class X1AMPExp33CfgPPO(X1AMPExp32CfgPPO):
    runner_class_name = "X1Exp33AMPOnPolicyRunner"

    class algorithm(X1AMPExp32CfgPPO.algorithm):
        anchor_loss_coef = 0.5
        standing_anchor_coef = 0.8
        standing_anchor_bank_size = 256
        standing_action_rms_threshold = 0.0

    class runner(X1AMPExp32CfgPPO.runner):
        experiment_name = "x1_amp_exp33"
        run_name = "exp3.3"
        max_iterations = 500
        save_interval = 50
        warm_start_mode = "all_weights"
