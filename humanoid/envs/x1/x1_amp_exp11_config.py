"""exp_1.1: bounded targets, posture priors and mixed standing/RSI resets."""
from .x1_amp_config import X1AMPCfg, X1AMPCfgPPO


class X1AMPExp11Cfg(X1AMPCfg):
    class commands(X1AMPCfg.commands):
        # Train forward/backward motion before adding lateral and yaw commands.
        class ranges(X1AMPCfg.commands.ranges):
            lin_vel_x = [-0.2, 0.65]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

    class control(X1AMPCfg.control):
        clip_pd_targets = True

    class amp(X1AMPCfg.amp):
        rsi_probability = 0.5
        standing_clip = "F1_B1_stand_to_walk"
        standing_frame = 11
        standing_joint_noise = 0.03
        stand_buffer_seconds = 1.5
        ground_clearance = 0.002

    class rewards(X1AMPCfg.rewards):
        class scales(X1AMPCfg.rewards.scales):
            waist_deviation = -0.5
            upper_deviation = -0.10
            leg_lateral_deviation = -0.25
            raw_target_violation = -0.5


class X1AMPExp11CfgPPO(X1AMPCfgPPO):
    class runner(X1AMPCfgPPO.runner):
        experiment_name = "x1_amp_exp11"
        run_name = "exp_1_1"
        max_iterations = 6000
