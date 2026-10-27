"""exp1.4: standing objectives and matched resets, warm-started from exp1.3."""
from .x1_amp_exp3_config import X1AMPExp3Cfg, X1AMPExp3CfgPPO


class X1AMPExp14Cfg(X1AMPExp3Cfg):
    class commands(X1AMPExp3Cfg.commands):
        zero_probability = .2

    class amp(X1AMPExp3Cfg.amp):
        rsi_speed_tolerance = .15
        expert_standing_probability = .2
        quiet_root_speed = .05
        quiet_angular_speed = .3
        quiet_joint_speed = .2

    class rewards(X1AMPExp3Cfg.rewards):
        stand_ramp_seconds = .5
        stand_linear_scale = .1
        stand_angular_scale = .3
        stand_joint_speed_scale = .5
        stand_pose_scale = .2
        stand_gravity_scale = .1

        class scales(X1AMPExp3Cfg.rewards.scales):
            stand_lin_vel = -.3
            stand_ang_vel = -.1
            stand_dof_vel = -.2
            stand_pose = -.3
            stand_orientation = -.2

    # Robot, PD, action/observation conventions and reference features are
    # inherited unchanged so checkpoint metadata stays strictly compatible.


class X1AMPExp14CfgPPO(X1AMPExp3CfgPPO):
    class algorithm(X1AMPExp3CfgPPO.algorithm):
        # Fresh Adam moments plus the trained policy's small std made 3e-5
        # overshoot on the first GPU update; adaptive LR can grow from here.
        learning_rate = 1e-5

    class runner(X1AMPExp3CfgPPO.runner):
        experiment_name = "x1_amp_exp14"
        run_name = "exp1.4"
        max_iterations = 1000
        save_interval = 100
