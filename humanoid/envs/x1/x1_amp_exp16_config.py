"""exp1.6: align moving AMP with requested speed and explore hip/knee motion."""
from .x1_amp_exp15_config import X1AMPExp15Cfg, X1AMPExp15CfgPPO


class X1AMPExp16Cfg(X1AMPExp15Cfg):
    class amp(X1AMPExp15Cfg.amp):
        motion_style_velocity_sigma = .12


class X1AMPExp16CfgPPO(X1AMPExp15CfgPPO):
    class policy(X1AMPExp15CfgPPO.policy):
        motion_exploration = {
            "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint",
                            "left_knee_pitch_joint", "right_knee_pitch_joint"],
            "std_minimum": .25,
            "speed_threshold": .05,
        }

    class runner(X1AMPExp15CfgPPO.runner):
        experiment_name = "x1_amp_exp16"
        run_name = "exp1.6"
        max_iterations = 1000
