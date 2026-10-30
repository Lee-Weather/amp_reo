"""Shared stability fixes and source policy for the two exp1.14 routes."""
from .x1_amp_exp113_config import X1AMPExp113Cfg, X1AMPExp113CfgPPO


class X1AMPExp114CommonCfg(X1AMPExp113Cfg):
    class rewards(X1AMPExp113Cfg.rewards):
        curriculum_initial_stand_weight = 1.
        reference_initial_weight = .25
        reference_warmup_iterations = 1000
        reference_pose_sigma = .35
        reference_velocity_sigma = 1.5
        reference_feet_sigma = .06

        class scales(X1AMPExp113Cfg.rewards.scales):
            stand_pose = -1.2
            raw_target_violation = -2.


class X1AMPExp114CommonCfgPPO(X1AMPExp113CfgPPO):
    class runner(X1AMPExp113CfgPPO.runner):
        policy_transfer_source_curriculum = "policy_transfer_standing_v1"
        policy_transfer_source_sha256 = "63ec2ca3aa3d6386b5a1a7928aa30e9c3053be6486a61bef0402357231fd42f0"
