"""exp3.4: deployment standing pose with the zero-command exposure pipeline fixed.

Difference from exp3.3:

  * standing pose is unchanged (pd_stand, FK-verified on the 29-DOF mirrored URDF:
    ``scripts/amp_tools/check_pose_fk.py`` -> symmetric feet, CoM 99 mm inside the
    support polygon), only the base height is corrected from 0.64 m to 0.611 m so
    the sole starts on the ground instead of 29 mm above it;
  * the zero-command pipeline is instrumented and, behind
    ``env.zero_exposure_fix``, repaired so that the 50% zero draws really reach
    ``standing_commands.requested``;
  * a short curriculum (``env.zero_curriculum``) keeps the zero share lower for the
    first iterations so the walking reward is not diluted from step one.
"""
from .x1_amp_exp33_config import X1AMPExp33Cfg, X1AMPExp33CfgPPO


class X1AMPExp34Cfg(X1AMPExp33Cfg):
    """Config: verified standing pose + repaired zero-command exposure."""

    class env(X1AMPExp33Cfg.env):
        # Phase A (diagnose) keeps this False and runs the exp3.3 pipeline with the
        # new metrics; Phase B flips it on and the 50% zero draws reach the policy.
        zero_exposure_fix = False
        # (iteration threshold, zero probability before, zero probability after)
        zero_curriculum = (100, 0.3, 0.5)
        # Forward kinematics on X1_29DOF_physically_mirrored.urdf: the lowest foot
        # vertex sits 0.611 m below base_link, so this is the height at which the
        # sole touches the ground.
        standing_pose_z = 0.611

    class commands(X1AMPExp33Cfg.commands):
        native_speed = 0.5953696966
        zero_probability = 0.5
        resampling_time = 6.

    class rewards(X1AMPExp33Cfg.rewards):
        standing_stability_angular_scale = -0.03
        standing_stability_angular_max = 0.15


class X1AMPExp34CfgPPO(X1AMPExp33CfgPPO):
    runner_class_name = "X1Exp34AMPOnPolicyRunner"

    class algorithm(X1AMPExp33CfgPPO.algorithm):
        anchor_loss_coef = 0.5
        standing_anchor_coef = 0.8
        standing_anchor_bank_size = 256
        standing_action_rms_threshold = 0.0

    class runner(X1AMPExp33CfgPPO.runner):
        experiment_name = "x1_amp_exp34"
        run_name = "exp3.4"
        max_iterations = 500
        save_interval = 50
        warm_start_mode = "all_weights"
        require_reward_transfer = False
        require_policy_transfer = False