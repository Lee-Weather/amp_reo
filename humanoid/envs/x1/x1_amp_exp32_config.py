"""exp3.2: fix zero-command exposure, bank anchor + hard rejection."""
from .x1_amp_exp31_config import X1AMPExp31Cfg, X1AMPExp31CfgPPO


class X1AMPExp32Cfg(X1AMPExp31Cfg):
    class commands(X1AMPExp31Cfg.commands):
        """Same 50% zero-command as exp3.1."""
        zero_probability = 0.5
        resampling_time = 6.

    class rewards(X1AMPExp31Cfg.rewards):
        standing_stability_angular_scale = -0.03
        standing_stability_max_angular = 0.15


class X1AMPExp32CfgPPO(X1AMPExp31CfgPPO):
    runner_class_name = "X1Exp32AMPOnPolicyRunner"

    class algorithm(X1AMPExp31CfgPPO.algorithm):
        anchor_loss_coef = 0.5

        # Standing-anchor bank: evaluate anchor on a fixed bank of zero-command
        # observations every update.  The bank is collected at warm_start time
        # by stepping the source policy through the env with vx=0.
        standing_anchor_coef = 0.8
        standing_anchor_bank_size = 256

        # Hard rejection: if the RMS action deviation on the bank exceeds this
        # threshold (in radians), the minibatch update is rolled back.
        # Set to 0 to disable (safe default for the first run).
        standing_action_rms_threshold = 0.0

    class runner(X1AMPExp31CfgPPO.runner):
        experiment_name = "x1_amp_exp32"
        run_name = "exp3.2"
        max_iterations = 500
        save_interval = 50
