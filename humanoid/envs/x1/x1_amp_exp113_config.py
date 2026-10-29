"""exp1.13: transfer the standing DH policy, unlock AMP and reward net progress."""
from .x1_amp_exp112_config import X1AMPExp112Cfg, X1AMPExp112CfgPPO


class X1AMPExp113Cfg(X1AMPExp112Cfg):
    class rewards(X1AMPExp112Cfg.rewards):
        motion_velocity_scale = .2

        class scales(X1AMPExp112Cfg.rewards.scales):
            backward_progress = 0.
            motion_progress = 1.5


class X1AMPExp113CfgPPO(X1AMPExp112CfgPPO):
    class algorithm(X1AMPExp112CfgPPO.algorithm):
        learning_rate = 1e-5
        # Protect the transferred policy while the new critic starts learning.
        policy_transfer_warmup_iterations = 200
        policy_transfer_warmup_max_lr = 1e-5

    class runner(X1AMPExp112CfgPPO.runner):
        experiment_name = "x1_amp_exp113"
        run_name = "exp1.13"
        max_iterations = 6000  # Additional updates; this experiment starts at 0.
        warm_start_mode = "policy_only"
        require_policy_transfer = True
        policy_transfer_source_curriculum = "scratch_standing_and_style_v1"
        policy_transfer_source_sha256 = "30c605883a48467fbe70e71d65b05033a62d665a9bc1e15e20d851071b38e0d9"
