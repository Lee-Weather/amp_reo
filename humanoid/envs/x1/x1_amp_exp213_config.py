"""exp2.13: heading-hold, deterministic startup and symmetric gait refinement.

This experiment keeps the exp2.12 network/observation dimensions (6468/101) and
reuses the DH topology. Heading feedback is injected into the otherwise unused
vy command column of the 98-D policy frame, so the checkpoint is warm-start
compatible with exp2.12/model_14500. The reward targets still use vy=0.
"""
from .x1_amp_config import X1AMPFinalCfg, X1AMPFinalCfgPPO

SOURCE_SHA256 = "5a9b7cfb5a1a007f19eb22ac55e656852245e2b61db3eb0bf9ac9a2c0620a7a2"


class X1AMPExp213Cfg(X1AMPFinalCfg):
    class amp(X1AMPFinalCfg.amp):
        # Half of the environments reproduce the deterministic deployment start.
        nominal_startup_fraction = 0.5

    class rewards(X1AMPFinalCfg.rewards):
        # Restore explicit reference/limits to a usable but still conservative
        # level (exp2.2 had reduced both to 0.1 for stability).
        guide_weight_override = 0.3
        raw_weight_override = 0.3
        # Heading feedback: root yaw relative to the reset pose, in radians.
        heading_scale = -0.5
        heading_tolerance = 0.15
        heading_clip = 0.8
        # Penalise a reference-swing foot that stays below 5 mm.
        swing_clearance_penalty = -0.25
        swing_clearance_threshold = 0.005
        # Discourage a persistent one-sided load distribution.
        contact_balance_scale = -0.5
        contact_balance_decay = 0.99
        # Stronger straight-line correction than exp2.12.
        class scales(X1AMPFinalCfg.rewards.scales):
            corridor_lateral = -0.3
            corridor_yaw = -0.4


class X1AMPExp213CfgPPO(X1AMPFinalCfgPPO):
    runner_class_name = "AMPOnPolicyRunner"

    class algorithm(X1AMPFinalCfgPPO.algorithm):
        learning_rate = 3e-5
        min_learning_rate = 1e-5
        max_learning_rate = 1e-4
        policy_transfer_warmup_iterations = 200
        policy_transfer_warmup_max_lr = 3e-5
        freeze_features_iterations = 0
        freeze_std_iterations = 0
        source_anchor_coef = 0.0
        source_max_pd_rms = 0.06

    class runner(X1AMPFinalCfgPPO.runner):
        # Base runner: warm start is an explicit policy-only transfer with a
        # fresh critic/D/AMP state, so the old anchor cannot freeze the policy.
        experiment_name = "x1_amp_exp213"
        run_name = "exp2.13"
        max_iterations = 1000
        save_interval = 100
        warm_start_mode = "policy_only"
        policy_transfer_source_sha256 = SOURCE_SHA256
        policy_transfer_source_iteration = 14500
        policy_transfer_source_task = "x1_amp_exp212"
        policy_transfer_source_curriculum = "nominal_startup_coverage_v1"
        policy_transfer_require_raw_reference_match = True
        require_policy_transfer = False
        require_reward_transfer = False