"""exp1.14: preserve DH and add a phase-conditioned reference teacher plus AMP."""
from .x1_amp_exp114_common_config import X1AMPExp114CommonCfg, X1AMPExp114CommonCfgPPO


class X1AMPExp114Cfg(X1AMPExp114CommonCfg):
    class env(X1AMPExp114CommonCfg.env):
        append_gait_phase = True
        num_single_obs = 98
        num_observations = X1AMPExp114CommonCfg.env.frame_stack*num_single_obs
        single_num_privileged_obs = num_single_obs+3
        num_privileged_obs = single_num_privileged_obs

    class amp(X1AMPExp114CommonCfg.amp):
        reference_cycles = [("F1_B3_continuous_walk", 27, 90), ("F1_B4_walk_backwards", 235, 305)]

    class rewards(X1AMPExp114CommonCfg.rewards):
        class scales(X1AMPExp114CommonCfg.rewards.scales):
            reference_pose = 2.
            reference_velocity = 1.
            reference_feet = 1.


class X1AMPExp114CfgPPO(X1AMPExp114CommonCfgPPO):
    class algorithm(X1AMPExp114CommonCfgPPO.algorithm):
        state_estimator_obs_index = X1AMPExp114Cfg.env.num_single_obs

    class runner(X1AMPExp114CommonCfgPPO.runner):
        experiment_name = "x1_amp_exp114"
        run_name = "exp1.14"
        warm_start_mode = "policy_append_phase"
