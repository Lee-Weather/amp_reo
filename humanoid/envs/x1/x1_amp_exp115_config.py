"""exp1.15: motion-normalized forward curriculum, initialized from stable DH."""
from .x1_amp_exp114_config import X1AMPExp114Cfg, X1AMPExp114CfgPPO


class X1AMPExp115Cfg(X1AMPExp114Cfg):
    class commands(X1AMPExp114Cfg.commands):
        zero_probability = .2
        backward_probability = 0.
        class ranges(X1AMPExp114Cfg.commands.ranges):
            lin_vel_x = [.3, .3]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

    class amp(X1AMPExp114Cfg.amp):
        style_start_iteration = 500
        style_full_iteration = 1500

    class rewards(X1AMPExp114Cfg.rewards):
        reference_initial_weight = .25
        reference_warmup_iterations = 500
        guidance_window_seconds = 2.
        guidance_tolerance = .5
        class scales(X1AMPExp114Cfg.rewards.scales):
            reference_amplitude = 2.
            reference_support = 1.
            net_displacement = 2.


class X1AMPExp115CfgPPO(X1AMPExp114CfgPPO):
    class runner(X1AMPExp114CfgPPO.runner):
        experiment_name = 'x1_amp_exp115'
        run_name = 'exp1.15'
        max_iterations = 1000
        warm_start_mode = 'policy_only'
        policy_transfer_source_curriculum = 'reference_guidance_v1'
        policy_transfer_source_sha256 = '5c7df155c40ea87f2bc3a23301f193d15fb0cb511b21961d390252d6af28e1d5'
