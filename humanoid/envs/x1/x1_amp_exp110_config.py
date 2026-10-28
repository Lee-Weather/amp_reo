"""exp1.10: restore the DH history encoder and supervised velocity estimator."""
from .x1_amp_exp19_config import X1AMPExp19Cfg, X1AMPExp19CfgPPO


class X1AMPExp110Cfg(X1AMPExp19Cfg):
    class env(X1AMPExp19Cfg.env):
        frame_stack = 66
        short_frame_stack = 5
        num_observations = frame_stack * X1AMPExp19Cfg.env.num_single_obs


class X1AMPExp110CfgPPO(X1AMPExp19CfgPPO):
    class policy(X1AMPExp19CfgPPO.policy):
        # Preserve the original DH actor, CNN and state-estimator topology.
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims = [256, 128, 64]
        in_channels = X1AMPExp110Cfg.env.frame_stack
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64

    class algorithm(X1AMPExp19CfgPPO.algorithm):
        state_estimator_loss_coef = 1.
        # _critic_obs appends scaled body linear velocity after the 96-D frame.
        state_estimator_obs_index = X1AMPExp110Cfg.env.num_single_obs

    class runner(X1AMPExp19CfgPPO.runner):
        policy_class_name = "ActorCriticDHAMP"
        experiment_name = "x1_amp_exp110"
        run_name = "exp1.10"
        resume = False
        max_iterations = 6000
        save_interval = 100
