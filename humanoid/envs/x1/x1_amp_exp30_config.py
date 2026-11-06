"""exp3.0: continue MLP 11250 on flat ground with step commands and heading.

The 480-D observation and MLP topology are unchanged. Heading error occupies
the unused vy command column; a heading correction occupies the wz command
column. The command is a step after the 1.5 s standing buffer; no speed ramp.
"""
from .x1_amp_exp19_config import X1AMPExp19Cfg, X1AMPExp19CfgPPO

SOURCE_SHA256 = "a558ff08e29c8840665353a5b30f24674562c21b6fdb6c0c47ec0f0cfcc9e451"
NATIVE_SPEED = 0.5953696966


class X1AMPExp30Cfg(X1AMPExp19Cfg):
    class commands(X1AMPExp19Cfg.commands):
        class ranges(X1AMPExp19Cfg.commands.ranges):
            lin_vel_x = [NATIVE_SPEED, NATIVE_SPEED]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]

        zero_probability = 0.2
        backward_probability = 0.0
        resampling_time = 6.

    class rewards(X1AMPExp19Cfg.rewards):
        heading_scale = -0.1
        heading_tolerance = 0.25
        heading_clip = 2.0
        heading_obs_scale = 1.0
        heading_command_gain = 0.5
        heading_command_clip = 0.3
        heading_command_threshold = 0.05


class X1AMPExp30CfgPPO(X1AMPExp19CfgPPO):
    runner_class_name = "AMPOnPolicyRunner"

    class algorithm(X1AMPExp19CfgPPO.algorithm):
        # Keep the exp1.9 adaptive range; the transfer rebuilds PPO Adam.
        learning_rate = 1e-5
        min_learning_rate = 1e-5
        max_learning_rate = 1e-4
        policy_transfer_warmup_iterations = 200
        policy_transfer_warmup_max_lr = 1e-5

    class runner(X1AMPExp19CfgPPO.runner):
        experiment_name = "x1_amp_exp30"
        run_name = "exp3.0"
        max_iterations = 1000
        save_interval = 100
        num_steps_per_env = 96
        warm_start_mode = "policy_amp_fresh_critic"
        policy_transfer_source_sha256 = SOURCE_SHA256
        policy_transfer_source_iteration = 11250
        policy_transfer_source_task = "x1_amp_exp19"
        policy_transfer_source_curriculum = None
        require_policy_transfer = False
        require_reward_transfer = True