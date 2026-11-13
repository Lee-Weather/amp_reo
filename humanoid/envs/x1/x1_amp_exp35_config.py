"""exp3.5: fix the buffer/episode deadlock found in exp3.4.

exp3.4 repaired the zero-command *pipeline* (``stationary.py`` skips zero
sampling on a degenerate ``lin_vel_x`` range) but the policy still regressed:
with ``resampling_time = 6 s`` and ``stand_buffer_seconds = 1.5 s``, the
1.5 s hold (75 policy steps) outlived the 71-step zero-command episodes, so the
walk command was never actually delivered and every reset replayed the same
"stand still from a dead stop" situation.

exp3.5 changes three things and nothing else:

  * ``resampling_time`` 6 s -> 1.0 s, so a command change lands *inside* a
    2-4 s episode and the policy sees "walk -> stop" transitions;
  * ``stand_buffer_seconds`` 1.5 s -> 0.4 s, so the settle hold cannot swallow
    the whole episode;
  * the standing reset pose returns to the motion-library stand frame (the
    exp3.2 baseline that can stand for 60 s), instead of the deployment
    ``pd_stand`` pose that sits outside the AMP discriminator's distribution.
    ``pd_stand`` stays as the FK-verified replay variant.
"""
from .x1_amp_exp34_config import X1AMPExp34Cfg, X1AMPExp34CfgPPO


class X1AMPExp35Cfg(X1AMPExp34Cfg):
    """Config: clip-frame standing pose, short command periods, short hold."""

    class env(X1AMPExp34Cfg.env):
        zero_exposure_fix = True
        # (iteration threshold, zero probability before, zero probability after)
        zero_curriculum = (60, 0.25, 0.35)
        # None keeps the inherited (motion-clip) standing height and pose.
        standing_pose_z = None

    class commands(X1AMPExp34Cfg.commands):
        native_speed = 0.5953696966
        zero_probability = 0.35
        resampling_time = 1.

    class amp(X1AMPExp34Cfg.amp):
        # 0.4 s settle hold; exp3.4 used 1.5 s, longer than a zero-episode.
        stand_buffer_seconds = 0.4

    class rewards(X1AMPExp34Cfg.rewards):
        standing_stability_angular_scale = -0.03
        standing_stability_angular_max = 0.15


class X1AMPExp35CfgPPO(X1AMPExp34CfgPPO):
    runner_class_name = "X1Exp35AMPOnPolicyRunner"

    class runner(X1AMPExp34CfgPPO.runner):
        experiment_name = "x1_amp_exp35"
        run_name = "exp3.5"
        max_iterations = 200
        save_interval = 50
        warm_start_mode = "all_weights"
        require_reward_transfer = False
        require_policy_transfer = False