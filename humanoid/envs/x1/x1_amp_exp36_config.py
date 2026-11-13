"""exp3.6: teach standing as *braking out of walking*, not cold-start balance.

exp3.3/3.4/3.5 all show the same signature: the base exp3.2 policy can hold a
zero command for 60 s when it enters standing *from walking*, but every reset
that starts from a dead-stop pose ends in a 1.4-1.9 s backward topple, and the
fine-tune that keeps resetting from a static pose drags the walking gait down
with it (0.4 m/s command -> -0.42 m/s body velocity in exp3.5).

exp3.6 keeps exp3.5's scheduling fixes and changes the standing reset
distribution instead of the zero-command share:

  * standing resets now start from a *motion-library walking state* (pose,
    velocity and gait phase drawn from the RSI bank) with a short zero-command
    hold, so the robot practises decelerating into a stand rather than holding a
    static pose the AMP discriminator does not contain;
  * the rest (clip-frame standing pose for the nominal cohort, 0.4 s hold,
    1 s command period, zero curriculum 0.25 -> 0.35) is inherited from exp3.5.
"""
from .x1_amp_exp35_config import X1AMPExp35Cfg, X1AMPExp35CfgPPO


class X1AMPExp36Cfg(X1AMPExp35Cfg):
    """Config: standing resets start in motion."""

    class env(X1AMPExp35Cfg.env):
        zero_exposure_fix = True
        zero_curriculum = (60, 0.25, 0.35)
        standing_pose_z = None
        # 1.0 = every standing reset starts from a walking state.
        moving_stand_fraction = 1.0

    class commands(X1AMPExp35Cfg.commands):
        zero_probability = 0.35
        # Long enough for the braking manoeuvre to finish before the next draw.
        resampling_time = 2.

    class amp(X1AMPExp35Cfg.amp):
        stand_buffer_seconds = 0.4


class X1AMPExp36CfgPPO(X1AMPExp35CfgPPO):
    runner_class_name = "X1Exp36AMPOnPolicyRunner"

    class runner(X1AMPExp35CfgPPO.runner):
        experiment_name = "x1_amp_exp36"
        run_name = "exp3.6"
        max_iterations = 200
        save_interval = 50