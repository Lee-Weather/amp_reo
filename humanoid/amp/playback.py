"""Deterministic playback commands for the AMP observation layout."""
import math

PLAY_SPEEDS = (0., 0.4, 0.6, 0., 0.4, 0.)
PLAY_STAGE_SECONDS = 10.


def speed_profile(dt, seconds=PLAY_STAGE_SECONDS):
    if not math.isfinite(dt) or not math.isfinite(seconds) or dt <= 0 or seconds <= 0:
        raise ValueError("Playback dt and stage duration must be positive and finite")
    return [(max(1, round(seconds / dt)), speed) for speed in PLAY_SPEEDS]


def command_at(step, profile):
    end = 0
    for stage, (steps, velocity) in enumerate(profile):
        end += steps
        if step < end:
            return stage, velocity
    return len(profile) - 1, 0.


def inject_command(env, vx, vy=0., wz=0.):
    """Update only the newest history frame, before the policy observes it."""
    if hasattr(env, "set_evaluation_command"):
        return env.set_evaluation_command(vx, vy, wz)
    for index, (name, value) in enumerate(zip(("lin_vel_x", "lin_vel_y", "ang_vel_yaw"), (vx, vy, wz))):
        env.command_ranges[name] = [value, value]
        env.commands[:, index] = value
    command = env.commands[:, :3] * env.commands_scale
    env.obs_history[-1][:, :3] = command
    width = env.cfg.env.num_single_obs
    env.obs_buf[:, -width:-width + 3] = command
    return env.obs_buf
