"""Bounded nominal PD and shared standing-command timing (no simulator import)."""
import torch


def bounded_pd(raw_target, position, velocity, lower, upper, kp, kd, torque_limit):
    target = raw_target.maximum(lower).minimum(upper)
    torque = (kp * (target-position) - kd * velocity).clamp(-torque_limit, torque_limit)
    return target, torque, (raw_target-target).square()


class StandingCommands:
    def __init__(self, count, device, hold_steps):
        self.hold_steps = hold_steps
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.requested = torch.zeros(count, 3, device=device)

    def reset(self, ids, standing):
        self.remaining[ids] = standing.long() * self.hold_steps

    def applied(self):
        return torch.where((self.remaining > 0)[:, None], torch.zeros_like(self.requested), self.requested)

    def advance(self, reset_mask):
        # A newly reset actor has not taken its first standing step yet.
        self.remaining.sub_(((self.remaining > 0) & ~reset_mask.bool()).long())


def sync_policy_commands(env):
    command = env.commands[:, :3] * env.commands_scale
    env.obs_history[-1][:, :3] = command
    width = env.cfg.env.num_single_obs
    env.obs_buf[:, -width:-width+3] = command
    if env.privileged_obs_buf is not None:
        env.privileged_obs_buf[:, :3] = command
    return env.obs_buf
