"""Command-matched reference sampling, without simulator dependencies."""
import math

import torch

from .motion import quat_inverse_rotate


def sample_commands(ranges, count, device, zero_probability):
    if not 0 <= zero_probability <= 1:
        raise ValueError("zero_probability must be in [0,1]")
    result = torch.empty(count, 3, device=device)
    for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
        lo, hi = ranges[name]
        result[:, i] = lo+(hi-lo)*torch.rand(count, device=device)
    # A fixed evaluation command must never be randomly replaced with zero.
    if ranges["lin_vel_x"][0] != ranges["lin_vel_x"][1]:
        result[torch.rand(count, device=device) < zero_probability] = 0
    return result


class CommandMotionSampler:
    def __init__(self, library, speed_tolerance=.15, expert_standing_probability=.2,
                 max_root_speed=.05, max_angular_speed=.3, max_joint_speed=.2,
                 exclude_quiet_for_moving=False, relative_speed_tolerance=None):
        if speed_tolerance <= 0 or not 0 <= expert_standing_probability <= 1:
            raise ValueError("Invalid reference sampling settings")
        self.library = library
        self.tolerance = speed_tolerance
        self.expert_standing_probability = expert_standing_probability
        if relative_speed_tolerance is not None and (not math.isfinite(relative_speed_tolerance) or relative_speed_tolerance <= 0):
            raise ValueError("Relative reference speed tolerance must be positive")
        self.exclude_quiet_for_moving = exclude_quiet_for_moving
        self.relative_speed_tolerance = relative_speed_tolerance
        ids, times = [], []
        minimum = (library.history_steps-1)*library.dt
        for i in range(len(library.names)):
            fps = float(library.fps[i])
            first = math.ceil(minimum*fps-1e-6)
            last = round(float(library.durations[i])*fps)
            time = torch.arange(first, last+1, device=library.device)/fps
            times.append(time)
            ids.append(torch.full((len(time),), i, device=library.device, dtype=torch.long))
        self.ids, self.times = torch.cat(ids), torch.cat(times)
        offsets = torch.arange(library.history_steps-1, -1, -1, device=library.device)*library.dt
        states = library.state(self.ids[:, None].expand(-1, library.history_steps), self.times[:, None]-offsets)
        root_speed = states[..., 7:10].norm(dim=-1)
        angular_speed = states[..., 10:13].norm(dim=-1)
        joint_speed = states[..., 42:].square().mean(-1).sqrt()
        # Every frame of the real history must be quiet, including the oldest.
        self.quiet = ((root_speed <= max_root_speed) & (angular_speed <= max_angular_speed)
                      & (joint_speed <= max_joint_speed)).all(-1)
        if not self.quiet.any():
            raise ValueError("No complete low-motion reference windows satisfy the thresholds")
        self.local_vx = quat_inverse_rotate(states[:, -1, 3:7], states[:, -1, 7:10])[:, 0]
        lengths = torch.bincount(self.ids, minlength=len(library.names)).clamp(min=1)
        self.prior = library.weights[self.ids]/lengths[self.ids]
        self.quiet_weights = self.prior*self.quiet

    def sample_matching(self, commands):
        selections = []
        for command in commands.split(512):
            vx = command[:, 0:1]
            tolerance = self.tolerance if self.relative_speed_tolerance is None else (.025+self.relative_speed_tolerance*vx.abs()).clamp(max=self.tolerance)
            allowed = (self.local_vx[None, :]-vx).abs() <= tolerance
            directional = (vx.abs() <= .05) | (vx*self.local_vx[None, :] > 0)
            allowed &= directional
            if self.exclude_quiet_for_moving:
                moving = vx.abs() > .05
                enough_speed = self.local_vx[None,:].abs() >= (.5*vx.abs()).clamp(max=.05)
                allowed &= ~moving | (~self.quiet[None,:] & enough_speed)
            zero = command.abs().amax(-1) <= 1e-6
            allowed[zero] = self.quiet
            weights = allowed*self.prior[None, :]
            if (weights.sum(-1) <= 0).any():
                raise ValueError("No reference state matches a requested command")
            selections.append(torch.multinomial(weights, 1).squeeze(-1))
        if not selections:
            return self.ids[:0], self.times[:0]
        selected = torch.cat(selections)
        return self.ids[selected], self.times[selected]

    def sample_expert(self, count):
        lib = self.library
        ids, times = lib.sample_ids_times(count)
        quiet = torch.rand(count, device=lib.device) < self.expert_standing_probability
        if quiet.any():
            selected = torch.multinomial(self.quiet_weights, int(quiet.sum()), replacement=True)
            ids[quiet], times[quiet] = self.ids[selected], self.times[selected]
        # Preserve real q/dq, root velocity and oldest-first reference history.
        return lib.history(ids, times)


def sample_directional_commands(ranges, count, device, zero_probability=.2, backward_probability=.3):
    """Event probabilities, preserving every speed and fixed evaluation commands."""
    if not 0 <= zero_probability < 1 or not 0 <= backward_probability <= 1-zero_probability:
        raise ValueError("Invalid command event probabilities")
    lo, hi = ranges["lin_vel_x"]
    if lo == hi:
        return sample_commands(ranges,count,device,zero_probability)
    if not lo < 0 < hi or tuple(ranges["lin_vel_y"]) != (0.,0.) or tuple(ranges["ang_vel_yaw"]) != (0.,0.):
        raise ValueError("Directional command sampler requires x-only signed ranges")
    result = torch.zeros(count,3,device=device)
    choice = torch.rand(count,device=device)
    negative = (choice >= zero_probability) & (choice < zero_probability+backward_probability)
    positive = choice >= zero_probability+backward_probability
    result[negative,0] = lo+(-lo)*torch.rand(int(negative.sum()),device=device)
    result[positive,0] = hi*torch.rand(int(positive.sum()),device=device)
    return result
