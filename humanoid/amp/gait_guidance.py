"""Motion-normalized guidance with explicit stationary counterexamples."""
import torch


def normalized_score(actual, target, scale, tolerance=.5):
    return (-.5*((actual-target)/(scale*tolerance)).square().mean(-1)).exp()


def span_improvement(actual, target, tolerance=.5):
    """Rest is exactly zero; excessive oscillation is not clipped into credit."""
    score = normalized_score(actual, target, target, tolerance)
    baseline = normalized_score(torch.zeros_like(actual), target, target, tolerance)
    return (score-baseline)/(1-baseline).clamp(min=1e-6)


class GaitWindow:
    """Bounded history, cleared on reset/command changes; no cross-episode credit."""
    def __init__(self, num_envs, steps, device):
        if steps < 2:
            raise ValueError("A gait window needs at least two samples")
        self.data = torch.zeros(steps, num_envs, 12, device=device)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.previous_command = torch.full((num_envs,), float('nan'), device=device)
        self.cursor = 0

    def reset(self, ids):
        self.count[ids] = 0
        self.previous_command[ids] = float('nan')

    def update(self, q, feet_x, root_xy, heading_xy, command):
        changed = (~torch.isfinite(self.previous_command)
                   | ((command-self.previous_command).abs() > 1e-6))
        self.count[changed] = 0
        self.previous_command.copy_(command)
        row = torch.cat((q, feet_x, root_xy, heading_xy), -1)
        self.data[self.cursor].copy_(row)
        self.count.add_(1).clamp_(max=len(self.data))
        oldest = self.data[(self.cursor+1) % len(self.data)]
        ready = self.count == len(self.data)
        spans = self.data[:, :, :8].amax(0)-self.data[:, :, :8].amin(0)
        displacement = ((root_xy-oldest[:, 8:10])*oldest[:, 10:12]).sum(-1)
        self.cursor = (self.cursor+1) % len(self.data)
        return spans, displacement, ready


class MotionScales:
    """Precompute real scaled-cycle statistics once, outside the physics loop."""
    def __init__(self, reference, fk, pitch_indices, speeds, phases=256):
        self.speeds = torch.tensor(speeds, device=reference.standing_q.device)
        tables = []
        for speed in speeds:
            phase = torch.arange(phases, device=self.speeds.device)/phases
            command = torch.full_like(phase, speed)
            q, dq = reference.sample(command, phase)
            feet = fk.key_positions(q)
            q = q[:, pitch_indices]
            dq = dq[:, pitch_indices]
            tables.append(dict(q=q.std(0, unbiased=False).clamp(min=.015),
                               dq=dq.square().mean(0).sqrt().clamp(min=.05),
                               feet=feet.flatten(-2).std(0, unbiased=False).clamp(min=.008),
                               span=torch.cat((q.amax(0)-q.amin(0),
                                               feet[:, :, 0].amax(0)-feet[:, :, 0].amin(0))).clamp(min=.02),
                               feet_min_z=feet[:, :, 2].amin(0)))
        self.tables = {key: torch.stack([t[key] for t in tables]) for key in tables[0]}

    def sample(self, command):
        ids = (command[:, None]-self.speeds[None]).abs().argmin(-1)
        return {key: value[ids] for key, value in self.tables.items()}
