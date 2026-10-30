"""Dense phase-free leg-transition feedback from unchanged real AMP histories."""
import math
import torch


class ReferenceTransition:
    def __init__(self, sampler, pose_sigma=.35, velocity_sigma=1.5, feet_sigma=.06):
        if not all(math.isfinite(x) and x > 0 for x in (pose_sigma, velocity_sigma, feet_sigma)):
            raise ValueError("Reference distance scales must be finite and positive")
        self.sampler = sampler
        lib = sampler.library
        self.pose_columns = [10+lib.dof_names.index(side+"_"+joint+"_joint")
                             for side in ("left", "right")
                             for joint in ("hip_pitch", "knee_pitch", "ankle_pitch")]
        self.velocity_columns = [column+len(lib.dof_names) for column in self.pose_columns]
        start = 10+2*len(lib.dof_names)
        self.feet_columns = [start+3*lib.key_body_names.index(side+"_ankle_roll_link")+axis
                             for side in ("left", "right") for axis in range(3)]
        self.pose_sigma, self.velocity_sigma, self.feet_sigma = pose_sigma, velocity_sigma, feet_sigma
        # Keep complete real histories; never erase expert velocities. Quiet
        # windows are not a teacher for a moving command.
        keep = ~sampler.quiet
        self.ids, self.times = sampler.ids[keep], sampler.times[keep]
        self.local_vx = sampler.local_vx[keep]
        self.histories = lib.history(self.ids, self.times)
        self.bank = self.project(self.histories)
        self.bank_squared = self.bank.square().sum(-1)

    def project(self, history, stationary=False):
        q, velocity, feet = (history[..., columns] for columns in
                             (self.pose_columns, self.velocity_columns, self.feet_columns))
        if stationary:
            q = q[:, -1:].expand_as(q)
            velocity = torch.zeros_like(velocity)
            feet = feet[:, -1:].expand_as(feet)
        groups = [q/self.pose_sigma, velocity/self.velocity_sigma, feet/self.feet_sigma]
        return torch.cat([group/math.sqrt(group.shape[-1]) for group in groups], -1).flatten(1)/math.sqrt(history.shape[1])

    def reward(self, history, commands):
        if (history.ndim != 3 or commands.ndim != 1 or len(history) != len(commands)
                or not torch.isfinite(history).all() or not torch.isfinite(commands).all()):
            raise ValueError("Reference transition feedback requires finite histories and x commands")
        moving = commands.abs() > .05
        tolerance = self.sampler.tolerance
        if self.sampler.relative_speed_tolerance is not None:
            tolerance = (.025+self.sampler.relative_speed_tolerance*commands.abs()).clamp(max=tolerance)[:, None]
        allowed = (self.local_vx[None, :]-commands[:, None]).abs() <= tolerance
        allowed &= commands[:, None]*self.local_vx[None, :] > 0
        allowed &= self.local_vx[None, :].abs() >= (.5*commands.abs()).clamp(max=.05)[:, None]
        # Zero/near-zero commands receive no guidance; avoid empty masks there.
        allowed[~moving] = True
        if not allowed.any(-1).all():
            raise ValueError("No real moving reference history covers a requested command")
        actual, stationary = self.project(history), self.project(history, stationary=True)
        queries = torch.cat((actual, stationary))
        distance = queries.square().sum(-1, keepdim=True)+self.bank_squared[None, :]-2*(queries @ self.bank.T)
        distance.clamp_(min=0)
        distance.masked_fill_(~torch.cat((allowed, allowed)), float("inf"))
        nearest = distance.min(-1).values
        score, baseline = (-.5*nearest).exp().split(len(history))
        # A held pose, or body sliding with fixed legs, has exactly zero bonus.
        # Actual body speed never gates the motion-learning signal.
        bonus = torch.where(moving, score-baseline, torch.zeros_like(score))
        return bonus, score, baseline
