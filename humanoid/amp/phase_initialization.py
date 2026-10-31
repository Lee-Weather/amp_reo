"""Phase-aligned forward reset states built from the retained AMP reference.

The reference clips are kinematic targets. Their measured stance foot can
slide, so contact consistency is reported instead of silently assumed. The
hard checks are finite states, joint limits, ground alignment and a plausible
root height; median stance slip is stored as a diagnostic for the preflight.
"""
import torch

from .motion import amp_features, quat_mul, quat_rotate


def canonical_heading(quaternion):
    """Remove yaw while retaining demonstrated roll and pitch."""
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    inverse = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw),
                           torch.sin(-yaw/2), torch.cos(-yaw/2)), -1)
    return quat_mul(inverse, quaternion)


class ForwardPhaseBank:
    """Uniform-phase q/dq, root state and AMP history at one forward speed."""

    def __init__(self, reference, library, clip_name, first, last, forward_speed,
                 hard_lower, hard_upper, phases=128, history_steps=3, dt=.02,
                 clearance=.002, contact_height=.01):
        if not 0 < forward_speed:
            raise ValueError("Forward phase bank requires a positive speed")
        if phases < 16 or history_steps < 2 or dt <= 0:
            raise ValueError("Invalid phase bank size or history length")
        if clearance < 0 or contact_height <= 0:
            raise ValueError("Invalid sole clearance tolerances")
        if clip_name not in library.names or reference.library is not library:
            raise ValueError("Phase bank clip must belong to the retained library")
        if library.grounding is None:
            raise ValueError("Phase bank requires the collision-foot grounding model")
        self.device, self.library, self.reference = library.device, library, reference
        self.phases, self.forward_speed = phases, forward_speed
        self.history_steps, self.dt = history_steps, dt
        self.clearance = clearance
        self.lower, self.upper = hard_lower, hard_upper

        clip = library.names.index(clip_name)
        fps = float(library.fps[clip])
        if not 0 <= first < last <= round(float(library.durations[clip])*fps):
            raise ValueError("Phase bank clip slice is outside the recorded motion")
        self.clip, self.first, self.last = clip, first, last

        grid = (torch.arange(phases, device=self.device, dtype=torch.float32)+.5)/phases
        speed = torch.full_like(grid, forward_speed)
        q, dq = reference.sample(speed, grid)
        frames = first+torch.round(grid*(last-first)).long()
        raw_rot = canonical_heading(library.frames[clip, frames, 3:7])
        root = torch.zeros(phases, 3, device=self.device)
        feet = self._foot_heights(root, raw_rot, q)
        root[:, 2] = clearance-feet.min(-1).values
        feet = feet+root[:, 2:3]
        angular = torch.zeros(phases, 3, device=self.device)
        velocity = torch.zeros(phases, 3, device=self.device)
        velocity[:, 0] = forward_speed
        states = torch.cat((root, raw_rot, velocity, angular, q, dq), -1)

        rate = float(reference.phase_rate(torch.tensor([forward_speed], device=self.device))[0])
        histories = []
        for offset in reversed(range(history_steps)):
            phase_k = (grid-offset*dt*rate).remainder(1)
            q_k, dq_k = reference.sample(speed, phase_k)
            z_k = clearance-self._foot_heights(torch.zeros_like(root), raw_rot, q_k).min(-1).values
            key = library.fk.key_positions(q_k)
            histories.append(amp_features(z_k[:, None], raw_rot, velocity, angular, q_k, dq_k, key))
        history = torch.stack(histories, 1)

        if not 0.2 < float(root[:, 2].mean()) < 1.2:
            raise ValueError("Ground-aligned phase bank root height is implausible")
        if float((q-self.lower).min()) < -1e-6 or float((self.upper-q).min()) < -1e-6:
            raise ValueError("Reference phase bank violates a joint limit")
        if not all(torch.isfinite(x).all() for x in (states, history)):
            raise ValueError("Reference phase bank contains non-finite values")

        stance = feet.argmin(-1)
        rows = torch.arange(phases, device=self.device)
        stance_v = self._foot_velocity(root, raw_rot, velocity, q, dq)[rows, stance]
        self.states, self.histories, self.phases_grid = states, history, grid
        self.report = {
            "clip": clip_name, "first_frame": int(first), "last_frame": int(last),
            "forward_speed_m_s": float(forward_speed), "phase_count": int(phases),
            "reference_period_s": float(last-first)/fps,
            "clearance_min_m": float(self._foot_heights(root, raw_rot, q).min()),
            "root_height_mean_m": float(root[:, 2].mean()),
            "root_height_min_m": float(root[:, 2].min()),
            "root_height_max_m": float(root[:, 2].max()),
            "stance_contact_fraction": float((feet[rows, stance] <= contact_height).float().mean()),
            "stance_slip_median_m_s": float(stance_v[:, :2].norm(dim=-1).median()),
            "stance_slip_p90_m_s": float(stance_v[:, :2].norm(dim=-1).quantile(.9)),
            "stance_slip_max_m_s": float(stance_v[:, :2].norm(dim=-1).max()),
            "stance_slip_exceeds_0_10_m_s": bool(float(stance_v[:, :2].norm(dim=-1).median()) > .10),
            "history_steps": int(history_steps),
        }

    def _foot_heights(self, root_position, root_quaternion, q):
        local, rotations = self.library.fk.key_transforms(q)
        heights = []
        for index, vertices in self.library.grounding.feet:
            foot_quaternion = quat_mul(root_quaternion, rotations[..., index, :])
            x, y, z, w = foot_quaternion.unbind(-1)
            row = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
            origin = root_position+quat_rotate(root_quaternion, local[..., index, :])
            heights.append((row@vertices.T).min(-1).values+origin[..., 2])
        return torch.stack(heights, -1)

    def _foot_velocity(self, root_position, root_quaternion, root_velocity, q, dq, delta=.001):
        local = self.library.fk.key_positions(q)[:, :2]
        forward = self.library.fk.key_positions(q+dq*delta)[:, :2]
        return (forward-local)/delta+root_velocity[:, None]

    def sample(self, count):
        if count <= 0:
            raise ValueError("Phase bank sample count must be positive")
        index = torch.randint(0, self.phases, (int(count),), device=self.device)
        return self.states[index].clone(), self.histories[index].clone(), self.phases_grid[index].clone()
