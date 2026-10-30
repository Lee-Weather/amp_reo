"""Periodic, speed-scaled guidance derived from trusted full-body AMP clips.

The AMP expert library is unchanged. These closed cycles are optional reward
targets, not feed-forward actions or a filter on the policy output.
"""
import torch

from .motion import quat_inverse_rotate


class PhaseReference:
    def __init__(self, library, standing_q, segments):
        if len(segments) != 2 or standing_q.shape != (len(library.dof_names),):
            raise ValueError("Provide forward/backward cycle slices and one standing pose")
        self.library = library
        self.standing_q = standing_q.clone()
        self.pitch_indices = [library.dof_names.index(side+"_"+joint+"_joint")
                              for side in ("left", "right")
                              for joint in ("hip_pitch", "knee_pitch", "ankle_pitch")]
        self.cycles, self.velocities, self.periods, self.nominal_speeds = [], [], [], []
        self.metadata = []
        for direction, (name, first, last) in enumerate(segments):
            clip = library.names.index(name)
            fps = float(library.fps[clip])
            limit = round(float(library.durations[clip])*fps)
            if not 0 <= first < last <= limit or last-first < 20:
                raise ValueError("A reference slice must contain one complete measured gait cycle")
            frames = library.frames[clip, first:last+1]
            count, n = len(frames), len(library.dof_names)
            q, velocity = frames[:, 13:13+n], frames[:, 13+n:]
            period = (last-first)/fps
            u = torch.linspace(0, 1, count, device=frames.device)[:, None]
            # Cubic endpoint correction closes both position and velocity.
            # Interpolation below then has a continuous first derivative at wrap.
            center_q, center_v = (q[0]+q[-1])/2, (velocity[0]+velocity[-1])/2
            a, b = center_q-q[0], center_q-q[-1]
            c, d = period*(center_v-velocity[0]), period*(center_v-velocity[-1])
            correction = (2*u**3-3*u**2+1)*a + (-2*u**3+3*u**2)*b
            correction += (u**3-2*u**2+u)*c + (u**3-u**2)*d
            derivative = (6*u**2-6*u)*a + (-6*u**2+6*u)*b
            derivative += (3*u**2-4*u+1)*c + (3*u**2-2*u)*d
            self.cycles.append(q+correction)
            self.velocities.append(velocity+derivative/period)
            self.periods.append(period)
            signed_speed = float(quat_inverse_rotate(frames[:-1, 3:7], frames[:-1, 7:10])[:, 0].mean())
            if signed_speed*(1 if direction == 0 else -1) <= .05:
                raise ValueError("Reference cycles must have the requested forward/backward direction")
            self.nominal_speeds.append(abs(signed_speed))
            self.metadata.append({"clip": name, "first_frame": first, "last_frame": last,
                                  "period_s": period, "nominal_vx": signed_speed,
                                  "maximum_closure_correction_rad": float(correction.abs().max())})
        self.periods = standing_q.new_tensor(self.periods)
        self.nominal_speeds = standing_q.new_tensor(self.nominal_speeds)

    def _scales(self, command):
        direction = (command < 0).long()
        ratio = command.abs()/self.nominal_speeds[direction]
        amplitude = ratio.clamp(max=1).sqrt()
        speed = torch.where(amplitude > 0, ratio/amplitude.clamp(min=1e-8), torch.zeros_like(ratio))
        return direction, amplitude, speed

    def phase_rate(self, command):
        direction, _, speed = self._scales(command)
        return speed/self.periods[direction]

    def sample(self, command, phase):
        if (command.ndim != 1 or command.shape != phase.shape
                or not torch.isfinite(command).all() or not torch.isfinite(phase).all()):
            raise ValueError("Reference command and phase must be finite arrays of equal shape")
        direction, amplitude, speed = self._scales(command)
        q = self.standing_q.expand(command.numel(), -1).clone()
        velocity = torch.zeros_like(q)
        for index in (0, 1):
            mask = direction == index
            if not mask.any():
                continue
            cycle, tangent = self.cycles[index], self.velocities[index]
            segments = len(cycle)-1
            t = phase[mask].remainder(1)*segments
            first = t.long().clamp(max=segments-1)
            u = (t-first).unsqueeze(-1)
            dt = self.periods[index]/segments
            p0, p1 = cycle[first], cycle[first+1]
            v0, v1 = tangent[first], tangent[first+1]
            position = (2*u**3-3*u**2+1)*p0 + (-2*u**3+3*u**2)*p1
            position += (u**3-2*u**2+u)*dt*v0 + (u**3-u**2)*dt*v1
            rate = ((6*u**2-6*u)*p0 + (-6*u**2+6*u)*p1)/dt
            rate += (3*u**2-4*u+1)*v0 + (3*u**2-2*u)*v1
            q[mask] = self.standing_q+amplitude[mask, None]*(position-self.standing_q)
            velocity[mask] = amplitude[mask, None]*speed[mask, None]*rate
        lower, upper = self.library.fk.lower, self.library.fk.upper
        inside = (q >= lower) & (q <= upper)
        return q.maximum(lower).minimum(upper), torch.where(inside, velocity, torch.zeros_like(velocity))

    def nearest_phase(self, q, velocity, command, candidates=64):
        phases = torch.arange(candidates, device=q.device, dtype=q.dtype)/candidates
        bank_phase = phases.expand(len(q), -1)
        bank_command = command[:, None].expand(-1, candidates)
        target, target_v = self.sample(bank_command.reshape(-1), bank_phase.reshape(-1))
        target = target.reshape(len(q), candidates, -1)[:, :, self.pitch_indices]
        target_v = target_v.reshape(len(q), candidates, -1)[:, :, self.pitch_indices]
        distance = ((q[:, None, self.pitch_indices]-target)/.35).square().mean(-1)
        distance += .1*((velocity[:, None, self.pitch_indices]-target_v)/1.5).square().mean(-1)
        return phases[distance.argmin(-1)]
