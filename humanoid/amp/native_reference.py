"""One native-tempo full-state cycle shared by phase guidance and AMP."""
import math
import hashlib
import json
from pathlib import Path

import torch

from .motion import quat_mul, quat_conjugate, quat_rotate, quat_inverse_rotate


def periodic_curve(values, period, displacement=None):
    """Close residual position and use shape-preserving periodic Hermite slopes."""
    count = len(values)-1
    displacement = torch.zeros_like(values[0]) if displacement is None else displacement
    u = torch.linspace(0, 1, count+1, device=values.device)[:, None]
    residual = values-u*displacement
    # Distribute the endpoint mismatch; this changes neither period nor amplitude scale.
    mismatch = residual[-1]-residual[0]
    residual = residual-(3*u.square()-2*u**3)*mismatch
    residual[-1] = residual[0]
    core = residual[:-1]
    forward = (torch.roll(core,-1,0)-core)/(period/count)
    backward = torch.roll(forward,1,0)
    same = forward*backward > 0
    slope = torch.where(same, 2*forward*backward/(forward+backward).where(same, torch.ones_like(forward)), 0.)
    slopes = torch.cat((slope,slope[:1]))+displacement/period
    return residual+u*displacement, slopes


def hermite(values, slopes, phase, period):
    n = len(values)-1
    phase = phase.remainder(1)
    t = phase*n
    i = t.long().clamp(max=n-1)
    u = (t-i)[:, None]
    dt = period/n
    a,b,va,vb = values[i],values[i+1],slopes[i],slopes[i+1]
    position = (2*u**3-3*u**2+1)*a+(-2*u**3+3*u**2)*b+(u**3-2*u**2+u)*dt*va+(u**3-u**2)*dt*vb
    velocity = ((6*u**2-6*u)*a+(-6*u**2+6*u)*b)/dt+(3*u**2-4*u+1)*va+(3*u**2-2*u)*vb
    return position,velocity


class NativeCycle:
    def __init__(self, library, clip='F1_B3_continuous_walk', first=27, last=90,
                 limit_margin_degrees=5., contact_correction=None):
        self.library, self.device = library, library.device
        cache_bytes = (Path(__file__).parent/'reference_data/exp20_sole_hulls.json').read_bytes()
        cache = json.loads(cache_bytes)
        self.sole_cache_sha256 = hashlib.sha256(json.dumps(cache,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.feet = []
        for i,vertices in library.grounding.feet:
            entry = cache[library.key_body_names[i]]
            if hashlib.sha256(vertices.detach().cpu().numpy().tobytes()).hexdigest()!=entry['vertices_sha256']:
                raise ValueError('Foot convex support cache does not match original collision vertices')
            self.feet.append((i,vertices[entry['indices']]))
        self.first,self.last,self.clip = first,last,clip
        index = library.names.index(clip)
        self.period = (last-first)/float(library.fps[index])
        raw = library.frames[index,first:last+1].clone()
        self.nominal_speed = float(quat_inverse_rotate(raw[:-1,3:7],raw[:-1,7:10])[:,0].mean())
        quat = raw[:,3:7]
        x,y,z,w = quat[0]
        yaw = torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        inverse_heading = raw.new_tensor([0.,0.,0.,1.])
        inverse_heading[2],inverse_heading[3] = torch.sin(-yaw/2),torch.cos(-yaw/2)
        rotation = inverse_heading.expand(len(raw),-1)
        raw[:,3:7] = quat_mul(rotation,quat)
        raw[:,:3] = quat_rotate(rotation,raw[:,:3]-raw[0,:3].clone())
        margin = torch.minimum(torch.full_like(library.fk.lower,math.radians(limit_margin_degrees)),
                               .1*(library.fk.upper-library.fk.lower))
        lower,upper = library.fk.lower+margin,library.fk.upper-margin
        source_q = raw[:,13:42].clone()
        values = raw[:,13:42].maximum(lower).minimum(upper)
        self.q,self.dq = periodic_curve(values,self.period)
        # Re-project after closing, then regenerate periodic slopes; no dq clamping.
        self.q,self.dq = periodic_curve(self.q.maximum(lower).minimum(upper),self.period)
        if contact_correction is not None:
            if not isinstance(contact_correction,dict) or 'dof_names' not in contact_correction:
                raise ValueError('Contact correction must declare its named joint order')
            names = contact_correction['dof_names']
            if (contact_correction.get('clip',clip)!=clip or contact_correction.get('first_frame',first)!=first
                    or contact_correction.get('last_frame',last)!=last):
                raise ValueError('Contact correction belongs to a different reference cycle')
            if len(set(names))!=len(names) or set(names)!=set(library.dof_names):
                raise ValueError('Contact correction joint names do not match the robot')
            order = [names.index(name) for name in library.dof_names]
            correction = torch.as_tensor(contact_correction['joint_correction'],device=self.device,dtype=self.q.dtype)[:,order]
            if correction.shape != self.q.shape or not torch.isfinite(correction).all():
                raise ValueError('Invalid native reference contact correction')
            self.q,self.dq = periodic_curve((self.q+correction).maximum(lower).minimum(upper),self.period)
        self.rot,self.rot_derivative = periodic_curve(raw[:,3:7],self.period)
        # Keep the measured root displacement; global x/y do not wrap to zero.
        displacement = raw[-1,:3]-raw[0,:3]
        displacement[2] = 0.
        self.root,self.root_derivative = periodic_curve(raw[:,:3],self.period,displacement)
        self.displacement = displacement
        phase = torch.arange(len(raw),device=self.device,dtype=raw.dtype)/(len(raw)-1)
        states = self.state(phase, ground=False)
        grounded = library.grounding.align(states[:,:3],states[:,3:7],states[:,13:42])
        self.root[:,2] = grounded[:,2]
        self.root,self.root_derivative = periodic_curve(self.root,self.period,displacement)
        self.pitch_indices = [library.dof_names.index(side+'_'+joint+'_joint') for side in ('left','right')
                              for joint in ('hip_pitch','knee_pitch','ankle_pitch')]
        self.metadata = dict(clip=clip,first_frame=first,last_frame=last,period_s=self.period,
            nominal_vx=self.nominal_speed,amplitude_scale=1.,time_scale=1.,
            max_joint_adjustment_rad=float((self.q-source_q).abs().max()),
            joint_margin_degrees=limit_margin_degrees,root_displacement=self.displacement.tolist())

    def state(self, phase, ground=False):
        if ground:
            raise ValueError('Do not ground per query without differentiating the resulting root trajectory')
        q,dq = hermite(self.q,self.dq,phase,self.period)
        root,velocity = hermite(self.root,self.root_derivative,phase,self.period)
        root = root+phase.floor()[:,None]*self.displacement
        quat,derivative = hermite(self.rot,self.rot_derivative,phase,self.period)
        norm = quat.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        unit = quat/norm
        derivative = (derivative-unit*(unit*derivative).sum(-1,keepdim=True))/norm
        omega = 2*quat_mul(derivative,quat_conjugate(unit))[:,:3]
        return torch.cat((root,unit,velocity,omega,q,dq),-1)

    def history(self, phases):
        offsets = torch.arange(self.library.history_steps-1,-1,-1,device=self.device)*self.library.dt/self.period
        states = self.state((phases[:,None]-offsets).flatten()).reshape(len(phases),len(offsets),-1)
        return self.library.features(states)

    def sample(self, command, phase):
        state = self.state(phase)
        return state[:,13:42],state[:,42:]

    def blended_pose(self,phase,standing,blend,derivative,ground_transition=True,phase_rate_scale=None):
        """Transition pose/geometry only: never an AMP expert or root dynamics claim."""
        native = self.state(phase)
        state = native.clone()
        state[:,13:42] = torch.lerp(standing[13:42].expand(len(phase),-1),native[:,13:42],blend[:,None])
        state[:,42:] = blend[:,None]*native[:,42:]+derivative[:,None]*(native[:,13:42]-standing[13:42])
        if phase_rate_scale is not None:
            state[:,42:] = (blend*phase_rate_scale)[:,None]*native[:,42:]+derivative[:,None]*(native[:,13:42]-standing[13:42])
        state[:,2] = torch.lerp(standing[2].expand(len(phase)),native[:,2],blend)
        start = standing[3:7].expand(len(phase),-1)
        target = torch.where((start*native[:,3:7]).sum(-1,keepdim=True)<0,-native[:,3:7],native[:,3:7])
        state[:,3:7] = torch.lerp(start,target,blend[:,None])
        state[:,3:7] /= state[:,3:7].norm(dim=-1,keepdim=True).clamp(min=1e-8)
        transitioning = (blend>0)&(blend<1)
        if ground_transition and transitioning.any():
            selected = state[transitioning].clone()
            selected[:,2] += self.library.grounding.clearance-self.sole_height(selected).amin(-1)
            state[transitioning,2] = selected[:,2]
        return state

    def world_vertices(self, states):
        local,rot = self.library.fk.key_transforms(states[:,13:42])
        result = []
        for i,vertices in self.feet:
            foot_q = quat_mul(states[:,3:7],rot[:,i])
            center = states[:,:3]+quat_rotate(states[:,3:7],local[:,i])
            result.append(center[:,None]+quat_rotate(foot_q[:,None].expand(-1,len(vertices),-1),
                vertices[None].expand(len(states),-1,-1)))
        return result

    def sole_height(self, states):
        local,rot = self.library.fk.key_transforms(states[:,13:42])
        heights = []
        for i,vertices in self.feet:
            quat = quat_mul(states[:,3:7],rot[:,i])
            x,y,z,w = quat.unbind(-1)
            row = torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1)
            origin = states[:,:3]+quat_rotate(states[:,3:7],local[:,i])
            heights.append((row@vertices.T).amin(-1)+origin[:,2])
        return torch.stack(heights,-1)

    def audit_transitions(self,standing):
        times = (torch.arange(101,device=self.device)+.5)*.02
        starts = torch.arange(8,device=self.device)/8
        times,starts = torch.meshgrid(times,starts,indexing='ij')
        times,starts = times.flatten(),starts.flatten()
        epsilon = .0001
        reports = {}
        for label,direction in (('start',1),('stop',-1)):
            def evaluate(time,geometry):
                u = (time/2).clamp(0,1)
                blend = 3*u.square()-2*u**3
                derivative = 3*u*(1-u)
                if direction<0: blend,derivative = 1-blend,-derivative
                return self.blended_pose(starts+time/self.period,standing,blend,derivative,geometry)
            state = evaluate(times,True)
            before,after = evaluate(times-epsilon,False),evaluate(times+epsilon,False)
            error = ((after[:,13:42]-before[:,13:42])/(2*epsilon)-state[:,42:]).abs().max()
            lower,upper = self.library.fk.lower,self.library.fk.upper
            violation = (lower-state[:,13:42]).clamp(min=0)+(state[:,13:42]-upper).clamp(min=0)
            height = self.sole_height(state).min()
            reports[label] = dict(samples=len(times),dq_max_error=float(error),
                minimum_sole_height_m=float(height),maximum_joint_speed_rad_s=float(state[:,42:].abs().max()),
                hard_limit_max_violation_rad=float(violation.max()),
                passed=bool(error<.02 and height>=-.003 and violation.max()<1e-5 and torch.isfinite(state).all()))
        return dict(cases=reports,passed=all(r['passed'] for r in reports.values()),
            scope='Pose derivatives, limits and grounded geometry across8 initial phases; not root dynamics or physical start/stop stability.')

    def audit(self, count=1260):
        phases = (torch.arange(count,device=self.device)+.5)/count
        states = self.state(phases)
        vertices = self.world_vertices(states)
        epsilon = .0005
        before = self.world_vertices(self.state(phases-epsilon/self.period))
        after = self.world_vertices(self.state(phases+epsilon/self.period))
        heights,speeds = [],[]
        rows = torch.arange(count,device=self.device)
        for v,a,b in zip(vertices,before,after):
            h,ids = v[:,:,2].min(-1)
            heights.append(h)
            speeds.append(((b-a)/(2*epsilon))[rows,ids,:2].norm(dim=-1))
        heights,speeds = torch.stack(heights,-1),torch.stack(speeds,-1)
        contact = heights < .008
        samples = speeds[contact]
        clearance = []
        for i in range(2):
            flag = heights[:,i] > .01
            # Three repeated cycles include raised runs spanning the phase seam.
            repeated = flag.repeat(3).cpu().numpy()
            import numpy as np
            edges = np.diff(np.r_[False,repeated,False].astype(int))
            starts,ends = np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
            durations = [(b-a)*self.period/count for a,b in zip(starts,ends) if count<=a<2*count]
            clearance.append(dict(p95_m=float(heights[:,i].quantile(.95)),
                longest_above_1cm_s=max(durations,default=0.)))
        # Separate derivative convergence from the material-point velocity probe.
        # A 0.5ms difference is too coarse for steep cubic segments near a knot.
        derivative_epsilon = .0001
        finite_before,finite_after = self.state(phases-derivative_epsilon/self.period),self.state(phases+derivative_epsilon/self.period)
        dq_error = ((finite_after[:,13:42]-finite_before[:,13:42])/(2*derivative_epsilon)-states[:,42:]).abs()
        local_v = quat_inverse_rotate(states[:,3:7],states[:,7:10])
        margin = torch.minimum(torch.full_like(self.library.fk.lower,math.radians(5)),
            .1*(self.library.fk.upper-self.library.fk.lower))
        margin_error = ((self.library.fk.lower+margin)-states[:,13:42]).clamp(min=0)+(states[:,13:42]-(self.library.fk.upper-margin)).clamp(min=0)
        report = dict(self.metadata,measured_body_vx=float(local_v[:,0].mean()),
            measured_body_vy=float(local_v[:,1].mean()),mean_abs_body_vy=float(local_v[:,1].abs().mean()),
            minimum_sole_height_m=float(heights.min()),stance_samples=int(contact.sum()),
            stance_slip_median_m_s=float(samples.median()) if len(samples) else None,
            stance_slip_p90_m_s=float(samples.quantile(.9)) if len(samples) else None,
            airborne_fraction=float((~contact).all(-1).float().mean()),feet=clearance,
            dq_finite_difference_max_error=float(dq_error.max()),dq_finite_difference_epsilon_s=derivative_epsilon,
            maximum_joint_speed_rad_s=float(states[:,42:].abs().max()),
            joint_margin_max_violation_rad=float(margin_error.max()))
        report['passed'] = bool(len(samples) and samples.median()<=.05 and samples.quantile(.9)<=.10
            and heights.min()>=-.003 and (~contact).all(-1).float().mean()<=.01
            and all(f['longest_above_1cm_s']>=.20 for f in clearance)
            and dq_error.max()<.01 and margin_error.max()<1e-5 and states[:,42:].abs().max()<=8.
            and abs(float(local_v[:,0].mean())-self.nominal_speed)<.02)
        return report

    def phase_rate(self, command):
        return torch.full_like(command,1/self.period)

    def nearest_phase(self, q, dq, command=None):
        phases = torch.arange(128,device=self.device)/128
        states = self.state(phases)
        distance = ((q[:,None,self.pitch_indices]-states[None,:,13:42][:,:,self.pitch_indices])/.35).square().mean(-1)
        return phases[distance.argmin(-1)]

    def expert(self, moving):
        """Match each sampled policy category; never silently fall back to raw clips."""
        phases = torch.rand(len(moving),device=self.device)
        result = self.history(phases)
        quiet = ~moving.bool()
        if quiet.any():
            if not hasattr(self,'quiet_histories'):
                raise ValueError('Quiet expert histories not initialized')
            index = torch.randint(len(self.quiet_histories),(int(quiet.sum()),),device=self.device)
            result[quiet] = self.quiet_histories[index]
        return result
