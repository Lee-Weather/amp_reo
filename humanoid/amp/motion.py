"""Named GMR motions: xyzw quaternions, world velocities, oldest-first history."""
import math
import pickle
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
    xyz = xyz + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def quat_inverse_rotate(q, v):
    return quat_rotate(quat_conjugate(q), v)


def quat_slerp(a, b, t):
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1.0)
    angle = torch.acos(dot)
    den = torch.sin(angle).clamp(min=1e-6)
    q = torch.where(dot > 0.9995, a + t * (b - a),
                    (torch.sin((1-t)*angle)*a + torch.sin(t*angle)*b) / den)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def amp_features(height, quat, lin_vel, ang_vel, q, dq, key_pos_b):
    gravity = torch.zeros_like(lin_vel)
    gravity[..., 2] = -1
    return torch.cat((height, quat_inverse_rotate(quat, gravity),
                      quat_inverse_rotate(quat, lin_vel), quat_inverse_rotate(quat, ang_vel),
                      q, dq, key_pos_b.flatten(-2)), -1)


class URDFKinematics:
    """Forward kinematics of selected URDF link origins in the base frame."""
    def __init__(self, urdf_path, dof_names, key_body_names, device="cpu"):
        root = ET.parse(urdf_path).getroot()
        joints = root.findall("joint")
        movable = {j.get("name"): j for j in joints if j.get("type") != "fixed"}
        if len(set(dof_names)) != len(dof_names) or set(movable) != set(dof_names):
            raise ValueError("Simulator DOF names must match URDF joints exactly")
        self.lower = torch.tensor([float(movable[n].find("limit").get("lower")) for n in dof_names], device=device)
        self.upper = torch.tensor([float(movable[n].find("limit").get("upper")) for n in dof_names], device=device)
        self.key_body_names = list(key_body_names)
        by_child = {j.find("child").get("link"): j for j in joints}
        roots = {x.get("name") for x in root.findall("link")} - set(by_child)
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.root_name = roots.pop()
        required = set()
        for body in key_body_names:
            while body != self.root_name:
                if body not in by_child:
                    raise ValueError("Unknown URDF key body: " + body)
                required.add(body)
                body = by_child[body].find("parent").get("link")
        self.chain = []
        visited = {self.root_name}
        while required:
            ready = [b for b in sorted(required) if by_child[b].find("parent").get("link") in visited]
            if not ready:
                raise ValueError("URDF joint tree contains a cycle")
            for body in ready:
                j = by_child[body]
                origin = j.find("origin")
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
                r, p, y = [float(x)/2 for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
                sr, cr, sp, cp, sy, cy = math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)
                quat = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
                axis_node = j.find("axis")
                axis = [float(x) for x in axis_node.get("xyz").split()] if axis_node is not None else [1., 0., 0.]
                axis = torch.tensor(axis, device=device)
                axis = axis / axis.norm()
                idx = dof_names.index(j.get("name")) if j.get("type") != "fixed" else None
                if j.get("type") not in ("fixed", "revolute", "continuous"):
                    raise ValueError("Only fixed/revolute joints supported")
                self.chain.append((body, j.find("parent").get("link"), torch.tensor(xyz, device=device),
                                   torch.tensor(quat, device=device), axis, idx))
                visited.add(body)
                required.remove(body)

    def key_positions(self, q):
        return self.key_transforms(q)[0]

    def key_transforms(self, q):
        shape = q.shape[:-1]
        pos = {self.root_name: q.new_zeros(*shape, 3)}
        identity = q.new_zeros(*shape, 4)
        identity[..., 3] = 1
        rot = {self.root_name: identity}
        for body, parent, xyz, origin_q, axis, idx in self.chain:
            pos[body] = pos[parent] + quat_rotate(rot[parent], xyz.expand(*shape, 3))
            rot[body] = quat_mul(rot[parent], origin_q.expand(*shape, 4))
            if idx is not None:
                half = q[..., idx:idx+1] / 2
                rot[body] = quat_mul(rot[body], torch.cat((axis * half.sin(), half.cos()), -1))
        return (torch.stack([pos[b] for b in self.key_body_names], -2),
                torch.stack([rot[b] for b in self.key_body_names], -2))


class MotionLibrary:
    def __init__(self, motion_dir, urdf_path, dof_names, key_body_names, dt=0.02,
                 history_steps=3, device="cpu", weights=None, limit_tolerance=0.05, ground_clearance=None):
        if dt <= 0 or history_steps < 2:
            raise ValueError("AMP requires positive dt and at least two history steps")
        self.device, self.dt, self.history_steps = device, dt, history_steps
        self.dof_names, self.key_body_names = list(dof_names), list(key_body_names)
        self.fk = URDFKinematics(urdf_path, self.dof_names, key_body_names, device)
        self.grounding = None
        if ground_clearance is not None:
            from .grounding import FootGrounding
            self.grounding = FootGrounding(urdf_path, self.fk, ground_clearance)
        paths = sorted(Path(motion_dir).glob("*.pkl"))
        if weights is not None:
            if set(weights) - {p.stem for p in paths}:
                raise ValueError("Motion weights contain unknown clip names")
            paths = [p for p in paths if weights.get(p.stem, 0) > 0]
        if not paths:
            raise ValueError("No reference motions found in " + str(motion_dir))
        clips, fps_list, lengths = [], [], []
        self.names = [p.stem for p in paths]
        for path in paths:
            # Only trusted local pickle files may be loaded.
            with path.open("rb") as stream:
                data = pickle.load(stream)
            names = list(data["dof_names"])
            if len(set(names)) != len(names) or set(names) != set(dof_names):
                raise ValueError(str(path) + ": joint names do not match URDF")
            fps = float(data["fps"])
            q = np.asarray(data["dof_pos"], dtype=np.float32)[:, [names.index(n) for n in dof_names]].copy()
            pos, rot = [np.asarray(data[k], dtype=np.float32).copy() for k in ("root_pos", "root_rot")]
            n = len(q)
            if q.shape != (n, len(dof_names)) or pos.shape != (n, 3) or rot.shape != (n, 4):
                raise ValueError(str(path) + ": invalid array shapes")
            if not np.isfinite(fps) or fps <= 0 or n < 3 or (n-1)/fps <= dt*(history_steps-1):
                raise ValueError(str(path) + ": invalid fps or clip too short")
            if not all(np.isfinite(x).all() for x in (q, pos, rot)):
                raise ValueError(str(path) + ": non-finite values")
            norm = np.linalg.norm(rot, axis=-1, keepdims=True)
            if np.any(norm < 1e-6):
                raise ValueError(str(path) + ": zero quaternion")
            rot /= norm
            for i in range(1, n):
                if np.dot(rot[i-1], rot[i]) < 0:
                    rot[i] *= -1
            qt = torch.tensor(q, device=device)
            clamped = qt.maximum(self.fk.lower).minimum(self.fk.upper)
            excess = (qt - clamped).abs().max().item()
            if excess > limit_tolerance:
                raise ValueError("{}: joint limit violation {:.4f} rad; check model/signs".format(path, excess))
            if excess > 1e-5:
                warnings.warn("{}: clamping small joint limit overshoot ({:.4f} rad)".format(path.name, excess))
            q = clamped.cpu().numpy()
            if self.grounding is not None:
                pos = self.grounding.align(torch.tensor(pos, device=device), torch.tensor(rot, device=device), clamped).cpu().numpy()
            vel, qvel = np.gradient(pos, 1/fps, axis=0), np.gradient(q, 1/fps, axis=0)
            rt = torch.tensor(rot, device=device)
            rdot = torch.tensor(np.gradient(rot, 1/fps, axis=0), device=device)
            omega = 2 * quat_mul(rdot, quat_conjugate(rt))[:, :3]
            clips.append(torch.cat((torch.tensor(pos, device=device), rt, torch.tensor(vel, device=device),
                                    omega, clamped, torch.tensor(qvel, device=device)), -1))
            fps_list.append(fps)
            lengths.append(n)
        self.frames = torch.stack([torch.cat((c, c[-1:].expand(max(lengths)-len(c), -1))) for c in clips])
        self.fps = torch.tensor(fps_list, device=device)
        self.durations = (torch.tensor(lengths, device=device) - 1) / self.fps
        self.weights = torch.tensor([weights[p.stem] if weights else 1. for p in paths], device=device)
        self.weights /= self.weights.sum()
        self.feature_dim = 10 + 2*len(dof_names) + 3*len(key_body_names)

    def sample_ids_times(self, count):
        ids = torch.multinomial(self.weights, count, replacement=True)
        minimum = (self.history_steps - 1) * self.dt
        return ids, minimum + torch.rand(count, device=self.device) * (self.durations[ids] - minimum)

    def state(self, ids, times):
        t = times.clamp(min=0).minimum(self.durations[ids]) * self.fps[ids]
        i = t.long()
        last = torch.round(self.durations[ids] * self.fps[ids]).long()
        a, b = self.frames[ids, i], self.frames[ids, (i+1).minimum(last)]
        f = (t-i).unsqueeze(-1)
        state = torch.lerp(a, b, f)
        state[..., 3:7] = quat_slerp(a[..., 3:7], b[..., 3:7], f)
        return state

    def features(self, states):
        n = len(self.dof_names)
        q, dq = states[..., 13:13+n], states[..., 13+n:]
        return amp_features(states[..., 2:3], states[..., 3:7], states[..., 7:10],
                            states[..., 10:13], q, dq, self.fk.key_positions(q))

    def history(self, ids, times):
        offsets = torch.arange(self.history_steps-1, -1, -1, device=self.device) * self.dt
        states = self.state(ids[:, None].expand(-1, self.history_steps), times[:, None]-offsets)
        return self.features(states)

    def sample(self, count):
        return self.history(*self.sample_ids_times(count))
