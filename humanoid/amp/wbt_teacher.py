"""Read-only WBT inference adapter; no policy transfer or learning state."""
import json

import numpy as np


def permutation(source_names, destination_names):
    if (len(set(source_names)) != len(source_names) or
            len(set(destination_names)) != len(destination_names) or
            set(source_names) != set(destination_names)):
        raise ValueError('Joint names must be unique and match exactly')
    return np.array([source_names.index(name) for name in destination_names])


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('Expected a unit xyzw quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class WBTTeacher:
    def __init__(self, bundle):
        with np.load(bundle, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        self.meta = json.loads(str(self.data.pop('metadata')))
        if self.meta['schema'] != 'x1_wbt_teacher_v1':
            raise ValueError('Unsupported teacher bundle')
        self.names = self.meta['reference_joint_names']
        self.actor_names = self.meta['actor_joint_names']
        self.to_actor = permutation(self.names, self.actor_names)
        self.default = np.asarray(self.meta['default_joint_pos'], dtype=np.float32)
        for key, value in self.data.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite teacher data: '+key)
        if self.data['mean'].shape != (160,) or np.any(self.data['divisor'] <= 0):
            raise ValueError('Invalid teacher normalizer')
        if len(self.names) != 29 or self.data['joint_pos'].shape[1:] != (29,):
            raise ValueError('Expected the X1 29DOF teacher')
        self.frames = len(self.data['joint_pos'])

    def observation(self, frame, root, q, dq, last_action, joint_names):
        if frame < 0 or frame >= self.frames:
            raise ValueError('Reference frame outside clip; looping is not implicit')
        order = permutation(joint_names, self.names)
        q, dq = np.asarray(q)[order], np.asarray(dq)[order]
        root, last_action = np.asarray(root), np.asarray(last_action)
        if root.shape != (13,) or q.shape != (29,) or dq.shape != (29,) or last_action.shape != (29,):
            raise ValueError('Teacher state dimensions disagree')
        rot = rotation_xyzw(root[3:7])
        ref = self.data
        anchor = self.meta['body_names'].index('base_link')
        refrot = rotation_xyzw(ref['body_quat_w'][frame, anchor][[1, 2, 3, 0]])
        relative = rot.T@refrot
        observation = np.concatenate((ref['joint_pos'][frame, self.to_actor],
            ref['joint_vel'][frame, self.to_actor],
            rot.T@(ref['body_pos_w'][frame, anchor]-root[:3]),
            relative[:, :2].reshape(-1), rot.T@root[7:10], rot.T@root[10:13],
            (q-self.default)[self.to_actor], dq[self.to_actor], last_action)).astype(np.float32)
        if observation.shape != (160,) or not np.isfinite(observation).all():
            raise ValueError('Invalid teacher observation')
        return observation

    def action(self, observation):
        # NumPy inference keeps Isaac Gym import ordering independent of torch.
        x = (np.asarray(observation, dtype=np.float32)-self.data['mean'])/self.data['divisor']
        for index in (0, 2, 4, 6):
            x = x@self.data['actor.{}.weight'.format(index)].T+self.data['actor.{}.bias'.format(index)]
            if index != 6:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        if x.shape != (29,) or not np.isfinite(x).all():
            raise ValueError('Invalid teacher action')
        return x

    def target(self, action, joint_names):
        # Actor slots differ from the reference arrays' joint order.
        actor_to_ref = permutation(self.actor_names, self.names)
        q = self.default+self.meta['action_scale']*np.asarray(action)[actor_to_ref]
        return q[permutation(self.names, joint_names)]


def student_action(target, default, scale, clip):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Student action scale must be positive')
    raw = (np.asarray(target)-np.asarray(default))/scale
    return np.clip(raw, -clip, clip), raw


def termination(root, reference_position, reference_quaternion, body_positions, reference_bodies, body_names):
    """WBT height/EE criteria plus stricter orientation and floor-height guards."""
    root = np.asarray(root)
    if not all(np.isfinite(v).all() for v in (root, body_positions, reference_bodies)):
        return 'nonfinite_state'
    if root[2] < .3:
        return 'base_below_0.3m'
    if abs(root[2]-reference_position[2]) > .25:
        return 'anchor_height_error_above_0.25m'
    rot = rotation_xyzw(root[3:7])
    refrot = rotation_xyzw(np.asarray(reference_quaternion)[[1, 2, 3, 0]])
    if abs(rot[2, 2]-refrot[2, 2]) > .8:
        return 'projected_gravity_z_difference_above_0.8'
    angle = np.arccos(np.clip((np.trace(rot.T@refrot)-1)/2, -1, 1))
    if angle > .8:
        return 'tracking_orientation_error_above_0.8rad'
    for side in ('left', 'right'):
        for part in ('ankle', 'wrist'):
            name = side+'_'+part+'_roll_link'
            i = body_names.index(name)
            if abs(body_positions[i, 2]-reference_bodies[i, 2]) > .25:
                return name+'_height_error_above_0.25m'
    return None
