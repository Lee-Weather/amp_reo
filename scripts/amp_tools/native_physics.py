"""Map verified Isaac Gym link inertia and passive joint properties to MuJoCo."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def merged_properties(model, names):
    """Aggregate fixed descendants into named PhysX bodies, in body coordinates."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): n for n in names}
    if -1 in ids:
        raise ValueError('PhysX body missing from MuJoCo')
    members = {bid: [] for bid in ids}
    for child in range(1, model.nbody):
        parent = child
        while parent not in ids and parent:
            if model.body_jntnum[parent]:
                raise ValueError('Cannot merge an independently moving body')
            parent = int(model.body_parentid[parent])
        if parent in ids:
            members[parent].append(child)
    result = {}
    for bid, children in members.items():
        mass = model.body_mass[children]
        total = float(mass.sum())
        com_world = (data.xipos[children]*mass[:, None]).sum(0)/total
        inertia_world = np.zeros((3, 3))
        for child in children:
            rot = data.ximat[child].reshape(3, 3)
            offset = data.xipos[child]-com_world
            inertia_world += rot@np.diag(model.body_inertia[child])@rot.T
            inertia_world += model.body_mass[child]*(np.eye(3)*offset.dot(offset)-np.outer(offset, offset))
        rot = data.xmat[bid].reshape(3, 3)
        result[ids[bid]] = dict(mass=total, com=(rot.T@(com_world-data.xpos[bid])).tolist(),
                               inertia=(rot.T@inertia_world@rot).tolist(), members=children)
    return result


def align_runtime(model, checkpoint, runtime_path, probe_path, asset_urdf=None):
    runtime = json.loads(Path(runtime_path).read_text(encoding='utf-8'))
    probe = json.loads(Path(probe_path).read_text(encoding='utf-8'))
    cfg, meta = checkpoint['config'], checkpoint['metadata']
    if 'asset_urdf_sha256' in runtime:
        if asset_urdf is None or hashlib.sha256(Path(asset_urdf).read_bytes()).hexdigest()!=runtime['asset_urdf_sha256']:
            raise ValueError('Runtime/asset URDF SHA256 mismatch')
    if runtime['asset_config'] != cfg['asset'] or runtime['sim_config'] != cfg['sim']:
        raise ValueError('Runtime export belongs to different asset or simulation configuration')
    names = meta['dof_names']
    if list(runtime['dofs']) != names:
        raise ValueError('Runtime/checkpoint joint order differs')
    for key in ('stiffness','damping','torque_limits'):
        if not np.allclose(runtime[key],meta['control'][key],rtol=0,atol=1e-6):
            raise ValueError('Runtime/checkpoint PD differs: '+key)
    if any(r['maximum_contact_force']>1e-6 for r in probe['results']) or probe['repeat_max_state_delta']>1e-6:
        raise ValueError('Passive damping probe is not contact-free/repeatable')
    if probe['damping_effect_max_joint_velocity_delta']<.01:
        raise ValueError('Passive damping has not been demonstrated in EFFORT mode')
    expected = runtime['final_bodies']
    grouped = merged_properties(model, [b['name'] for b in expected])
    armature = float(cfg['asset']['armature'])
    before = []
    for body in expected:
        current = grouped[body['name']]
        delta = np.asarray(body['inertia'])-np.asarray(current['inertia'])
        if abs(body['mass']-current['mass'])>1e-5 or np.max(np.abs(np.asarray(body['com'])-current['com']))>1e-6:
            raise ValueError('Mass/COM mismatch requires a separate conversion: '+body['name'])
        if np.max(np.abs(delta-armature*np.eye(3)))>1e-6:
            raise ValueError('Inertia difference is not verified link-space armature: '+body['name'])
        before.append(float(np.max(np.abs(delta))))
    # Exactly one isotropic increment for each surviving PhysX body, including
    # the floating base. Adding it to fixed descendants would double count it.
    for body in expected:
        bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body['name'])
        model.body_inertia[bid] += armature
    for name in names:
        actual = runtime['dofs'][name]
        if actual['driveMode'] != 3 or actual['stiffness'] != 0:
            raise ValueError('Requires the audited EFFORT actuator configuration')
        jid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        adr = model.jnt_dofadr[jid]
        model.dof_armature[adr] = actual['armature']
        model.dof_damping[adr] = actual['damping']
        model.dof_frictionloss[adr] = actual['friction']
    mujoco.mj_setConst(model,mujoco.MjData(model))
    after = merged_properties(model,[b['name'] for b in expected])
    residual = max(float(np.max(np.abs(np.asarray(b['inertia'])-after[b['name']]['inertia']))) for b in expected)
    if residual>1e-6:
        raise ValueError('Post-alignment full inertia audit failed')
    return dict(mode='verified_physx_link_inertia_v1',runtime_checkpoint_sha256=runtime['checkpoint_sha256'],
        runtime_path=str(Path(runtime_path).resolve()),probe_path=str(Path(probe_path).resolve()),
        compared_merged_bodies=len(expected),link_inertia_increment=armature,
        before_full_inertia_max_error=max(before),after_full_inertia_max_error=residual,
        joint_armature=sorted({p['armature'] for p in runtime['dofs'].values()}),
        passive_joint_damping=sorted({p['damping'] for p in runtime['dofs'].values()}),
        scope='Mass/COM/full inertia and passive DOF properties; contact solvers are not equivalent.')
