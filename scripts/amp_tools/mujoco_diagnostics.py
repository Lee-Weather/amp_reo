"""Contact forces in world axes, summed on each foot (not contact counts)."""
import numpy as np


def foot_contact_forces(model, data, foot_geoms):
    import mujoco
    forces = np.zeros((len(foot_geoms), 3))
    flags = np.zeros(len(foot_geoms), dtype=int)
    groups = [set(group) for group in foot_geoms]
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, local)
        world = contact.frame.reshape(3, 3).T @ local[:3]
        for side, geoms in enumerate(groups):
            sign = int(contact.geom2 in geoms)-int(contact.geom1 in geoms)
            if sign and np.linalg.norm(world)>1e-8:
                forces[side] += sign*world
                flags[side] = 1
    return forces, flags


def roll_pitch_yaw(rotation):
    return np.array([np.arctan2(rotation[2,1], rotation[2,2]),
        np.arcsin(np.clip(-rotation[2,0], -1., 1.)),
        np.arctan2(rotation[1,0], rotation[0,0])])
