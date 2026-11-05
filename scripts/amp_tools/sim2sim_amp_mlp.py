"""MuJoCo sim2sim for the exp1.9-style MLP AMP policy (5x96 -> 29)."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
import torch
from sim2sim_amp import prepare_mjcf, rotation_from_wxyz
from mujoco_diagnostics import foot_contact_forces, roll_pitch_yaw
from native_physics import align_runtime


def frame_features(command, angular, rotation, q, dq, previous, default, scales, command_scale):
    return np.concatenate((
        [command * command_scale[0], 0., 0.],
        angular * scales['ang_vel'],
        -rotation[2],
        (q - default) * scales['dof_pos'],
        dq * scales['dof_vel'],
        previous,
    )).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--mjcf-source', type=Path, required=True)
    p.add_argument('--meshes', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--video-path', type=Path)
    p.add_argument('--csv-path', type=Path)
    p.add_argument('--training-commit', required=True)
    p.add_argument('--runtime-properties', type=Path, required=True)
    p.add_argument('--passive-probe', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--vx', type=float, default=0.5953696966)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    np.random.seed(5)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    iteration = a.checkpoint.stem.removeprefix('model_')
    video_path = a.video_path or a.checkpoint.parent / ('sim2sim_mlp_' + iteration + '.mp4')
    csv_path = a.csv_path or a.checkpoint.parent / ('sim2sim_mlp_' + iteration + '.csv')
    if video_path.exists() or csv_path.exists():
        raise FileExistsError('Preserve existing recording and CSV')
    video_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, meta = ck['config'], ck['metadata']
    names = meta['dof_names']
    n = len(names)
    control = meta['control']
    dt = float(meta['policy_dt'])
    simdt = float(cfg['sim']['dt'])
    decimation = round(dt / simdt)
    frame_stack = int(cfg['env']['frame_stack'])
    single = int(cfg['env']['num_single_obs'])
    assert n == 29 and single == 96 and frame_stack == 5
    assert meta['num_obs'] == frame_stack * single
    assert abs(decimation * simdt - dt) < 1e-7

    sys.path.insert(0, str(a.root))
    from humanoid.amp.models import ActorCritic
    from humanoid.amp.exp20_commands import SmoothCommands
    policy_cfg = copy.deepcopy(cfg['policy'])
    policy_cfg.pop('motion_exploration', None)
    policy = ActorCritic(meta['num_obs'], cfg['env']['num_privileged_obs'], n, **policy_cfg)
    policy.load_state_dict(ck['model_state_dict'], strict=True)
    policy.eval()
    actor = torch.jit.script(policy.actor.eval())
    actor.save(str(a.out_dir / 'actor.pt'))
    with torch.no_grad():
        probe = torch.randn(4, meta['num_obs']) * .1
        torch.testing.assert_close(actor(probe), policy.act_inference(probe), rtol=1e-5, atol=1e-6)

    default = np.array([float(cfg['init_state']['default_joint_angles'][k]) for k in names])
    standing = np.array(control['standing_state'], dtype=np.float64)
    kp = np.array(control['stiffness'], dtype=np.float64)
    kd = np.array(control['damping'], dtype=np.float64)
    effort = np.array(control['torque_limits'], dtype=np.float64)
    hard = np.array(control['hard_limits'], dtype=np.float64)
    action_scale = float(cfg['control']['action_scale'])
    clip_actions = float(cfg['normalization']['clip_actions'])
    clip_obs = float(cfg['normalization']['clip_observations'])
    scales = cfg['normalization']['obs_scales']
    command_scale = [scales['lin_vel'], scales['lin_vel'], scales['ang_vel']]

    mjcf = prepare_mjcf(a.mjcf_source, a.meshes.resolve().as_posix(), a.out_dir / 'mjcf', names)
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    m.opt.timestep = simdt
    m.opt.integrator = 0
    d = mujoco.MjData(m)
    if m.nu != n:
        raise ValueError(f'Actuator count {m.nu} != {n}')
    mj_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    missing = [k for k in names if k not in mj_joints]
    if missing:
        raise ValueError('MJCF is missing joints: ' + str(missing))
    jid = {k: mj_joints.index(k) for k in names}
    qpos_adr = np.array([m.jnt_qposadr[jid[k]] for k in names], dtype=int)
    dof_adr = np.array([m.jnt_dofadr[jid[k]] for k in names], dtype=int)
    actuator_joint = [mj_joints[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    assert sorted(actuator_joint) == sorted(names)
    actuator_index = np.array([actuator_joint.index(k) for k in names], dtype=int)
    urdf = Path(cfg['asset']['file'].format(LEGGED_GYM_ROOT_DIR=a.root))
    alignment = align_runtime(m, ck, a.runtime_properties, a.passive_probe, asset_urdf=urdf)

    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link') for s in ('left', 'right')]
    foot_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] != 0] for bid in feet]

    d.qpos[0:3] = standing[0:3]
    d.qpos[3:7] = [1., 0., 0., 0.]
    d.qpos[qpos_adr] = standing[13:42]
    d.qvel[:] = 0.
    mujoco.mj_forward(m, d)

    def soles():
        result = []
        for geoms in foot_geoms:
            points = []
            for g in geoms:
                mid = m.geom_dataid[g]
                start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                points.append(m.mesh_vert[start:start + count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            result.append(float(np.concatenate(points)[:, 2].min()))
        return result

    z_shift = cfg['amp']['ground_clearance'] - min(soles())
    d.qpos[2] += z_shift
    mujoco.mj_forward(m, d)
    hold = round(cfg['amp']['stand_buffer_seconds'] / dt)
    schedule = SmoothCommands(1, 'cpu', hold, dt, 2.0, a.vx)
    schedule.reset(torch.tensor([0]), torch.tensor([True]))
    schedule.requested[0, 0] = a.vx
    previous = np.zeros(n)

    def state():
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, base, velocity, 1)
        return (np.array(d.qpos[qpos_adr]), np.array(d.qvel[dof_adr]), velocity,
                rotation_from_wxyz(d.qpos[3:7]))

    def frame():
        q, dq, velocity, rotation = state()
        return frame_features(float(schedule.value[0]), velocity[:3], rotation, q, dq,
                              previous, default, scales, command_scale)

    history = collections.deque([np.zeros(single, dtype=np.float32) for _ in range(frame_stack - 1)]
                                + [frame()], maxlen=frame_stack)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 25., (1280, 720))
    assert video.isOpened(), 'MP4 writer unavailable'
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.elevation = -15.
    camera.azimuth = 90.

    columns = (['step', 'time_s', 'command', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                'sole_left', 'sole_right', 'upright', 'fallen']
               + ['q_' + k for k in names] + ['dq_' + k for k in names]
               + ['action_' + k for k in names] + ['tau_' + k for k in names]
               + ['roll', 'pitch', 'yaw'])
    rows = []
    fallen = False
    fall_time = None
    frames = 0
    try:
        for step in range(round(a.seconds / dt)):
            obs = np.clip(np.concatenate(history), -clip_obs, clip_obs)
            with torch.no_grad():
                action = actor(torch.from_numpy(obs)[None])[0].numpy()
            action = np.clip(action, -clip_actions, clip_actions)
            assert np.isfinite(action).all()
            previous = action
            target = np.clip(default + action_scale * action, hard[:, 0], hard[:, 1])
            for _ in range(decimation):
                torque = np.clip(kp * (target - d.qpos[qpos_adr]) - kd * d.qvel[dof_adr], -effort, effort)
                d.ctrl[actuator_index] = torque
                mujoco.mj_step(m, d)
            assert np.isfinite(d.qvel).all()
            q, dq, velocity, rotation = state()
            height = soles()
            time = (step + 1) * dt
            contact_force, contact_flags = foot_contact_forces(m, d, foot_geoms)
            if (d.qpos[2] < .35 or rotation[2, 2] < .5) and not fallen:
                fallen = True
                fall_time = time
            rows.append([step, time, float(schedule.value[0]), *d.qpos[:3], *velocity[3:], *velocity[:3],
                         *height, rotation[2, 2], int(fallen), *q, *dq, *action, *torque,
                         *roll_pitch_yaw(rotation)])
            if step % 2 == 0:
                camera.lookat[:] = [d.qpos[0], d.qpos[1], max(.4, d.qpos[2])]
                renderer.update_scene(d, camera=camera)
                pixels = renderer.render()[:, :, ::-1].copy()
                text = 'MLP %s | t=%.2fs | cmd=%.3f | vx=%.3f%s' % (
                    a.checkpoint.parent.name, time, float(schedule.value[0]), velocity[3],
                    ' | FALL' if fallen else '')
                cv2.putText(pixels, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (30, 30, 240) if fallen else (20, 20, 20), 2)
                video.write(pixels)
                frames += 1
            schedule.advance(torch.tensor([False]))
            history.append(frame())
            if step % 250 == 0:
                print('SIM2SIM step={} time={:.2f} fallen={}'.format(step, time, fallen), flush=True)
            if fallen and time >= fall_time + 3.:
                break
    finally:
        video.release()
        renderer.close()
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    numeric = np.asarray(rows)
    summary = dict(
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        training_commit=a.training_commit, mujoco_version=mujoco.__version__,
        policy_input=frame_stack * single, policy_output=n, policy_dt=dt, sim_dt=simdt,
        frame_stack=frame_stack, num_single_obs=single, requested_vx=a.vx,
        stand_seconds=hold * dt, command_transition_seconds=2.0,
        duration_s=float(numeric[-1, 1]), video_frames=frames, video_fps=25,
        fallen=fallen, fall_time_s=fall_time,
        video_path=str(video_path.resolve()), csv_path=str(csv_path.resolve()),
        physics_alignment=alignment, action_scale=action_scale,
        controller='substep bounded PD; no LPF; no noise or pushes',
        termination='base z<0.35 or upright<0.5, record3s after first fall; no resets',
        limits=['MLP exp1.9 policy has no phase input; command ramp is imposed for comparison.',
                'MuJoCo contact differs from PhysX; one nominal rollout only.'])
    (a.out_dir / 'sim2sim_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()