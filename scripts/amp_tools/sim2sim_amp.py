"""Headless MuJoCo sim2sim check for a DH AMP checkpoint.

Rebuilds the 98-dim single-frame observation (66-frame history), the bounded
nominal PD controller and the phase clock from the checkpoint metadata, then
drives the exported TorchScript actor in MuJoCo. This is a qualitative
cross-simulator check, not an Isaac Gym replacement.
"""
import argparse
import collections
import csv
import json
import math
import pathlib
import re
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def rotation_from_wxyz(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def prepare_mjcf(source, meshes, destination, joint_names):
    raw = (source / "x1_29dof_raw.xml").read_text(encoding="utf-8")
    raw = re.sub(r'meshdir="[^"]*"', 'meshdir="{}"'.format(meshes), raw, count=1)
    raw = re.sub(r'file="../../meshes/([^"]+)"', r'file="{}/\1"'.format(meshes), raw)
    raw = raw.replace(' content_type="model/stl"', "")
    raw = re.sub(r' actuatorfrcrange="[^"]*"', "", raw)
    raw = re.sub(r"<compiler ", '<compiler autolimits="true" ', raw, count=1)
    actuators = "\n".join('    <motor joint="{}" gear="1"/>'.format(name) for name in joint_names)
    raw = raw.replace("</mujoco>", "  <actuator>\n{}\n  </actuator>\n</mujoco>".format(actuators))
    scene = (source / "x1_29dof_scene.xml").read_text(encoding="utf-8")
    scene = re.sub(r"<compiler ", '<compiler autolimits="true" ', scene, count=1)
    scene = re.sub(r'timestep="[^"]*"', 'timestep="{}"'.format(0.002), scene, count=1)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "x1_29dof_raw.xml").write_text(raw, encoding="utf-8")
    (destination / "x1_29dof_scene.xml").write_text(scene, encoding="utf-8")
    return destination / "x1_29dof_scene.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--policy", type=pathlib.Path, required=True)
    parser.add_argument("--mjcf-source", type=pathlib.Path, required=True)
    parser.add_argument("--meshes", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--stand-seconds", type=float, default=1.5)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--integrator", choices=("euler", "implicitfast"), default="euler")
    args = parser.parse_args()
    torch.set_num_threads(1)
    import mujoco

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg, meta = checkpoint["config"], checkpoint["metadata"]
    names = list(meta["dof_names"])
    action_dim = len(names)
    default = np.array([float(cfg["init_state"]["default_joint_angles"][n]) for n in names], dtype=np.float64)
    standing = np.array(meta["control"]["standing_state"], dtype=np.float64)
    kp = np.array(meta["control"]["stiffness"], dtype=np.float64)
    kd = np.array(meta["control"]["damping"], dtype=np.float64)
    tau_limit = np.array(meta["control"]["torque_limits"], dtype=np.float64)
    hard = np.array(meta["control"]["hard_limits"], dtype=np.float64)
    action_scale = float(cfg["control"]["action_scale"])
    clip_actions = float(cfg["normalization"]["clip_actions"])
    clip_obs = float(cfg["normalization"]["clip_observations"])
    scales = cfg["normalization"]["obs_scales"]
    commands_scale = np.array([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]], dtype=np.float64)
    policy_dt = float(meta["policy_dt"])
    sim_dt = float(args.sim_dt)
    decimation = int(round(policy_dt / sim_dt))
    if abs(decimation * sim_dt - policy_dt) > 1e-6:
        raise ValueError("sim-dt must divide the policy period exactly")
    frame_stack = int(cfg["env"]["frame_stack"])
    single = int(cfg["env"]["num_single_obs"])

    mjcf = prepare_mjcf(args.mjcf_source, args.meshes.resolve().as_posix(), args.out_dir / "mjcf", names)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    model.opt.timestep = sim_dt
    model.opt.integrator = 3 if args.integrator == "implicitfast" else 0
    data = mujoco.MjData(model)
    if model.nu != action_dim:
        raise ValueError("Actuator count {} != action dim {}".format(model.nu, action_dim))
    mj_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    missing = [n for n in names if n not in mj_joints]
    if missing:
        raise ValueError("MJCF is missing joints: " + str(missing))
    jid = {n: mj_joints.index(n) for n in names}
    qpos_adr = np.array([model.jnt_qposadr[jid[n]] for n in names], dtype=int)
    dof_adr = np.array([model.jnt_dofadr[jid[n]] for n in names], dtype=int)
    actuator_joint = [mj_joints[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    if sorted(actuator_joint) != sorted(names):
        raise ValueError("Actuators do not cover the checkpoint joints one-to-one")
    actuator_index = np.array([actuator_joint.index(n) for n in names], dtype=int)
    # Isaac Gym applies the configured armature and no joint damping beyond PD.
    model.dof_armature[dof_adr] = float(cfg["asset"].get("armature", 0.0))
    model.dof_damping[dof_adr] = 0.0
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body < 0:
        raise ValueError("base_link body is missing")
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s + "_ankle_roll_link")
                   for s in ("left", "right")]
    foot_geoms = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == body] for body in foot_bodies]

    def sole_height(side):
        lowest = math.inf
        for g in foot_geoms[side]:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].reshape(-1, 3)
            world = data.geom_xpos[g] + vertices.dot(data.geom_xmat[g].reshape(3, 3).T)
            lowest = min(lowest, float(world[:, 2].min()))
        return lowest

    sys.path.insert(0, str(args.root))
    from humanoid.amp.motion import MotionLibrary
    from humanoid.amp.phase_reference import PhaseReference

    motion_dir = cfg["amp"]["motion_dir"].format(LEGGED_GYM_ROOT_DIR=str(args.root))
    urdf = cfg["asset"]["file"].format(LEGGED_GYM_ROOT_DIR=str(args.root))
    library = MotionLibrary(motion_dir, urdf, names, cfg["amp"]["key_body_names"], dt=policy_dt,
                            history_steps=cfg["amp"]["history_steps"], device="cpu",
                            weights=cfg["amp"]["motion_weights"],
                            limit_tolerance=cfg["amp"]["limit_tolerance"],
                            ground_clearance=cfg["amp"].get("ground_clearance"))
    reference = PhaseReference(library, torch.tensor(standing[13:42], dtype=torch.float32),
                               cfg["amp"]["reference_cycles"])
    phase_rate = float(reference.phase_rate(torch.tensor([args.vx]))[0])

    policy = torch.jit.load(str(args.policy), map_location="cpu")
    policy.eval()

    data.qpos[0:3] = standing[0:3]
    w, x, y, z = 0., 0., 0., 1.
    data.qpos[3:7] = [standing[6], standing[3], standing[4], standing[5]]
    data.qpos[qpos_adr] = standing[13:42]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    if data.qpos[2] < 0.2:
        raise ValueError("Standing pose starts too low: " + str(float(data.qpos[2])))
    initial_soles = (sole_height(0), sole_height(1))
    if min(initial_soles) < 0.0:
        # Align the kinematic standing pose to the MuJoCo ground plane.
        data.qpos[2] += -min(initial_soles)
        mujoco.mj_forward(model, data)
        initial_soles = (sole_height(0), sole_height(1))
    print("INIT root_z {:.4f} sole_left {:.4f} sole_right {:.4f} contacts {}".format(
        float(data.qpos[2]), initial_soles[0], initial_soles[1], data.ncon), flush=True)

    history = collections.deque([np.zeros(single, dtype=np.float32) for _ in range(frame_stack)],
                                maxlen=frame_stack)
    last_action = np.zeros(action_dim, dtype=np.float64)
    phase = 0.0
    command = 0.0
    stand_steps = int(round(args.stand_seconds / policy_dt))
    total_steps = int(round(args.seconds / policy_dt))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video = None
    renderer = None
    if args.render:
        import cv2
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            video = cv2.VideoWriter(str(args.out_dir / "sim2sim.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
        except Exception as error:  # Headless GL may be unavailable.
            print("RENDER_UNAVAILABLE", repr(error))
            renderer, video = None, None
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 3.2, -18.0, 90.0

    def observation():
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body, velocity, 1)
        rotation = rotation_from_wxyz(data.qpos[3:7])
        q = np.array(data.qpos[qpos_adr], dtype=np.float64)
        dq = np.array(data.qvel[dof_adr], dtype=np.float64)
        angle = 2 * math.pi * phase
        features = [command * commands_scale[0], 0., 0.,
                    velocity[3] * scales["ang_vel"], velocity[4] * scales["ang_vel"],
                    velocity[5] * scales["ang_vel"],
                    -rotation[2, 0], -rotation[2, 1], -rotation[2, 2],
                    (q - default) * scales["dof_pos"], dq * scales["dof_vel"], last_action,
                    np.zeros(2)]
        frame = np.concatenate([np.atleast_1d(np.asarray(f, dtype=np.float64)) for f in features])
        if abs(command) > 0.05:
            frame[-2:] = [math.sin(angle), math.cos(angle)]
        else:
            frame[-2:] = [0., 0.]
        if frame.size != single:
            raise ValueError("Observation frame {} != {}".format(frame.size, single))
        return frame.astype(np.float32), q, dq

    columns = (["step", "time_s", "command", "phase", "x", "y", "z", "vx", "vy", "vz",
                "wx", "wy", "wz", "sole_left", "sole_right", "upright"]
               + ["q_" + n for n in names] + ["dq_" + n for n in names]
               + ["tau_" + n for n in names] + ["action_" + n for n in names])
    rows = []
    fallen = False
    fall_time = None
    for step in range(total_steps):
        if step == stand_steps:
            command = args.vx
            with torch.no_grad():
                q_current = torch.tensor(np.array(data.qpos[qpos_adr], dtype=np.float32)[None])
                dq_current = torch.tensor(np.array(data.qvel[dof_adr], dtype=np.float32)[None])
                phase = float(reference.nearest_phase(q_current, dq_current,
                                                      torch.tensor([args.vx], dtype=torch.float32))[0])
        frame, q, dq = observation()
        if step == 0:
            history.clear()
            history.extend([np.zeros(single, dtype=np.float32) for _ in range(frame_stack)])
        history.append(frame)
        observation_vector = np.concatenate(list(history)).astype(np.float32)
        observation_vector = np.clip(observation_vector, -clip_obs, clip_obs)
        with torch.no_grad():
            action = policy(torch.from_numpy(observation_vector).unsqueeze(0))[0].numpy()
        action = np.clip(action, -clip_actions, clip_actions).astype(np.float64)
        last_action = action
        if step % 25 == 0:
            print("STEP {} action_absmax {:.3f} action_mean {:.3f}".format(
                step, float(np.abs(action).max()), float(action.mean())), flush=True)
        target = np.clip(default + action_scale * action, hard[:, 0], hard[:, 1])
        for _ in range(decimation):
            # Isaac Gym re-evaluates the PD law on every physics substep while
            # holding the policy action fixed.
            q_sub = np.array(data.qpos[qpos_adr], dtype=np.float64)
            dq_sub = np.array(data.qvel[dof_adr], dtype=np.float64)
            torque = np.clip(kp * (target - q_sub) - kd * dq_sub, -tau_limit, tau_limit)
            data.ctrl[actuator_index] = torque
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qvel).all():
            print("NUMERICAL_DIVERGENCE at step", step, flush=True)
            break
        position = np.array(data.qpos[0:3], dtype=np.float64)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body, velocity, 1)
        rotation = rotation_from_wxyz(data.qpos[3:7])
        upright = float(rotation[2, 2])
        sole = (sole_height(0), sole_height(1))
        rows.append([step, step * policy_dt, command, phase, *position, *velocity[:3],
                     *velocity[3:6], *sole, upright, *q, *dq, *torque, *action])
        if position[2] < 0.15 and not fallen:
            fallen, fall_time = True, step * policy_dt
        if renderer is not None and step % 2 == 0 and video is not None:
            camera.lookat[:] = [position[0], position[1], max(0.4, position[2])]
            renderer.update_scene(data, camera=camera)
            video.write(renderer.render()[:, :, ::-1])
        if fallen:
            break

    if video is not None:
        video.release()
        if hasattr(renderer, "close"):
            renderer.close()
    if not rows:
        summary = {"numerical_divergence": True, "requested_vx": args.vx}
        (args.out_dir / "sim2sim_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                           encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    with (args.out_dir / "sim2sim.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    numeric = np.asarray(rows, dtype=np.float64)
    settled = numeric[numeric[:, 1] >= args.stand_seconds + 2.0]
    summary = {
        "checkpoint": str(args.checkpoint), "policy": str(args.policy),
        "requested_vx": args.vx, "stand_seconds": args.stand_seconds,
        "policy_dt": policy_dt, "decimation": decimation, "sim_dt": sim_dt,
        "phase_rate_hz": phase_rate,
        "duration_s": float(numeric[-1, 1] + policy_dt),
        "fallen": fallen, "fall_time_s": fall_time,
        "final_height_m": float(numeric[-1, 6]),
        "min_height_m": float(numeric[:, 6].min()),
        "max_upright": float(numeric[:, 15].max()),
        "distance_x_m": float(numeric[-1, 4] - numeric[0, 4]),
        "settled_mean_vx_m_s": float(settled[:, 7].mean()) if len(settled) else None,
        "settled_mean_abs_vy_m_s": float(np.abs(settled[:, 8]).mean()) if len(settled) else None,
        "settled_mean_abs_wz_rad_s": float(np.abs(settled[:, 12]).mean()) if len(settled) else None,
        "torque_saturation_fraction": float(np.mean(
            np.abs(numeric[:, -2 * action_dim:-action_dim]) >= tau_limit - 1e-6)),
        "sole_p95_m": {"left": float(np.percentile(settled[:, 13], 95)) if len(settled) else None,
                       "right": float(np.percentile(settled[:, 14], 95)) if len(settled) else None},
    }
    (args.out_dir / "sim2sim_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                       encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
