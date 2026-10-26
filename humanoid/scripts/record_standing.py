"""Film the exp_1.1 fixed-PD gate, including failures before any teleport.

This diagnostic suppresses automatic resets, but preserves termination checks
and records the FIRST failure. Physics continues afterwards only to show the
fall; those frames cannot turn a failed trial into a pass. No policy is loaded.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import isaacgym
from isaacgym import gymapi
import cv2
import numpy as np
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def angles(root):
    x, y, z, w = root[:, 3:7].T
    return np.rad2deg(np.column_stack((
        np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))))


def text(frame, message, xy, color=(235, 235, 235), scale=.75):
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def main():
    # Keep the framework's --seed distinct from this script's --seeds.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 7, 11])
    parsed, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+rest
    args = get_args()
    if args.task != "x1_amp_exp11":
        raise ValueError("Use --task=x1_amp_exp11")
    if parsed.seconds <= 0:
        raise ValueError("seconds must be positive")
    os.makedirs(parsed.output_dir, exist_ok=True)
    video_path = os.path.join(parsed.output_dir, "fixed_pd_standing.mp4")
    if os.path.exists(video_path):
        raise FileExistsError(video_path)
    cfg, _ = task_registry.get_cfgs(args.task)
    args.num_envs = cfg.env.num_envs = 3
    cfg.noise.add_noise = False
    cfg.amp.rsi_probability = 0.
    cfg.env.capture_replay_state = True
    cfg.env.enable_camera_sensors = True
    cfg.env.camera_width, cfg.env.camera_height = 960, 540
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    env.command_ranges = {k: [0., 0.] for k in
                          ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
    camera_properties = gymapi.CameraProperties()
    camera_properties.width, camera_properties.height = 960, 540
    cameras = [env.camera_handle] + [
        env.gym.create_camera_sensor(env.envs[i], camera_properties)
        for i in (1, 2)]
    if any(camera < 0 for camera in cameras):
        raise RuntimeError("Camera creation failed")
    # The exact fixed reference target used by validate_exp11 --phase static.
    action = ((env.standing_state[13:42]-env.default_dof_pos)
              / cfg.control.action_scale).expand(3, -1)
    if action.abs().max() > cfg.normalization.clip_actions:
        raise ValueError("The fixed standing target would be action-clipped")
    reset_idx = env.reset_idx
    fps, stride = 25, round(1/(25*env.dt))
    if abs(stride*env.dt-1/fps) > 1e-6:
        raise ValueError("Camera timing requires a 25 Hz divisor of control rate")
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        video_path], stdin=subprocess.PIPE)
    summaries, frame_count = [], 0

    def compose(seed, time_s, root, failures, end=False):
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        canvas = np.full((1080, 1920, 3), 24, dtype=np.uint8)
        attitude = angles(root)
        for i, camera in enumerate(cameras):
            raw = env.gym.get_camera_image(env.sim, env.envs[i], camera, gymapi.IMAGE_COLOR)
            panel = cv2.cvtColor(np.asarray(raw, dtype=np.uint8).reshape(540, 960, 4), cv2.COLOR_RGBA2BGR)
            failure = failures[i]
            status = "FAIL @ {:.2f}s".format(failure["time_s"]) if failure else (
                "PASS" if end else "RUNNING")
            color = (80, 80, 255) if failure else (80, 230, 120)
            cv2.rectangle(panel, (0, 0), (960, 98), (22, 22, 22), -1)
            text(panel, "seed={}  env={}  t={:.2f}s  {}".format(seed, i, time_s, status), (18, 34), color)
            text(panel, "h={:.3f}m  roll={:+.1f}deg  pitch={:+.1f}deg".format(
                root[i, 2], *attitude[i]), (18, 72), scale=.7)
            y, x = (i//2)*540, (i % 2)*960
            canvas[y:y+540, x:x+960] = panel
        lines = [
            ("exp_1.1  FIXED PD STANDING ACCEPTANCE", (235, 235, 235)),
            ("No learned policy | 29 controlled joints", (235, 235, 235)),
            ("Seed {} | 3 perturbations | {:.2f}/{:.2f}s".format(seed, time_s, parsed.seconds), (235, 235, 235)),
            ("B1 frame 11 | joint noise +/-0.03 rad", (235, 235, 235)),
            ("Original gains, gravity and failure thresholds", (235, 235, 235)),
            ("Ground clearance +2mm | command = 0", (235, 235, 235)),
            ("FAIL = first original termination signal", (80, 180, 255)),
            ("Automatic resets suppressed for this video", (80, 180, 255)),
            ("After FAIL: continuing fall is visual only", (80, 180, 255)),
            ("No reset or post-failure recovery counts as PASS", (80, 180, 255)),
        ]
        for j, (message, color) in enumerate(lines):
            text(canvas, message, (985, 585+j*43), color, .69)
        return canvas

    try:
        with open(os.path.join(parsed.output_dir, "standing_diag.csv"), "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "seed", "env", "time_s", "height_m", "roll_deg", "pitch_deg",
                "termination", "timeout", "first_failure_s", "nonfoot_contact_n",
                "effective_target_outside_hard_limits"])
            writer.writeheader()
            for seed in parsed.seeds:
                env.reset_idx = reset_idx
                torch.manual_seed(seed)
                env.reset()
                env.set_evaluation_command(0.)
                root = env.root_states.detach().cpu().numpy().copy()
                for i, camera in enumerate(cameras):
                    origin = env.gym.get_env_origin(env.envs[i])
                    local = root[i, :3]-np.array([origin.x, origin.y, origin.z])
                    eye = gymapi.Vec3(float(local[0]+1.35), float(local[1]-1.65), 1.45)
                    target = gymapi.Vec3(float(local[0]), float(local[1]), .70)
                    env.gym.set_camera_location(camera, env.envs[i], eye, target)
                failures = [None]*3
                # A recording-only hook: termination is still computed normally.
                # Restore the bound reset method before initializing each trial.
                env.reset_idx = lambda env_ids: None
                for step in range(round(parsed.seconds/env.dt)):
                    _, _, _, done, info = env.step(action)
                    root = info["replay_state"]["root"].detach().cpu().numpy()
                    if not np.isfinite(root).all() or not torch.isfinite(env.dof_pos).all():
                        raise RuntimeError("Nonfinite standing state")
                    time_s = (step+1)*env.dt
                    attitude = angles(root)
                    nonfoot = env.contact_forces[:, env.termination_contact_indices].norm(dim=-1).max(dim=-1).values.cpu().numpy()
                    outside = ((env.pd_target_effective < env.hard_dof_limits[:, 0]-1e-6)
                               | (env.pd_target_effective > env.hard_dof_limits[:, 1]+1e-6)).sum(-1).cpu().numpy()
                    if outside.any():
                        raise RuntimeError("Effective PD target outside hard limits")
                    for i in range(3):
                        if bool(done[i]) and failures[i] is None:
                            failures[i] = dict(time_s=round(time_s, 4), height_m=float(root[i, 2]),
                                               roll_deg=float(attitude[i, 0]), pitch_deg=float(attitude[i, 1]),
                                               nonfoot_contact_n=float(nonfoot[i]),
                                               timeout=bool(info["replay_state"]["timeout"][i]))
                        writer.writerow(dict(seed=seed, env=i, time_s=time_s, height_m=float(root[i, 2]),
                                             roll_deg=float(attitude[i, 0]), pitch_deg=float(attitude[i, 1]),
                                             termination=int(done[i]), timeout=int(info["replay_state"]["timeout"][i]),
                                             first_failure_s="" if failures[i] is None else failures[i]["time_s"],
                                             nonfoot_contact_n=float(nonfoot[i]),
                                             effective_target_outside_hard_limits=int(outside[i])))
                    if (step+1) % stride == 0:
                        frame = compose(seed, time_s, root, failures)
                        # Isaac updates rendered link transforms after simulation;
                        # a pre-simulation t=0 image can still show the spawn pose.
                        if step+1 == stride:
                            if not summaries:
                                cv2.imwrite(os.path.join(parsed.output_dir, "preview_start.jpg"), frame)
                            for _ in range(fps):
                                encoder.stdin.write(frame.tobytes())
                                frame_count += 1
                        encoder.stdin.write(frame.tobytes())
                        frame_count += 1
                final = compose(seed, time_s, root, failures, end=True)
                for _ in range(fps):
                    encoder.stdin.write(final.tobytes())
                    frame_count += 1
                summary = dict(seed=seed, passed=sum(f is None for f in failures), trials=3, first_failures=failures)
                summaries.append(summary)
                print("STANDING_RESULT "+json.dumps(summary), flush=True)
    finally:
        env.reset_idx = reset_idx
        encoder.stdin.close()
        returncode = encoder.wait()
        env.gym.destroy_sim(env.sim)
    if returncode:
        raise RuntimeError("ffmpeg failed with {}".format(returncode))
    metadata = dict(task=args.task, controller="fixed_reference_PD_no_policy", auto_reset=False,
                    post_failure_frames="visualization only; first failure is latched",
                    seconds_per_trial=parsed.seconds, fps=fps, frames=frame_count,
                    trials=summaries, standing_control=env.checkpoint_control_metadata())
    with open(os.path.join(parsed.output_dir, "standing_metadata.json"), "w") as stream:
        json.dump(metadata, stream, indent=2)
    print("STANDING_VIDEO_OK frames={} path={}".format(frame_count, video_path), flush=True)


if __name__ == "__main__":
    main()
