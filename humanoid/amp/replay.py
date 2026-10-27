"""Opt-in video and terminal-state diagnostics; never imported by training."""
import csv
import os
import subprocess
import json
import math

import numpy as np


class ReplayRecorder:
    def __init__(self, env, runner, output_dir, render=True, metadata=None):
        self.env, self.runner = env, runner
        self.output_dir = output_dir
        self.metadata = metadata or {}
        self.render = render
        self.width = getattr(env.cfg.env, "camera_width", 1920)
        self.height = getattr(env.cfg.env, "camera_height", 1080)
        self.stride = 2
        self.rows = []
        self.episode_id = self.clip_count = 0
        self.last_yaw = None
        self.yaw_accum = 0.
        self.video = None
        os.makedirs(output_dir, exist_ok=True)
        if any(os.path.exists(os.path.join(output_dir, name)) for name in ("isaac_diag.csv", "play_output.mp4")):
            raise FileExistsError("Use a new output_dir to preserve earlier replay artifacts")
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        self.feet_names = [body_names[index] for index in env.feet_indices.tolist()]
        self.left = next(i for i, name in enumerate(self.feet_names) if name.startswith("left_"))
        self.right = next(i for i, name in enumerate(self.feet_names) if name.startswith("right_"))
        if render:
            from isaacgym import gymapi
            import cv2
            self.cv2 = cv2
            actor = env.gym.get_actor_handle(env.envs[0], 0)
            body = env.gym.get_actor_rigid_body_handle(env.envs[0], actor, 0)
            rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
            env.gym.attach_camera_to_body(env.camera_handle, env.envs[0], body,
                                         gymapi.Transform(gymapi.Vec3(2., -2., 1.5), rotation), gymapi.FOLLOW_POSITION)
            self.video = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", "{}x{}".format(self.width, self.height), "-r", str(1 / (env.dt * self.stride)), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", os.path.join(output_dir, "play_output.mp4")], stdin=subprocess.PIPE)
        self.stream = open(os.path.join(output_dir, "isaac_diag.csv"), "w", newline="")
        self.writer = None
        self.frames = 0

    def write(self, step, reward, done, info, requested=0., stage=0):
        env = self.env
        state = {k: v[0].detach().cpu().numpy().reshape(-1) for k, v in info["replay_state"].items()}
        style_score = self.runner.discriminator.style_reward(info["amp_obs"])[0].item()
        style = style_score * self.runner.disc_cfg["style_reward_scale"] * env.dt
        style_raw = style
        multiplier = float(info["amp_style_multiplier"][0]) if "amp_style_multiplier" in info else 1.
        style *= multiplier
        weight = self.runner.disc_cfg["task_reward_weight"]
        row = dict(step=step, time_s=(step + 1) * env.dt, done=int(done[0].item()),
                   timeout=int(state["timeout"][0]), episode_step=int(state["episode_step"][0]),
                   task_reward=reward[0].item(), style_reward=style, style_score=style_score,
                   style_reward_raw=style_raw, amp_style_multiplier=multiplier,
                   mixed_reward=weight * reward[0].item() + (1 - weight) * style)
        for key, labels in (("root", ["x", "y", "z", "qx", "qy", "qz", "qw", "world_vx", "world_vy", "world_vz", "world_wx", "world_wy", "world_wz"]),
                            ("velocity", ["vx", "vy", "vz"]), ("omega", ["wx", "wy", "wz"]),
                            ("gravity", ["gx", "gy", "gz"])):
            row.update(zip(labels, state[key].tolist()))
        for label, value in zip(("cmd_vx", "cmd_vy", "cmd_wz"), state["command"].tolist()):
            row[label] = value
        for key in ("q", "dq", "torque", "action", "target"):
            row.update((key + "_" + name, float(value)) for name, value in zip(env.dof_names, state[key]))
        if "raw_target" in state:
            row.update(standing_reset=int(state["standing_reset"][0]), stand_steps_left=int(state["stand_steps_left"][0]))
            for key, prefix in (("raw_target", "pd_target_raw"), ("effective_target", "pd_target_effective"),
                                ("target_clip_counts", "pd_target_clip_counts")):
                row.update((prefix+"_"+name, float(value)) for name, value in zip(env.dof_names, state[key]))
        # Compatible names used by the supplied play.py and diagnostic tools.
        qx, qy, qz, qw = state["root"][3:7]
        roll = math.atan2(2 * (qw*qx + qy*qz), 1 - 2 * (qx*qx + qy*qy))
        pitch = math.asin(np.clip(2 * (qw*qy - qz*qx), -1, 1))
        yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        if self.last_yaw is None:
            self.last_yaw = yaw
        self.yaw_accum += math.atan2(math.sin(yaw-self.last_yaw), math.cos(yaw-self.last_yaw))
        self.last_yaw = yaw
        row.update(active_stage=stage, command_requested=requested, command_x=row["cmd_vx"],
                   command_applied=row["cmd_vx"], command_observed=float(state["command_observed"][0]),
                   cmd_linear_x=row["cmd_vx"], cmd_linear_y=row["cmd_vy"], cmd_angular_z=row["cmd_wz"],
                   base_vel_x=row["vx"], base_vel_y=row["vy"], base_vel_z=row["vz"],
                   base_ang_vel_x=row["wx"], base_ang_vel_y=row["wy"], base_ang_vel_z=row["wz"],
                   base_euler_x=roll, base_euler_y=pitch, base_euler_z=yaw, base_yaw=yaw,
                   base_height=row["z"], base_pos_x=row["x"], base_pos_y=row["y"],
                   yaw_from_episode_start=self.yaw_accum, episode_id=self.episode_id, terminal_snapshot=1)
        clips = int(state["torque_clip_step"][0])
        self.clip_count += clips
        row.update(torque_clip_step=clips, clip_count=self.clip_count,
                   action_clip_step=int((np.abs(state["action"]) >= env.cfg.normalization.clip_actions-1e-6).sum()))
        for key, prefix in (("q", "pos"), ("dq", "vel"), ("torque", "effort"), ("target", "pos_des_raw")):
            row.update((prefix + "_" + name, float(value)) for name, value in zip(env.dof_names, state[key]))
        if "effective_target" in state:
            row.update(("pos_des_effective_"+name, float(value)) for name, value in zip(env.dof_names, state["effective_target"]))
        for index in range(2):
            row["foot{}_contact_n".format(index)] = float(np.linalg.norm(state["contact"].reshape(2, 3)[index]))
            row["foot{}_z".format(index)] = float(state["feet"].reshape(2, 13)[index, 2])
        for index, side in ((self.left, "l"), (self.right, "r")):
            for axis, suffix in enumerate(("x", "y", "z")):
                row["foot_" + suffix + "_" + side] = float(state["feet"].reshape(2, 13)[index, axis])
            row["foot_force_" + side] = float(state["contact"].reshape(2, 3)[index, 2])
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows.append(row)
        if row["done"]:
            self.episode_id += 1
            self.last_yaw = None
            self.yaw_accum = 0.
        if not self.render or (step + 1) % self.stride:
            return
        # Render the current actor (an auto-reset actor on terminal steps).
        from isaacgym import gymapi
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        frame = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR)
        img = self.cv2.cvtColor(np.asarray(frame, dtype=np.uint8).reshape(self.height, self.width, 4), self.cv2.COLOR_RGBA2BGR)
        self.overlay(img, row)
        self.video.stdin.write(img.tobytes())
        self.frames += 1

    def close(self):
        self.stream.close()
        if self.video:
            self.video.stdin.close()
            if self.video.wait() != 0:
                raise RuntimeError("Video encoder failed")
        summary = []
        for stage in sorted({row["active_stage"] for row in self.rows}):
            rows = [row for row in self.rows if row["active_stage"] == stage]
            stats = dict(stage=stage, steps=len(rows), requested=rows[0]["command_requested"],
                         average_command=float(np.mean([r["command_applied"] for r in rows])),
                         average_vx=float(np.mean([r["base_vel_x"] for r in rows])),
                         rmse_vx=float(np.sqrt(np.mean([(r["base_vel_x"]-r["command_applied"])**2 for r in rows]))),
                         resets=sum(r["done"] for r in rows), timeouts=sum(r["timeout"] for r in rows))
            summary.append(stats)
            print("Segment {stage}: cmd={requested:.2f} | applied={average_command:.3f} | real={average_vx:.3f} | RMSE={rmse_vx:.3f} | resets={resets} (timeouts={timeouts})".format(**stats))
        metadata = dict(self.metadata, control_dt=self.env.dt, physics_dt=self.env.cfg.sim.dt,
                        video_fps=1 / (self.env.dt*self.stride), video_frames=self.frames,
                        video_enabled=self.render, captured_steps=len(self.rows),
                        resets=sum(r["done"] for r in self.rows), timeouts=sum(r["timeout"] for r in self.rows),
                        torque_clip_count=self.clip_count, torque_clip_scope="joint saturation events across all physics substeps",
                        segments=summary)
        with open(os.path.join(self.output_dir, "replay_metadata.json"), "w") as stream:
            json.dump(metadata, stream, indent=2)
        print("Replay files closed: {} frames".format(self.frames), flush=True)

    def overlay(self, image, row):
        left, right = row["foot_force_l"] > 1., row["foot_force_r"] > 1.
        state = "DOUBLE SUPPORT" if left and right else "FLIGHT PHASE" if not left and not right else "SINGLE SUPPORT"
        avg = np.mean([r["base_vel_x"] for r in self.rows])
        lines = [
            ("CMD: {:.2f} | REAL: {:.2f} | AVG: {:.2f}".format(row["command_applied"], row["vx"], avg), (255, 255, 0)),
            ("VEL_Y: {:+.2f} | VEL_Z: {:+.2f} | VEL_YAW: {:+.2f}".format(row["vy"], row["vz"], row["wz"]), (0, 255, 255)),
            ("L-FOOT: {} ({:.1f} N)".format("ON" if left else "OFF", row["foot_force_l"]), (0, 255, 0) if left else (0, 0, 255)),
            ("R-FOOT: {} ({:.1f} N)".format("ON" if right else "OFF", row["foot_force_r"]), (0, 255, 0) if right else (0, 0, 255)),
            ("STATE: " + state, (0, 255, 255)),
            ("STAGE: {} | REQUEST: {:.2f} | t: {:.2f}s | RESET: {}".format(row["active_stage"], row["command_requested"], row["time_s"], row["done"]), (255, 255, 255)),
        ]
        scale = self.width / 1920.
        for index, (text, color) in enumerate(lines):
            position = (int(770*scale), int((60+50*index)*scale))
            self.cv2.putText(image, text, position, self.cv2.FONT_HERSHEY_SIMPLEX, .9*scale, (0, 0, 0), 4, self.cv2.LINE_AA)
            self.cv2.putText(image, text, position, self.cv2.FONT_HERSHEY_SIMPLEX, .9*scale, color, 2, self.cv2.LINE_AA)
