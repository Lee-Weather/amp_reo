"""Bounded X1 AMP playback with explicit commands and optional JIT export."""
import argparse
import json
import os
import sys
from datetime import datetime

import isaacgym  # Must be imported before torch.
import torch

from humanoid.envs import *
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.utils import get_args, task_registry
from humanoid.amp.playback import speed_profile, command_at, inject_command


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, help="Optional limit; default is the full six-stage profile")
    parser.add_argument("--vx", type=float, help="Use a constant speed instead of the six-stage profile")
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--yaw_rate", type=float, default=0.)
    parser.add_argument("--export_policy", type=str)
    parser.add_argument("--output_dir", type=str, help="Record video and pre-reset diagnostics for env 0")
    parser.add_argument("--no_video", action="store_true", help="Still write CSV, summary and metadata")
    parser.add_argument("--stage_seconds", type=float, default=10.)
    parser.add_argument("--stand_buffer_seconds", type=float, default=1.5)
    parser.add_argument("--checkpoint_path", type=str, help="Load a local checkpoint without optimizer states")
    parser.add_argument("--reset_mode", choices=("standing", "rsi"), default="standing")
    play, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.task not in ("x1_amp", "x1_amp_exp11"):
        raise ValueError("Use --task=x1_amp or --task=x1_amp_exp11")
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs or 1
    if cfg.env.num_envs != 1:
        raise ValueError("Diagnostic playback requires --num_envs=1")
    render = not play.no_video and os.environ.get("PLAY_RENDER", "1") != "0"
    cfg.env.enable_camera_sensors = render
    cfg.env.camera_width, cfg.env.camera_height = 1920, 1080
    cfg.env.capture_replay_state = True
    if play.stand_buffer_seconds < 0:
        raise ValueError("stand_buffer_seconds must be non-negative")
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.amp.reference_state_initialization = play.reset_mode == "rsi"
    shared_buffer = args.task == "x1_amp_exp11"
    if shared_buffer:
        cfg.amp.rsi_probability = 1. if play.reset_mode == "rsi" else 0.
        cfg.amp.stand_buffer_seconds = play.stand_buffer_seconds
    initial_vx = play.vx if play.vx is not None else 0.
    cfg.commands.ranges.lin_vel_x = [initial_vx, initial_vx]
    cfg.commands.ranges.lin_vel_y = [play.vy, play.vy]
    cfg.commands.ranges.ang_vel_yaw = [play.yaw_rate, play.yaw_rate]
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if play.checkpoint_path:
        args.resume = False
        train_cfg.runner.resume = False
        runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
        runner.load(os.path.abspath(play.checkpoint_path), load_optimizer=False)
    else:
        train_cfg.runner.resume = True
        runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root="default")
    policy = runner.get_inference_policy(device=env.device)
    profile = speed_profile(env.dt, play.stage_seconds)
    if play.vx is not None:
        profile = [(sum(count for count, _ in profile), play.vx)]
    total_steps = play.steps if play.steps is not None else sum(count for count, _ in profile)
    if play.vx is not None:
        profile = [(total_steps, play.vx)]
    if total_steps <= 0:
        raise ValueError("steps must be positive")
    if play.export_policy:
        target = os.path.abspath(play.export_policy)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # Export a copy so playback remains on the simulator's device.
        import copy
        torch.jit.script(copy.deepcopy(runner.actor_critic.actor).cpu()).save(target)
        metadata = runner._metadata()
        metadata.update({"observation_frame": ["command_vx_vy_wz", "base_angular_velocity",
                                               "projected_gravity", "joint_position_minus_default",
                                               "joint_velocity", "previous_action"],
                         "history_order": "oldest_to_newest", "frame_stack": cfg.env.frame_stack,
                         "obs_scales": {key: getattr(env.obs_scales, key)
                                        for key in ("lin_vel", "ang_vel", "dof_pos", "dof_vel")},
                         "clip_actions": cfg.normalization.clip_actions,
                         "clip_observations": cfg.normalization.clip_observations,
                         "stiffness": env.p_gains.tolist(), "damping": env.d_gains.tolist(),
                         "torque_limits": env.torque_limits.tolist()})
        with open(target+".json", "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
        print("Exported policy:", target)
    from humanoid.amp.replay import ReplayRecorder
    output_dir = play.output_dir or os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "play_output",
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    recorder = ReplayRecorder(env, runner, output_dir, render=render, metadata={
        "task": args.task, "seed": cfg.seed, "profile": profile, "steps": total_steps,
        "checkpoint": runner.current_learning_iteration, "load_run": args.load_run,
        "checkpoint_path": os.path.abspath(play.checkpoint_path) if play.checkpoint_path else None,
        "stand_buffer_seconds": play.stand_buffer_seconds,
        "pushes": cfg.domain_rand.push_robots, "noise": cfg.noise.add_noise,
        "reference_state_initialization": cfg.amp.reference_state_initialization,
        "reset_mode": play.reset_mode, "policy_metadata": runner._metadata(),
    })
    stand_buffer = 0
    try:
        with torch.no_grad():
            for step in range(total_steps):
                stage, requested = command_at(step, profile)
                applied = 0. if stand_buffer and not shared_buffer else requested
                obs = inject_command(env, applied, 0. if stand_buffer else play.vy,
                                     0. if stand_buffer else play.yaw_rate)
                obs, _, reward, done, info = env.step(policy(obs))
                recorder.write(step, reward, done, info, requested=requested, stage=stage)
                if done[0] and not shared_buffer:
                    stand_buffer = round(play.stand_buffer_seconds / env.dt)
                elif stand_buffer:
                    stand_buffer -= 1
                if (step+1) % 100 == 0:
                    print("step={} mean_reward={:.5f} resets={}".format(step+1, reward.mean().item(), done.sum().item()), flush=True)
    finally:
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()
