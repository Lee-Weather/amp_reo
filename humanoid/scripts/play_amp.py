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
    if args.task not in ("x1_amp", "x1_amp_exp11", "x1_amp_exp3", "x1_amp_exp14", "x1_amp_exp15", "x1_amp_exp16", "x1_amp_exp17", "x1_amp_exp18", "x1_amp_exp19", "x1_amp_exp110", "x1_amp_exp111", "x1_amp_exp112", "x1_amp_exp113", "x1_amp_exp113t"):
        raise ValueError("Use a registered x1_amp task")
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    # The shared helper applies --seed only to train_cfg, not env_cfg.
    # Set it before make_env() seeds the simulator and reset perturbations.
    if args.seed is not None:
        cfg.seed = args.seed
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
    shared_buffer = args.task in ("x1_amp_exp11", "x1_amp_exp3", "x1_amp_exp14", "x1_amp_exp15", "x1_amp_exp16", "x1_amp_exp17", "x1_amp_exp18", "x1_amp_exp19", "x1_amp_exp110", "x1_amp_exp111", "x1_amp_exp112", "x1_amp_exp113", "x1_amp_exp113t")
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
        torch.jit.script(copy.deepcopy(runner.actor_critic.get_policy_module()).cpu()).save(target)
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
    dynamic_tracking = hasattr(cfg.rewards, "tracking_min_std")
    recorder = ReplayRecorder(env, runner, output_dir, render=render, metadata={
        "task": args.task, "seed": cfg.seed, "profile": profile, "steps": total_steps,
        "checkpoint": runner.current_learning_iteration, "load_run": args.load_run,
        "checkpoint_path": os.path.abspath(play.checkpoint_path) if play.checkpoint_path else None,
        "stand_buffer_seconds": play.stand_buffer_seconds,
        "style_reward_scale": runner.disc_cfg["style_reward_scale"],
        "task_reward_weight": runner.disc_cfg["task_reward_weight"],
        "tracking_lin_sigma": None if dynamic_tracking else getattr(cfg.rewards, "tracking_lin_sigma", cfg.rewards.tracking_sigma),
        "tracking_lin_rule": ({"type": "speed_dependent_std_squared",
                               "minimum_std": cfg.rewards.tracking_min_std,
                               "maximum_std": cfg.rewards.tracking_max_std,
                               "maximum_speed": cfg.rewards.tracking_max_speed} if dynamic_tracking else {"type": "fixed_denominator"}),
        "standing_style_multiplier": getattr(cfg.amp, "standing_style_multiplier", 1.),
        "motion_style_velocity_sigma": None if args.task in ("x1_amp_exp113", "x1_amp_exp113t") else getattr(cfg.amp, "motion_style_velocity_sigma", None),
        "moving_style_rule": "ungated_v1" if args.task in ("x1_amp_exp113", "x1_amp_exp113t") else "inherited",
        "motion_progress_speed_scale": getattr(cfg.rewards, "motion_velocity_scale", None),
        "training_motion_exploration": getattr(train_cfg.policy, "motion_exploration", None),
        "tracking_ang_sigma": cfg.rewards.tracking_sigma,
        "tracking_lateral_std": getattr(cfg.rewards,"tracking_lateral_std",None),
        "foot_scuff_velocity_reference": "lowest_material_point_from_COM" if args.task in ("x1_amp_exp17", "x1_amp_exp18", "x1_amp_exp19", "x1_amp_exp110", "x1_amp_exp111", "x1_amp_exp112", "x1_amp_exp113", "x1_amp_exp113t") else "rigid_body_COM",
        "backward_reward_speed_scale": getattr(cfg.rewards,"backward_speed_scale",None),
        "braking_leg_gate_scales": ({"linear":cfg.rewards.brake_leg_linear_scale,"angular":cfg.rewards.brake_leg_angular_scale}
                                    if hasattr(cfg.rewards,"brake_leg_linear_scale") else None),
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
