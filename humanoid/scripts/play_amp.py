"""Bounded X1 AMP playback with explicit commands and optional JIT export."""
import argparse
import hashlib
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
    parser.add_argument('--source_policy',action='store_true',help='exp2.0 only: evaluate the verified7000 source via policy migration, not target checkpoint resume')
    parser.add_argument("--reset_mode", choices=("standing", "rsi"), default="standing")
    parser.add_argument("--transition_trace", type=str,
                        help="Capture an exact nominal startup/action trace; stop before crossing a reset")
    play, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.task not in ("x1_amp", "x1_amp_base"):
        raise ValueError("Use --task=x1_amp or --task=x1_amp_base")
    if play.transition_trace:
        if (args.task != "x1_amp" or not play.checkpoint_path or play.reset_mode != "standing"
                or play.vx is None or play.vx <= 0 or play.vy != 0 or play.yaw_rate != 0):
            raise ValueError("Transition capture requires x1_amp, checkpoint_path, standing reset and native forward vx")
        if os.path.exists(play.transition_trace):
            raise FileExistsError(play.transition_trace)
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    if play.source_policy:
        raise ValueError("--source_policy belonged to the removed exp2.0 task")
    if args.task == "x1_amp" and play.vx is not None and abs(play.vx)>1e-7:
        if abs(play.vx-cfg.commands.native_speed)>.002:
            raise ValueError("x1_amp uses its native forward speed; no other steady walking speeds")
        play.vx = cfg.commands.native_speed
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
    cfg.amp.reference_state_initialization = play.reset_mode == "rsi" and args.task != "x1_amp"
    if play.transition_trace:
        # A matched-state probe uses a declared nominal condition, independently
        # of the training cohort size (25% of a single environment rounds to 0).
        from humanoid.amp.nominal_startup import configure_nominal_probe
        configure_nominal_probe(cfg)
    shared_buffer = False  # Historical exp1.x shared-buffer tasks were removed.
    initial_vx = play.vx if play.vx is not None else 0.
    cfg.commands.ranges.lin_vel_x = [initial_vx, initial_vx]
    cfg.commands.ranges.lin_vel_y = [play.vy, play.vy]
    cfg.commands.ranges.ang_vel_yaw = [play.yaw_rate, play.yaw_rate]
    if args.task == "x1_amp":
        cfg.commands.guidance_evaluation = True
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if args.task == "x1_amp":
        env.evaluation_reset_mode = play.reset_mode
    if play.checkpoint_path:
        args.resume = False
        train_cfg.runner.resume = False
        runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root=None)
        if play.source_policy:
            runner.warm_start(os.path.abspath(play.checkpoint_path))
        else:
            runner.load(os.path.abspath(play.checkpoint_path), load_optimizer=False)
    else:
        train_cfg.runner.resume = True
        runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root="default")
    policy = runner.get_inference_policy(device=env.device)
    profile = speed_profile(env.dt, play.stage_seconds)
    if args.task == "x1_amp" and play.vx is None:
        count = max(1, round(play.stage_seconds/env.dt))
        profile = [(count, 0.), (count, cfg.commands.native_speed), (count, 0.)]
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
        if getattr(cfg.env, "append_gait_phase", False):
            metadata["observation_frame"] += ["sin_phase", "cos_phase"]
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
        'source_policy_baseline':play.source_policy,
        'evaluation_friction_range':None,
        'training_origin':runner.training_origin,
        "stand_buffer_seconds": play.stand_buffer_seconds,
        "style_reward_scale": runner.disc_cfg["style_reward_scale"],
        "task_reward_weight": runner.disc_cfg["task_reward_weight"],
        "tracking_lin_sigma": None if dynamic_tracking else getattr(cfg.rewards, "tracking_lin_sigma", cfg.rewards.tracking_sigma),
        "tracking_lin_rule": ({"type": "speed_dependent_std_squared",
                               "minimum_std": cfg.rewards.tracking_min_std,
                               "maximum_std": cfg.rewards.tracking_max_std,
                               "maximum_speed": cfg.rewards.tracking_max_speed} if dynamic_tracking else {"type": "fixed_denominator"}),
        "standing_style_multiplier": getattr(cfg.amp, "standing_style_multiplier", 1.),
        "motion_style_velocity_sigma": None if args.task == "x1_amp" else getattr(cfg.amp, "motion_style_velocity_sigma", None),
        "moving_style_rule": ('native_category_mask_v1' if args.task=='x1_amp' else "inherited"),
        'command_transition_seconds':getattr(cfg.commands,'transition_seconds',None),
        'reference_acceptance':getattr(env,'reference_report',None),
        'transition_acceptance':getattr(env,'transition_report',None),
        "motion_progress_speed_scale": getattr(cfg.rewards, "motion_velocity_scale", None),
        "training_motion_exploration": getattr(train_cfg.policy, "motion_exploration", None),
        "tracking_ang_sigma": cfg.rewards.tracking_sigma,
        "tracking_lateral_std": getattr(cfg.rewards,"tracking_lateral_std",None),
        "foot_scuff_velocity_reference": "lowest_material_point_from_COM" if args.task == "x1_amp" else "rigid_body_COM",
        "backward_reward_speed_scale": getattr(cfg.rewards,"backward_speed_scale",None),
        "braking_leg_gate_scales": ({"linear":cfg.rewards.brake_leg_linear_scale,"angular":cfg.rewards.brake_leg_angular_scale}
                                    if hasattr(cfg.rewards,"brake_leg_linear_scale") else None),
        "pushes": cfg.domain_rand.push_robots, "noise": cfg.noise.add_noise,
        "reference_state_initialization": cfg.amp.reference_state_initialization,
        "reset_mode": play.reset_mode, "policy_metadata": runner._metadata(),
    })
    stand_buffer = 0
    trace = None
    try:
        with torch.no_grad():
            for step in range(total_steps):
                stage, requested = command_at(step, profile)
                applied = 0. if stand_buffer and not shared_buffer else requested
                obs = inject_command(env, applied, 0. if stand_buffer else play.vy,
                                     0. if stand_buffer else play.yaw_rate)
                before = obs[0].detach().cpu().numpy().copy() if play.transition_trace else None
                if play.transition_trace and trace is None:
                    from humanoid.amp.transition_trace import TransitionTrace
                    props = env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])
                    if not props or any(abs(p.friction-.8)>1e-6 or abs(p.contact_offset-.001)>1e-7
                                        or abs(p.rest_offset)>1e-7 for p in props):
                        raise ValueError("Transition capture nominal shape properties did not persist")
                    root = env.root_states[0].detach().cpu().numpy().copy()
                    root[:3] -= env.env_origins[0].detach().cpu().numpy()
                    with open(play.checkpoint_path, "rb") as stream:
                        digest = hashlib.sha256(stream.read()).hexdigest()
                    trace = TransitionTrace(dict(engine="isaacgym", checkpoint_sha256=digest,
                        dof_names=list(env.dof_names), num_observations=env.num_obs, policy_dt=float(env.dt),
                        physics_dt=float(cfg.sim.dt), initial_phase=float(env.gait_phase[0]),
                        requested_vx=play.vx, friction=.8, contact_offset_m=.001,
                        root_velocity_reference="engine_reported_world"), root,
                        env.dof_pos[0].detach().cpu().numpy(), env.dof_vel[0].detach().cpu().numpy(), before)
                obs, _, reward, done, info = env.step(policy(obs))
                recorder.write(step, reward, done, info, requested=requested, stage=stage)
                if trace is not None:
                    state = {key: value[0].detach().cpu().numpy().copy() for key, value in info["replay_state"].items()}
                    state["root"][:3] -= env.env_origins[0].detach().cpu().numpy()
                    trace.append(before, state["action"], state["root"], state["q"], state["dq"],
                                 state["contact"], bool(done[0]))
                    if done[0]:
                        break
                if done[0] and not shared_buffer:
                    stand_buffer = round(play.stand_buffer_seconds / env.dt)
                elif stand_buffer:
                    stand_buffer -= 1
                if (step+1) % 100 == 0:
                    print("step={} mean_reward={:.5f} resets={}".format(step+1, reward.mean().item(), done.sum().item()), flush=True)
    finally:
        if recorder:
            recorder.close()
        if trace is not None and trace.rows:
            trace.save(play.transition_trace)


if __name__ == "__main__":
    main()
