"""Bounded X1 AMP playback with explicit commands and optional JIT export."""
import argparse
import json
import os
import sys

import isaacgym  # Must be imported before torch.
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--yaw_rate", type=float, default=0.)
    parser.add_argument("--export_policy", type=str)
    play, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    if args.task != "x1_amp":
        raise ValueError("Use --task=x1_amp for this script")
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs or 1
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.amp.reference_state_initialization = False
    cfg.commands.ranges.lin_vel_x = [play.vx, play.vx]
    cfg.commands.ranges.lin_vel_y = [play.vy, play.vy]
    cfg.commands.ranges.ang_vel_yaw = [play.yaw_rate, play.yaw_rate]
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg, log_root="default")
    policy = runner.get_inference_policy(device=env.device)
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
    with torch.no_grad():
        for step in range(play.steps):
            obs, _, reward, done, _ = env.step(policy(env.get_observations()))
            if (step+1) % 100 == 0:
                print("step={} mean_reward={:.5f} resets={}".format(step+1, reward.mean().item(), done.sum().item()), flush=True)


if __name__ == "__main__":
    main()
