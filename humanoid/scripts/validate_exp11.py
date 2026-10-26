"""Bounded Isaac Gym checks for exp_1.1; never starts a long training run."""
import argparse
import copy
import os
import sys
import tempfile
import json

import isaacgym
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.amp.runner import AMPOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", choices=("static", "smoke", "candidates", "timing"), required=True)
    parser.add_argument("--standing_frame", type=int)
    parser.add_argument("--standing_pose_json", type=str, help="Optional calibrated state for the static gate")
    parsed, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+rest
    args = get_args()
    if args.task != "x1_amp_exp11":
        raise ValueError("Use --task=x1_amp_exp11")
    cfg, training = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs or (3 if parsed.phase == "static" else 64)
    cfg.noise.add_noise = False
    cfg.env.capture_replay_state = True
    if parsed.standing_frame is not None:
        cfg.amp.standing_frame = parsed.standing_frame
    if parsed.phase in ("static", "candidates", "timing"):
        cfg.amp.rsi_probability = 0.
    if parsed.phase == "timing":
        cfg.commands.resampling_time = .2
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    if parsed.standing_pose_json:
        with open(parsed.standing_pose_json) as stream:
            pose = torch.tensor(json.load(stream), device=env.device)
        pose[:3] = env.motion_library.grounding.align(pose[:3], pose[3:7], pose[13:42])
        env.standing_state = pose
    assert torch.all(env.hard_dof_limits[:, 0] <= env.dof_pos_limits[:, 0])
    assert torch.all(env.hard_dof_limits[:, 1] >= env.dof_pos_limits[:, 1])
    if parsed.phase == "timing":
        from humanoid.amp.control import sync_policy_commands
        env.reset()
        env.standing_commands.remaining[:] = 3
        env.standing_commands.requested[:] = torch.tensor([.4, 0., 0.], device=env.device)
        env.commands[:] = env.standing_commands.applied()
        sync_policy_commands(env)
        for step in range(30):
            before = env.commands.clone()
            if step < 3:
                assert not before.any()
            _, _, _, _, info = env.step(torch.zeros_like(env.actions))
            snapshot = info["replay_state"]
            torch.testing.assert_close(snapshot["command"], before)
            torch.testing.assert_close(snapshot["command_observed"], before)
        env.episode_length_buf[:] = env.max_episode_length
        _, _, _, done, info = env.step(torch.zeros_like(env.actions))
        assert done.all() and info["time_outs"].all()
        assert (env.standing_commands.remaining == env.standing_commands.hold_steps).all()
        assert not env.commands.any()
        training.runner.resume = False
        runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=training, log_root=None)
        os.makedirs("logs/exp11_validation", exist_ok=True)
        runner.save("logs/exp11_validation/model_0.pt")
        print("EXP11_TIMING_OK buffer=3 resampling=.2s timeout_boundary=OK", flush=True)
        return
    if parsed.phase == "candidates":
        from humanoid.envs.base.legged_robot import LeggedRobot
        from humanoid.amp.motion import quat_mul
        candidates = [("F1_B1_stand_to_walk", f) for f in (5, 11, 20, 30)] + [("F1_B2_walk_to_stand", f) for f in (250, 260, 270, 276, 280)]
        repeats = env.num_envs//len(candidates)
        assert repeats >= 3
        labels = torch.arange(env.num_envs, device=env.device) % len(candidates)
        states = []
        for clip, frame in candidates:
            index = env.motion_library.names.index(clip)
            state = env.motion_library.state(torch.tensor([index], device=env.device), torch.tensor([frame/50.], device=env.device))[0]
            qx, qy, qz, qw = state[3:7]
            yaw = torch.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
            inverse = state.new_zeros(4)
            inverse[2], inverse[3] = torch.sin(-yaw/2), torch.cos(-yaw/2)
            state[3:7] = quat_mul(inverse, state[3:7])
            state[:2] = 0
            state[7:13] = 0
            state[42:] = 0
            states.append(state)
        states = torch.stack(states)[labels]
        target = states[:, 13:42].clone()
        torch.manual_seed(5)
        states[:, 13:42] += (2*torch.rand_like(target)-1)*.03
        states[:, 13:42] = states[:, 13:42].maximum(env.hard_dof_limits[:, 0]).minimum(env.hard_dof_limits[:, 1])
        env.command_ranges = {name: [0., 0.] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
        env._reset_reference = states
        env.reset_standing[:] = True
        env.standing_commands.remaining[:] = 75
        LeggedRobot.reset_idx(env, torch.arange(env.num_envs, device=env.device))
        env._reset_reference = None
        failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        action = (target-env.default_dof_pos)/cfg.control.action_scale
        for _ in range(round(5./env.dt)):
            _, _, _, done, _ = env.step(action)
            failed |= done.bool()
        for i, (clip, frame) in enumerate(candidates):
            ids = labels == i
            print("CANDIDATE {} {} pass={}/{}".format(clip, frame, int((~failed[ids]).sum()), int(ids.sum())), flush=True)
        return
    if parsed.phase == "static":
        env.command_ranges = {name: [0., 0.] for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
        for seed in (5, 7, 11):
            torch.manual_seed(seed)
            env.reset()
            action = ((env.standing_state[13:42]-env.default_dof_pos)/cfg.control.action_scale).expand(env.num_envs, -1)
            minimum = 1e9
            for _ in range(round(5./env.dt)):
                _, _, reward, done, info = env.step(action)
                minimum = min(minimum, info["replay_state"]["root"][:, 2].min().item())
                if done.any():
                    print("STATIC_FAILED seed={} height={:.4f} roll_pitch={}".format(seed, minimum, env.base_euler_xyz[:, :2].tolist()), flush=True)
                    raise AssertionError("Standing candidate is not statically stable")
                assert torch.isfinite(reward).all()
            print("STATIC_PASS seed={} min_height={:.4f}".format(seed, minimum), flush=True)
        print("EXP11_STATIC_OK", flush=True)
        return

    # True 50/50 random branch coverage over many independent resets.
    stand_count = 0
    for _ in range(10):
        env.reset()
        stand_count += env.reset_standing.sum().item()
    ratio = stand_count/(10*env.num_envs)
    assert .35 < ratio < .65, ratio
    # Saturating commands must still feed bounded targets into PD.
    env.step(torch.full_like(env.actions, 5.))
    assert (env.pd_target_effective <= env.hard_dof_limits[:, 1]+1e-6).all()
    assert (env.pd_target_effective >= env.hard_dof_limits[:, 0]-1e-6).all()
    assert env.target_clip_counts.sum() > 0
    env.reset()
    training.runner.resume = False
    runner, _, _ = task_registry.make_alg_runner(env, args=args, train_cfg=training, log_root=None)
    with tempfile.TemporaryDirectory(prefix="exp11_smoke_") as tmp:
        runner.log_dir = tmp
        runner.learn(20)
        path = os.path.join(tmp, "model_20.pt")
        restored = AMPOnPolicyRunner(env, copy.deepcopy(runner.config), None, args.rl_device)
        restored.load(path)
        assert restored.current_learning_iteration == 20
        restored.learn(2)
        assert restored.current_learning_iteration == 22
        assert torch.isfinite(env.get_observations()).all()
        state = torch.load(path, map_location="cpu")
        state["metadata"]["schema_version"] = 1
        incompatible = os.path.join(tmp, "incompatible.pt")
        torch.save(state, incompatible)
        try:
            restored.load(incompatible)
        except ValueError:
            pass
        else:
            raise AssertionError("Old control metadata was accepted")
    print("EXP11_SMOKE_OK ratio={:.3f} iteration=22".format(ratio), flush=True)


if __name__ == "__main__":
    main()
