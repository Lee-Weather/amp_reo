"""Frozen-policy GPU diagnostic: IID versus held zero-mean target perturbations.

This script does not update policy/discriminator parameters or change production
sampling. Sustained velocity is a diagnostic, not a gait acceptance criterion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import isaacgym  # Must precede torch.
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.amp.playback import inject_command


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--bias-degrees', type=float, default=2.)
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    assert args.task == 'x1_amp_exp18' and args.seed == 5 and args.num_envs == 4096
    assert custom.steps >= 400 and 0 < custom.bias_degrees <= 3
    output = Path(custom.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing diagnostic')
    model_path = Path(custom.model)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_hash == custom.sha256
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.seed = 5
    cfg.env.num_envs = 4096
    cfg.env.capture_replay_state = True
    cfg.env.enable_camera_sensors = False
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.amp.reference_state_initialization = False
    cfg.amp.rsi_probability = 0.
    cfg.commands.ranges.lin_vel_x = [-.2, -.2]
    cfg.commands.ranges.lin_vel_y = [0., 0.]
    cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _, _ = task_registry.make_alg_runner(env, args.task, args=args, log_root=None)
    runner.load(str(model_path), load_optimizer=False)
    actor = runner.actor_critic
    actor.eval()
    runner.discriminator.eval()
    frozen = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    frozen_disc = {name: tensor.clone() for name, tensor in runner.discriminator.state_dict().items()}
    device = env.device
    assert str(device).startswith('cuda') and args.sim_device.startswith('cuda')
    n = env.num_envs
    # Interleave cohorts across the spatial grid; all remain seed=5.
    cohort = torch.arange(n, device=device) % 4
    names = ['original_iid', 'extra_white', 'held_10', 'held_20']
    joints = [side+'_'+joint+'_pitch_joint' for side in ('left', 'right')
              for joint in ('hip', 'knee', 'ankle')]
    indices = [env.dof_names.index(name) for name in joints]
    bias_std = math.radians(custom.bias_degrees) / cfg.control.action_scale
    generator = torch.Generator(device=device)
    generator.manual_seed(5)
    held = torch.zeros(n, len(indices), device=device)
    bias_age = torch.zeros(n, dtype=torch.long, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    streak = torch.zeros(n, dtype=torch.long, device=device)
    maximum_streak = streak.clone()
    count = torch.zeros(n, device=device)
    sum_vx = count.clone()
    squared_error = count.clone()
    displacement = count.clone()
    falls = torch.zeros(n, dtype=torch.long, device=device)
    lift_streak = torch.zeros(n, 2, dtype=torch.bool, device=device)
    lift_count = torch.zeros(n, 2, dtype=torch.long, device=device)
    sample_bias_square = torch.zeros(4, device=device)
    sample_bias_count = torch.zeros(4, device=device)
    inject_command(env, -.2)
    env.reset()
    initial_pose = env.dof_pos.clone()
    initial_root = env.root_states[:, :3].clone()-env.env_origins
    with torch.no_grad():
        for step in range(custom.steps):
            obs = inject_command(env, -.2)
            active = env.commands[:, 0] < -.05
            distribution = actor.distribution(obs)
            # Identical sampling law to the original runner in cohort 0.
            white = torch.randn(distribution.loc.shape, device=device, generator=generator)
            actions = distribution.loc + distribution.scale*white
            for group, period in ((1, 1), (2, 10), (3, 20)):
                mask = (cohort == group) & active
                refresh = mask & (bias_age % period == 0)
                held[~active] = 0.
                if refresh.any():
                    held[refresh] = torch.randn(int(refresh.sum()), len(indices), device=device,
                                                 generator=generator)*bias_std
                selected = held*mask.unsqueeze(-1)
                actions[:, indices] += selected
                sample_bias_square[group] += selected.square().sum()
                sample_bias_count[group] += mask.sum()*len(indices)
            bias_age = torch.where(active, bias_age+1, torch.zeros_like(bias_age))
            _, _, _, done, info = env.step(actions)
            state = info['replay_state']
            assert torch.isfinite(state['velocity']).all()
            assert (state['command_observed']-state['command']).abs().max() < 1e-6
            failure = done.bool() & ~state['timeout'].bool()
            falls += failure.long()
            # Never count a reset recovery as continuous success from standing.
            alive &= ~done.bool()
            held[done.bool()] = 0.
            bias_age[done.bool()] = 0
            valid = alive & (state['root'][:, 2]-env.env_origins[:, 2] > .45)
            valid &= (state['gravity'][:, 2] < -.7) & (state['command'][:, 0] < -.05)
            vx = state['velocity'][:, 0]
            backwards = valid & (vx <= -.1) & (vx >= -.3)
            streak = torch.where(backwards, streak+1, torch.zeros_like(streak))
            maximum_streak = torch.maximum(maximum_streak, streak)
            # Match the main replay's 1.5s buffer + 2s settling exclusion.
            if (step+1)*env.dt >= 3.5:
                count += valid.float()
                sum_vx += vx*valid
                squared_error += (vx+.2).square()*valid
            displacement += vx*valid*env.dt
            lifted = (env.sole_height > .015) & valid.unsqueeze(-1)
            lift_count += (lifted & ~lift_streak).long()
            lift_streak = lifted
            if (step+1) % 100 == 0:
                print('PROBE_PROGRESS step=%d survivors=%d' % (step+1, int(alive.sum())), flush=True)
    assert all(torch.equal(tensor, frozen[name]) for name, tensor in actor.state_dict().items())
    assert all(torch.equal(tensor, frozen_disc[name]) for name, tensor in runner.discriminator.state_dict().items())
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    cases = []
    for group, name in enumerate(names):
        mask = cohort == group
        samples = count[mask].sum()
        survived = mask & alive
        durable = survived & (maximum_streak*env.dt >= 1.)
        case = dict(name=name, environments=int(mask.sum()), first_episode_survival_fraction=float(alive[mask].float().mean()),
                    non_timeout_terminations=int(falls[mask].sum()), valid_settled_samples=int(samples),
                    mean_vx=float(sum_vx[mask].sum()/samples.clamp(min=1)),
                    rmse_vx=float((squared_error[mask].sum()/samples.clamp(min=1)).sqrt()),
                    survivor_02s_backward_fraction=float((survived & (maximum_streak*env.dt >= .2)).sum()/mask.sum()),
                    survivor_1s_backward_fraction=float(durable.sum()/mask.sum()),
                    survivor_1s_backward_and_both_feet_lifted_twice_fraction=float(
                        (durable & (lift_count >= 2).all(-1)).sum()/mask.sum()),
                    mean_integrated_body_vx_m=float(displacement[mask].mean()),
                    extra_raw_target_rms_degrees=float((sample_bias_square[group]/sample_bias_count[group].clamp(min=1)).sqrt()
                                                      *cfg.control.action_scale*180/math.pi),
                    initial_q_rms=float(initial_pose[mask].square().mean().sqrt()),
                    initial_root_height_mean=float(initial_root[mask, 2].mean()))
        cases.append(case)
    result = dict(checkpoint_iteration=runner.current_learning_iteration, checkpoint_sha256=model_hash,
                  seed=5, num_envs=n, task=args.task, steps=custom.steps, control_dt=env.dt,
                  standing_buffer_seconds=cfg.amp.stand_buffer_seconds, noise=False, pushes=False,
                  extra_bias_degrees=custom.bias_degrees, joints=joints, cases=cases,
                  weights_unchanged=True, policy_metadata=runner._metadata(),
                  limits=['One seed and one frozen policy, not a training comparison or gait acceptance.',
                          'White/held cohorts have equal nominal added marginal variance; actual state-dependent action distributions differ.',
                          'Velocity streaks and foot lifts can include recovery steps; they do not establish natural or stable gait.',
                          'Original IID has lower total variance than the three perturbation cohorts.',
                          'First-episode survivor metrics exclude every environment that reset; full failure counts remain reported.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('TEMPORAL_PROBE_OK '+json.dumps(cases), flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
