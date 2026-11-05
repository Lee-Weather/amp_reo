"""Observe production PPO rollouts and directional gradients without updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rollouts', type=int, default=48)
    custom, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    args = get_args()
    assert args.task == 'x1_amp_exp18' and args.seed == 5 and args.num_envs == 4096
    output = Path(custom.output)
    assert not output.exists() and custom.rollouts >= 48
    model = Path(custom.model)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == custom.sha256
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.seed = 5
    cfg.env.num_envs = 4096
    cfg.env.enable_camera_sensors = False
    # Keep actual training observation noise, directional commands and mixed RSI.
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _, _ = task_registry.make_alg_runner(env, args.task, args=args, log_root=None)
    runner.load(str(model), load_optimizer=False)
    actor = runner.actor_critic
    snapshots = [{k: t.clone() for k, t in obj.state_dict().items()}
                 for obj in (actor, runner.discriminator)]
    sums = {}
    windows = []
    gradients = []
    sampling = torch.Generator(device=env.device)
    sampling.manual_seed(5)
    for rollout in range(custom.rollouts):
        batch = runner.collect_rollout()
        command = batch['obs'][:, actor.motion_command_index]/env.obs_scales.lin_vel
        vx = batch['critic_obs'][:, -3]/env.obs_scales.lin_vel
        # Raw GAE and the globally normalized advantages actually used by PPO.
        raw = batch['returns']-batch['values']
        assert torch.allclose(batch['advantages'], (raw-raw.mean())/(raw.std(unbiased=False)+1e-8), atol=1e-5)
        masks = {'zero': command.abs() <= 1e-6, 'backward': command < -.05,
                 'slow_forward': (command > .05) & (command <= .2), 'forward': command > .2}
        masks.update(backward_rest=masks['backward'] & (vx.abs() < .03),
                     backward_wrong=masks['backward'] & (vx > .05),
                     backward_moving=masks['backward'] & (vx <= -.1) & (vx >= -.3))
        values = dict(raw_advantage=raw, normalized_advantage=batch['advantages'],
                      value=batch['values'], returns=batch['returns'], vx=vx,
                      task_reward=batch['task_rewards'], style_reward=batch['style_rewards'],
                      style_zero=(batch['style_scores'] == 0).float())
        window = dict(rollout=rollout+1, raw_advantage_std=float(raw.std(unbiased=False)),
                      groups={})
        for name, mask in masks.items():
            count = int(mask.sum())
            window['groups'][name] = dict(samples=count,
                raw_mean=float(raw[mask].mean()) if count else None,
                raw_std=float(raw[mask].std(unbiased=False)) if count else None,
                normalized_positive_fraction=float((batch['advantages'][mask] > 0).float().mean()) if count else None)
            if name not in sums:
                sums[name] = dict(samples=0, sums={key: 0. for key in values},
                                  squares={key: 0. for key in values}, positive=0)
            aggregate = sums[name]
            aggregate['samples'] += count
            aggregate['positive'] += int((batch['advantages'][mask] > 0).sum())
            for key, tensor in values.items():
                aggregate['sums'][key] += float(tensor[mask].double().sum())
                aggregate['squares'][key] += float(tensor[mask].double().square().sum())
        windows.append(window)
        if rollout+1 in (8, 24, 48):
            vectors = {}
            for name in ('zero', 'backward', 'slow_forward', 'forward'):
                ids = masks[name].nonzero(as_tuple=False).flatten()
                ids = ids[torch.randperm(len(ids), device=env.device, generator=sampling)[:4096]]
                if not len(ids):
                    continue
                distribution = actor.distribution(batch['obs'][ids])
                # At the unchanged policy ratio=1, this has PPO's actor gradient.
                # No entropy, value loss, optimizer step or per-group rescaling.
                loss = -(distribution.log_prob(batch['actions'][ids]).sum(-1)*batch['advantages'][ids]).mean()
                grad = torch.autograd.grad(loss, tuple(actor.actor.parameters()))
                vectors[name] = torch.cat([g.detach().flatten() for g in grad])
            entry = dict(rollout=rollout+1, conditional_actor_gradient_norms={
                name: float(vector.norm()) for name, vector in vectors.items()}, cosine={})
            if 'backward' in vectors:
                for name in ('zero', 'slow_forward', 'forward'):
                    if name in vectors:
                        a, b = vectors['backward'], vectors[name]
                        entry['cosine']['backward_vs_'+name] = float(torch.dot(a, b)/(a.norm()*b.norm()).clamp(min=1e-12))
            gradients.append(entry)
        if (rollout+1) % 8 == 0:
            print('ADVANTAGE_AUDIT_PROGRESS rollout=%d/%d' % (rollout+1, custom.rollouts), flush=True)
    for obj, snapshot in zip((actor, runner.discriminator), snapshots):
        assert all(torch.equal(tensor, snapshot[name]) for name, tensor in obj.state_dict().items())
    assert len(runner.optimizer.state) == 0 and len(runner.disc_optimizer.state) == 0
    groups = {}
    for name, aggregate in sums.items():
        count = aggregate['samples']
        means = {k: v/max(count, 1) for k, v in aggregate['sums'].items()}
        groups[name] = dict(samples=count, means=means,
            stds={k: max(0., v/max(count, 1)-means[k]**2)**.5 for k, v in aggregate['squares'].items()},
            normalized_positive_fraction=aggregate['positive']/max(count, 1))
    result = dict(checkpoint_iteration=runner.current_learning_iteration, checkpoint_sha256=custom.sha256,
                  task=args.task, num_envs=4096, seed=5, rollouts=custom.rollouts,
                  total_transitions=custom.rollouts*runner.num_steps_per_env*4096,
                  simulated_seconds_per_env=custom.rollouts*runner.num_steps_per_env*env.dt,
                  noise=cfg.noise.add_noise, rsi_probability=cfg.amp.rsi_probability,
                  weights_and_normalizers_unchanged=True, no_optimizer_steps=True,
                  groups=groups, windows=windows, sampled_actor_gradients=gradients,
                  limits=['Frozen policy/critic/discriminator, not an ablation of advantage normalization.',
                          'Velocity is the pre-step critic input; grouping includes RSI, switching and recovery.',
                          'A negative group mean advantage or smaller gradient alone does not prove a PPO bug.',
                          'Gradient snapshots use up to 4096 samples/group, one seed, conditional rather than frequency-weighted means.',
                          'Global advantage normalization changes with each rollout; pooled statistics mix these windows.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('DIRECTIONAL_ADVANTAGE_AUDIT_OK', flush=True)
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
