"""Score all real reference endpoint histories with frozen saved discriminators."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from humanoid.amp.models import Discriminator
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.stationary import CommandMotionSampler


def main():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    record = Path(__file__).resolve().parent
    paths = [ROOT/'czy/data/exp1.6/model_9000.pt', record/'interim/model_9400.pt']
    states = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    cfg, meta = states[1]['config']['amp'], states[1]['metadata']
    motions = ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'
    urdf = ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf'
    lib = MotionLibrary(motions, urdf, meta['dof_names'], meta['key_body_names'],
                        meta['policy_dt'], meta['history_steps'], weights=cfg['motion_weights'],
                        ground_clearance=cfg['ground_clearance'])
    sampler = CommandMotionSampler(lib, cfg['rsi_speed_tolerance'], cfg['expert_standing_probability'],
                                   cfg['quiet_root_speed'], cfg['quiet_angular_speed'], cfg['quiet_joint_speed'],
                                   exclude_quiet_for_moving=True,
                                   relative_speed_tolerance=cfg['rsi_relative_speed_tolerance'])
    ids, vx, quiet = sampler.ids.numpy(), sampler.local_vx.numpy(), sampler.quiet.numpy()
    prior = sampler.prior.numpy()
    quiet_prior = prior*quiet
    mixture = (1-cfg['expert_standing_probability'])*prior/prior.sum()
    mixture += cfg['expert_standing_probability']*quiet_prior/quiet_prior.sum()
    assert np.isclose(mixture.sum(), 1)
    expert = lib.history(sampler.ids, sampler.times)
    groups = {name: ids == i for i, name in enumerate(lib.names)}
    groups.update(quiet=quiet, negative_moving=vx<-.05, positive_moving=vx>.05)
    for command in (-.2, -.1, .1, .4, .6, .65):
        tolerance = min(cfg['rsi_speed_tolerance'], .025+cfg['rsi_relative_speed_tolerance']*abs(command))
        allowed = ((np.abs(vx-command)<=tolerance) & (command*vx>0) & ~quiet
                   & (np.abs(vx)>=min(.5*abs(command),.05)))
        if not allowed.any():
            raise ValueError('No candidates for '+str(command))
        groups['command_'+str(command)] = allowed
    scored = []
    with torch.no_grad():
        for path, state in zip(paths, states):
            assert state['metadata'] == meta
            disc = Discriminator(lib.feature_dim, lib.history_steps,
                                 state['config']['discriminator']['hidden_dims'])
            disc.load_state_dict(state['discriminator_state_dict'])
            disc.eval()
            output = disc(expert).numpy()
            style = disc.style_reward(expert).numpy()
            assert np.isfinite(output).all() and np.isfinite(style).all()
            values = {}
            for name, mask in groups.items():
                p = prior[mask]/prior[mask].sum()
                values[name] = dict(windows=int(mask.sum()),
                    expert_mixture_mass=float(mixture[mask].sum()),
                    conditional_prior_mean_vx=float(np.dot(p,vx[mask])),
                    conditional_prior_mean_D=float(np.dot(p,output[mask])),
                    conditional_prior_mean_style=float(np.dot(p,style[mask])),
                    D_quantiles=np.quantile(output[mask],[.05,.5,.95]).tolist(),
                    style_quantiles=np.quantile(style[mask],[.05,.5,.95]).tolist())
            scored.append(dict(iteration=state['iter'],
                               checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), groups=values))
    report = dict(scope='Frozen discriminator reference audit, no training or physics', seed=5,
                  history_steps=lib.history_steps, feature_dim=lib.feature_dim,
                  resource_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [urdf]+sorted(motions.glob('*.pkl'))},
                  reference_config=cfg, scores=scored,
                  limits=['Endpoint grids approximate the continuously sampled expert time distribution.',
                          'Conditional prior means use the actual clip prior restricted to each group.',
                          'Groups overlap; their masses must not be added.',
                          'No command-conditioned discriminator is used; candidate groups only diagnose support.',
                          'Expert recognition does not prove physically feasible imitation or policy success.'])
    (record/'expert_discriminator_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(iteration=s['iteration'], groups={k:s['groups'][k] for k in
                       ('negative_moving','positive_moving','command_-0.2','command_0.6')}) for s in scored]))


if __name__ == '__main__':
    main()
