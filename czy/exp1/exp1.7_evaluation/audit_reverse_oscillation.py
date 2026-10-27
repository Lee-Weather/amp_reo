"""Audit reward credit in existing frozen rollouts; no new simulation."""
import argparse
import json
from launch_cpu_compare_9000_9400 import remote, RECORD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-final', action='store_true')
    args = parser.parse_args()
    if args.gpu_final:
        job = json.loads((RECORD/'replay_job.json').read_text())
        state = dict(server_dir=job['server_dir'], checkpoint_sha256={'10000':job['checkpoint_sha256']})
        config_iteration = 10000
        case_paths = [(10000, 'logs/replay/speed_check_0/isaac_diag.csv')]
        scope = 'Final GPU standing reverse rollout, fixed 3.60-19.78s window; no resets'
        destination = 'reverse_oscillation_reward_audit_final_gpu.json'
    else:
        state = json.loads((RECORD/'cpu_compare_9000_9400.json').read_text())
        config_iteration = 9400
        case_paths = [(n,'logs/cpu_%d_reverse/isaac_diag.csv'%n) for n in (9000,9400)]
        scope = 'Existing same-CPU standing reverse rollouts, fixed 3.60-19.78s window; no resets'
        destination = 'reverse_oscillation_reward_audit.json'
    program = '''
import csv,hashlib,json,pathlib
import numpy as np
import torch
state=json.loads(__STATE__);root=pathlib.Path(state['server_dir'])
model=root/('checkpoints/model_%d.pt'%__CONFIG_ITERATION__)
assert hashlib.sha256(model.read_bytes()).hexdigest()==state['checkpoint_sha256'][str(__CONFIG_ITERATION__)]
checkpoint=torch.load(model,map_location='cpu');cfg=checkpoint['config']
r=cfg['rewards'];scales=r['scales'];dt=checkpoint['metadata']['policy_dt']
mix=cfg['discriminator']['task_reward_weight']
assert not r['only_positive_rewards']
cases=[]
for iteration,relative_path in __CASE_PATHS__:
    path=root/relative_path
    rows=list(csv.DictReader(path.open()))
    data={key:np.array([float(row[key]) for row in rows]) for key in rows[0]}
    take=(data['time_s']>=3.6)&(data['time_s']<=19.8)
    assert not data['done'][take].any() and np.allclose(data['command_applied'][take],-.2)
    assert (data['gz'][take]<-.7).all() and (data['base_height'][take]>.45).all()
    v=data['base_vel_x'][take];cmd=data['command_applied'][take]
    std=r['tracking_min_std']+(r['tracking_max_std']-r['tracking_min_std'])*np.minimum(np.abs(cmd)/r['tracking_max_speed'],1)
    linear=np.exp(-(v-cmd)**2/std**2-data['base_vel_y'][take]**2/r['tracking_lateral_std']**2)
    angular=np.exp(-data['base_ang_vel_z'][take]**2/r['tracking_sigma'])
    bonus=np.clip(-v/np.abs(cmd),0,1)
    components={'backward_bonus_mixed':mix*scales['backward_progress']*dt*bonus,
                'linear_tracking_mixed':mix*scales['tracking_lin_vel']*dt*linear,
                'angular_tracking_mixed':mix*scales['tracking_ang_vel']*dt*angular,
                'task_mixed':mix*data['task_reward'][take],
                'style_mixed':(1-mix)*data['style_reward'][take],
                'total_mixed':data['mixed_reward'][take]}
    assert np.allclose(components['task_mixed']+components['style_mixed'],components['total_mixed'])
    z=(v-cmd)/np.abs(cmd)
    huber=np.where(np.abs(z)<=1,.5*z*z,np.abs(z)-.5)
    candidates={'signed_progress_upper_cap':np.minimum(-v/np.abs(cmd),1),
                'signed_huber_weight_1':1-2*huber,
                'signed_huber_weight_1_5':1.5*(1-2*huber),
                'signed_huber_weight_2':2*(1-2*huber)}
    counterfactual={}
    for name,candidate in candidates.items():
        correction=mix*scales['backward_progress']*dt*(candidate-bonus)
        counterfactual[name]={'mean_bonus':float(candidate.mean()),
            'task_mixed_per_step':float((components['task_mixed']+correction).mean()),
            'total_mixed_per_step_with_same_frozen_D':float((components['total_mixed']+correction).mean())}
    phases={}
    for name,mask in [('reverse',v<-.05),('forward',v>.05),('near_zero',np.abs(v)<=.05)]:
        phases[name]={'fraction':float(mask.mean()),'mean_vx':float(v[mask].mean()) if mask.any() else None,
                      'mean_bonus':float(bonus[mask].mean()) if mask.any() else None}
    cases.append({'iteration':iteration,'samples':int(take.sum()),'mean_vx':float(v.mean()),
        'mean_abs_vx':float(np.abs(v).mean()),'mean_bonus':float(bonus.mean()),
        'mixed_reward_components_per_step':{k:float(a.mean()) for k,a in components.items()},
        'phases':phases,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'untrained_counterfactual_candidates':counterfactual})
print(json.dumps({'scope':__SCOPE__,
    'config_iteration':__CONFIG_ITERATION__,'policy_dt':dt,'task_mix':mix,'cases':cases,
    'config_checkpoint_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
    'limits':['Reconstructed components only; remaining task rewards are not independently decomposed.',
              'A positive clipped bonus under zero net speed demonstrates a loophole, not sole causal attribution.',
              'Counterfactual scores retain the same trajectory and discriminator; no learning or physics response is predicted.',
              'Candidate concavity does not prove whole-reward ordering, stability, naturalness or safe termination incentives.',
              'No training or reward code was changed.']}))
'''.replace('__STATE__', repr(json.dumps(state)))
    program = program.replace('__CONFIG_ITERATION__', str(config_iteration)).replace('__CASE_PATHS__', repr(case_paths)).replace('__SCOPE__', repr(scope))
    result = json.loads(remote(program, python='/home/robot/Anaconda/envs/F1/bin/python'))
    (RECORD/destination).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
