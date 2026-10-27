"""Audit reward credit in existing frozen CPU rollouts; no new simulation."""
import json
from launch_cpu_compare_9000_9400 import remote, RECORD


def main():
    state = json.loads((RECORD/'cpu_compare_9000_9400.json').read_text())
    program = '''
import csv,hashlib,json,pathlib
import numpy as np
import torch
state=json.loads(__STATE__);root=pathlib.Path(state['server_dir'])
model=root/'checkpoints/model_9400.pt'
assert hashlib.sha256(model.read_bytes()).hexdigest()==state['checkpoint_sha256']['9400']
checkpoint=torch.load(model,map_location='cpu');cfg=checkpoint['config']
r=cfg['rewards'];scales=r['scales'];dt=checkpoint['metadata']['policy_dt']
mix=cfg['discriminator']['task_reward_weight']
assert not r['only_positive_rewards']
cases=[]
for iteration in (9000,9400):
    path=root/('logs/cpu_%d_reverse/isaac_diag.csv'%iteration)
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
    phases={}
    for name,mask in [('reverse',v<-.05),('forward',v>.05),('near_zero',np.abs(v)<=.05)]:
        phases[name]={'fraction':float(mask.mean()),'mean_vx':float(v[mask].mean()) if mask.any() else None,
                      'mean_bonus':float(bonus[mask].mean()) if mask.any() else None}
    cases.append({'iteration':iteration,'samples':int(take.sum()),'mean_vx':float(v.mean()),
        'mean_abs_vx':float(np.abs(v).mean()),'mean_bonus':float(bonus.mean()),
        'mixed_reward_components_per_step':{k:float(a.mean()) for k,a in components.items()},
        'phases':phases,'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps({'scope':'Existing same-CPU standing reverse rollouts, fixed 3.60-19.78s window; no resets',
    'config_iteration':9400,'policy_dt':dt,'task_mix':mix,'cases':cases,
    'limits':['Reconstructed components only; remaining task rewards are not independently decomposed.',
              'A positive clipped bonus under zero net speed demonstrates a loophole, not sole causal attribution.',
              'CPU rollouts are not final GPU acceptance; no training or reward code was changed.']}))
'''.replace('__STATE__', repr(json.dumps(state)))
    result = json.loads(remote(program, python='/home/robot/Anaconda/envs/F1/bin/python'))
    (RECORD/'reverse_oscillation_reward_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
