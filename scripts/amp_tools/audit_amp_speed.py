"""Score consistently time-scaled real reference histories with frozen AMP discriminators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.motion import MotionLibrary, quat_inverse_rotate
from humanoid.amp.models import Discriminator
from humanoid.amp.gait_rewards import style_multiplier, moving_speed_quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,nargs='+',required=True)
    parser.add_argument('--resource-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    provenance=json.loads(args.resource_manifest.read_text())
    assert provenance['resource_verification']['all_remote_resource_hashes_match']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in provenance['resource_sha256'].items())
    checkpoints=[torch.load(p,map_location='cpu',weights_only=False) for p in args.models]
    meta=checkpoints[0]['metadata'];cfg=checkpoints[0]['config']['amp']
    lib=MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized',
                      ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf',
                      meta['dof_names'],meta['key_body_names'],meta['policy_dt'],meta['history_steps'],
                      weights=cfg['motion_weights'],ground_clearance=cfg['ground_clearance'])
    offsets=torch.arange(lib.history_steps-1,-1,-1)*lib.dt
    examples={}
    for clip_name,commands,minimum_speed in [('F1_B4_walk_backwards',[-.2,-.15,-.1,-.075],.15),
                                            ('F1_B3_continuous_walk',[.1,.4,.6],.3)]:
        clip=lib.names.index(clip_name);fps=float(lib.fps[clip])
        times=torch.arange(round(float(lib.durations[clip])*fps)+1)/fps
        ids=torch.full((len(times),),clip,dtype=torch.long)
        state=lib.state(ids,times)
        vx=quat_inverse_rotate(state[:,3:7],state[:,7:10])[:,0]
        mask=(times>.12)&(times<lib.durations[clip]-.12)&(vx*commands[0]>0)&(vx.abs()>minimum_speed)
        ids,times,vx=ids[mask],times[mask],vx[mask]
        assert len(ids)>100
        original=lib.history(ids,times)
        def retime(ratio):
            sample_times=times[:,None]-offsets[None,:]*ratio[:,None]
            assert (sample_times>=0).all()
            states=lib.state(ids[:,None].expand(-1,lib.history_steps),sample_times)
            # A single positive time scale applies to positions/history times
            # and all linear/angular/joint velocities in each real history.
            states[:,:,7:13]*=ratio[:,None,None]
            states[:,:,42:]*=ratio[:,None,None]
            return lib.features(states)
        identity=retime(torch.ones_like(vx))
        assert torch.max(torch.abs(identity-original))<1e-6
        rows=[]
        for command in commands:
            ratio=command/vx;scaled=retime(ratio)
            error=float((scaled[:,-1,4]-command).abs().max());assert error<1e-6
            rows.append(dict(command=command,ratio=ratio,scaled=scaled,endpoint_vx_error_max=error))
        examples[clip_name]=dict(vx=vx,original=original,rows=rows)
    results=[]
    with torch.no_grad():
        for path,state in zip(args.models,checkpoints):
            assert state['metadata']==meta
            disc=Discriminator(lib.feature_dim,lib.history_steps,state['config']['discriminator']['hidden_dims'])
            disc.load_state_dict(state['discriminator_state_dict']);disc.eval()
            groups={}
            for name,example in examples.items():
                original_style=disc.style_reward(example['original'])
                rows=[]
                for row in example['rows']:
                    command=torch.full_like(example['vx'],row['command'])
                    raw=disc(row['scaled']);style=disc.style_reward(row['scaled'])
                    old_gate=style_multiplier(command,example['vx'],torch.zeros_like(command))*moving_speed_quality(command,example['vx'],cfg['motion_style_velocity_sigma'])
                    new_vx=row['scaled'][:,-1,4]
                    new_gate=style_multiplier(command,new_vx,torch.zeros_like(command))*moving_speed_quality(command,new_vx,cfg['motion_style_velocity_sigma'])
                    assert torch.isfinite(raw).all() and torch.isfinite(style).all()
                    rows.append(dict(command_vx=row['command'],examples=len(command),
                        time_scale_p05_p50_p95=torch.quantile(row['ratio'],torch.tensor([.05,.5,.95])).tolist(),
                        endpoint_vx_error_max=row['endpoint_vx_error_max'],original_mean_style=float(original_style.mean()),
                        retimed_mean_D=float(raw.mean()),retimed_mean_style=float(style.mean()),
                        retimed_zero_style_fraction=float((style<=0).float().mean()),
                        original_mean_shaped_style=float((original_style*old_gate).mean()),
                        retimed_mean_shaped_style=float((style*new_gate).mean())))
                groups[name]=rows
            results.append(dict(iteration=state['iter'],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=groups))
    result=dict(scope='Frozen discriminator counterfactual reference-speed audit; no physics or training',
                resource_manifest_sha256=hashlib.sha256(args.resource_manifest.read_bytes()).hexdigest(),
                resource_sha256=provenance['resource_sha256'],results=results,
                method='Constant time scale within each history: sample original at t-r*offset and multiply root linear/angular and joint velocities by r.',
                limits=['Each history has its own constant scale; this is not a stitched continuous retimed motion file.',
                        'Time scaling preserves local kinematic consistency but does not preserve gravity-dependent dynamics.',
                        'High reference scores do not establish policy reachability, exploration sufficiency or physical imitation feasibility.',
                        'Shaped style is an example score before style scale, policy dt and task/style mixing.',
                        'No expert data, discriminator or training settings were modified.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
