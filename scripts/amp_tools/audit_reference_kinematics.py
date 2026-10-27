"""Measure near-floor material-point motion in real references, without contact assumptions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.motion import MotionLibrary, URDFKinematics, quat_mul, quat_rotate, quat_inverse_rotate
from humanoid.amp.grounding import FootGrounding


def audit(model):
    torch.set_num_threads(1)
    checkpoint=torch.load(model,map_location='cpu',weights_only=False)
    meta=checkpoint['metadata'];cfg=checkpoint['config']['amp']
    motions=ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'
    urdf=ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf'
    lib=MotionLibrary(motions,urdf,meta['dof_names'],meta['key_body_names'],meta['policy_dt'],meta['history_steps'],
                      weights=cfg['motion_weights'],ground_clearance=cfg['ground_clearance'])
    feet=[side+'_ankle_roll_link' for side in ('left','right')]
    fk=URDFKinematics(urdf,meta['dof_names'],feet)
    grounding=FootGrounding(urdf,fk,cfg['ground_clearance'])
    fk.chain=[(a,b,p.double(),q.double(),axis.double(),i) for a,b,p,q,axis,i in fk.chain]
    resources=[urdf]+sorted(motions.glob('*.pkl'))
    xml=ET.parse(urdf).getroot()
    for link in xml.findall('link'):
        if link.get('name') in feet:
            resources.append((urdf.parent/link.find('collision/geometry/mesh').get('filename')).resolve())
    result={}
    for clip,name in enumerate(lib.names):
        n=round(float(lib.durations[clip]*lib.fps[clip]))+1
        state=lib.frames[clip,:n].double()
        pos,rot,velocity=state[:,:3],state[:,3:7],state[:,7:10]
        omega=torch.cat((state[:,10:13],torch.zeros(n,1,dtype=torch.float64)),-1)
        q,dq=state[:,13:42],state[:,42:]
        qdot=.5*quat_mul(omega,rot)
        def transforms(delta):
            root_q=rot+delta*qdot;root_q=root_q/root_q.norm(dim=-1,keepdim=True)
            local,local_q=fk.key_transforms(q+delta*dq)
            rq=root_q[:,None,:].expand(-1,2,-1)
            return pos[:,None,:]+delta*velocity[:,None,:]+quat_rotate(rq,local),quat_mul(rq,local_q)
        origin,rotation=transforms(0.)
        selected=[];height=[]
        for index,raw_mesh in grounding.feet:
            mesh=raw_mesh.double();x,y,z,w=rotation[:,index].unbind(-1)
            row=torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1)
            lowest,ids=(row@mesh.T).min(-1)
            selected.append(mesh[ids]);height.append(lowest+origin[:,index,2])
        material=torch.stack(selected,1);height=torch.stack(height,1)
        def point_velocity(epsilon):
            plus,qp=transforms(epsilon);minus,qm=transforms(-epsilon)
            return (plus+quat_rotate(qp,material)-minus-quat_rotate(qm,material))/(2*epsilon)
        point=point_velocity(1e-4);check=point_velocity(2e-4)
        sensitivity=float((point-check).abs().max());assert sensitivity<1e-3
        speed=point[:,:,:2].norm(dim=-1).numpy();height=height.numpy()
        local_vx=quat_inverse_rotate(rot,state[:,7:10])[:,0].numpy()
        times=np.arange(n)/float(lib.fps[clip])
        interior=(times>=.12)&(times<float(lib.durations[clip])-.12)
        groups={'moving':interior&(np.abs(local_vx)>.05),
                'matched_backward_02':interior&(local_vx>=-.325)&(local_vx<=-.075),
                'matched_forward_06':interior&(local_vx>=.45)&(local_vx<=.75)}
        stats={}
        for group,mask in groups.items():
            if not mask.any():continue
            per_foot={}
            for i,side in enumerate(('left','right')):
                near=mask&(height[:,i]<=.01)
                values=speed[near,i]
                per_foot[side]={'near_floor_samples':int(near.sum()),'near_floor_fraction':float(near.sum()/mask.sum()),
                    'point_horizontal_speed_p05_p50_p95_m_s':np.quantile(values,[.05,.5,.95]).tolist() if len(values) else None,
                    'point_speed_above_010_fraction':float((values>.10).mean()) if len(values) else None}
            stats[group]={'samples':int(mask.sum()),'mean_reference_local_vx':float(local_vx[mask].mean()),'feet':per_foot}
        result[name]={'frames':n,'duration_s':float(lib.durations[clip]),
                      'epsilon_sensitivity_max_m_s':sensitivity,'groups':stats}
    return dict(source_iteration=checkpoint['iter'],source_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                resource_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in resources},
                clips=result,selection=dict(near_floor_max_height_m=.01,horizontal_speed_threshold_m_s=.10,
                                           boundary_exclusion_s=.12,quantile_probabilities=[.05,.5,.95]),
                method='Double precision directional FK derivative of the fixed current-lowest collision vertex along reference root origin velocity, world angular velocity and joint dq.',
                root_velocity_convention='Reference root velocity comes from the derivative of root_pos, so it is origin velocity; do not apply Gym COM subtraction.',
                limits=['Near floor <=1cm is a geometric selection, not observed contact or measured physical slip.',
                        'No forces or dynamics are reconstructed. A lowest mesh vertex can differ from a solver contact patch.',
                        'Grounding and small joint-limit clamping match the production MotionLibrary.',
                        'Each reference has different speeds and phases; this is not a controlled causal comparison or policy acceptance.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:clip['groups'] for name,clip in result['clips'].items()}))


if __name__=='__main__':main()
