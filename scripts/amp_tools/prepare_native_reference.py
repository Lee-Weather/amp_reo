"""CPU-only constrained leg correction; writes evidence even when acceptance fails."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.amp.motion import MotionLibrary
from humanoid.amp.native_reference import NativeCycle,periodic_curve,hermite


def prepare(model,output,iterations=250,initial=None):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    ck = torch.load(model,map_location='cpu',weights_only=False)
    meta = ck['metadata']
    lib = MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized',
        ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf',
        meta['dof_names'],meta['key_body_names'],ground_clearance=.002)
    reference = NativeCycle(lib)
    before = reference.audit()
    n = len(reference.q)-1
    phases = torch.arange(n)/n
    original = reference.state(phases).detach()
    samples = 4*n
    dense = (torch.arange(samples)+.5)/samples
    epsilon = .0005
    all_phases = torch.cat((dense,dense-epsilon/reference.period,dense+epsilon/reference.period))
    physical = reference.state(all_phases).detach()
    initial_vertices = reference.world_vertices(physical[:samples])
    initial_height = torch.stack([v[:,:,2].amin(-1) for v in initial_vertices],-1)
    contact = (initial_height<.008).float()
    legs = [i for i,name in enumerate(meta['dof_names']) if any(s in name for s in ('hip_','knee_','ankle_'))]
    delta = torch.nn.Parameter(torch.zeros(n,len(legs)))
    if initial:
        payload = json.loads(Path(initial).read_text())
        order = [payload['dof_names'].index(name) for name in meta['dof_names']]
        initial_delta = torch.tensor(payload['joint_correction'])[:-1,order][:,legs]
        with torch.no_grad(): delta.copy_(initial_delta)
    optimizer = torch.optim.LBFGS([delta],lr=.5,max_iter=iterations,line_search_fn='strong_wolfe',tolerance_grad=1e-7)
    lower,upper = lib.fk.lower,lib.fk.upper
    margin = torch.minimum(torch.full_like(lower,torch.pi/36),.1*(upper-lower))
    rows = torch.arange(samples)
    def candidate():
        q = original[:,13:42].clone()
        q[:,legs] = (q[:,legs]+delta.clamp(-.35,.35)).maximum((lower+margin)[legs]).minimum((upper-margin)[legs])
        return q
    calls = [0]
    def closure():
        optimizer.zero_grad()
        q = candidate()
        values,slopes = periodic_curve(torch.cat((q,q[:1])),reference.period)
        q,dq = hermite(values,slopes,all_phases,reference.period)
        states = torch.cat((physical[:,:13],q,dq),-1)
        vertices = reference.world_vertices(states)
        heights,velocity,rotation = [],[],[]
        for v in vertices:
            h,ids = v[:samples,:,2].min(-1)
            heights.append(h)
            minus,plus = v[samples:2*samples],v[2*samples:]
            material = (plus-minus)/(2*epsilon)
            velocity.append(material[rows,ids,:2].square().sum(-1))
            # Multiple sole material points penalize unwanted stance rotation.
            rotation.append(material[:,:,:2].square().sum(-1).mean(-1))
        h = torch.stack(heights,-1)
        h = h-h.min(-1,keepdim=True).values+.002
        v,r = torch.stack(velocity,-1),torch.stack(rotation,-1)
        loss = ((v+.15*r)*contact).sum()/contact.sum()/.1**2
        loss += ((h/.005).square()*contact).sum()/contact.sum()
        loss += .2*((h-initial_height)/.01).square().mean()
        loss += .02*(delta/.20).square().mean()
        loss += .01*((torch.roll(delta,-1,0)-delta)/.05).square().mean()
        loss += .10*((torch.roll(delta,-1,0)-2*delta+torch.roll(delta,1,0))/.03).square().mean()
        loss += 5*((dq[:samples].abs()-7.5).clamp(min=0)/.5).square().mean()
        loss.backward()
        calls[0] += 1
        return loss
    optimizer.step(closure)
    q = candidate().detach()
    correction = torch.cat((q-original[:,13:42],(q-original[:,13:42])[:1]))
    payload = dict(dof_names=meta['dof_names'],joint_correction=correction.tolist())
    refined = NativeCycle(lib,contact_correction=payload)
    after = refined.audit()
    result = dict(source_model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        method='native_tempo_constrained_leg_correction_v2',optimizer_evaluations=calls[0],
        before=before,after=after,passed=after['passed'],dof_names=meta['dof_names'],joint_correction=correction.tolist(),
        maximum_extra_joint_correction_rad=float(correction.abs().max()),
        limitation='Kinematic consistency only; no dynamics, policy training or physical stability tested.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='joint_correction'},indent=2))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=250)
    p.add_argument('--initial',type=Path)
    a = p.parse_args()
    prepare(a.model,a.output,a.iterations,a.initial)
