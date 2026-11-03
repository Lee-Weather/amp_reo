"""Continuous per-reference-foot swing shaping and actual clearance-run events."""
import torch
from .exp20_rewards import huber

def dense_swing(height,target,force,weight):
    force=force.clamp(min=0)
    weight=weight.reshape(-1,1).clamp(min=1)
    phase=((target-.005)/.015).clamp(0,1)
    gate=(force.sum(-1)/(.5*weight[:,0])).clamp(0,1)
    height_score=1/(1+huber((height-target)/.01))
    unload=1/(1+(force/(.1*weight)).square())
    other=(force.flip(-1)/(.25*weight)).clamp(0,1)
    phase_weight=phase/phase.sum(-1,keepdim=True).clamp(min=1e-6)
    components=torch.stack((.6*height_score,.2*unload,.2*other),-1)
    contributions=components*phase_weight[...,None]*gate[:,None,None]
    return contributions.sum((-1,-2)),dict(height=height,target=target,force=force,
        phase=phase,contributions=contributions,body_weight=weight)

class ClearanceRuns:
    def __init__(self,count,device,dt):
        self.dt=dt
        self.seconds=torch.zeros(count,2,device=device)
        self.counted=torch.zeros(count,2,dtype=torch.bool,device=device)

    def update(self,height,eligible):
        airborne=(height>.01)&eligible[:,None]
        self.seconds=torch.where(airborne,self.seconds+self.dt,torch.zeros_like(self.seconds))
        self.counted&=airborne
        events=airborne&(self.seconds>=.08-1e-7)&~self.counted
        self.counted|=events
        return events

    def reset(self,ids):
        self.seconds[ids]=0;self.counted[ids]=False
