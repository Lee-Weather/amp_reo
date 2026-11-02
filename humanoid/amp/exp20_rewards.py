"""Continuous full-body similarity, swing/load agreement and uncapped costs."""
import torch


def huber(error):
    error = error.abs()
    return torch.where(error<=1,.5*error.square(),error-.5)


def similarity(actual,target,scale):
    if scale<=0: raise ValueError('Similarity scale must be positive')
    return (1/(1+huber((actual-target)/scale))).mean(-1)


def raw_violation(raw,lower,upper,scale=.1):
    excess = (lower-raw).clamp(min=0)+(raw-upper).clamp(min=0)
    return huber(excess/scale)


def margin_cost(q,lower,upper):
    margin = torch.minimum(torch.full_like(lower,torch.pi/36),.1*(upper-lower))
    distance = torch.minimum(q-lower,upper-q)
    return huber((margin-distance).clamp(min=0)/margin).mean(-1)


def support_terms(height,target,force,weight,point_speed):
    force = force.clamp(min=0)
    weight = weight.reshape(-1,1).clamp(min=1)
    swing = ((target-.005)/.015).clamp(0,1)
    stance = 1-swing
    total_gate = (force.sum(-1)/(.5*weight[:,0])).clamp(0,1)
    load = force/(force.sum(-1,keepdim=True)+1e-6)
    desired = stance/stance.sum(-1,keepdim=True).clamp(min=1e-6)
    clearance = ((height-.005)/(target-.005).clamp(min=.005)).clamp(0,1)
    height_quality = 1/(1+huber((height-target)/.01))
    unload = 1/(1+(force/(.1*weight)).square())
    other_support = (force.flip(-1)/(.25*weight)).clamp(0,1)
    swing_score = (swing*clearance*height_quality*unload*other_support).sum(-1)/swing.sum(-1).clamp(min=1e-6)
    support_score = (stance/(1+huber(height/.01)+huber((load-desired)/.25))).sum(-1)/stance.sum(-1).clamp(min=1e-6)
    actual_contact = force>20
    slip_weight = stance*actual_contact
    slip = (slip_weight*huber(point_speed/.10)).sum(-1)/slip_weight.sum(-1).clamp(min=1e-6)
    return swing_score*total_gate,support_score*total_gate,slip
