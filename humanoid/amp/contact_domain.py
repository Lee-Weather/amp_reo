"""Static contact-offset populations, with explicit original and near-surface groups."""
import torch


def sample_offsets(count, interval):
    low,high=map(float,interval)
    if count<1 or not 0<low<=.01<=high:
        raise ValueError('Contact offsets must be positive and include original0.01m')
    values=low+(high-low)*torch.rand(count,device='cpu')
    near=count//2;nominal=count//4
    values[:near]=low
    values[near:near+nominal]=.01
    return values


def set_contact_offsets(props,value):
    if not 0<float(value)<=.02:
        raise ValueError('Invalid contact offset')
    for shape in props:
        shape.contact_offset=float(value)
        shape.rest_offset=0.
    return props
