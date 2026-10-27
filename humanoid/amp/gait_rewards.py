"""Reward geometry shared by CPU diagnostics and the next AMP experiment."""
import torch


def tracking_denominator(command_vx, minimum_std=.08, maximum_std=.30, maximum_speed=.65):
    fraction = (command_vx.abs()/maximum_speed).clamp(max=1.)
    return (minimum_std+(maximum_std-minimum_std)*fraction).square()


def style_multiplier(command_vx, actual_vx, stand_gate, standing_floor=.2, motion_threshold=.05):
    """Keep quiet AMP at zero; moving AMP requires progress in the requested direction."""
    standing = 1-(1-standing_floor)*stand_gate
    progress = (actual_vx*command_vx.sign()/command_vx.abs().clamp(min=motion_threshold)).clamp(0, 1)
    return standing*torch.where(command_vx.abs() > motion_threshold, progress, torch.ones_like(progress))


def sole_heights(feet_state, vertices, ground_height):
    """Lowest collision vertex in world z, relative to the plane for each foot."""
    heights = []
    for i, mesh in enumerate(vertices):
        x, y, z, w = feet_state[:, i, 3:7].unbind(-1)
        zrow = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
        heights.append((zrow @ mesh.T).min(-1).values+feet_state[:, i, 2]-ground_height)
    return torch.stack(heights, -1)


def scuff_cost(height, horizontal_speed, command_vx, clearance=.015):
    deficit = ((clearance-height)/clearance).clamp(min=0, max=1).square()
    moving = (command_vx.abs() > .05).float()*(command_vx.abs()/.2).clamp(max=1.)
    return (deficit*horizontal_speed).sum(-1)*moving
