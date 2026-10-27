"""Reward geometry shared by CPU diagnostics and the next AMP experiment."""
import math
import torch
from .motion import quat_rotate


def tracking_denominator(command_vx, minimum_std=.08, maximum_std=.30, maximum_speed=.65):
    fraction = (command_vx.abs()/maximum_speed).clamp(max=1.)
    return (minimum_std+(maximum_std-minimum_std)*fraction).square()


def style_multiplier(command_vx, actual_vx, stand_gate, standing_floor=.2, motion_threshold=.05):
    """Keep quiet AMP at zero; moving AMP requires progress in the requested direction."""
    standing = 1-(1-standing_floor)*stand_gate
    progress = (actual_vx*command_vx.sign()/command_vx.abs().clamp(min=motion_threshold)).clamp(0, 1)
    return standing*torch.where(command_vx.abs() > motion_threshold, progress, torch.ones_like(progress))


def moving_speed_quality(command_vx, actual_vx, sigma=.12, motion_threshold=.05):
    """Suppress moving-style rewards for both under- and overspeed; keep zero unchanged."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Moving style speed tolerance must be positive")
    quality = torch.exp(-(actual_vx-command_vx).square()/sigma**2)
    return torch.where(command_vx.abs() > motion_threshold, quality, torch.ones_like(quality))


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


def lowest_sole_kinematics(feet_state, vertices, com_local, ground_height):
    """Height and velocity of the same lowest material vertex; Gym v is COM velocity."""
    heights, speeds = [], []
    for i, mesh in enumerate(vertices):
        q = feet_state[:,i,3:7]
        x,y,z,w = q.unbind(-1)
        row = torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1)
        height, ids = (row@mesh.T).min(-1)
        heights.append(height+feet_state[:,i,2]-ground_height)
        arm = quat_rotate(q, mesh[ids]-com_local[i])
        velocity = feet_state[:,i,7:10]+torch.cross(feet_state[:,i,10:13],arm,dim=-1)
        speeds.append(velocity[:,:2].norm(dim=-1))
    return torch.stack(heights,-1), torch.stack(speeds,-1)


def backward_progress(command_vx, actual_vx):
    """Bounded directional progress bonus, without overspeed or rest credit."""
    progress = (-actual_vx/command_vx.abs().clamp(min=.05)).clamp(0,1)
    return torch.where(command_vx < -.05,progress,torch.zeros_like(progress))


def backward_velocity_reward(command_vx, actual_vx, speed_scale=.2):
    """Concave signed credit: rest is zero and a zero-net oscillation cannot gain credit."""
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError("Backward reward speed scale must be positive")

    def huber(value):
        magnitude = value.abs()
        return torch.where(magnitude <= 1, .5*value.square(), magnitude-.5)

    reward = 2*(huber(command_vx/speed_scale)-huber((actual_vx-command_vx)/speed_scale))
    return torch.where(command_vx < -.05, reward, torch.zeros_like(reward))


def braking_leg_gate(planar_velocity, yaw_rate, pending, linear_scale=.08, angular_scale=.15):
    """Relax leg quiet/pose costs only during an unconfirmed moving-to-zero transition."""
    if not all(math.isfinite(s) and s > 0 for s in (linear_scale, angular_scale)):
        raise ValueError("Braking gate scales must be positive")
    settled = torch.exp(-planar_velocity.square().sum(-1)/linear_scale**2
                        -yaw_rate.square()/angular_scale**2)
    return torch.where(pending, settled, torch.ones_like(settled))
