"""CPU-testable heading, clearance and load helpers for exp2.13."""
import torch


def _wrap_to_pi(angle):
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def heading_error(yaw, reference):
    """Wrapped yaw error in [-pi, pi]."""
    return _wrap_to_pi(yaw - reference)


def heading_observation(error, clip, scale):
    """Clip the heading error and scale it into the unused vy observation slot."""
    if clip <= 0:
        raise ValueError("Heading clip must be positive")
    return error.clamp(-float(clip), float(clip)) * scale


def clearance_shortfall(height, reference_phase, threshold):
    """Reference-swing-weighted penalty for a foot below the clearance floor."""
    threshold = max(float(threshold), 1e-6)
    deficit = ((threshold - height) / threshold).clamp(0., 1.)
    return (deficit.square() * reference_phase).sum(-1) / reference_phase.sum(-1).clamp(min=1e-6)


def contact_share(load):
    """Per-foot share of the total vertical contact load (both zero -> 0.5)."""
    load = load.clamp(min=0.)
    total = load.sum(-1, keepdim=True)
    return torch.where(total > 1e-6, load / (total + 1e-6), torch.full_like(load, 0.5))