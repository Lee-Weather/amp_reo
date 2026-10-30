"""Signed, per-joint phase credit without a history-window eligibility gate."""
import torch


def huber_terms(actual, target, scale):
    error = ((actual-target)/scale.clamp(min=1e-6)).abs()
    return torch.where(error <= 1., .5*error.square(), error-.5)


def phase_quality(actual, target, scale):
    # Diagnostic only: large joint errors do not exponentially suppress all
    # other joints. This positive score is not paid to a stationary policy.
    return (1/(1+huber_terms(actual, target, scale))).mean(-1)


def phase_progress(previous, actual, target, scale, dt):
    """Compare BOTH states to the same current-phase target; rest is zero.

    Per-coordinate signed rates are bounded before averaging. This is a
    shaping heuristic, not a claim of potential-based policy invariance.
    """
    if dt <= 0:
        raise ValueError("Phase progress requires positive control dt")
    improvement = (huber_terms(previous, target, scale)-huber_terms(actual, target, scale))/dt
    return torch.tanh(improvement).mean(-1)


def phase_velocity_credit(actual, target, scale):
    held = huber_terms(torch.zeros_like(actual), target, scale)
    error = huber_terms(actual, target, scale)
    # Keep negative credit for motion away from/too fast for the target.
    # Static dq=0 is exactly zero for every phase, not only on average.
    return ((held-error)/(1+held)).clamp(-1., 1.).mean(-1)


def phase_support_credit(height, force, target_lift, velocity_credit):
    """Soft phase-specific clearance, stance support and swing unloading."""
    lift = target_lift.clamp(min=0)
    swing = lift/(lift+.015)
    clearance = (height.clamp(min=0)/lift.clamp(min=.015)).clamp(max=1.)
    force = force.clamp(min=0)
    stance = force.flip(-1)/(force.flip(-1)+20.)
    unloaded = 20./(force+20.)
    support = (swing*clearance*stance*unloaded).sum(-1)/swing.sum(-1).clamp(min=1e-6)
    return support*velocity_credit.clamp(min=0.)
