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


def static_cycle_baseline(reference_cycle, scale):
    """Per-phase error of the cycle-optimal constant pose under Huber loss.

    A Huber location estimate with a symmetric loss is minimised by the
    median, so the baseline is independent of the policy. A policy that holds
    that constant receives exactly zero credit at every phase.
    """
    if reference_cycle.ndim != 2 or reference_cycle.shape[0] < 2:
        raise ValueError("A reference cycle must be a non-degenerate 2-D array")
    if (scale <= 0).any():
        raise ValueError("Reference error scales must be positive")
    static = torch.median(reference_cycle.detach(), dim=0).values
    return huber_terms(static.expand_as(reference_cycle), reference_cycle.detach(), scale)


def absolute_phase_credit(actual, reference, static_error, scale,
                          negative_floor=-.05, positive_ceiling=1.):
    """Signed per-coordinate Huber credit against a policy-independent baseline.

    Persistent deviation is charged directly, without the `tanh` improvement
    rate used by exp1.16. Bounded credit keeps the dense cost from rewarding
    an early fall; the caller must still pair it with an event-level failure
    penalty larger than the worst-case accumulated negative credit.
    """
    if not negative_floor <= 0 < positive_ceiling:
        raise ValueError("Absolute phase credit bounds must straddle zero")
    if static_error.shape != reference.shape:
        raise ValueError("Static baseline must share the reference shape")
    actual_error = huber_terms(actual, reference, scale)
    credit = (static_error-actual_error).mean(-1)
    return credit.clamp(negative_floor, positive_ceiling)


def static_velocity_baseline(reference_velocity, scale):
    """Error of the zero joint-velocity baseline for every phase."""
    if (scale <= 0).any():
        raise ValueError("Reference velocity scales must be positive")
    return huber_terms(torch.zeros_like(reference_velocity), reference_velocity.detach(), scale)


def foot_support_score(actual_height, target_height, foot_force, nominal_weight,
                       height_scale=.02, load_scale=.25, load_fraction=.5):
    """Positive per-foot clearance/load agreement with a total-load gate.

    The reference and actual heights must both be minimum collision-sole
    heights. Swing/stance targets are continuous, and each foot is scored
    independently; an unloaded (airborne) robot cannot collect support credit.
    """
    if (height_scale <= 0) or (load_scale <= 0) or not 0 < load_fraction <= 1:
        raise ValueError("Support scales and load fraction must be positive")
    if actual_height.shape != target_height.shape or foot_force.shape != target_height.shape:
        raise ValueError("Support height and force shapes must match")
    target = target_height.clamp(min=0)
    target_load = torch.exp(-(target/.012).square())
    force = foot_force.clamp(min=0)
    total = force.sum(-1)
    share = force/(total[:, None]+1e-6)
    height_error = huber_terms(actual_height, target, torch.full_like(target, height_scale))
    load_error = huber_terms(share, target_load, torch.full_like(target_load, load_scale))
    gate = (total/(load_fraction*nominal_weight.clamp(min=1.))).clamp(0, 1)
    return (1/(1+height_error+load_error)).mean(-1)*gate
