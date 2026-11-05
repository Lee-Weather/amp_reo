"""Read-only privileged-velocity interventions for an existing DH policy."""
import torch


def diagnostic_action(model, observations, actual_scaled_velocity, mode, step):
    if mode not in ('estimated', 'oracle_all', 'oracle_first5') or step < 0:
        raise ValueError('Unknown velocity intervention or negative policy step')
    estimated = model.estimate_velocity(observations)
    if (actual_scaled_velocity.shape != estimated.shape
            or not torch.isfinite(actual_scaled_velocity).all().item()
            or not torch.isfinite(estimated).all().item()):
        raise ValueError('Invalid diagnostic velocity dimensions or values')
    overridden = mode == 'oracle_all' or (mode == 'oracle_first5' and step < 5)
    used = actual_scaled_velocity if overridden else estimated
    if overridden:
        short = observations[..., -model.num_short_obs:]
        encoded = model.long_history(observations.reshape(-1, model.in_channels, model.num_proprio_obs))
        action = model.actor(torch.cat((short, used, encoded), dim=-1))
    else:
        # Preserve the original path exactly in the baseline and after a startup-only intervention.
        action = model.act_inference(observations)
    return action, estimated, used, overridden
