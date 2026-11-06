"""exp3.1 runner: KL anchor for zero-command standing stability."""
import copy
import torch
from .runner import AMPOnPolicyRunner


class X1Exp31AMPOnPolicyRunner(AMPOnPolicyRunner):
    def warm_start(self, path, expected_sha256=None):
        """Save source policy (11250) snapshot as anchor."""
        super().warm_start(path, expected_sha256)
        self.policy_anchor = copy.deepcopy(
            self.actor_critic.get_policy_module()
        )
        self.policy_anchor.eval()
        self.policy_anchor.requires_grad_(False)

    def policy_regularization_loss(self, obs):
        """KL anchor loss for zero-command states.

        When the observed vx command is near zero, constrain the current
        policy output to stay close to the source policy (11250).
        """
        cfg = self.config.get('algorithm', {})
        coef = cfg.get('anchor_loss_coef', 0.)
        if coef <= 0 or not hasattr(self, 'policy_anchor') or self.policy_anchor is None:
            return torch.tensor(0., device=obs.device)

        threshold = cfg.get('anchor_zero_threshold', 0.01)
        zero_mask = obs[:, 0].abs() < threshold

        if not zero_mask.any():
            return torch.tensor(0., device=obs.device)

        zero_obs = obs[zero_mask]
        with torch.no_grad():
            anchor_actions = self.policy_anchor(zero_obs)

        cur_dist = self.actor_critic.distribution(zero_obs)
        cur_actions = cur_dist.rsample()
        sigma = cur_dist.scale.detach()
        diff = (cur_actions - anchor_actions) / sigma
        loss = diff.square().mean()

        return coef * loss
