"""exp3.2 runner: fixed bank anchor + hard rejection for zero-command stability.

Key differences from exp3.1:
  - Generates a standing-anchor observation bank at warm_start by stepping
    the source policy through the env with vx=0.  This bank is used *every*
    update, regardless of whether the training batch contains zero-command
    samples (fixes the "zero-applied-fraction ≈ 0" problem).
  - Hard rejection: if the RMS action deviation on the bank exceeds
    `standing_action_rms_threshold`, the minibatch update is rolled back.
  - Exports the bank-anchor loss and deviation as training metrics.
"""
import copy
import torch
from .runner import AMPOnPolicyRunner


class X1Exp32AMPOnPolicyRunner(AMPOnPolicyRunner):
    def warm_start(self, path, expected_sha256=None):
        """1) Load model / optimiser, 2) freeze anchor copy, 3) collect bank."""
        super().warm_start(path, expected_sha256)
        # ── anchor copy of the source policy ──────────────────────
        self.policy_anchor = copy.deepcopy(
            self.actor_critic.get_policy_module()
        ).eval().requires_grad_(False)

        # ── collect zero-command observations from the env ─────────
        cfg = self.config.get("algorithm", {})
        bank_size = cfg.get("standing_anchor_bank_size", 256)
        if bank_size > 0 and hasattr(self.env, "generate_standing_observations"):
            self.standing_anchor_obs = self.env.generate_standing_observations(
                self.policy_anchor, num_obs=bank_size
            )
            print(f"STANDING_ANCHOR_BANK_COLLECTED {bank_size}x"
                  f"{list(self.standing_anchor_obs.shape)}",
                  flush=True)
            # restore env to a clean state
            self.env.reset()
        else:
            self.standing_anchor_obs = None

    # ──────────────────────────────────────────────────────────────
    # Anchor loss  (called per minibatch from _update_policy_minibatch)
    # ──────────────────────────────────────────────────────────────
    def policy_regularization_loss(self, obs):
        """
        Batch anchor: constrain zero-command samples to stay close to source.

        Returns:
            Tensor scalar loss, or 0 if no anchor / no zero-command samples.
        """
        # 1) zero-command samples in the current batch ─────────────
        cfg = self.config.get("algorithm", {})
        batch_loss = torch.tensor(0., device=obs.device)
        coef = cfg.get("anchor_loss_coef", 0.)
        if coef > 0 and hasattr(self, "policy_anchor") and self.policy_anchor is not None:
            threshold = cfg.get("anchor_zero_threshold", 0.01)
            zero_mask = obs[:, 0].abs() < threshold
            if zero_mask.any():
                zero_obs = obs[zero_mask]
                with torch.no_grad():
                    anchor_acts = self.policy_anchor(zero_obs)
                cur_dist = self.actor_critic.distribution(zero_obs)
                cur_acts = cur_dist.rsample()
                sigma = cur_dist.scale.detach()
                diff = (cur_acts - anchor_acts) / sigma
                batch_loss = coef * diff.square().mean()

        # 2) standing bank anchor (every update, independent of batch) ──
        bank_loss = torch.tensor(0., device=obs.device)
        bank_coef = cfg.get("standing_anchor_coef", 0.)
        if (bank_coef > 0
                and hasattr(self, "standing_anchor_obs")
                and self.standing_anchor_obs is not None
                and len(self.standing_anchor_obs) > 0):
            bank_obs = self.standing_anchor_obs.to(obs.device)
            with torch.no_grad():
                bank_target = self.policy_anchor(bank_obs)
            cur_dist = self.actor_critic.distribution(bank_obs)
            cur_acts = cur_dist.rsample()
            sigma = cur_dist.scale.detach()
            diff = (cur_acts - bank_target) / sigma
            bank_loss = bank_coef * diff.square().mean()

        return batch_loss + bank_loss

    # ──────────────────────────────────────────────────────────────
    # Update wrapper  —  hard rejection with rollback
    # ──────────────────────────────────────────────────────────────
    def update_policy(self, batch):
        # snapshot state before the update
        threshold = self.config.get("algorithm", {}).get(
            "standing_action_rms_threshold", 0.)
        if threshold > 0 and hasattr(self, "standing_anchor_obs") and self.standing_anchor_obs is not None:
            before_state = copy.deepcopy(self.actor_critic.state_dict())
            before_opt = copy.deepcopy(self.optimizer.state_dict())

        metrics = super().update_policy(batch)

        # check deviation and roll back if needed
        if threshold > 0 and hasattr(self, "standing_anchor_obs") and self.standing_anchor_obs is not None:
            with torch.no_grad():
                bank_obs = self.standing_anchor_obs.to(batch["obs"].device)
                delta = (self.actor_critic.act_inference(bank_obs)
                         - self.policy_anchor(bank_obs))
                rms = float(delta.square().mean().sqrt())

            if rms > threshold:
                self.actor_critic.load_state_dict(before_state)
                self.optimizer.load_state_dict(before_opt)
                # re-evaluate after rollback
                with torch.no_grad():
                    delta = (self.actor_critic.act_inference(bank_obs)
                             - self.policy_anchor(bank_obs))
                    rollback_rms = float(delta.square().mean().sqrt())
                metrics["anchor_guarded_rollback"] = 1.
                metrics["anchor_rms_after_rollback"] = rollback_rms
            else:
                metrics["anchor_guarded_rollback"] = 0.

            metrics["anchor_bank_rms"] = rms

        # always log anchor loss on the batch
        if hasattr(self, "policy_anchor") and self.policy_anchor is not None:
            metrics["anchor_loss"] = float(
                self.policy_regularization_loss(batch["obs"]).detach()
            )

        metrics["anchor_zero_requested"] = float(
            (batch["obs"][:, 0].abs() < 0.01).float().mean()
        )

        return metrics
