"""Source-relative constraints and consistent low-noise PPO for exp2.2."""
import copy
import torch
from .conservative_runner import ConservativeAMPOnPolicyRunner


class AnchoredAMPOnPolicyRunner(ConservativeAMPOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.actor_critic.exploration_gain=self.alg_cfg['exploration_gain']

    def _freeze_inference(self):
        critic_only=super()._freeze_inference()
        # Features open only after both the minimum warmup and a qualified gate.
        frozen=(self.current_learning_iteration<self.alg_cfg['freeze_features_iterations']
                or self.env.gait_curriculum.stage==0)
        if frozen:
            for name,param in self.actor_critic.named_parameters():
                if name.startswith(('long_history.','state_estimator.')):param.requires_grad_(False)
        return critic_only

    def policy_regularization_loss(self,obs):
        obs=obs[:512]
        with torch.no_grad():
            target=self.policy_anchor(obs)
            sigma=self.actor_critic.distribution(obs).scale.detach()
        delta=(self.actor_critic.act_inference(obs)-target)/sigma
        quiet=obs[:,-98].abs()<.1
        weights=torch.where(quiet,4.,1.)
        current=(delta.square().mean(-1)*weights).mean()
        bank=self.env.source_anchor_observations
        with torch.no_grad():expected=self.policy_anchor(bank)
        bank_loss=((self.actor_critic.act_inference(bank)-expected)/sigma.mean(0)).square().mean()
        return self.alg_cfg['source_anchor_coef']*(current+bank_loss)

    def update_policy(self,batch):
        critic_only=self._freeze_inference()
        before=None
        if not critic_only:
            before=(copy.deepcopy(self.actor_critic.state_dict()),copy.deepcopy(self.optimizer.state_dict()))
        metrics=super().update_policy(batch)
        with torch.no_grad():
            obs=self.env.source_anchor_observations
            delta=self.actor_critic.act_inference(obs)-self.policy_anchor(obs)
            raw_pd=(delta.square().mean(-1).sqrt()*self.env.cfg.control.action_scale)
            maximum=float(raw_pd.max())
        rejected=not critic_only and maximum>self.alg_cfg['source_max_pd_rms']
        if rejected:
            self.actor_critic.load_state_dict(before[0]);self.optimizer.load_state_dict(before[1])
            with torch.no_grad():
                delta=self.actor_critic.act_inference(obs)-self.policy_anchor(obs)
            metrics['source_action_rms_delta']=float((self.actor_critic.act_inference(batch['obs'][:256])-self.policy_anchor(batch['obs'][:256])).detach().square().mean().sqrt())
        metrics.update(source_bank_pd_rms=float(delta.square().mean().sqrt()*self.env.cfg.control.action_scale),
            source_bank_proposed_max_pd_rms=maximum,source_guard_rejected=float(rejected),
            source_anchor_loss=float(self.policy_regularization_loss(batch['obs']).detach()),
            exploration_gain=self.actor_critic.exploration_gain,
            effective_noise_std=float(self.actor_critic.distribution(batch['obs'][:256]).scale.detach().mean()),
            features_frozen=float(not next(self.actor_critic.long_history.parameters()).requires_grad))
        return metrics
