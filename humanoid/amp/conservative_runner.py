"""Conservative transfer hooks; the original AMP runner remains the default."""
import copy
import json
import os
import torch

from .runner import AMPOnPolicyRunner


class ConservativeAMPOnPolicyRunner(AMPOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy_anchor=None

    def warm_start(self,*args,**kwargs):
        super().warm_start(*args,**kwargs)
        self.policy_anchor=copy.deepcopy(self.actor_critic.get_policy_module()).eval().requires_grad_(False)

    def _freeze_inference(self):
        critic_only=self.current_learning_iteration<self.alg_cfg['critic_only_iterations']
        for name,param in self.actor_critic.named_parameters():
            enabled=name.startswith('critic.') or not critic_only
            if name=='log_std' and self.current_learning_iteration<self.alg_cfg['freeze_std_iterations']: enabled=False
            param.requires_grad_(enabled)
        return critic_only

    def collect_rollout(self):
        self._freeze_inference()
        return super().collect_rollout()

    def update_policy(self,batch):
        critic_only=self._freeze_inference()
        if critic_only:
            losses=[]
            for _ in range(self.alg_cfg['num_learning_epochs']):
                for ids in torch.tensor_split(torch.randperm(len(batch['obs']),device=self.device),self.alg_cfg['num_mini_batches']):
                    value=self.actor_critic.evaluate(batch['critic_obs'][ids])
                    loss=(value-batch['returns'][ids]).square().mean()*self.alg_cfg['value_loss_coef']
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite critic preheat')
                    self.optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(),self.alg_cfg['max_grad_norm'])
                    self.optimizer.step();losses.append(loss.item())
            metrics=dict(value_loss=sum(losses)/len(losses),surrogate_loss=0.,entropy=0.,kl=0.,kl_max=0.,
                         kl_early_stop=0.,policy_updates=0.,critic_updates=len(losses),state_estimator_loss=0.)
        else: metrics=super().update_policy(batch)
        with torch.no_grad():
            obs=batch['obs'][:256]
            delta=self.actor_critic.act_inference(obs)-self.policy_anchor(obs)
            metrics['source_action_rms_delta']=float(delta.square().mean().sqrt())
        metrics['critic_only']=float(critic_only)
        metrics['std_frozen']=float(self.current_learning_iteration<self.alg_cfg['freeze_std_iterations'])
        return metrics

    def update_discriminator(self,amp,valid=None,moving=None):
        if self.current_learning_iteration<self.alg_cfg['critic_only_iterations'] or not self.env.discriminator_training_enabled():
            # Keep labels paired with actual replay rows, without training D or
            # advancing its normalizer while imitation is disabled.
            data,labels=amp[valid.bool()],moving[valid.bool()]
            data,labels=data[-self.replay.capacity:],labels[-self.replay.capacity:]
            cursor,count=self.replay.cursor,len(data);first=min(count,self.replay.capacity-cursor)
            self.replay_categories[cursor:cursor+first]=labels[:first]
            self.replay_categories[:count-first]=labels[first:]
            self.replay.insert(data)
            return dict(disc_loss=0.,grad_penalty=0.,expert_score=0.,policy_score=0.,discriminator_samples=0.,discriminator_active=0.)
        return dict(super().update_discriminator(amp,valid,moving),discriminator_active=1.)

    def after_training_iteration(self):
        report=self.env.complete_training_rollout(self.current_learning_iteration)
        if report and report['next_stage']!=report['stage']:
            if self.log_dir:
                self.save(os.path.join(self.log_dir,'stable_model_'+str(self.current_learning_iteration)+'.pt'))
            self.env.gait_curriculum.commit(report['next_stage'])
            print('EXP21_STAGE_CHANGED '+json.dumps(report),flush=True)
        return {}

    def save(self,path,infos=None):
        details=dict(infos or {})
        details['exp21']=dict(learning_state=self.env.checkpoint_learning_state(),
            anchor_state=self.policy_anchor.state_dict() if self.policy_anchor is not None else None)
        super().save(path,details)

    def load(self,path,load_optimizer=True):
        infos=super().load(path,load_optimizer)
        state=infos['exp21']
        self.env.restore_learning_state(state['learning_state'])
        self.policy_anchor=copy.deepcopy(self.actor_critic.get_policy_module()).eval().requires_grad_(False)
        self.policy_anchor.load_state_dict(state['anchor_state'],strict=True)
        self._freeze_inference()
        return infos
