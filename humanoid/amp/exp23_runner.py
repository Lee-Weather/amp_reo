"""Explicit exp2.2 continuation, preserving its critic, Adam and7000 anchor."""
import contextlib
import copy
import io
import json
import torch
from .anchored_runner import AnchoredAMPOnPolicyRunner

class Exp23AMPOnPolicyRunner(AnchoredAMPOnPolicyRunner):
    def warm_start(self,path,expected_sha256=None):
        # Reuse the strict robot/control/source compatibility contract before
        # restoring training heads. Suppress its fresh-critic success message.
        with contextlib.redirect_stdout(io.StringIO()):
            super().warm_start(path,expected_sha256)
        state=torch.load(path,map_location=self.device)
        saved=state['infos']['exp21']
        curriculum=saved['learning_state']['curriculum']
        if curriculum['stage']!=0 or state['disc_optimizer_state_dict']['state']:
            raise ValueError('Expected the audited AMP-paused exp2.2 source')
        for section in ('model_state_dict','discriminator_state_dict'):
            if any(not torch.isfinite(v).all() for v in state[section].values()):
                raise ValueError('Nonfinite continuation source')
        for entry in state['optimizer_state_dict']['state'].values():
            if any(not torch.isfinite(v).all() for v in entry.values() if torch.is_tensor(v)):
                raise ValueError('Nonfinite continuation Adam state')
        self.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
        self.discriminator.load_state_dict(state['discriminator_state_dict'],strict=True)
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(state['disc_optimizer_state_dict'])
        for group in self.optimizer.param_groups:group['lr']=self.alg_cfg['learning_rate']
        self.policy_anchor.load_state_dict(saved['anchor_state'],strict=True)
        self.env.gait_curriculum.baseline=copy.deepcopy(curriculum['last_window'])
        self.training_origin.update(mode='anchored_continuation_preserve_critic_optimizers',
            retained=['actor','long_history','state_estimator','log_std','critic','discriminator',
                'normalizer','policy_optimizer','disc_optimizer','original7000_anchor'],
            reset=['curriculum_windows','iteration_counter','replay'],
            parent_training_origin=state['training_origin'])
        self._freeze_inference()
        print('ANCHORED_CONTINUATION_OK '+json.dumps(dict(self.training_origin,
            source_anchor_coef=self.alg_cfg['source_anchor_coef'],
            source_critic_adam_max_step=max(float(v['step']) for v in self.optimizer.state.values()))),flush=True)
