"""Explicit continuation of trained D, both Adam states and original anchor."""
import contextlib
import copy
import io
import json
import os
import torch
from .exp24_runner import Exp24AMPOnPolicyRunner
from .conservative_runner import ConservativeAMPOnPolicyRunner

class Exp25AMPOnPolicyRunner(Exp24AMPOnPolicyRunner):
    stage_change_marker='EXP25_STAGE_CHANGED'

    def warm_start(self,path,expected_sha256=None):
        # Validate the exact robot, controls, reference and source SHA through
        # the shared transfer contract; exp2.3's D-empty guard is inapplicable.
        with contextlib.redirect_stdout(io.StringIO()):
            ConservativeAMPOnPolicyRunner.warm_start(self,path,expected_sha256)
        source=torch.load(path,map_location=self.device)
        saved=source['infos']['exp21'];curriculum=saved['learning_state']['curriculum']
        expected_d=self.cfg.get('policy_transfer_source_discriminator_updates',900)
        if not source['disc_optimizer_state_dict']['state'] or curriculum['d_updates']!=expected_d:
            raise ValueError('Source discriminator update count differs from audited continuation contract')
        for section in ('model_state_dict','discriminator_state_dict'):
            if any(not torch.isfinite(v).all() for v in source[section].values()):raise ValueError('Nonfinite source model')
        for section in ('optimizer_state_dict','disc_optimizer_state_dict'):
            if any(not torch.isfinite(v).all() for entry in source[section]['state'].values()
                    for v in entry.values() if torch.is_tensor(v)):raise ValueError('Nonfinite source Adam')
        self.actor_critic.load_state_dict(source['model_state_dict'],strict=True)
        self.discriminator.load_state_dict(source['discriminator_state_dict'],strict=True)
        self.optimizer.load_state_dict(source['optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(source['disc_optimizer_state_dict'])
        for group in self.optimizer.param_groups:group['lr']=self.alg_cfg['learning_rate']
        self.policy_anchor.load_state_dict(saved['anchor_state'],strict=True)
        self.current_learning_iteration=self.session_start_iteration=source['iter']
        self.env.gait_curriculum.migrate(curriculum,source['iter'])
        self.stop_reason=None
        self.training_origin.update(mode='anchored_continuation_preserve_critic_optimizers',
            new_iteration=source['iter'],retained=['actor','critic','long_history','state_estimator','log_std',
                'discriminator','normalizer','policy_optimizer','disc_optimizer','original7000_anchor'],
            reset=['replay','curriculum_windows','stop_flags'],
            curriculum_migration=self.cfg.get('curriculum_migration','exp24_to_pooled_low_amp_exp25'),
            parent_training_origin=source['training_origin'])
        window=curriculum['last_window']
        self.best_record=dict(iteration=source['iter'],score=self._stability_score(window),
            file=os.path.basename(path),source_input=True,standing=window['standing'],rsi=window['rsi'])
        self._freeze_inference()
        print('ANCHORED_CONTINUATION_OK '+json.dumps(dict(self.training_origin,
            source_anchor_coef=self.alg_cfg['source_anchor_coef'],
            retained_discriminator_updates=self.env.gait_curriculum.d_updates,
            source_critic_adam_max_step=max(float(v['step']) for v in self.optimizer.state.values()),
            additional_updates=self.cfg['max_iterations'])),flush=True)
