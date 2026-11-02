"""Overnight continuation with D preheat, checkpoints and a wall-time limit."""
import contextlib
import io
import json
import os
import time
import torch
from .exp23_runner import Exp23AMPOnPolicyRunner
from .runner import AMPOnPolicyRunner

class Exp24AMPOnPolicyRunner(Exp23AMPOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        self.wall_started_at=time.monotonic()
        self.session_start_iteration=200
        self.best_record=None
        self.stop_reason=None
        super().__init__(*args,**kwargs)

    def warm_start(self,path,expected_sha256=None):
        with contextlib.redirect_stdout(io.StringIO()):super().warm_start(path,expected_sha256)
        source=torch.load(path,map_location='cpu')
        self.current_learning_iteration=self.session_start_iteration=source['iter']
        curriculum=source['infos']['exp21']['learning_state']['curriculum']
        self.env.gait_curriculum.migrate(curriculum,source['iter'])
        self.training_origin.update(new_iteration=source['iter'],reset=['replay'],
            curriculum_migration='exp23_stage0_to_exp24_v1')
        window=curriculum['last_window']
        self.best_record=dict(iteration=source['iter'],score=self._stability_score(window),
            file=os.path.basename(path),source_input=True,standing=window['standing'],rsi=window['rsi'])
        self._freeze_inference()
        print('ANCHORED_CONTINUATION_OK '+json.dumps(dict(self.training_origin,
            source_anchor_coef=self.alg_cfg['source_anchor_coef'],
            source_critic_adam_max_step=max(float(v['step']) for v in self.optimizer.state.values()),
            max_wall_seconds=self.cfg['max_wall_seconds'],additional_updates=self.cfg['max_iterations'])),flush=True)

    @staticmethod
    def _stability_score(window):
        return min(window[n]['mean_seconds'] for n in ('standing','rsi'))-5*max(window[n]['non_timeout_rate'] for n in ('standing','rsi'))

    def update_discriminator(self,amp,valid=None,moving=None):
        metrics=AMPOnPolicyRunner.update_discriminator(self,amp,valid,moving)
        active=bool(self.replay.size)
        if active:self.env.gait_curriculum.d_updates+=self.disc_cfg['num_updates']
        return dict(metrics,discriminator_active=float(active),
            discriminator_preheat=float(self.env.gait_curriculum.weights['amp']==0),
            discriminator_total_updates=float(self.env.gait_curriculum.d_updates))

    def after_training_iteration(self):
        report=self.env.complete_training_rollout(self.current_learning_iteration)
        saved=False
        if report is not None:
            score=self._stability_score(report)
            if self.best_record is None or score>self.best_record['score']:
                self.best_record=dict(iteration=self.current_learning_iteration,score=score,
                    file='model_'+str(self.current_learning_iteration)+'.pt',source_input=False,
                    standing=report['standing'],rsi=report['rsi'])
                saved=True
            if report['next_stage']!=report['stage']:
                # Save the last model under the previous stage before commit.
                if self.log_dir:self.save(os.path.join(self.log_dir,'stable_model_'+str(self.current_learning_iteration)+'.pt'))
                self.env.gait_curriculum.commit(report['next_stage'])
                print(getattr(self,'stage_change_marker','EXP24_STAGE_CHANGED')+' '+json.dumps(report),flush=True)
        if self.current_learning_iteration-self.session_start_iteration==50:saved=True
        if saved and self.log_dir:self.save(os.path.join(self.log_dir,'model_'+str(self.current_learning_iteration)+'.pt'))
        if self.log_dir and self.best_record is not None:
            with open(os.path.join(self.log_dir,'best_checkpoint.json'),'w') as stream:json.dump(self.best_record,stream,indent=2)
        return dict(overnight_elapsed_seconds=time.monotonic()-self.wall_started_at,
            best_stability_score=self.best_record['score'] if self.best_record else 0.)

    def training_stop_reason(self):
        reason=self.env.gait_curriculum.stop_reason
        if time.monotonic()-self.wall_started_at>=self.cfg['max_wall_seconds']:reason='wall_time_limit_5_5_hours'
        self.stop_reason=reason
        return reason

    def save(self,path,infos=None):
        details=dict(infos or {})
        details['exp24']=dict(best_record=self.best_record,session_start_iteration=self.session_start_iteration,
            stop_reason=self.stop_reason or self.env.gait_curriculum.stop_reason)
        super().save(path,details)

    def load(self,path,load_optimizer=True):
        infos=super().load(path,load_optimizer)
        state=infos['exp24'];self.best_record=state['best_record']
        self.session_start_iteration=state['session_start_iteration'];self.stop_reason=None
        return infos
