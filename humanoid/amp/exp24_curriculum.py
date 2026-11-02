"""Overnight AMP stages, independent D warmup and bounded stagnation exit."""
import copy
from .gait_curriculum import GaitCurriculum

class Exp24Curriculum(GaitCurriculum):
    def __init__(self):
        super().__init__()
        self.baseline=None
        self.run_start_iteration=200
        self.stage_started_at=200
        self.d_updates=0
        self.ever_amp_activated=False
        self.stop_reason=None

    def migrate(self,state,iteration):
        for name in GaitCurriculum().__dict__:setattr(self,name,copy.deepcopy(state[name]))
        if self.stage!=0 or self.last_iteration!=iteration or self.window_start!=iteration:
            raise ValueError('Expected completed exp2.3 stage0 source window')
        self.baseline=copy.deepcopy(state['last_window'])
        self.run_start_iteration=self.stage_started_at=iteration
        self.good_windows=self.bad_windows=0

    @property
    def weights(self):
        return dict(rsi_probability=.5,guide=.1,raw=.1,amp=(0.,.01,.03,.05)[self.stage],slip=0.)

    def _qualified(self,report,seconds,failures):
        if self.baseline is None or report['full_speed_fraction']<.2:return False
        for name in ('standing','rsi'):
            b,source=report[name],self.baseline[name]
            if (b['n']<100 or b['mean_seconds']<max(seconds,.95*source['mean_seconds'])
                    or b['non_timeout_rate']>min(failures,source['non_timeout_rate']+.05)):return False
        return True

    def finish(self,iteration):
        good,bad=self.good_windows,self.bad_windows
        report=super().finish(iteration)
        if report is None:return None
        thresholds=((14.,.5),(16.,.35),(18.,.2),(18.,.2))
        qualified=self._qualified(report,*thresholds[self.stage])
        retained=self._qualified(report,*thresholds[max(0,self.stage-1)])
        self.good_windows=good+1 if qualified else 0
        self.bad_windows=0 if retained else bad+1
        next_stage=self.stage
        required=3 if self.stage>=2 else 2
        held=iteration-self.stage_started_at
        if (self.stage<3 and self.good_windows>=required and self.d_updates>=100
                and (self.stage==0 or held>=200)):next_stage+=1
        if self.stage>0 and self.bad_windows>=2:next_stage=self.stage-1
        report.update(qualified=qualified,retention_passed=retained,next_stage=next_stage,
            discriminator_updates=self.d_updates,stage_held_updates=held)
        recent=[r for r in self.history if r['first_iteration']>self.run_start_iteration]
        if len(recent)>=2 and all(all(r[n]['n']>=100 and r[n]['mean_seconds']<10
                for n in ('standing','rsi')) for r in recent[-2:]):
            self.stop_reason='two_windows_both_branches_below10s'
        if (len(recent)>=10 and not self.ever_amp_activated and next_stage==0
                and all(r[n]['n']>=100 for r in recent[-10:] for n in ('standing','rsi'))):
            old,new=recent[-10:-5],recent[-5:]
            improved=False
            for name in ('standing','rsi'):
                def average(windows,key):
                    total=sum(r[name]['n'] for r in windows)
                    return sum(r[name][key]*r[name]['n'] for r in windows)/max(1,total)
                improved|=average(new,'mean_seconds')>=average(old,'mean_seconds')+.3
                improved|=average(new,'non_timeout_rate')<=average(old,'non_timeout_rate')-.03
            if not improved:self.stop_reason='500_updates_amp_off_without_mean_or_failure_improvement'
        report['stop_reason']=self.stop_reason
        return report

    def commit(self,stage):
        if stage not in (0,1,2,3):raise ValueError('Invalid exp2.4 stage')
        super().commit(stage)
        self.stage_started_at=self.last_iteration
        self.ever_amp_activated|=stage>0

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if self.stage not in (0,1,2,3):raise ValueError('Invalid saved exp2.4 stage')
