"""One low AMP stage: pooled100-update entry, smooth ramp and hysteresis."""
import copy
from .gait_curriculum import GaitCurriculum

class Exp25Curriculum(GaitCurriculum):
    def __init__(self):
        super().__init__()
        self.baseline=None
        self.run_start_iteration=1100
        self.stage_started_at=1100
        self.d_updates=0
        self.ever_amp_activated=False
        self.stop_reason=None
        self.activation_iteration=None
        self.unsafe_windows=0

    def migrate(self,state,iteration):
        if state['stage']!=0 or state['last_iteration']!=iteration or state['window_start']!=iteration:
            raise ValueError('Expected completed AMP-paused exp2.4 source')
        for key in GaitCurriculum().__dict__:setattr(self,key,copy.deepcopy(state[key]))
        self.baseline=copy.deepcopy(state['last_window'])
        self.run_start_iteration=self.stage_started_at=iteration
        self.d_updates=state['d_updates']
        self.good_windows=self.bad_windows=0
        self.stop_reason=None

    @property
    def weights(self):
        ramp=0. if self.activation_iteration is None else min(1.,max(0.,(self.last_iteration-self.activation_iteration)/100.))
        return dict(rsi_probability=.5,guide=.1,raw=.1,amp=.01*ramp,slip=0.)

    def finish(self,iteration):
        control_steps,full_speed_steps=self.control_steps,self.full_speed_steps
        report=super().finish(iteration)
        if report is None:return None
        # The base50-update window remains logged; decisions use disjoint pairs
        # from this experiment only, excluding all inherited source windows.
        report.update(qualified=False,next_stage=self.stage,stop_reason=self.stop_reason,
            control_steps=control_steps,full_speed_steps=full_speed_steps)
        recent=[w for w in self.history if w['first_iteration']>self.run_start_iteration]
        if len(recent)>=2 and all(all(w[n]['n']>=100 and w[n]['mean_seconds']<10
                for n in ('standing','rsi')) for w in recent[-2:]):
            self.stop_reason='two_windows_both_branches_below10s'
        if (iteration-self.run_start_iteration)%100==0 and len(recent)>=2:
            pair=recent[-2:]
            pooled={}
            for name in ('standing','rsi'):
                count=sum(w[name]['n'] for w in pair)
                pooled[name]=dict(n=count,**{key:sum(w[name][key]*w[name]['n'] for w in pair)/max(1,count)
                    for key in ('mean_seconds','non_timeout_rate')})
            # Weight command coverage by the observed control-step counts.
            controls=sum(w['control_steps'] for w in pair)
            coverage=sum(w['full_speed_steps'] for w in pair)/max(1,controls)
            valid=all(b['n']>=200 for b in pooled.values())
            qualified=valid and coverage>=.2 and self.baseline is not None and self.d_updates>=100
            if self.baseline is not None:
                qualified=qualified and all(b['mean_seconds']>=max(15.,.95*self.baseline[n]['mean_seconds'])
                    and b['non_timeout_rate']<=min(.55,self.baseline[n]['non_timeout_rate']+.05)
                    for n,b in pooled.items())
            report.update(pooled100=pooled,pooled_full_speed_fraction=coverage,qualified=qualified)
            if self.stage==0 and qualified:report['next_stage']=1
            unsafe=valid and any(b['mean_seconds']<13 or b['non_timeout_rate']>.6 for b in pooled.values())
            if self.ever_amp_activated:
                self.unsafe_windows=self.unsafe_windows+1 if unsafe else 0
                if self.unsafe_windows>=2:self.stop_reason='two_100update_windows_after_amp_stability_regression'
        report['stop_reason']=self.stop_reason
        return report

    def commit(self,stage):
        if self.stage!=0 or stage!=1:raise ValueError('exp2.5 has one AMP entry only')
        super().commit(stage)
        self.activation_iteration=self.stage_started_at=self.last_iteration
        self.ever_amp_activated=True

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if self.stage not in (0,1):raise ValueError('Invalid exp2.5 stage')
