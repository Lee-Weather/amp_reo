"""AMP stays active; ramp only swing and stop on sustained source regression."""
import copy
from .gait_curriculum import GaitCurriculum

class Exp26Curriculum(GaitCurriculum):
    def __init__(self):
        super().__init__()
        self.stage=1
        self.baseline=None
        self.run_start_iteration=3100
        self.d_updates=0
        self.ever_amp_activated=True
        self.stop_reason=None
        self.unsafe_windows=0

    def migrate(self,state,iteration):
        if (state['stage']!=1 or state['last_iteration']!=iteration or state['window_start']!=iteration
                or not state['ever_amp_activated']):raise ValueError('Expected completed AMP-active source')
        self.last_iteration=self.window_start=self.run_start_iteration=iteration
        self.baseline=copy.deepcopy(state['last_window']['pooled100'])
        self.d_updates=state['d_updates']
        self.stop_reason=None

    @property
    def weights(self):
        return dict(rsi_probability=.5,guide=.1,raw=.1,amp=.01,slip=0.)

    @property
    def swing_multiplier(self):
        return 1.+min(1.,max(0.,(self.last_iteration-self.run_start_iteration)/200.))

    def finish(self,iteration):
        controls,full=self.control_steps,self.full_speed_steps
        report=super().finish(iteration)
        if report is None:return None
        report.update(next_stage=1,control_steps=controls,full_speed_steps=full,
            swing_multiplier=self.swing_multiplier,stop_reason=self.stop_reason)
        if len(self.history)>=2 and all(all(w[n]['n']>=100 and w[n]['mean_seconds']<10
                for n in ('standing','rsi')) for w in self.history[-2:]):
            self.stop_reason='two_windows_both_branches_below10s'
        if (iteration-self.run_start_iteration)%100==0 and len(self.history)>=2:
            pair=self.history[-2:];pooled={}
            for name in ('standing','rsi'):
                count=sum(w[name]['n'] for w in pair)
                pooled[name]=dict(n=count,**{key:sum(w[name][key]*w[name]['n'] for w in pair)/max(1,count)
                    for key in ('mean_seconds','non_timeout_rate')})
            valid=all(b['n']>=200 for b in pooled.values()) and self.baseline is not None
            unsafe=valid and any(b['mean_seconds']<self.baseline[n]['mean_seconds']-2.
                or b['non_timeout_rate']>self.baseline[n]['non_timeout_rate']+.1 for n,b in pooled.items())
            self.unsafe_windows=self.unsafe_windows+1 if unsafe else 0
            if self.unsafe_windows>=2:self.stop_reason='two_100update_windows_source_regression'
            report.update(pooled100=pooled,qualified=valid and not unsafe,
                unsafe_windows=self.unsafe_windows,baseline_nonregression_checked=valid)
        report['stop_reason']=self.stop_reason
        return report

    def commit(self,stage):
        raise ValueError('exp2.6 holds AMP stage1 throughout')

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if self.stage!=1:raise ValueError('exp2.6 requires AMP stage1')
