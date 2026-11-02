"""Low-weight AMP entry after200 new updates and two qualified windows."""
from .gait_curriculum import GaitCurriculum

class Exp23Curriculum(GaitCurriculum):
    def __init__(self):
        super().__init__()
        self.baseline=None

    @property
    def weights(self):
        return dict(rsi_probability=.5,guide=.1,raw=.1,
            amp=.01 if self.stage==1 and self.last_iteration>=200 else 0.,slip=0.)

    def finish(self,iteration):
        good,bad=self.good_windows,self.bad_windows
        report=super().finish(iteration)
        if report is None:return None
        branches=[report['standing'],report['rsi']]
        qualified=all(b['n']>=100 and b['mean_seconds']>=14 and b['non_timeout_rate']<=.5 for b in branches)
        qualified=qualified and report['full_speed_fraction']>=.2 and self.baseline is not None
        if self.baseline is not None:
            for name,b in zip(('standing','rsi'),branches):
                source=self.baseline[name]
                qualified=qualified and b['mean_seconds']>=.95*source['mean_seconds']
                qualified=qualified and b['non_timeout_rate']<=source['non_timeout_rate']+.05
        self.good_windows=good+1 if qualified else 0
        self.bad_windows=0 if qualified else bad+1
        next_stage=self.stage
        if self.stage==0 and iteration>=200 and self.good_windows>=2:next_stage=1
        if self.stage==1 and self.bad_windows>=2:next_stage=0
        report.update(qualified=qualified,next_stage=next_stage,entry_earliest_new_iteration=200,
            entry_good_windows=self.good_windows,baseline_nonregression_checked=self.baseline is not None)
        return report

    def commit(self,stage):
        if stage not in (0,1):raise ValueError('exp2.3 supports only paused/low AMP stages')
        super().commit(stage)

    def load_state_dict(self,state):
        super().load_state_dict(state)
        if self.stage not in (0,1):raise ValueError('Invalid exp2.3 stage')
