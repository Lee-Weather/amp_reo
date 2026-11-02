"""Serializable, disjoint completed-episode gates for conservative gait transfer."""
import copy


STAGES = (
    (.7,.10,.10,0.,0.), (.7,.25,.10,0.,0.), (.5,.25,.10,0.,0.),
    (.5,.25,.10,.05,0.), (.5,.50,.10,.05,0.), (.5,.50,.25,.05,0.),
    (.2,.50,.25,.05,0.), (.2,.50,.25,.10,.1), (.2,1.,.25,.10,.1),
    (.2,1.,.5,.10,.1), (.2,1.,.5,.25,.5), (0.,1.,.5,.25,.5),
    (0.,1.,1.,.25,.5), (0.,1.,1.,.5,1.),
)


class GaitCurriculum:
    def __init__(self):
        self.stage=0
        self.good_windows=0
        self.bad_windows=0
        self.window_start=0
        self.last_iteration=0
        self.pending=[[0.,0.,0.,0.],[0.,0.,0.,0.]]
        self.control_steps=0
        self.full_speed_steps=0
        self.history=[]
        self.last_window=None

    @property
    def weights(self):
        return dict(zip(('rsi_probability','guide','raw','amp','slip'),STAGES[self.stage]))

    def add(self, branches, control_steps, full_speed_steps):
        if len(branches)!=2 or any(len(row)!=4 for row in branches):
            raise ValueError('Expected standing/RSI count, seconds, failure and early-failure sums')
        for dst,src in zip(self.pending,branches):
            for i,v in enumerate(src): dst[i]+=float(v)
        self.control_steps+=int(control_steps)
        self.full_speed_steps+=int(full_speed_steps)

    def finish(self,iteration):
        if iteration<self.last_iteration: raise ValueError('Curriculum iteration moved backwards')
        self.last_iteration=iteration
        if iteration-self.window_start<50: return None
        branches=[]
        for count,seconds,fail,early in self.pending:
            branches.append(dict(n=int(count),mean_seconds=seconds/max(1,count),
                non_timeout_rate=fail/max(1,count),early_failure_rate=early/max(1,count)))
        coverage=self.full_speed_steps/max(1,self.control_steps)
        required=[branches[0]] if self.weights['rsi_probability']==0 else branches
        qualified=all(b['n']>=100 and b['mean_seconds']>=15 and b['non_timeout_rate']<=.2 for b in required)
        qualified=qualified and coverage>=.2
        self.good_windows=self.good_windows+1 if qualified else 0
        bad=any(b['n']>=100 and b['mean_seconds']<10 for b in required)
        self.bad_windows=self.bad_windows+1 if bad else 0
        next_stage=self.stage
        if self.good_windows>=3 and self.stage<len(STAGES)-1: next_stage+=1
        elif self.bad_windows>=2 and self.stage>0: next_stage-=1
        report=dict(first_iteration=self.window_start+1,last_iteration=iteration,
            standing=branches[0],rsi=branches[1],full_speed_fraction=coverage,
            stage=self.stage,next_stage=next_stage,qualified=qualified)
        self.last_window=report
        self.history.append(report)
        self.history=self.history[-20:]
        self.pending=[[0.,0.,0.,0.],[0.,0.,0.,0.]]
        self.control_steps=self.full_speed_steps=0
        self.window_start=iteration
        return report

    def commit(self,stage):
        if abs(stage-self.stage)>1 or not 0<=stage<len(STAGES): raise ValueError('Invalid curriculum transition')
        self.stage=stage
        self.good_windows=self.bad_windows=0

    def state_dict(self): return copy.deepcopy(self.__dict__)

    def load_state_dict(self,state):
        fresh=type(self)()
        if set(state)!=set(fresh.__dict__): raise ValueError('Incomplete curriculum checkpoint')
        if not 0<=state['stage']<len(STAGES): raise ValueError('Invalid saved curriculum stage')
        self.__dict__.update(copy.deepcopy(state))
