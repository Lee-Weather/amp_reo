"""Event-driven smooth commands; simulation time advances once per control step."""
import torch


class SmoothCommands:
    def __init__(self,count,device,hold_steps,dt,duration=2.,speed=.5953696966):
        if duration<=0 or dt<=0 or speed<=0 or hold_steps<0:
            raise ValueError('Invalid smooth command settings')
        self.requested = torch.zeros(count,3,device=device)
        self.remaining = torch.zeros(count,dtype=torch.long,device=device)
        self.active_goal = torch.zeros(count,device=device)
        self.start = torch.zeros_like(self.active_goal)
        self.value = torch.zeros_like(self.active_goal)
        self.derivative = torch.zeros_like(self.active_goal)
        self.elapsed = torch.full_like(self.active_goal,duration)
        self.elapsed_steps = torch.full((count,),round(duration/dt)+1,dtype=torch.long,device=device)
        self.started = torch.zeros(count,dtype=torch.bool,device=device)
        self.transition = torch.zeros_like(self.started)
        self.hold_steps,self.dt,self.duration,self.speed = hold_steps,dt,duration,speed

    def reset(self,ids,standing):
        self.remaining[ids] = standing.long()*self.hold_steps
        for value in (self.active_goal,self.start,self.value,self.derivative): value[ids] = 0
        self.elapsed[ids] = self.duration
        self.elapsed_steps[ids] = round(self.duration/self.dt)+1
        self.transition[ids] = False
        self.started[ids] = False

    def advance(self,reset_mask):
        valid = ~reset_mask.bool()
        self.remaining.sub_(((self.remaining>0)&valid).long())
        ready = valid & (self.remaining==0)
        self.started = ready & ((self.requested[:,0]-self.active_goal).abs()>1e-7)
        self.start[self.started] = self.value[self.started]
        self.active_goal[self.started] = self.requested[self.started,0]
        self.elapsed_steps[ready] += 1
        self.elapsed_steps[self.started] = 0
        self.elapsed = (self.elapsed_steps*self.dt).clamp(max=self.duration)
        u = (self.elapsed/self.duration).clamp(0,1)
        smooth = 3*u.square()-2*u**3
        self.value[ready] = (self.start+(self.active_goal-self.start)*smooth)[ready]
        self.derivative = (self.active_goal-self.start)*6*u*(1-u)/self.duration
        self.transition = ready & (u<1) & ((self.active_goal-self.start).abs()>1e-7)
        self.derivative[~self.transition] = 0
        self.value[self.remaining>0] = 0

    def applied(self):
        result = torch.zeros_like(self.requested)
        result[:,0] = self.value
        return result

    def blend(self):
        return (self.value/self.speed).clamp(0,1),self.derivative/self.speed
