"""Independently verify signed reverse and braking leg rewards in real GPU steps."""
import argparse
import json
import math
from pathlib import Path
import sys

import isaacgym
import torch

from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.amp.playback import inject_command


def huber(x):
    return .5*x*x if abs(x)<=1 else abs(x)-.5


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    custom,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    args=get_args();assert args.task in ('x1_amp_exp18','x1_amp_exp19','x1_amp_exp110') and args.seed==5 and args.num_envs==1
    cfg,train_cfg=task_registry.get_cfgs(args.task);cfg.seed=5;cfg.env.num_envs=1
    cfg.env.capture_replay_state=True;cfg.env.enable_camera_sensors=False
    cfg.noise.add_noise=False;cfg.domain_rand.push_robots=False
    cfg.amp.reference_state_initialization=False;cfg.amp.rsi_probability=0.
    cfg.commands.ranges.lin_vel_x=[0.,0.]
    assert cfg.rewards.backward_speed_scale==.2 and cfg.rewards.scales.backward_progress==1.5
    assert cfg.rewards.brake_leg_linear_scale==.08 and cfg.rewards.brake_leg_angular_scale==.15
    if args.task in ('x1_amp_exp19','x1_amp_exp110'):assert cfg.rewards.tracking_sigma==.04
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,_,_=task_registry.make_alg_runner(env,args.task,args=args,log_root=None)
    runner.load(custom.model,load_optimizer=False);policy=runner.get_inference_policy()
    reports=[]
    with torch.no_grad():
        for name,stages in [('zero',[(0.,150)]),('reverse',[(-.2,400)]),('braking',[(.4,250),(0.,250)])]:
            inject_command(env,stages[0][0]);env.reset()
            errors={k:0. for k in ('backward_progress','stand_dof_vel','stand_pose','leg_gate')}
            falls=0;pending_steps=0;min_gate=1.;negative_bonus_steps=0
            for requested,steps in stages:
                for _ in range(steps):
                    old={k:env.episode_sums[k].clone() for k in errors if k!='leg_gate'}
                    obs=inject_command(env,requested)
                    _,_,_,done,info=env.step(policy(obs));s=info['replay_state']
                    falls+=int(bool(done[0]) and not bool(s['timeout'][0]))
                    if bool(done[0]):continue
                    cmd=float(s['command'][0,0]);vx,vy=[float(v) for v in s['velocity'][0,:2]]
                    wz=float(s['omega'][0,2]);pending=bool(env.stop_pending[0])
                    gate=math.exp(-(vx*vx+vy*vy)/.08**2-wz*wz/.15**2) if pending else 1.
                    actual_gate=float(env._braking_leg_gate()[0]);errors['leg_gate']=max(errors['leg_gate'],abs(gate-actual_gate))
                    pending_steps+=int(pending);min_gate=min(min_gate,gate)
                    valid=float(s['root'][0,2]-env.env_origins[0,2])>.45 and float(s['gravity'][0,2])<-.7
                    bonus=2*(huber(cmd/.2)-huber((vx-cmd)/.2)) if cmd<-.05 and valid else 0.
                    negative_bonus_steps+=int(bonus<0)
                    q=s['q'][0,env.leg_indices].double();dq=s['dq'][0,env.leg_indices].double()
                    target=env.standing_state[13:42][env.leg_indices].double()
                    ramp=float(env.stand_gate[0])
                    expected={'backward_progress':bonus,
                              'stand_dof_vel':ramp*gate*float(dq.square().mean())/.5**2,
                              'stand_pose':ramp*gate*float((q-target).square().mean())/.2**2}
                    for key,value in expected.items():
                        actual=float((env.episode_sums[key]-old[key])[0]/env.reward_scales[key])
                        errors[key]=max(errors[key],abs(actual-value))
            assert all(v<1e-4 for v in errors.values()),errors
            reports.append(dict(case=name,stages=stages,errors=errors,non_timeout_terminations=falls,
                                pending_steps=pending_steps,minimum_leg_gate=min_gate,negative_bonus_steps=negative_bonus_steps))
    assert reports[-1]['pending_steps']>0 and reports[-1]['minimum_leg_gate']<.5
    result=dict(passed=True,checkpoint=runner.current_learning_iteration,cases=reports,
                scope='GPU reward consistency and short fixed-command stability, not full gait acceptance')
    Path(custom.output).write_text(json.dumps(result,indent=2)+'\n')
    print('EXP18_RUNTIME_OK '+json.dumps(result),flush=True)
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
