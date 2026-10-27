"""Check real GPU reward components and COM-point velocities before a long run."""
import argparse
import json
from pathlib import Path
import sys

import isaacgym
import torch
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.amp.playback import inject_command
from humanoid.amp.motion import quat_mul, quat_rotate


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    custom,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    args=get_args();assert args.task=='x1_amp_exp17' and args.seed==5 and args.num_envs==1
    cfg,train_cfg=task_registry.get_cfgs(args.task);cfg.seed=5;cfg.env.num_envs=1
    cfg.env.capture_replay_state=True;cfg.env.enable_camera_sensors=False
    cfg.noise.add_noise=False;cfg.domain_rand.push_robots=False
    cfg.amp.reference_state_initialization=False;cfg.amp.rsi_probability=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    runner,_,_=task_registry.make_alg_runner(env,args.task,args=args,log_root=None)
    runner.load(custom.model,load_optimizer=False);policy=runner.get_inference_policy()
    cases=[]
    with torch.no_grad():
        for requested in (0.,-.2,.4):
            inject_command(env,requested);env.reset()
            errors={'backward_progress':0.,'tracking_lin_vel':0.,'tracking_ang_vel':0.,'point_speed':0.,'feet_scuff':0.}
            reward_names=('backward_progress','tracking_lin_vel','tracking_ang_vel','feet_scuff')
            falls=0
            selection_disagreements=0
            quaternion_norm_error=0.
            for _ in range(150):
                old={k:env.episode_sums[k].clone() for k in reward_names}
                obs=inject_command(env,requested)
                _,_,_,done,info=env.step(policy(obs))
                s=info['replay_state'];cmd=s['command'][0];v=s['velocity'][0]
                valid=(float(s['root'][0,2]-env.env_origins[0,2])>.45 and float(s['gravity'][0,2])<-.7
                       and (not bool(done[0]) or bool(s['timeout'][0])))
                bonus=min(1.,max(0.,-float(v[0])/max(.05,abs(float(cmd[0]))))) if float(cmd[0])<-.05 and valid else 0.
                metric=info['training_metrics']['Tracking/backward_progress_bonus']
                if metric.numel():errors['backward_progress']=max(errors['backward_progress'],abs(float(metric[0])-bonus))
                feet=s['feet'][0].double();speeds=[];heights=[]
                for i,mesh in enumerate(env.sole_vertices):
                    q=feet[i,3:7];mesh=mesh.double();world=quat_rotate(q.expand(len(mesh),-1),mesh)
                    height,precise_index=world[:,2].min(0)
                    # Validate the same fixed material point selected by the
                    # production float32 min, including nearly flat-foot ties.
                    raw=s['feet'][0,i,3:7];x,y,z,w=raw.unbind(-1)
                    row=torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
                    index=(row@env.sole_vertices[i].T).argmin()
                    selection_disagreements+=int(index!=precise_index)
                    quaternion_norm_error=max(quaternion_norm_error,abs(float(q.norm())-1))
                    vertex=mesh[index]
                    heights.append(height+feet[i,2]-env.env_origins[0,2])
                    com=env.foot_com_local[i].double();arm=vertex-com
                    omega=torch.cat((feet[i,10:13],feet.new_zeros(1)))
                    derivative=.5*quat_mul(omega,q);epsilon=1e-4
                    qp=q+epsilon*derivative;qm=q-epsilon*derivative;qp/=qp.norm();qm/=qm.norm()
                    # Differentiate one fixed material point, moving from COM.
                    point_v=feet[i,7:10]+(quat_rotate(qp,arm)-quat_rotate(qm,arm))/(2*epsilon)
                    speeds.append(point_v[:2].norm())
                if abs(float(cmd[0]))>.05:
                    actual=info['training_metrics']['Gait/lowest_point_speed'].double()
                    errors['point_speed']=max(errors['point_speed'],float((actual-torch.stack(speeds)).abs().max()))
                if not bool(done[0]):
                    std=.08+.22*min(abs(float(cmd[0]))/.65,1.)
                    linear=torch.exp(-(v[0]-cmd[0])**2/std**2-(v[1]-cmd[1])**2/.1**2)
                    angular=torch.exp(-(s['omega'][0,2]-cmd[2])**2/.09)
                    deficit=((.015-torch.stack(heights))/.015).clamp(0,1).square()
                    moving=(abs(float(cmd[0]))>.05)*min(abs(float(cmd[0]))/.2,1.)
                    scuff=float((deficit*torch.stack(speeds)).sum())*moving
                    for key,expected in [('backward_progress',bonus),('tracking_lin_vel',float(linear)),
                                         ('tracking_ang_vel',float(angular)),('feet_scuff',scuff)]:
                        actual=float((env.episode_sums[key]-old[key])[0]/env.reward_scales[key])
                        errors[key]=max(errors[key],abs(actual-expected))
                falls+=int(bool(done[0]) and not bool(s['timeout'][0]))
            assert all(value<1e-4 for value in errors.values()),errors
            cases.append(dict(vx=requested,steps=150,errors=errors,non_timeout_terminations=falls,
                              selection_precision_disagreements=selection_disagreements,
                              quaternion_norm_max_error=quaternion_norm_error))
    result=dict(passed=True,cases=cases,foot_com_local=env.foot_com_local.tolist(),checkpoint=runner.current_learning_iteration,
                scope='GPU runtime reward and kinematics consistency only; not gait acceptance')
    Path(custom.output).write_text(json.dumps(result,indent=2)+'\n')
    print('RUNTIME_REWARD_OK '+json.dumps(result),flush=True)
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
