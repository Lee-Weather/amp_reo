"""No-learning32-phase native RSI feasibility and reset consistency probe."""
from pathlib import Path
import sys
import json
import isaacgym
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from humanoid.envs import *
from humanoid.utils import get_args,task_registry
from humanoid.amp.gait_curriculum import STAGES


def main():
    args=get_args()
    if args.task!='x1_amp_exp21' or args.num_envs!=32 or not args.warm_start:
        raise ValueError('Use exp2.1,32 environments and its verified source')
    cfg,train=task_registry.get_cfgs(args.task)
    cfg.seed=5;cfg.noise.add_noise=False
    train.runner.policy_transfer_diagnostic_mode=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    env.diagnostic_reset_phases=torch.arange(32,device=env.device)/32
    runner,_,_=task_registry.make_alg_runner(env,args=args,train_cfg=train,log_root=None)
    env.reset()
    phase=env.gait_phase.clone()
    expected=env.reference_guide.state(phase)
    torch.testing.assert_close(env.dof_pos,expected[:,13:42])
    torch.testing.assert_close(env.dof_vel,expected[:,42:])
    torch.testing.assert_close(env.root_states[:,7:13],expected[:,7:13])
    assert not env.reset_standing.any() and not env.standing_commands.remaining.any()
    assert (env.commands[:,0]>.99*cfg.commands.native_speed).all()
    names=['reference_pose','reference_lateral','reference_waist','reference_upper','reference_velocity',
           'reference_feet','reference_swing','reference_support','stance_slip','joint_margin','raw_target_violation']
    term_sums=torch.zeros(len(names),2,device=env.device)
    total_sums=torch.zeros(3,device=env.device)
    compute=env.compute_reward
    def audit_reward():
        before={n:env.episode_sums[n].clone() for n in names}
        compute()
        low=torch.stack([env.episode_sums[n]-before[n] for n in names])
        stage=env.gait_curriculum.stage
        env.gait_curriculum.stage=len(STAGES)-1
        full=torch.stack([getattr(env,'_reward_'+n)()*env.reward_scales[n] for n in names])
        env.gait_curriculum.stage=stage
        term_sums[:,0].add_(low.sum(-1));term_sums[:,1].add_(full.sum(-1))
        total_sums[0]+=env.rew_buf.sum()
        total_sums[1]+=(env.rew_buf+full.sum(0)-low.sum(0)).sum()
        total_sums[2]+=env.num_envs
    env.compute_reward=audit_reward
    policy=runner.get_inference_policy(device=env.device)
    failed=torch.zeros(32,device=env.device,dtype=torch.bool)
    first=torch.full((32,),10.,device=env.device)
    max_force=0.;minimum_height=1.
    with torch.no_grad():
        for step in range(500):
            obs,_,reward,done,info=env.step(policy(env.get_observations()))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            bad=done.bool()&~info['time_outs'].bool()
            first[bad&~failed]=(step+1)*env.dt
            failed|=bad
            if step<5:
                max_force=max(max_force,float(env.contact_forces[:,env.feet_indices,2].abs().max()))
                minimum_height=min(minimum_height,float(env.root_states[:,2].min()))
    report=dict(learning_updates=0,phases=32,seconds=500*env.dt,
        reset_q_dq_root_velocity_phase_consistent=True,all_values_finite=True,
        initial_max_foot_force_N=max_force,initial_min_root_height_m=minimum_height,
        phase_first_failure_seconds=first.cpu().tolist(),failure_fraction=float(failed.float().mean()),
        failure_observed=failed.cpu().tolist(),observation_horizon_seconds=10.,
        phase_survival_at_1s=float((first>1).float().mean()),phase_survival_at_3s=float((first>3).float().mean()),
        interpretation='Source-policy recovery diagnostic, not a proof of reference dynamics or learned gait.')
    # Also expose the distribution used by training. Deterministic playback
    # alone can conceal failures caused by stochastic action/observation noise.
    cfg.noise.add_noise=True
    env.reset()
    sampled_failed=torch.zeros(32,device=env.device,dtype=torch.bool)
    sampled_first=torch.full((32,),10.,device=env.device)
    with torch.no_grad():
        for step in range(500):
            actions=runner.actor_critic.distribution(env.get_observations()).sample()
            obs,_,reward,done,info=env.step(actions)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            bad=done.bool()&~info['time_outs'].bool()
            sampled_first[bad&~sampled_failed]=(step+1)*env.dt
            sampled_failed|=bad
    report['training_noise_probe']=dict(failure_observed=sampled_failed.cpu().tolist(),
        phase_first_failure_seconds=sampled_first.cpu().tolist(),
        phase_survival_at_1s=float((sampled_first>1).float().mean()),
        phase_survival_at_3s=float((sampled_first>3).float().mean()),
        mean_censored_survival_seconds=float(sampled_first.mean()),
        observation_noise=True,source_action_distribution_unchanged=True)
    report['real_physics_reward_audit']=dict(samples=int(total_sums[2]),
        mean_low_weight_task_reward=float(total_sums[0]/total_sums[2]),
        mean_full_weight_counterfactual_task_reward=float(total_sums[1]/total_sums[2]),
        new_terms={n:dict(low=float(term_sums[i,0]/total_sums[2]),full=float(term_sums[i,1]/total_sums[2])) for i,n in enumerate(names)},
        scope='Measured simulator states, contacts, torques and actions under7000. Full-weight counterfactual re-evaluates only the11 changed pure reward terms on identical snapshots; no second simulation or cumulative-return prediction.')
    # Invalid reset/state and explosive first-step contacts block training. A
    # source policy can still fail later: that is what this validation trains.
    report['reset_feasibility_passed']=minimum_height>.45 and max_force<20*36.46*9.81 and report['phase_survival_at_1s']>=.75
    output=ROOT/'logs/exp21_gpu_preflight.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('EXP21_GPU_PREFLIGHT '+json.dumps(report),flush=True)
    if not report['reset_feasibility_passed']:raise RuntimeError('Native RSI resets need repair before training')


if __name__=='__main__':main()
