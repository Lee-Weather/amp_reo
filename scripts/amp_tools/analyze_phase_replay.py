"""Audit recorded phase targets, guidance credit, limits and the pre-fall state."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_gait import geometry, frequency
from humanoid.amp.motion import MotionLibrary, URDFKinematics
from humanoid.amp.grounding import FootGrounding
from humanoid.amp.phase_reference import PhaseReference
from humanoid.amp.gait_guidance import MotionScales
from humanoid.amp.phase_rewards import phase_progress, phase_velocity_credit, phase_support_credit, huber_terms


def corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    job = json.loads((a.output_dir/'replay_job.json').read_text())
    assert hashlib.sha256(a.model.read_bytes()).hexdigest() == job['checkpoint_sha256']
    for name in ('motion.py', 'grounding.py', 'phase_reference.py', 'phase_rewards.py', 'gait_guidance.py'):
        path = 'humanoid/amp/'+name
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == job['source_file_sha256'][path]
    with a.csv.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    d = {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    assert all(np.isfinite(v).all() for v in d.values())
    names, dt = ck['metadata']['dof_names'], ck['metadata']['policy_dt']
    def matrix(prefix):
        return np.column_stack([d[prefix+n] for n in names])
    q, dq, rq, rdq = [matrix(prefix) for prefix in ('q_', 'dq_', 'reference_q_', 'reference_dq_')]
    raw, effective = matrix('pd_target_raw_'), matrix('pd_target_effective_')
    cmd, phase = d['command_applied'], d['reference_phase']
    root = torch.tensor(np.column_stack([d[k] for k in ('x','y','z','qx','qy','qz','qw')]), dtype=torch.float32)
    urdf = ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf'
    cfg = ck['config']['amp']
    lib = MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized', urdf, names, cfg['key_body_names'],
        dt=dt, history_steps=cfg['history_steps'], weights=cfg['motion_weights'],
        limit_tolerance=cfg['limit_tolerance'], ground_clearance=cfg['ground_clearance'])
    standing = torch.tensor(ck['metadata']['control']['standing_state'], dtype=torch.float32)
    reference = PhaseReference(lib, standing[13:42], cfg['reference_cycles'])
    fk = URDFKinematics(urdf, names, [s+'_ankle_roll_link' for s in ('left','right')])
    grounding = FootGrounding(urdf, fk, cfg['ground_clearance'])
    local, world, sole = geometry(root, torch.tensor(q, dtype=torch.float32), fk, grounding)
    tq, tdq, trq = [torch.tensor(v, dtype=torch.float32) for v in (q,dq,rq)]
    reproduced_q, reproduced_dq = reference.sample(torch.tensor(cmd,dtype=torch.float32), torch.tensor(phase,dtype=torch.float32))
    target_error = dict(q_max_abs=float(np.abs(reproduced_q.numpy()-rq).max()),
                        dq_max_abs=float(np.abs(reproduced_dq.numpy()-rdq).max()))
    assert max(target_error.values()) < 1e-4, target_error
    ref_feet = fk.key_positions(trq)
    ref_rot = standing[3:7].expand(len(q), -1)
    ref_root = grounding.align(torch.zeros(len(q),3), ref_rot, trq)
    _, _, reference_sole = geometry(torch.cat((ref_root,ref_rot),-1), trq, fk, grounding)
    scales = MotionScales(reference, fk, reference.pitch_indices, [.1,.3,.4,.6,.65,-.2]).sample(torch.tensor(cmd,dtype=torch.float32))
    ids = reference.pitch_indices
    previous = torch.roll(tq,1,0)
    pose_credit = phase_progress(previous[:,ids],tq[:,ids],trq[:,ids],scales['q'],dt).numpy()
    feet_tensor = torch.tensor(local).flatten(-2)
    feet_credit = phase_progress(torch.roll(feet_tensor,1,0), feet_tensor, ref_feet.flatten(-2),scales['feet'],dt).numpy()
    velocity_credit = phase_velocity_credit(tdq[:,ids],torch.tensor(rdq[:,ids],dtype=torch.float32),scales['dq']).numpy()
    support_credit = phase_support_credit(torch.tensor(sole),torch.tensor(np.column_stack([d['foot_force_'+s] for s in ('l','r')]),dtype=torch.float32),
        ref_feet[:,:,2]-scales['feet_min_z'],torch.tensor(velocity_credit)).numpy()
    same = np.r_[False,(d['episode_id'][1:]==d['episode_id'][:-1]) & (np.abs(cmd[1:]-cmd[:-1])<1e-6)]
    moving = (cmd>.05) & ~d['done'].astype(bool) & (d['episode_step']>175) & (d['z']>.45) & (d['gz']<-.7) & same
    limits = np.array(ck['metadata']['control']['hard_limits'])
    transition_valid = same & (d['episode_step']>1)
    per_joint = {}
    for i, name in enumerate(names):
        m = moving
        clip = d['pd_target_clip_counts_'+name]/ck['config']['control']['decimation']
        per_joint[name] = dict(actual_mean_deg=float(np.rad2deg(q[m,i].mean())), reference_mean_deg=float(np.rad2deg(rq[m,i].mean())),
            mean_offset_deg=float(np.rad2deg((q[m,i]-rq[m,i]).mean())), tracking_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((q[m,i]-rq[m,i])**2)))),
            q_correlation=corr(q[m,i],rq[m,i]), dq_correlation=corr(dq[m,i],rdq[m,i]), raw_clip_fraction=float(clip[m].mean()),
            actual_near_hard_bound_1deg=float(((q[m,i]-limits[i,0]<np.deg2rad(1)) | (limits[i,1]-q[m,i]<np.deg2rad(1))).mean()),
            actual_frequency=frequency(q[(d['time_s']>=4)&(d['time_s']<18),i],dt),
            reference_frequency=frequency(rq[(d['time_s']>=4)&(d['time_s']<18),i],dt))
        if i in ids:
            j=ids.index(i)
            error=huber_terms(tq[:,ids],trq[:,ids],scales['q']).numpy()[:,j]
            improvement=(huber_terms(previous[:,ids],trq[:,ids],scales['q']).numpy()[:,j]-error)/dt
            per_joint[name].update(mean_normalized_huber_error=float(error[m].mean()),
                signed_progress_mean=float(np.tanh(improvement[m]).mean()),
                progress_saturated_abs_over_09_fraction=float((np.abs(np.tanh(improvement[m]))>.9).mean()))
    phase_increment=np.remainder(np.diff(phase),1.)
    phase_rate=reference.phase_rate(torch.tensor(cmd,dtype=torch.float32)).numpy()
    mm=moving[1:] & same[1:]
    phase_clock_error=np.abs(phase_increment[mm]-dt*phase_rate[1:][mm])
    falls=np.flatnonzero(d['done'].astype(bool) & ~d['timeout'].astype(bool))
    fall_context=[]
    for index in falls:
        event=dict(time_s=float(d['time_s'][index]), windows={})
        for seconds in (4.,2.,1.,.5):
            mask=(d['time_s']>=d['time_s'][index]-seconds)&(d['time_s']<d['time_s'][index])&(d['episode_id']==d['episode_id'][index])
            event['windows'][str(seconds)]=dict(mean_vx=float(d['base_vel_x'][mask].mean()), mean_abs_vy=float(np.abs(d['base_vel_y'][mask]).mean()),
                maximum_abs_roll_deg=float(np.rad2deg(np.abs(d['base_euler_x'][mask])).max()),
                minimum_height=float(d['z'][mask].min()), minimum_force_left=float(d['foot_force_l'][mask].min()),minimum_force_right=float(d['foot_force_r'][mask].min()))
        for seconds in (4.,2.,1.,.5,.2,0.):
            k=max(0,index-round(seconds/dt)); event.setdefault('snapshots',[]).append(dict(time_s=float(d['time_s'][k]),
                vx=float(d['base_vel_x'][k]),vy=float(d['base_vel_y'][k]),height=float(d['z'][k]),roll_deg=float(np.rad2deg(d['base_euler_x'][k])),
                pitch_deg=float(np.rad2deg(d['base_euler_y'][k])),left_force=float(d['foot_force_l'][k]),right_force=float(d['foot_force_r'][k]),
                left_knee_deg=float(np.rad2deg(q[k,names.index('left_knee_pitch_joint')])),right_knee_deg=float(np.rad2deg(q[k,names.index('right_knee_pitch_joint')]))))
        fall_context.append(event)
    offline=dict(pose_mean=float(pose_credit[moving].mean()),feet_mean=float(feet_credit[moving].mean()),
        velocity_mean=float(velocity_credit[moving].mean()),support_mean=float(support_credit[moving].mean()),
        positive_velocity_fraction=float((velocity_credit[moving]>0).mean()),
        zero_support_fraction=float((support_credit[moving]<1e-6).mean()),
        support_target_link_lift_vs_grounded_sole_mae_m=float(np.abs((ref_feet[:,:,2]-scales['feet_min_z']).numpy()[moving]-reference_sole[moving]).mean()))
    windows=np.zeros(len(q),dtype=bool)
    for start in (4.,24.,44.):windows|=(d['time_s']>=start)&(d['time_s']<start+14.)
    clearance={}
    for index,side in enumerate(('left','right')):
        raised=(sole[:,index]>.01)&windows
        desired=(reference_sole[:,index]>.01)&windows
        def runs(flags):
            edges=np.diff(np.r_[False,flags,False].astype(int))
            durations=(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*dt
            valid=durations[durations>=.08-1e-7]
            return dict(count_over_80ms=len(valid),median_duration_over_80ms_s=float(np.median(valid)) if len(valid) else None)
        clearance[side]=dict(policy_fraction_above_1cm=float(raised[windows].mean()),
            reference_fraction_above_1cm=float(desired[windows].mean()),policy_runs=runs(raised),reference_runs=runs(desired),
            raised_frames_in_reference_raised_phase_fraction=float((raised&desired).sum()/max(1,raised.sum())))
    # A closed motion with no net error improvement can nevertheless receive
    # positive saturated rate credit: slow approach followed by fast retreat.
    path=torch.cat((torch.linspace(.5,.3,91),torch.linspace(.3,.5,11)[1:]))[:,None]
    exploit=phase_progress(path[:-1],path[1:],torch.zeros_like(path[1:]),torch.full_like(path[1:],.1),dt)
    counterexample=dict(description='Synthetic fixed target0, position0.3..0.5rad, 90-step approach and10-step return, scale0.1rad; illustrative, not measured policy intent.',
        initial_position=float(path[0]),final_position=float(path[-1]),mean_progress_credit=float(exploit.mean()),
        net_error_improvement=0.,phase_tracking_achieved=False)
    # Proposed exp1.17 state costs, evaluated only as offline reward ordering.
    # No policy is optimized and kinematic reference examples are not physics.
    candidate={}
    mean_pose=trq[moving].mean(0).expand_as(trq)
    twice_q,twice_dq=reference.sample(torch.tensor(cmd,dtype=torch.float32),torch.tensor(phase*2,dtype=torch.float32))
    cases={'recorded_policy':(tq,tdq),'exact_phase_reference':(trq,torch.tensor(rdq,dtype=torch.float32)),
           'fixed_standing':(standing[13:42].expand_as(trq),torch.zeros_like(tdq)),
           'fixed_cycle_mean':(mean_pose,torch.zeros_like(tdq)),
           'twenty_percent_amplitude':(standing[13:42]+.2*(trq-standing[13:42]),.2*torch.tensor(rdq,dtype=torch.float32)),
           'double_phase_rate':(twice_q,2*twice_dq)}
    for label,(cq,cv) in cases.items():
        pose=-2*huber_terms(cq[:,ids],trq[:,ids],scales['q'].clamp(min=.1)).mean(-1)
        velocity=-.5*huber_terms(cv[:,ids],torch.tensor(rdq[:,ids],dtype=torch.float32),scales['dq'].clamp(min=.2)).mean(-1)
        feet=-huber_terms(fk.key_positions(cq).flatten(-2),ref_feet.flatten(-2),scales['feet'].clamp(min=.02)).mean(-1)
        candidate[label]=dict(pose=float(pose[moving].mean()),velocity=float(velocity[moving].mean()),
                             feet=float(feet[moving].mean()),total=float((pose+velocity+feet)[moving].mean()))
    report=dict(checkpoint_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),
        training_commit=job['training_commit'], settled_samples=int(moving.sum()),target_reproduction=target_error,
        phase_clock_max_increment_error=float(phase_clock_error.max()), reference_rate_hz=float(np.median(phase_rate[moving])),
        per_joint=per_joint,offline_guidance_pre_dt_unweighted=offline,phase_clearance=clearance,fall_context=fall_context,
        closed_cycle_rate_credit_counterexample=counterexample,proposed_absolute_cost_offline_audit=candidate,
        limitations=['One seed5 60-second replay; causality of the fall is not established by correlation.',
            'Offline reward reconstruction uses recorded transitions; raw PD violation cost recorded only on the last physics substep is not the complete training cost.',
            'Support guidance is a product; its small value does not alone prove its PPO gradient or exploration effect.'])
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'phase_alignment_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    plot(a.output_dir,d,q,rq,sole,reference_sole,names)
    print(json.dumps(dict(settled_samples=report['settled_samples'],target_reproduction=target_error,offline=offline,
        phase_clock_max_increment_error=report['phase_clock_max_increment_error'],counterexample=counterexample,candidate=candidate,
        selected_joints={n:per_joint[n] for n in [names[i] for i in ids]+['lumbar_pitch_joint']},falls=fall_context),indent=2))


def plot(directory,d,q,rq,sole,reference_sole,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
    mask=(d['time_s']>=8)&(d['time_s']<16)
    for i,side in enumerate(('left','right')):
        for row,joint in enumerate(('hip_pitch','knee_pitch')):
            index=names.index(side+'_'+joint+'_joint')
            axes[row,i].plot(d['time_s'][mask],np.rad2deg(q[mask,index]),label='Policy')
            axes[row,i].plot(d['time_s'][mask],np.rad2deg(rq[mask,index]),label='Logged same-phase reference')
            axes[row,i].set(title=side+' '+joint,ylabel='deg');axes[row,i].legend()
        axes[2,i].plot(d['time_s'][mask],sole[mask,i]*1000,label='Policy minimum sole')
        axes[2,i].plot(d['time_s'][mask],reference_sole[mask,i]*1000,label='Grounded kinematic reference')
        axes[2,i].set(title=side+' sole clearance',ylabel='mm');axes[2,i].legend()
    near=(d['time_s']>=15)&(d['time_s']<=20)
    axes[3,0].plot(d['time_s'][near],np.rad2deg(d['base_euler_x'][near]),label='base roll')
    axes[3,0].plot(d['time_s'][near],np.rad2deg(d['base_euler_y'][near]),label='base pitch')
    axes[3,0].set(title='Pre-fall orientation',ylabel='deg');axes[3,0].legend()
    for side in ('l','r'):axes[3,1].plot(d['time_s'][near],d['foot_force_'+side][near],label='foot '+side)
    axes[3,1].set(title='Pre-fall vertical contact force',ylabel='N');axes[3,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('seconds')
    fig.suptitle('exp1.16: actual logged phase targets and the first fall')
    fig.savefig(directory/'phase_alignment_analysis.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
