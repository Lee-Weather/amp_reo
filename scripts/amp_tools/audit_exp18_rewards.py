"""Evaluate exp1.8 rewards on immutable parent CSVs before running physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.gait_rewards import backward_progress, backward_velocity_reward, braking_leg_gate


def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    if not all(np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError('Nonfinite CSV')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--main-csv', type=Path, required=True)
    parser.add_argument('--reverse-csv', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.model, map_location='cpu', weights_only=False)
    source = json.loads((args.source_record/'replay_summary.json').read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    assert digest == source['checkpoint_sha256']
    assert hashlib.sha256(args.main_csv.read_bytes()).hexdigest() == source['file_sha256']['isaac_diag.csv']
    assert hashlib.sha256(args.reverse_csv.read_bytes()).hexdigest() == source['auxiliary_checks'][1]['csv_sha256']
    r, meta = state['config']['rewards'], state['metadata']
    dt, mix = meta['policy_dt'], state['config']['discriminator']['task_reward_weight']
    assert not r['only_positive_rewards']
    reverse = load_csv(args.reverse_csv)
    mask = (reverse['time_s'] >= 3.6) & (reverse['time_s'] <= 19.8)
    assert not reverse['done'][mask].any() and np.allclose(reverse['cmd_vx'][mask], -.2)
    assert (reverse['gz'][mask] < -.7).all() and (reverse['z'][mask] > .45).all()
    cmd = torch.tensor(reverse['cmd_vx'][mask], dtype=torch.float64)
    v = torch.tensor(reverse['vx'][mask], dtype=torch.float64)
    old = backward_progress(cmd, v).numpy()*r['scales']['backward_progress']
    new = 1.5*backward_velocity_reward(cmd, v).numpy()
    correction = mix*dt*(new-old)
    new_task = mix*reverse['task_reward'][mask]+correction
    new_total = reverse['mixed_reward'][mask]+correction
    assert new_total.mean() > 0  # Immediate termination at zero return is not preferable on this full recorded window.
    data = load_csv(args.main_csv)
    names = meta['dof_names']
    legs = [n for n in names if any(j in n for j in ('hip_', 'knee_', 'ankle_'))]
    target = np.array(meta['control']['standing_state'])[13:42]
    leg_target = target[[names.index(n) for n in legs]]
    q = np.column_stack([data['q_'+n] for n in legs])
    dq = np.column_stack([data['dq_'+n] for n in legs])
    pending = False; moving = False; zero_count = 0; hold = 0
    gates = []; costs = []; pendings = []
    for i in range(len(data)):
        zero = max(abs(data[k][i]) for k in ('cmd_vx','cmd_vy','cmd_wz')) <= 1e-6
        zero_count = zero_count+1 if zero else 0
        pending = (pending or (zero and moving)) and zero
        moving = not zero
        slow = np.hypot(data['vx'][i], data['vy'][i]) <= .02 and abs(data['wz'][i]) <= .03
        hold = hold+1 if zero and slow else 0
        if pending and hold >= round(.2/dt): pending = False
        gate = float(braking_leg_gate(torch.tensor([[data['vx'][i],data['vy'][i]]]),
                                     torch.tensor([data['wz'][i]]),torch.tensor([bool(pending)]))[0])
        ramp = min(zero_count*dt/r['stand_ramp_seconds'],1.)
        cost = ramp*(r['scales']['stand_dof_vel']*np.mean(dq[i]**2)/r['stand_joint_speed_scale']**2
                     +r['scales']['stand_pose']*np.mean((q[i]-leg_target)**2)/r['stand_pose_scale']**2)
        gates.append(gate);costs.append(cost);pendings.append(pending)
        if data['done'][i]:pending=False;moving=False;zero_count=0;hold=0
    gates, costs = np.array(gates), np.array(costs)
    delta = mix*dt*costs*(gates-1)
    quiet_windows = ((data['time_s']>=3.6)&(data['time_s']<9.8)) | ((data['time_s']>=34)&(data['time_s']<39.8)) | ((data['time_s']>=54)&(data['time_s']<59.8))
    assert np.max(np.abs(delta[quiet_windows])) == 0
    stops = []
    for start in (30.,50.):
        take = (data['time_s']>=start)&(data['time_s']<start+2)
        stops.append(dict(start_s=start,mean_leg_gate=float(gates[take].mean()),
                          old_weighted_leg_cost=float((mix*dt*costs[take]).mean()),
                          new_weighted_leg_cost=float((mix*dt*costs[take]*gates[take]).mean())))
    result = dict(source_iteration=state['iter'], source_model_sha256=digest,
                  reverse_csv_sha256=hashlib.sha256(args.reverse_csv.read_bytes()).hexdigest(),
                  main_csv_sha256=hashlib.sha256(args.main_csv.read_bytes()).hexdigest(),
                  candidate=dict(backward_speed_scale=.2,backward_weight=1.5,leg_linear_scale=.08,leg_angular_scale=.15),
                  reverse=dict(samples=int(mask.sum()),mean_vx=float(v.mean()),old_mean_bonus=float(old.mean()),
                               new_mean_bonus=float(new.mean()),new_task_mixed_mean=float(new_task.mean()),
                               new_total_mixed_mean=float(new_total.mean()),negative_step_fraction=float((new_total<0).mean()),
                               recorded_window_discounted_return=float(np.dot(new_total,.99**np.arange(len(new_total)))),
                               immediate_zero_return_termination_worse_for_recorded_window=bool(np.dot(new_total,.99**np.arange(len(new_total)))>0)),
                  braking=stops,late_quiet_reward_delta_max=float(np.abs(delta[quiet_windows]).max()),
                  limits=['Same trajectory and discriminator; no prediction of learned physics or future terminations.',
                          'Standing-state metadata rounds joint targets to 1e-6rad.',
                          'Positive recorded-window return is not a guarantee against termination incentives at every state.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
