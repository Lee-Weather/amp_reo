"""CPU audit of exp1.17/18's actual bounded absolute phase objectives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyze_gait import geometry
from humanoid.amp.motion import MotionLibrary, URDFKinematics
from humanoid.amp.grounding import FootGrounding
from humanoid.amp.phase_reference import PhaseReference
from humanoid.amp.phase_rewards import (huber_terms, static_cycle_baseline,
    static_velocity_baseline, absolute_phase_credit, foot_support_score)
from humanoid.amp.gait_rewards import sole_heights, motion_velocity_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    job = json.loads((args.output_dir/'replay_job.json').read_text())
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == job['checkpoint_sha256']
    verified = {}
    for path in job['source_file_sha256']:
        if path.startswith('humanoid/amp/') or path.endswith(('exp117_env.py', 'exp117_config.py')):
            actual = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            assert actual == job['source_file_sha256'][path], path
            verified[path] = actual
    for path, expected in job['resource_file_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    with args.csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    data = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
    assert all(np.isfinite(v).all() for v in data.values())
    names = ck['metadata']['dof_names']
    def matrix(prefix):
        return torch.tensor(np.column_stack([data[prefix+n] for n in names]), dtype=torch.float32)
    q, dq, rq, rdq, raw = [matrix(p) for p in ('q_', 'dq_', 'reference_q_', 'reference_dq_', 'pd_target_raw_')]
    time, cmd = data['time_s'], data['command_applied']
    events = np.ones(len(time), dtype=bool)
    events[1:] = ((np.diff(data['episode_id']) != 0) | (np.abs(np.diff(cmd)) > 1e-7)
        | (np.diff(data['active_stage']) != 0)
        | ((data['stand_steps_left'][:-1] > 0) & (data['stand_steps_left'][1:] == 0)))
    last = np.maximum.accumulate(np.where(events, time, -np.inf))
    mask = ((time-last >= 2-1e-6) & ~data['done'].astype(bool)
        & (data['stand_steps_left'] == 0) & (data['z'] > .45) & (cmd > .05))
    assert (data['gz'][mask] < -.7).all(), 'Reference-valid gate is off in settled samples'
    cfg, acfg = ck['config']['rewards'], ck['config']['amp']
    urdf = ROOT/'resources/robots/x1_29/urdf/X1_29DOF_physically_mirrored.urdf'
    lib = MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized', urdf,
        names, acfg['key_body_names'], dt=ck['metadata']['policy_dt'], history_steps=acfg['history_steps'],
        weights=acfg['motion_weights'], limit_tolerance=acfg['limit_tolerance'], ground_clearance=acfg['ground_clearance'])
    standing = torch.tensor(ck['metadata']['control']['standing_state'], dtype=torch.float32)
    reference = PhaseReference(lib, standing[13:42], acfg['reference_cycles'])
    fk = URDFKinematics(urdf, names, ['left_ankle_roll_link', 'right_ankle_roll_link'])
    grounding = FootGrounding(urdf, fk, acfg['ground_clearance'])
    phase = torch.tensor(data['reference_phase'], dtype=torch.float32)
    reproduced = reference.sample(torch.tensor(cmd, dtype=torch.float32), phase)
    error = dict(q_max_abs=float((reproduced[0]-rq).abs().max()), dq_max_abs=float((reproduced[1]-rdq).abs().max()))
    assert max(error.values()) < 1e-4, error
    rates = reference.phase_rate(torch.tensor(cmd, dtype=torch.float32)).numpy()
    continuous = mask[1:] & mask[:-1] & ~events[1:]
    phase_error = np.abs(np.remainder(np.diff(phase.numpy()),1)[continuous]
        -ck['metadata']['policy_dt']*rates[1:][continuous])
    phases = (torch.arange(256, dtype=torch.float32)+.5)/256
    cq, cdq = reference.sample(torch.full_like(phases, .3), phases)
    ids = (phase*256).floor().long().clamp(max=255)
    pitch = reference.pitch_indices
    lateral = [i for i,n in enumerate(names) if any(s in n for s in ('hip_roll','hip_yaw','ankle_roll'))]
    waist = [i for i,n in enumerate(names) if n.startswith('lumbar_')]
    specifications = {
        'pose': (q[:,pitch], rq[:,pitch], cq[:,pitch], cfg['reference_pose_scale']),
        'velocity': (dq[:,pitch], rdq[:,pitch], cdq[:,pitch], cfg['reference_velocity_scale']),
        'feet': (fk.key_positions(q).flatten(-2), fk.key_positions(rq).flatten(-2), fk.key_positions(cq).flatten(-2), cfg['reference_feet_scale']),
        'lateral': (q[:,lateral], rq[:,lateral], cq[:,lateral], cfg['reference_lateral_scale']),
        'waist': (q[:,waist], rq[:,waist], cq[:,waist], cfg['reference_waist_scale'])}
    objectives = {}
    coupling = .4+.6*torch.tensor(np.clip(data['base_vel_x']/.3, 0, 1), dtype=torch.float32)
    for label, (actual,target,cycle,scale) in specifications.items():
        scale = torch.tensor([scale])
        baseline = (static_velocity_baseline if label == 'velocity' else static_cycle_baseline)(cycle, scale)[ids]
        unbounded = (baseline-huber_terms(actual,target,scale)).mean(-1)
        credit = absolute_phase_credit(actual,target,baseline,scale, cfg['phase_credit_negative_floor'],cfg['phase_credit_positive_ceiling'])
        objectives[label] = dict(unbounded_mean=float(unbounded[mask].mean()), credit_mean=float(credit[mask].mean()),
            negative_floor_fraction=float((unbounded[mask] <= cfg['phase_credit_negative_floor']).float().mean()),
            positive_fraction=float((credit[mask] > 0).float().mean()),
            weighted_mean_before_dt=float((credit*coupling)[mask].mean()*cfg['scales']['reference_'+label]))
    limits = torch.tensor(ck['metadata']['control']['hard_limits'])
    excess = (limits[:,0]-raw).clamp(min=0)+(raw-limits[:,1]).clamp(min=0)
    norm = excess/cfg['raw_target_scale']
    summed = torch.where(norm<=1, .5*norm.square(), norm-.5).sum(-1)
    raw_cost = dict(uncapped_huber_sum_mean=float(summed[mask].mean()),
        cap_saturated_fraction=float((summed[mask]>=cfg['raw_target_max_sum']).float().mean()),
        capped_cost_mean=float(summed.clamp(max=cfg['raw_target_max_sum'])[mask].mean()),
        weighted_mean_before_dt=float(summed.clamp(max=cfg['raw_target_max_sum'])[mask].mean()*cfg['scales']['raw_target_violation']))
    root = torch.tensor(np.column_stack([data[k] for k in ('x','y','z','qx','qy','qz','qw')]), dtype=torch.float32)
    _, _, sole = geometry(root,q,fk,grounding)
    local,rot = fk.key_transforms(rq)
    zeros = torch.zeros_like(local)
    target = sole_heights(torch.cat((local,rot,zeros,zeros),-1), [v for _,v in grounding.feet], 0)
    target = target-target.min(-1,keepdim=True).values+acfg['ground_clearance']
    mass = sum(float(n.attrib['value']) for n in ET.parse(urdf).getroot().findall('./link/inertial/mass'))
    force = torch.tensor(np.column_stack([data['foot_force_l'],data['foot_force_r']]), dtype=torch.float32)
    support = foot_support_score(torch.tensor(sole), target, force, torch.tensor(mass*9.81),
        cfg['support_height_scale'],cfg['support_load_scale'],cfg['support_load_fraction'])
    # Illustrative reward ordering only: constant flat feet with balanced support.
    flat = foot_support_score(torch.full_like(target,.002),target,torch.full_like(force,mass*9.81/2),torch.tensor(mass*9.81),
        cfg['support_height_scale'],cfg['support_load_scale'],cfg['support_load_fraction'])
    support_report = dict(recorded_score_mean=float(support[mask].mean()),
        weighted_mean_before_dt=float((support*coupling)[mask].mean()*cfg['scales']['reference_support']),
        synthetic_flat_balanced_score_mean=float(flat[mask].mean()), nominal_urdf_mass_kg=mass,
        mass_assumption='Nominal URDF; verify domain_rand mass settings before using for randomized training.')
    posture = {}
    for i,n in enumerate(names):
        posture[n] = dict(mean_deg=float(q[mask,i].mean().rad2deg()),reference_mean_deg=float(rq[mask,i].mean().rad2deg()),
            rmse_deg=float((q[mask,i]-rq[mask,i]).square().mean().sqrt().rad2deg()),
            near_bound_1deg_fraction=float((((q[:,i]-limits[i,0])<np.deg2rad(1))|((limits[i,1]-q[:,i])<np.deg2rad(1)))[mask].float().mean()))
    report = dict(training_commit=job['training_commit'],model_sha256=job['checkpoint_sha256'],
        csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),source_hashes=verified,
        samples=int(mask.sum()),target_reproduction=error,phase_objectives=objectives,raw_target_cost=raw_cost,
        phase_clock_max_increment_error=float(phase_error.max()),reference_rate_hz=float(np.median(rates[mask])),
        support=support_report,posture=posture,
        progress_weighted_before_dt=float(motion_velocity_reward(torch.tensor(cmd),torch.tensor(data['base_vel_x']),cfg['motion_velocity_scale'])[mask].mean()*cfg['scales']['motion_progress']),
        replay_style_mean=float(data['style_score'].mean()),replay_style_reward_mean=float(data['style_reward'].mean()),
        replay_task_reward_mean=float(data['task_reward'].mean()),
        limitations=['Single deterministic replay; reward ordering establishes mechanisms, not unique causal attribution.',
          'Raw PD target is the last physics substep, not averaged substep clipping counts.',
          'Support reconstruction assumes nominal URDF mass; synthetic flat support is not a simulated policy.',
          'Baseline reproduces implemented cycle median; a median is not generally the exact Huber minimizer.'])
    (args.output_dir/'reward_mechanism_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('posture','source_hashes')},indent=2))


if __name__ == '__main__':
    main()
