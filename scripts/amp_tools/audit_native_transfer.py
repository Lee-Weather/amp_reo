"""Audit the actual native-cycle AMP route and imported PhysX/MuJoCo properties."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from sim2sim_amp import prepare_mjcf
from native_physics import align_runtime, merged_properties


def physics_audit(ck, runtime, source, meshes, output, runtime_dir, root):
    xml = prepare_mjcf(source, meshes.resolve().as_posix(), output/'mjcf', ck['metadata']['dof_names'])
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = [p['name'] for p in runtime['final_bodies']]
    groups = merged_properties(model, names)
    errors = []
    for actual in runtime['final_bodies']:
        expected = groups[actual['name']]
        inertia = np.asarray(actual['inertia'])
        err = float(np.max(np.abs(inertia-np.asarray(expected['inertia']))))
        errors.append(dict(name=actual['name'], mass_error_kg=abs(actual['mass']-expected['mass']),
            com_max_error_m=float(np.max(np.abs(np.asarray(actual['com'])-expected['com']))),
            inertia_max_error_kg_m2=err,
            inertia_relative_frobenius=float(np.linalg.norm(inertia-np.asarray(expected['inertia']))/np.linalg.norm(inertia)),
            physx_inertia_eigenvalues=np.linalg.eigvalsh(inertia).tolist(),
            merged_mujoco_body_count=len(expected['members'])))
    errors.sort(key=lambda p: p['inertia_max_error_kg_m2'], reverse=True)
    prior = {p['name']: p for p in runtime['before_recompute_inertia']}
    recompute = [{k: p[k] for k in ('name',)} | dict(
        inertia_max_change=float(np.max(np.abs(np.asarray(p['inertia'])-prior[p['name']]['inertia']))),
        mass_change=abs(p['mass']-prior[p['name']]['mass']),
        com_max_change=float(np.max(np.abs(np.asarray(p['com'])-prior[p['name']]['com']))))
        for p in runtime['final_bodies']]
    recompute.sort(key=lambda p: p['inertia_max_change'], reverse=True)
    control = ck['metadata']['control']
    control_errors = {k: float(np.max(np.abs(np.asarray(runtime[k])-control[k])))
                      for k in ('stiffness', 'damping', 'torque_limits')}
    def passive_error(model):
        probe = json.loads((runtime_dir/'passive_probe.json').read_text())
        case = probe['results'][0]
        initial, target = np.asarray(case['start']), np.asarray(case['end'])
        joints = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ck['metadata']['dof_names']]
        qa, da = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        model.opt.timestep=probe['dt']
        model.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER
        data=mujoco.MjData(model)
        data.qpos[:3]=initial[:3]
        data.qpos[3:7]=initial[[6,3,4,5]]
        data.qpos[qa]=initial[13:42]
        data.qvel[da]=initial[42:]
        mujoco.mj_forward(model,data)
        for _ in range(probe['steps_per_case']):
            mujoco.mj_step(model,data)
        err=data.qvel[da]-target[42:]
        return dict(joint_velocity_RMSE=float(np.sqrt(np.mean(err**2))),joint_velocity_max_error=float(np.max(np.abs(err))),
                    maximum_contact_count=int(data.ncon))
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ck['metadata']['dof_names']]
    adr=model.jnt_dofadr[joints]
    model.dof_armature[adr]=ck['config']['asset']['armature']
    model.dof_damping[adr]=0.
    model.dof_frictionloss[adr]=0.
    old_motion=passive_error(model)
    urdf=Path(ck['config']['asset']['file'].format(LEGGED_GYM_ROOT_DIR=root))
    alignment=align_runtime(model,ck,runtime_dir/'runtime.json',runtime_dir/'passive_probe.json',asset_urdf=urdf)
    new_motion=passive_error(model)
    return dict(physx_body_count=len(names), mujoco_body_count=model.nbody-1,
        maximum_mass_error_kg=max(p['mass_error_kg'] for p in errors),
        maximum_com_error_m=max(p['com_max_error_m'] for p in errors),
        maximum_inertia_error_kg_m2=errors[0]['inertia_max_error_kg_m2'],
        bodies=errors, recompute_inertia_changes=recompute, controller_max_errors=control_errors,
        dofs=runtime['dofs'], shapes=runtime['shapes'],alignment=alignment,
        passive_motion_comparison=dict(legacy=old_motion,aligned=new_motion),
        remaining_scope='Contact solvers/material combination and geometry decomposition are not proven equivalent.')


def stats(tensor):
    tensor = torch.as_tensor(tensor).float()
    if not tensor.numel():
        return None
    return dict(mean=float(tensor.mean()), p10=float(tensor.quantile(.1)),
                p50=float(tensor.quantile(.5)), p90=float(tensor.quantile(.9)),
                minimum=float(tensor.min()), maximum=float(tensor.max()))


def feature_groups(n, total):
    return dict(height=[0], gravity=list(range(1, 4)), linear_velocity=list(range(4, 7)),
        angular_velocity=list(range(7, 10)), joint_position=list(range(10, 10+n)),
        joint_velocity=list(range(10+n, 10+2*n)), key_positions=list(range(10+2*n, total)))


def analyze_group(disc, actual, expert, groups):
    if not len(actual):
        return dict(samples=0)
    with torch.no_grad():
        scores = disc(actual)
        actual_n, expert_n = disc.normalizer(actual), disc.normalizer(expert)
    normalized = actual_n.detach().requires_grad_(True)
    value = disc.network(normalized.flatten(1)).squeeze(-1)
    grad = torch.autograd.grad(value.sum(), normalized)[0]
    result = dict(samples=len(actual), policy_D=stats(scores), expert_D=stats(disc(expert).detach()),
        policy_style=stats(disc.style_reward(actual)), expert_style=stats(disc.style_reward(expert)), groups={})
    for name, cols in groups.items():
        hybrid = actual.clone()
        hybrid[:, :, cols] = expert[:, :, cols]
        with torch.no_grad():
            delta = disc(hybrid)-scores
        result['groups'][name] = dict(standardized_difference=float((actual_n[:, :, cols]-expert_n[:, :, cols]).abs().mean()),
            normalized_D_gradient=float(grad[:, :, cols].abs().mean()), replace_group_delta_D=float(delta.mean()),
            normalization_clipped_fraction=float((actual_n[:, :, cols].abs()>=9.999).float().mean()))
    return result


def amp_audit(ck, root, runtime_dir):
    from humanoid.amp.motion import MotionLibrary, quat_inverse_rotate
    from humanoid.amp.native_reference import NativeCycle
    from humanoid.amp.stationary import CommandMotionSampler
    from humanoid.amp.models import Discriminator
    cfg, meta = ck['config'], ck['metadata']
    paths = lambda path: Path(path.format(LEGGED_GYM_ROOT_DIR=root))
    lib = MotionLibrary(paths(cfg['amp']['motion_dir']), paths(cfg['asset']['file']), meta['dof_names'],
        cfg['amp']['key_body_names'], dt=meta['policy_dt'], history_steps=cfg['amp']['history_steps'],
        weights=cfg['amp']['motion_weights'], limit_tolerance=cfg['amp']['limit_tolerance'],
        ground_clearance=cfg['amp']['ground_clearance'])
    for stem, sha in meta['training']['reference_source_sha256'].items():
        assert hashlib.sha256((paths(cfg['amp']['motion_dir'])/(stem+'.pkl')).read_bytes()).hexdigest()==sha
    correction = json.loads(paths(cfg['amp']['native_correction_file']).read_text())
    assert hashlib.sha256(json.dumps(correction,sort_keys=True,separators=(',',':')).encode()).hexdigest()==meta['training']['native_reference']['correction_sha256']
    ref = NativeCycle(lib, cfg['amp']['native_clip'], cfg['amp']['native_first_frame'],
                      cfg['amp']['native_last_frame'], contact_correction=correction)
    # CommandMotionSampler is used ONLY to construct the production quiet bank.
    sampler = CommandMotionSampler(lib, speed_tolerance=cfg['amp']['rsi_speed_tolerance'],
        expert_standing_probability=cfg['amp']['expert_standing_probability'],
        max_root_speed=cfg['amp']['quiet_root_speed'], max_angular_speed=cfg['amp']['quiet_angular_speed'],
        max_joint_speed=cfg['amp']['quiet_joint_speed'], exclude_quiet_for_moving=True,
        relative_speed_tolerance=cfg['amp']['rsi_relative_speed_tolerance'])
    ref.quiet_histories = lib.history(sampler.ids[sampler.quiet], sampler.times[sampler.quiet])
    disc = Discriminator(lib.feature_dim, lib.history_steps, cfg['discriminator']['hidden_dims'])
    disc.load_state_dict(ck['discriminator_state_dict'], strict=True)
    disc.eval()
    samples = np.load(runtime_dir/'samples.npz')
    states = torch.from_numpy(samples['state']).float()
    amp = torch.from_numpy(samples['amp']).float()
    reconstructed = lib.features(states)
    latest_error = (reconstructed-amp[:, -1]).abs()
    valid = torch.from_numpy(samples['valid']).bool()
    done = torch.from_numpy(samples['done']).bool()
    indices = torch.arange(lib.history_steps-1, len(states))
    offsets = torch.arange(lib.history_steps-1, -1, -1)
    previous_done = torch.stack([done[indices-i] for i in range(1, lib.history_steps)]).any(0)
    continuous = indices[~previous_done]
    histories = reconstructed[continuous[:, None]-offsets]
    history_error = float((histories-amp[continuous]).abs().max())
    score_error = float((disc(amp).detach()-torch.from_numpy(samples['disc_score'])).abs().max())
    groups = feature_groups(len(meta['dof_names']), lib.feature_dim)
    labels = np.full(len(states), 'quiet', dtype='<U10')
    moving, transition = samples['moving'].astype(bool), samples['transition'].astype(bool)
    command = samples['command'][:, 0]
    labels[moving & ~transition] = 'steady'
    labels[transition] = 'transition'
    labels[samples['hold'] > 0] = 'buffer'
    alive = ~done & (states[:, 2]>.45) & (amp[:, -1, 3]<-.7)
    matched = ref.history(torch.from_numpy(samples['phase']).float())
    category_expert = ref.expert(torch.from_numpy(moving))
    style = torch.from_numpy(samples['style_score']).float()
    multiplier = torch.from_numpy(samples['multiplier']).float()
    actual_reward = style*cfg['discriminator']['style_reward_scale']*lib.dt*multiplier
    task = torch.from_numpy(samples['task_reward']).float()
    condition = {}
    for label in ('buffer', 'transition', 'steady', 'quiet'):
        mask = torch.from_numpy(labels==label)
        eligible = mask & valid & alive
        detail = analyze_group(disc, amp[eligible],
            matched[eligible] if label=='steady' else category_expert[eligible], groups)
        detail.update(total_samples=int(mask.sum()), valid_samples=int((mask&valid).sum()),
            actual_style_reward=stats(actual_reward[mask]), task_reward=stats(task[mask]),
            multiplier=stats(multiplier[mask]), raw_style=stats(style[mask]))
        if int(eligible.sum()):
            detail['valid_style_zero_fraction']=float((style[eligible]<=0).float().mean())
            detail['valid_style_to_abs_task_mean_ratio']=float(actual_reward[eligible].mean()/task[eligible].abs().mean().clamp(min=1e-8))
        if label=='steady' and int(eligible.sum()):
            target=ref.state(torch.from_numpy(samples['phase']).float())
            qerr=states[eligible,13:42]-target[eligible,13:42]
            detail['joint_position_errors']=sorted([dict(name=name,rmse_rad=float(qerr[:,i].square().mean().sqrt()),
                mean_bias_rad=float(qerr[:,i].mean())) for i,name in enumerate(meta['dof_names'])],key=lambda p:p['rmse_rad'],reverse=True)
            detail['local_vx']=stats(torch.from_numpy(samples['velocity'])[eligible,0])
            detail['abs_local_vy']=stats(torch.from_numpy(samples['velocity'])[eligible,1].abs())
            detail['abs_local_wz']=stats(torch.from_numpy(samples['omega'])[eligible,2].abs())
            eligible_indices=continuous[eligible[continuous]]
            actual_states=states[eligible_indices[:,None]-offsets].clone()
            phases=torch.from_numpy(samples['phase']).float()[eligible_indices]
            target_states=ref.state((phases[:,None]-offsets*lib.dt/ref.period).flatten()).reshape(-1,3,71)
            base_D=disc(lib.features(actual_states)).detach()
            replacements={}
            subsets=dict(upper=[i for i,n in enumerate(meta['dof_names']) if any(k in n for k in ('shoulder','elbow','wrist'))],
                waist=[i for i,n in enumerate(meta['dof_names']) if 'lumbar' in n],
                legs=[i for i,n in enumerate(meta['dof_names']) if any(k in n for k in ('hip','knee','ankle'))])
            with torch.no_grad():
                for name,indices_q in subsets.items():
                    hybrid=actual_states.clone()
                    for offset_q in (13,42):
                        columns=[offset_q+i for i in indices_q]
                        hybrid[:,:,columns]=target_states[:,:,columns]
                    new_history=lib.features(hybrid)
                    replacements[name]=dict(joint_count=len(indices_q),mean_delta_D=float((disc(new_history)-base_D).mean()),
                                             style_mean=float(disc.style_reward(new_history).mean()))
            detail['q_dq_and_FK_consistent_replacements']=replacements
        condition[label] = detail
    # Expert-only self-checks use exactly the same curve/history production path.
    phases = torch.arange(512)/512
    expert = ref.history(phases)
    ref_states = ref.state((phases[:,None]-torch.arange(2,-1,-1)*lib.dt/ref.period).flatten()).reshape(512,3,-1)
    independent_history = lib.features(ref_states)
    root_quat = states[:, 3:7]
    world_key = torch.from_numpy(samples['key_world']).float()
    sim_local = quat_inverse_rotate(root_quat[:,None].expand(-1,len(lib.key_body_names),-1),world_key-states[:,None,:3])
    expected_local = lib.fk.key_positions(states[:,13:42])
    velocity_error = float((quat_inverse_rotate(root_quat,states[:,7:10])-torch.from_numpy(samples['velocity'])).abs().max())
    normal_var = disc.normalizer.var.detach()
    return dict(samples=len(states), training_updates=0, feature_dim=lib.feature_dim, history_steps=lib.history_steps,
        dt=lib.dt, runtime_non_timeout_resets=int((done&~torch.from_numpy(samples['timeout']).bool()).sum()),
        same_state_latest_feature_max_error=float(latest_error.max()),
        same_state_feature_errors={k:float(latest_error[:,cols].max()) for k,cols in groups.items()},
        same_state_history_max_error=history_error, continuous_history_samples=len(continuous),
        same_state_D_max_error=score_error, simulator_FK_key_position_max_error_m=float((sim_local-expected_local).abs().max()),
        velocity_coordinate_max_error=velocity_error,
        expert_history_rebuild_max_error=float((expert-independent_history).abs().max()),
        native_expert_D=stats(disc(expert).detach()), native_expert_style=stats(disc.style_reward(expert)),
        quiet_expert_D=stats(disc(ref.quiet_histories).detach()), quiet_expert_style=stats(disc.style_reward(ref.quiet_histories)),
        normalizer_minimum_variance=float(normal_var.min()), conditions=condition,
        limits='Single seed nominal short diagnostic inference, not a training-window estimate or PPO gradient-share measurement. Feature replacement can be nonphysical; local D sensitivity is not a causal intervention.',
        passed_feature_reconstruction=bool(latest_error.max()<2e-4 and history_error<2e-4 and score_error<2e-4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--mjcf-source', type=Path, required=True)
    p.add_argument('--meshes', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    sys.path.insert(0, str(a.root.resolve()))
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    runtime = json.loads((a.runtime_dir/'runtime.json').read_text())
    assert runtime['checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    physics = physics_audit(ck, runtime, a.mjcf_source, a.meshes, a.output_dir, a.runtime_dir, a.root)
    (a.output_dir/'physics_audit.json').write_text(json.dumps(physics,indent=2)+'\n',encoding='utf-8')
    amp = amp_audit(ck, a.root, a.runtime_dir)
    (a.output_dir/'amp_audit.json').write_text(json.dumps(amp,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(physics={k:physics[k] for k in ('physx_body_count','maximum_mass_error_kg','maximum_com_error_m','maximum_inertia_error_kg_m2')},
        amp={k:amp[k] for k in ('samples','passed_feature_reconstruction','same_state_latest_feature_max_error','same_state_history_max_error','same_state_D_max_error','runtime_non_timeout_resets')})))


if __name__ == '__main__':
    main()
