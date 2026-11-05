"""Capture production student inputs before teacher-driven env.step, without learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.wbt_teacher import WBTTeacher, permutation, rotation_xyzw, termination
from humanoid.amp.teacher_dataset import canonical_action, validate_dataset, audit_conditions
from humanoid.amp.reference_condition import ReferenceCondition


def initialize_reference(env, teacher, gymtorch, torch, quat_rotate_inverse):
    """Shared teacher/student playback initialization; no policy or physics step."""
    env.evaluation_reset_mode = 'standing'
    env.reset()
    names = list(env.dof_names)
    order = permutation(teacher.names, names)
    bodies = teacher.meta['body_names']
    body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], n) for n in bodies]
    if min(body_ids) < 0:
        raise ValueError('Teacher tracked bodies are missing')
    anchor = bodies.index('base_link')
    ref = teacher.data
    root = np.concatenate((ref['body_pos_w'][0, anchor], ref['body_quat_w'][0, anchor][[1, 2, 3, 0]],
        ref['body_lin_vel_w'][0, anchor], ref['body_ang_vel_w'][0, anchor]))
    env.root_states[0] = torch.as_tensor(root, device=env.device)
    env.root_states[0, :3] += env.env_origins[0]
    env.dof_pos[0] = torch.as_tensor(ref['joint_pos'][0, order], device=env.device)
    env.dof_vel[0] = torch.as_tensor(ref['joint_vel'][0, order], device=env.device)
    ids = torch.zeros(1, device=env.device, dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_states), gymtorch.unwrap_tensor(ids), 1)
    env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(ids), 1)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.gait_phase[:] = env.reference_guide.nearest_phase(env.dof_pos, env.dof_vel)
    env._bank_pending = None
    for value in (env.actions, env.last_actions, env.last_last_actions):
        value.zero_()
    for frame in env.obs_history:
        frame.zero_()
    env.compute_observations()
    env.set_evaluation_command(env.cfg.commands.native_speed)
    return names, order, bodies, body_ids, anchor, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--teacher-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--condition-profile', type=Path, help='Opt-in B1 progress semantics; native mode is unchanged')
    args = parser.parse_args()
    lock = json.loads(args.teacher_lock.read_text(encoding='utf-8'))
    if hashlib.sha256(args.bundle.read_bytes()).hexdigest() != lock['bundle_sha256']:
        raise ValueError('Teacher bundle does not match the frozen input')
    teacher = WBTTeacher(args.bundle)
    condition = ReferenceCondition(args.condition_profile) if args.condition_profile else None
    if teacher.meta['name'] != lock['name'] or teacher.meta['normalizer'] != lock['normalizer']:
        raise ValueError('Teacher implementation differs from the frozen input')
    if condition is not None and (condition.frames != teacher.frames or
            condition.meta['reference_sha256'] not in lock['sources'].values() or
            abs(condition.meta['policy_dt']-teacher.meta['policy_dt']) > 1e-9):
        raise ValueError('Condition does not match the locked reference or control clock')
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError('max-steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    import isaacgym  # Must precede torch and environment registration.
    from isaacgym import gymtorch
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    import humanoid.envs
    from humanoid.utils import get_args, task_registry
    from humanoid.amp.nominal_startup import configure_nominal_probe
    torch.set_num_threads(1)
    sys.argv = [sys.argv[0], '--task=x1_amp', '--num_envs=1', '--seed=5', '--headless']
    cfg, _ = task_registry.get_cfgs('x1_amp')
    cfg.env.capture_replay_state = True
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    configure_nominal_probe(cfg)
    cfg.commands.guidance_evaluation = True
    cfg.commands.ranges.lin_vel_x = [cfg.commands.native_speed]*2
    cfg.commands.ranges.lin_vel_y = cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env, _ = task_registry.make_env('x1_amp', args=get_args(), env_cfg=cfg)
    rows, reason = [], None
    try:
        names, order, bodies, body_ids, anchor, ref = initialize_reference(env, teacher, gymtorch, torch, quat_rotate_inverse)
        default = env.default_dof_pos[0].cpu().numpy().copy()
        limits = env.hard_dof_limits.cpu().numpy().copy()
        if env.motor_offsets.abs().max().item() > 1e-7:
            raise ValueError('Canonical target labels require zero motor offsets')
        if abs(env.dt-teacher.meta['policy_dt']) > 1e-7:
            raise ValueError('Teacher/student policy clocks differ')
        reference_phases = env.reference_guide.nearest_phase(
            torch.as_tensor(ref['joint_pos'][:, order], device=env.device),
            torch.as_tensor(ref['joint_vel'][:, order], device=env.device)).cpu().numpy()
        last = np.zeros(29, dtype=np.float32)
        steps = min(args.max_steps or teacher.frames-1, teacher.frames-1)
        with torch.no_grad():
            for i in range(steps):
                observation = (condition.apply(env, i) if condition is not None else
                               env.set_evaluation_command(cfg.commands.native_speed))
                student_obs = observation[0].cpu().numpy().copy()
                pre_root = env.root_states[0].cpu().numpy().copy()
                origin = env.env_origins[0].cpu().numpy()
                pre_root[:3] -= origin
                q, dq = [v[0].cpu().numpy().copy() for v in (env.dof_pos, env.dof_vel)]
                teacher_obs = teacher.observation(i, pre_root, q, dq, last, names)
                teacher_action = teacher.action(teacher_obs)
                requested_target = teacher.target(teacher_action, names)
                sent, expected = canonical_action(requested_target, default, cfg.control.action_scale,
                    cfg.normalization.clip_actions, limits)
                previous = env.actions[0].cpu().numpy().copy()
                command = env.commands[0, :3].cpu().numpy().copy()
                phase_features = student_obs[-2:].copy()
                phase = env.gait_phase[0].cpu().numpy().copy()
                reference_rotation = rotation_xyzw(ref['body_quat_w'][i, anchor][[1, 2, 3, 0]])
                velocity = reference_rotation.T@ref['body_lin_vel_w'][i, anchor]
                omega = reference_rotation.T@ref['body_ang_vel_w'][i, anchor]
                requested_command = env.standing_commands.requested[0].cpu().numpy().copy()
                next_obs, _, _, done, info = env.step(torch.as_tensor(sent[None], device=env.device))
                if condition is not None and not bool(done[0]):
                    next_obs = condition.apply(env, i+1)
                terminal = info['replay_state']  # State and targets captured BEFORE any automatic reset.
                actual = terminal['effective_target'][0].cpu().numpy().copy()
                if not np.allclose(actual, expected, rtol=0, atol=1e-6):
                    raise ValueError('Production PD did not apply the expected bounded target')
                post_root = terminal['root'][0].cpu().numpy().copy()
                post_root[:3] -= origin
                row = dict(student_observations=student_obs,
                    next_student_observations=next_obs[0].cpu().numpy().copy(),
                    teacher_observations=teacher_obs, teacher_actions=teacher_action,
                    teacher_requested_targets=requested_target, effective_targets=actual,
                    label_actions=((actual-default)/cfg.control.action_scale).astype(np.float32),
                    applied_actions=terminal['action'][0].cpu().numpy().copy(), previous_actions=previous,
                    pre_roots=pre_root, pre_q=q, pre_dq=dq, post_roots=post_root,
                    applied_commands=command, requested_commands=requested_command,
                    student_phase=phase, phase_features=phase_features,
                    reference_commands=np.array([velocity[0], velocity[1], omega[2]], dtype=np.float32),
                    reference_native_phase=np.array(reference_phases[i]),
                    time_s=np.array(i*teacher.meta['policy_dt']), done=np.array(bool(done[0])),
                    episode_step=terminal['episode_step'][0].cpu().numpy().copy())
                rows.append(row)
                last = teacher_action.copy()
                if bool(done[0]):
                    reason = 'production_environment_terminated'
                else:
                    pos = env.rigid_state[0, body_ids, :3].cpu().numpy().copy()-origin
                    reason = termination(post_root, ref['body_pos_w'][i+1, anchor],
                        ref['body_quat_w'][i+1, anchor], pos, ref['body_pos_w'][i+1], bodies)
                if (i+1) % 50 == 0 or reason:
                    print('DATASET_STEP '+json.dumps(dict(step=i+1, reason=reason, vx_command=float(command[0]),
                        reference_vx=float(velocity[0]))), flush=True)
                if reason:
                    break
        meta = dict(schema='x1_teacher_student_dataset_v1', teacher_lock=lock,
            dof_names=names, frame_width=cfg.env.num_single_obs, frame_stack=cfg.env.frame_stack,
            policy_dt=teacher.meta['policy_dt'], observation_clip=cfg.normalization.clip_observations,
            observation_scales={key:float(getattr(env.obs_scales, key)) for key in ('ang_vel', 'dof_pos', 'dof_vel')},
            command_scale=env.commands_scale.cpu().tolist(), default_dof_pos=default.tolist(),
            action_scale=cfg.control.action_scale, action_clip=cfg.normalization.clip_actions, hard_limits=limits.tolist(),
            task='x1_amp', seed=5, input_source=('production env.step plus explicit online reference-condition adapter'
                if condition is not None else 'actual production env.step and compute_observations'),
            initial_state='reference frame0 injected after production standing reset; zero histories then one real frame',
            behavior_action='canonical inverse of effective target, not unclipped teacher raw action',
            phase_clock='b1_finite_reference_progress_v1' if condition is not None else 'unchanged production command_scaled_v1',
            command_profile='reference causal0.4s vx envelope; vy/wz0' if condition is not None else 'fixed native forward request with production hold/ramp',
            condition_profile=None if condition is None else dict(sha256=condition.sha256, metadata=condition.meta),
            nominal_conditions='noise/push off; friction0.8/contact_offset0.001',
            stand_buffer_seconds=cfg.amp.stand_buffer_seconds, transition_seconds=cfg.commands.transition_seconds,
            native_period=env.reference_guide.period, native_speed=cfg.commands.native_speed,
            termination_reason=reason, completed_reference=reason is None and len(rows)==teacher.frames-1,
            expected_rows=teacher.frames-1, max_steps=args.max_steps, training_updates=0,
            sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                'humanoid/envs/x1/x1_amp_env.py','humanoid/envs/base/legged_robot.py',
                'humanoid/envs/x1/x1_amp_config.py',
                'humanoid/amp/exp20_commands.py','humanoid/amp/exp29_phase.py')})
        data = {key:np.stack([row[key] for row in rows]) for key in rows[0]}
        data['metadata'] = np.array(json.dumps(meta, allow_nan=False))
        with (args.output/'dataset.npz').open('xb') as stream:
            np.savez_compressed(stream, **data)
        # Save the evidence first; failing audits must not erase the rejected dataset.
        report = dict(metadata=meta, integrity=validate_dataset(data), conditions=audit_conditions(data))
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('TEACHER_DATASET_RECORDED '+json.dumps(report, allow_nan=False), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
