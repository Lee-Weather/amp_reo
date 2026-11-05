"""Teacher takeover after a fixed student prefix; no reset, no learning."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanoid.amp.wbt_teacher import permutation


def target_equivalent_history(teacher, target, joint_names):
    reference = np.asarray(target)[permutation(joint_names, teacher.names)]
    return ((reference-teacher.default)/teacher.meta['action_scale'])[teacher.to_actor].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--takeover-step', type=int, required=True, choices=[20, 40])
    parser.add_argument('--history', required=True, choices=['target_equivalent', 'shadow'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import isaacgym
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_rotate_inverse
    import torch
    import humanoid.envs
    from humanoid.utils import get_args, task_registry
    from humanoid.amp.nominal_startup import configure_nominal_probe
    from humanoid.amp.reference_condition import ReferenceCondition
    from humanoid.amp.wbt_teacher import WBTTeacher, termination
    from humanoid.amp.teacher_dataset import canonical_action
    from humanoid.amp.student_distillation import (verify_inputs, build_student, trusted_load,
        state_digest, SCHEMA, frozen_digest, freeze_features)
    from collect_wbt_student import initialize_reference
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, demonstration, meta, source, _ = verify_inputs(plan, ROOT)
    checkpoint = trusted_load(args.checkpoint, args.checkpoint_sha256)
    if checkpoint.get('schema') != SCHEMA or checkpoint['plan'] != plan or checkpoint['updates'] != 200:
        raise ValueError('Takeover student must be the exact trained200 pilot')
    source_model = build_student(source)
    freeze_features(source_model)
    frozen = frozen_digest(source_model)
    model = build_student(checkpoint)
    if frozen_digest(model) != frozen:
        raise ValueError('Takeover student changed frozen branches')
    teacher = WBTTeacher(paths['teacher_bundle'])
    profile = ReferenceCondition(paths['profile'])
    torch.set_num_threads(1)
    torch.manual_seed(5)
    sys.argv = [sys.argv[0], '--task=x1_amp', '--num_envs=1', '--seed=5', '--headless']
    cfg, _ = task_registry.get_cfgs('x1_amp')
    cfg.env.capture_replay_state = True
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    configure_nominal_probe(cfg)
    cfg.commands.guidance_evaluation = True
    cfg.commands.ranges.lin_vel_x = [cfg.commands.native_speed]*2
    cfg.commands.ranges.lin_vel_y = cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    args.output.mkdir(parents=True, exist_ok=False)
    env, _ = task_registry.make_env('x1_amp', args=get_args(), env_cfg=cfg)
    rows, reason = [], None
    try:
        names, order, bodies, body_ids, anchor, ref = initialize_reference(env, teacher, gymtorch, torch, quat_rotate_inverse)
        if names != meta['dof_names']:
            raise ValueError('Takeover joint order changed')
        default = env.default_dof_pos[0].cpu().numpy().copy()
        limits = env.hard_dof_limits.cpu().numpy().copy()
        if (env.motor_offsets.abs().max().item() > 1e-7 or cfg.control.action_scale != meta['action_scale']
                or cfg.normalization.clip_actions != meta['action_clip']
                or not np.allclose(default, meta['default_dof_pos'], rtol=0, atol=1e-7)
                or not np.allclose(limits, meta['hard_limits'], rtol=0, atol=1e-7)):
            raise ValueError('Takeover PD contract differs from locked data')
        model.to(env.device).eval()
        before = state_digest(model.state_dict())
        last = np.zeros(len(names), dtype=np.float32)
        previous_target = None
        with torch.no_grad():
            for i in range(profile.frames-1):
                observation = profile.apply(env, i)
                obs = observation[0].cpu().numpy().copy()
                if i == 0 and not np.allclose(obs, demonstration['student_observations'][0], rtol=0, atol=2e-5):
                    raise ValueError('Takeover initial observation changed')
                origin = env.env_origins[0].cpu().numpy()
                pre = env.root_states[0].cpu().numpy().copy()
                pre[:3] -= origin
                q, dq = [value[0].cpu().numpy().copy() for value in (env.dof_pos, env.dof_vel)]
                previous = env.actions[0].cpu().numpy().copy()
                if i <= args.takeover_step and i > 0 and args.history == 'target_equivalent':
                    last = target_equivalent_history(teacher, previous_target, names)
                teacher_input = teacher.observation(i, pre, q, dq, last, names)
                raw_teacher = teacher.action(teacher_input)
                canonical, expected = canonical_action(teacher.target(raw_teacher, names), default,
                    meta['action_scale'], meta['action_clip'], limits)
                student = model.act_inference(observation)
                if not torch.isfinite(student).all().item():
                    raise ValueError('Nonfinite student takeover prefix action')
                is_teacher = i >= args.takeover_step
                sent = torch.as_tensor(canonical[None], device=env.device) if is_teacher else student
                _, _, _, done, info = env.step(sent)
                actual = info['replay_state']
                post = actual['root'][0].cpu().numpy().copy()
                post[:3] -= origin
                post_q = actual['q'][0].cpu().numpy().copy()
                target = actual['effective_target'][0].cpu().numpy().copy()
                if is_teacher and not np.allclose(target, expected, rtol=0, atol=1e-6):
                    raise ValueError('Teacher takeover did not execute its bounded target')
                rows.append(dict(step=i+1, teacher_controls=is_teacher, observation=obs,
                    previous_action=previous, pre_root=pre, pre_q=q, pre_dq=dq,
                    post_root=post, q=post_q, applied_action=actual['action'][0].cpu().numpy().copy(),
                    effective_target=target, teacher_observation=teacher_input, teacher_raw_action=raw_teacher,
                    teacher_canonical_action=canonical, student_raw_action=student[0].cpu().numpy().copy(),
                    joint_rmse=np.sqrt(np.mean((post_q-ref['joint_pos'][i+1, order])**2)),
                    root_error=np.linalg.norm(post[:3]-ref['body_pos_w'][i+1, anchor]), done=bool(done[0])))
                previous_target = target
                # After takeover, preserve the actual raw teacher command history, as in its original policy.
                last = raw_teacher.copy()
                if bool(done[0]):
                    reason = 'production_environment_terminated'
                else:
                    positions = env.rigid_state[0, body_ids, :3].cpu().numpy().copy()-origin
                    reason = termination(post, ref['body_pos_w'][i+1, anchor],
                        ref['body_quat_w'][i+1, anchor], positions, ref['body_pos_w'][i+1], bodies)
                if (i+1) % 50 == 0 or reason:
                    print('TAKEOVER_STEP '+json.dumps(dict(step=i+1, teacher=is_teacher, reason=reason)), flush=True)
                if reason:
                    break
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        np.testing.assert_allclose(arrays['previous_action'][1:], arrays['applied_action'][:-1], rtol=0, atol=1e-6)
        np.testing.assert_allclose(arrays['observation'][1:, :-meta['frame_width']],
            arrays['observation'][:-1, meta['frame_width']:], rtol=0, atol=1e-6)
        unchanged = before == state_digest(model.state_dict())
        full = reason is None and len(rows) == profile.frames-1
        summary = dict(schema='x1_b1_teacher_takeover_v1', seed=5, training_updates=0,
            history_initialization=args.history, takeover_step=args.takeover_step, takeover_time_s=args.takeover_step*env.dt,
            steps=len(rows), seconds=len(rows)*env.dt, completed_reference=full, termination_reason=reason,
            joint_rmse_mean_rad=float(arrays['joint_rmse'].mean()), root_error_max_m=float(arrays['root_error'].max()),
            teacher_control_steps=int(arrays['teacher_controls'].sum()), model_unchanged=unchanged,
            checkpoint_sha256=args.checkpoint_sha256, profile_sha256=profile.sha256,
            actual_student_feedback_verified=True, no_reset_or_state_injection_at_takeover=True,
            limitation='A scheduled teacher takeover test does not establish arbitrary-state recovery or independent generalization.')
        np.savez_compressed(args.output/'trajectory.npz', **arrays, metadata=np.array(json.dumps(summary)))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        print('TAKEOVER_RESULT '+json.dumps(summary), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
