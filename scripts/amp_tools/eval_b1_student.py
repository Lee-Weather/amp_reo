"""Student-only closed-loop B1 replay: teacher supplies initial/reference state only."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Omit for untrained source baseline')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--max-steps', type=int, help='Partial smoke only; cannot pass the full-skill gate')
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.checkpoint_sha256):
        parser.error('Student checkpoint requires its recorded SHA256')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('max-steps must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import isaacgym  # MUST precede torch.
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_rotate_inverse
    import torch
    import humanoid.envs
    from humanoid.utils import get_args, task_registry
    from humanoid.amp.nominal_startup import configure_nominal_probe
    from humanoid.amp.wbt_teacher import WBTTeacher, termination
    from humanoid.amp.reference_condition import ReferenceCondition
    from humanoid.amp.student_distillation import (verify_inputs, build_student, trusted_load,
        freeze_features, frozen_digest, state_digest, SCHEMA, sha256)
    from collect_wbt_student import initialize_reference
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    paths, demonstration, meta, source, _ = verify_inputs(plan, ROOT)
    model = build_student(source)
    freeze_features(model)
    original_frozen = frozen_digest(model)
    checkpoint_sha = sha256(paths['checkpoint'])
    if args.checkpoint:
        checkpoint = trusted_load(args.checkpoint, args.checkpoint_sha256)
        if (checkpoint.get('schema') != SCHEMA or checkpoint.get('plan') != plan
                or checkpoint.get('amp_resume_compatible') is not False
                or checkpoint.get('observation_semantics') != 'b1_finite_reference_progress_v1'
                or not 0 < checkpoint.get('updates', 0) <= plan['budget']['updates']):
            raise ValueError('Student checkpoint does not belong to this pilot')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if frozen_digest(model) != original_frozen:
            raise ValueError('Student checkpoint altered frozen branches')
        checkpoint_sha = args.checkpoint_sha256
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
        if names != meta['dof_names'] or env.num_obs != meta['frame_width']*meta['frame_stack']:
            raise ValueError('Playback environment observation/joint contract changed')
        for actual, expected in ((env.default_dof_pos[0].cpu().numpy(), meta['default_dof_pos']),
                                 (env.hard_dof_limits.cpu().numpy(), meta['hard_limits'])):
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError('Playback PD contract changed')
        if (env.motor_offsets.abs().max().item() > 1e-7
                or cfg.control.action_scale != meta['action_scale']
                or cfg.normalization.clip_actions != meta['action_clip']):
            raise ValueError('Playback action scaling changed')
        model.to(env.device).eval()
        before = state_digest(model.state_dict())
        steps = min(args.max_steps or profile.frames-1, profile.frames-1)
        with torch.no_grad():
            for i in range(steps):
                observation = profile.apply(env, i)
                if i == 0:
                    initial_error = float(np.abs(observation[0].cpu().numpy()-demonstration['student_observations'][0]).max())
                    if initial_error > 2e-5:
                        raise ValueError('Student initial observation differs from the verified demonstration')
                pre = env.root_states[0].cpu().numpy().copy()
                origin = env.env_origins[0].cpu().numpy()
                pre[:3] -= origin
                # The sole action producer is the student; no teacher queries or replayed labels.
                raw_action = model.act_inference(observation)
                if not torch.isfinite(raw_action).all().item():
                    raise ValueError('Student produced nonfinite action')
                obs = observation[0].cpu().numpy().copy()
                previous = env.actions[0].cpu().numpy().copy()
                _, _, _, done, info = env.step(raw_action)
                actual = info['replay_state']
                post = actual['root'][0].cpu().numpy().copy()
                post[:3] -= origin
                q = actual['q'][0].cpu().numpy().copy()
                expected_q = ref['joint_pos'][i+1, order]
                rows.append(dict(step=i+1, observation=obs, previous_action=previous,
                    raw_action=raw_action[0].cpu().numpy().copy(),
                    applied_action=actual['action'][0].cpu().numpy().copy(),
                    effective_target=actual['effective_target'][0].cpu().numpy().copy(),
                    pre_root=pre, post_root=post, q=q,
                    root_error=np.linalg.norm(post[:3]-ref['body_pos_w'][i+1, anchor]),
                    joint_rmse=np.sqrt(np.mean((q-expected_q)**2)), done=bool(done[0])))
                if bool(done[0]):
                    reason = 'production_environment_terminated'
                else:
                    positions = env.rigid_state[0, body_ids, :3].cpu().numpy().copy()-origin
                    reason = termination(post, ref['body_pos_w'][i+1, anchor],
                        ref['body_quat_w'][i+1, anchor], positions, ref['body_pos_w'][i+1], bodies)
                if (i+1) % 50 == 0 or reason:
                    print('STUDENT_STEP '+json.dumps(dict(step=i+1, reason=reason)), flush=True)
                if reason:
                    break
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        if len(rows) > 1 and not np.allclose(arrays['previous_action'][1:], arrays['applied_action'][:-1], rtol=0, atol=1e-6):
            raise ValueError('Playback did not feed student actions back into history')
        width = meta['frame_width']
        n = len(names)
        feedback = arrays['observation'][:, -width+9+2*n:-width+9+3*n]
        if not np.allclose(feedback, arrays['previous_action'], rtol=0, atol=1e-6):
            raise ValueError('Observed previous-action slot differs from executed student actions')
        if len(rows) > 1 and not np.allclose(arrays['observation'][1:, :-width],
                arrays['observation'][:-1, width:], rtol=0, atol=1e-6):
            raise ValueError('Student replay rewrote past history frames')
        if len(rows) > 1 and not np.allclose(arrays['observation'][1:, -width:-width+3],
                profile.data['commands'][1:len(rows)]*np.asarray(meta['command_scale']), rtol=0, atol=1e-6):
            raise ValueError('Playback condition profile drifted')
        unchanged = before == state_digest(model.state_dict())
        full = reason is None and len(rows) == profile.frames-1
        summary = dict(schema='x1_b1_student_replay_v1', checkpoint_sha256=checkpoint_sha,
            untrained_source_baseline=args.checkpoint is None, steps=len(rows), seconds=len(rows)*env.dt,
            completed_reference=full, termination_reason=reason, training_updates=0,
            action_source='student.act_inference only', teacher_action_queries=0,
            student_action_feedback_verified=True, model_unchanged=unchanged,
            initial_observation_max_error=initial_error, history_shift_verified=True,
            joint_rmse_mean_rad=float(arrays['joint_rmse'].mean()),
            root_error_max_m=float(arrays['root_error'].max()),
            horizontal_displacement_m=float(np.linalg.norm(arrays['post_root'][-1, :2]-arrays['pre_root'][0, :2])),
            profile_sha256=profile.sha256, initial_state=meta['initial_state'], seed=5,
            acceptance=dict(completed_384_steps=full,
                joint_rmse_below_012=bool(arrays['joint_rmse'].mean() <= .12),
                root_error_below_035=bool(arrays['root_error'].max() <= .35)),
            limitation='Single fixed-seed nominal finite skill; no perturbation, looping, joystick or terminal handoff proof.')
        summary['passed'] = all(summary['acceptance'].values()) and unchanged
        np.savez_compressed(args.output/'trajectory.npz', **arrays, metadata=np.array(json.dumps(summary)))
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('STUDENT_REPLAY_RESULT '+json.dumps(summary), flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
