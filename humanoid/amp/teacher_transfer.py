"""Exact MLP-to-DH action-function transfer, with robot/PD convention checks."""
import hashlib
import io
import json

import torch


def apply_mlp_teacher(runner, checkpoint, expected_sha256):
    if runner.cfg.get('teacher_sha256') != expected_sha256:
        raise ValueError('Task does not authorize this teacher')
    if runner.current_learning_iteration != 0 or runner.optimizer.state or runner.disc_optimizer.state:
        raise ValueError('Teacher initialization requires fresh optimizer state')
    if (runner.training_origin or {}).get('mode') != 'policy_transfer_fresh_critic_amp_optimizers':
        raise ValueError('Initialize the stable DH source before applying the teacher')
    contents = checkpoint.getvalue() if isinstance(checkpoint, io.BytesIO) else checkpoint.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Teacher checkpoint hash mismatch')
    teacher = torch.load(io.BytesIO(contents), map_location=runner.device)
    target = runner._metadata()
    for key in ('dof_names', 'key_body_names', 'history_steps', 'policy_dt', 'default_dof_pos', 'action_scale', 'control'):
        if teacher['metadata'].get(key) != target.get(key):
            raise ValueError('Teacher convention mismatch: '+key)
    if teacher['metadata'].get('num_obs') != 480 or target.get('num_obs') != 6468:
        raise ValueError('Teacher transfer requires5x96 core input into66x98 DH')
    if (teacher['config']['normalization'] != runner.config['normalization']
            or teacher['config']['control'] != runner.config['control']):
        raise ValueError('Teacher observation scaling/clipping or control changed')
    source, fresh = teacher['model_state_dict'], runner.actor_critic.state_dict()
    result = dict(fresh)
    columns = torch.cat([torch.arange(96, device=runner.device)+i*98 for i in range(5)])
    for key, value in fresh.items():
        if not key.startswith('actor.'):
            continue
        old = source[key]
        if not torch.isfinite(old).all() or old.dtype != value.dtype:
            raise ValueError('Invalid teacher tensor: '+key)
        if key == 'actor.0.weight':
            if old.shape != (512, 480) or value.shape != (512, 557):
                raise ValueError('Unexpected teacher/DH actor shape')
            mapped = torch.zeros_like(value)
            mapped[:, columns] = old
            result[key] = mapped
        else:
            if old.shape != value.shape:
                raise ValueError('Teacher actor topology differs')
            result[key] = old
    # Preserve the stable source's exploration, CNN and estimator. Only the
    # deterministic actor is transplanted; zero columns initially ignore
    # phase/estimated velocity/CNN, with all branches trainable afterwards.
    runner.actor_critic.load_state_dict(result, strict=True)
    runner.training_origin['teacher'] = dict(source_sha256=digest, source_iteration=teacher['iter'],
        method='exact_mlp_5x96_to_dh_5x98_zero_auxiliary_columns_v1',
        retained_dh=['long_history', 'state_estimator', 'log_std'], backward_skill_claimed=False)
    print('TEACHER_TRANSFER_OK '+json.dumps(runner.training_origin['teacher']), flush=True)
