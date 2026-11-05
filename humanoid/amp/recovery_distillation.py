"""Masked, episode-balanced continuation of the bounded B1 actor pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .reference_condition import ReferenceCondition
from .student_distillation import (verify_inputs as verify_base, trusted_load, sha256,
    build_student, freeze_features, frozen_digest)
from .wbt_teacher import rotation_xyzw

SCHEMA = 'x1_b1_recovery_bc_v1'
DATA_SCHEMA = 'x1_b1_recovery_supervision_v1'
WARM_SHA = '1fb31b74ae8b4c4d4293f180e3deafcbddaea145174b7fd62f4e6ced5ae22765'
DATA_SHA = '5ec9d6cc1d37c495110bd200f7d8d7a8c9e3a7887c94801a6b9974a859bb4441'
SEMANTICS = 'b1_finite_reference_progress_v1'


def validate_plan(plan):
    if (plan.get('schema') != SCHEMA or plan.get('experiment') != 'exp2.14'
            or plan.get('seed') != 5 or plan.get('trainable') != 'actor_only'
            or plan.get('independent_validation_episodes') != 0
            or plan.get('sampling') != '32_nominal_32_recovery_without_replacement_per_batch'
            or plan.get('budget') != dict(updates=200, batch_size=64, learning_rate=3e-5,
                max_grad_norm=1., max_wall_seconds=600, save_interval=50,
                max_raw_target_shift_rms_rad=.20)
            or plan.get('acceptance') != dict(nominal_rmse_rad_max=.06,
                recovery_rmse_rad_max=.06, recovery_relative_max=.7,
                closed_loop_steps=384, closed_loop_mean_joint_rmse_rad_max=.12,
                closed_loop_max_root_error_m=.35, zero_early_termination=True)):
        raise ValueError('Recovery plan differs from the fixed round budget/contract')
    if (plan['inputs']['warm_start']['sha256'] != WARM_SHA
            or plan['inputs']['recovery']['sha256'] != DATA_SHA):
        raise ValueError('Recovery input identity differs from the verified candidate')


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_data(data, meta, base_meta, profile):
    """Check context causality even for rows explicitly excluded from supervision."""
    count, width, frames, joints = 768, meta['frame_width'], meta['frame_stack'], len(meta['dof_names'])
    if (meta.get('schema') != DATA_SCHEMA or meta.get('requires_label_mask') is not True
            or meta.get('seed') != 5 or meta.get('training_updates') != 0
            or meta.get('independent_validation_episodes') != 0
            or meta.get('context_rows') != 768 or meta.get('supervised_rows') != 748
            or meta.get('original_supervised_rows') != 384 or meta.get('recovery_supervised_rows') != 364
            or meta['episodes'][1]['case'] != 'step20_target_equivalent'
            or meta['episodes'][1]['student_context_steps'] != 20):
        raise ValueError('Unknown recovery supervision provenance/mask contract')
    for key in ('frame_width', 'frame_stack', 'dof_names', 'policy_dt', 'action_scale',
                'teacher_lock', 'condition_profile', 'observation_scales', 'command_scale', 'observation_clip'):
        if meta[key] != base_meta[key]:
            raise ValueError('Recovery/base contract mismatch: '+key)
    default, limits = np.array(meta['default_dof_pos']), np.array(meta['hard_limits'])
    for key in ('default_dof_pos', 'hard_limits'):
        if not np.allclose(meta[key], base_meta[key], rtol=0, atol=1e-7):
            raise ValueError('Recovery/base PD mismatch: '+key)
    shapes = dict(student_observations=(count, width*frames), pre_roots=(count, 13),
        pre_q=(count, joints), pre_dq=(count, joints), previous_actions=(count, joints),
        applied_actions=(count, joints), effective_targets=(count, joints),
        label_actions=(count, joints), label_mask=(count,), episode_id=(count,), reference_step=(count,))
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError('Invalid recovery array: '+key)
    mask = data['label_mask']
    expected_mask = np.r_[np.ones(384, dtype=bool), np.arange(384) >= 20]
    if mask.dtype != np.bool_ or not np.array_equal(mask, expected_mask):
        raise ValueError('Recovery label mask must exclude exactly the student prefix')
    if (not np.array_equal(data['episode_id'], np.repeat([0, 1], 384))
            or not np.array_equal(data['reference_step'], np.tile(np.arange(384), 2))):
        raise ValueError('Recovery episode/reference clock changed')
    obs = data['student_observations'].reshape(count, frames, width)
    frame_error = 0.
    for episode in (0, 1):
        section = slice(episode*384, (episode+1)*384)
        history = obs[section]
        previous = data['previous_actions'][section]
        if (np.any(history[0, :-1] != 0) or np.any(previous[0] != 0)
                or not np.allclose(history[1:, :-1], history[:-1, 1:], rtol=0, atol=1e-6)
                or not np.allclose(previous[1:], data['applied_actions'][section][:-1], rtol=0, atol=1e-6)):
            raise ValueError('Recovery causal history/previous action crossed an episode boundary')
        roots = data['pre_roots'][section]
        rotations = np.stack([rotation_xyzw(root[3:7]).T for root in roots])
        angular = np.einsum('nij,nj->ni', rotations, roots[:, 10:13])
        gravity = rotations @ np.array([0., 0., -1.])
        scales = meta['observation_scales']
        expected = np.concatenate((profile.data['commands'][:384]*np.array(meta['command_scale']),
            angular*scales['ang_vel'], gravity, (data['pre_q'][section]-default)*scales['dof_pos'],
            data['pre_dq'][section]*scales['dof_vel'], previous, profile.data['features'][:384]), axis=1)
        error = float(np.abs(history[:, -1]-np.clip(expected, -meta['observation_clip'], meta['observation_clip'])).max())
        frame_error = max(frame_error, error)
    if frame_error > 2e-5:
        raise ValueError('Recovery latest frame differs from actual pre-action state')
    clip, scale = base_meta['action_clip'], meta['action_scale']
    expected_targets = np.clip(default+scale*np.clip(data['applied_actions'], -clip, clip), limits[:, 0], limits[:, 1])
    labels = data['label_actions'][mask]
    if (np.abs(data['applied_actions']).max() > clip+1e-6
            or not np.allclose(data['effective_targets'], expected_targets, rtol=0, atol=1e-6)
            or np.abs(labels).max() > clip+1e-6
            or not np.allclose(default+scale*labels, data['effective_targets'][mask], rtol=0, atol=1e-6)
            or not np.allclose(labels, data['applied_actions'][mask], rtol=0, atol=1e-6)):
        raise ValueError('Recovery labels do not encode actually executed bounded targets')
    _, inverse, counts = np.unique(data['student_observations'][mask], axis=0, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        if np.ptp(data['effective_targets'][mask][inverse == group], axis=0).max() > 1e-4:
            raise ValueError('Conflicting labels for identical student inputs')
    return dict(context_rows=count, supervised_rows=int(mask.sum()), masked_rows=int((~mask).sum()),
        causal_history_verified=True, latest_frame_max_error=frame_error, conflicting_duplicate_groups=0)


def verify_inputs(plan, root):
    validate_plan(plan)
    paths = {}
    for name, item in plan['inputs'].items():
        path = (root/item['path']).resolve()
        if root.resolve() not in path.parents or sha256(path) != item['sha256']:
            raise ValueError('Recovery input SHA/path mismatch: '+name)
        paths[name] = path
    for relative, expected in plan['source_sha256'].items():
        if hashlib.sha256((root/relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('Recovery source changed: '+relative)
    paths['base_plan'] = root/plan['base_plan']['path']
    if sha256(paths['base_plan']) != plan['base_plan']['sha256']:
        raise ValueError('Frozen base plan SHA mismatch')
    base_plan = json.loads(paths['base_plan'].read_text(encoding='utf-8'))
    base_paths, demonstration, base_meta, original, _ = verify_base(base_plan, root)
    for name, item in base_plan['inputs'].items():
        if plan['inputs'].get(name) != item:
            raise ValueError('Recovery physical inputs differ from the frozen base')
    warm = trusted_load(paths['warm_start'], WARM_SHA)
    if (warm.get('schema') != 'x1_b1_bc_pilot_v1' or warm.get('updates') != 200
            or warm.get('plan') != base_plan or warm.get('observation_semantics') != SEMANTICS
            or warm.get('amp_resume_compatible') is not False or warm['config'] != original['config']):
        raise ValueError('Warm start is not the verified BC200 policy')
    original_model, warm_model = build_student(original), build_student(warm)
    freeze_features(original_model); freeze_features(warm_model)
    if frozen_digest(original_model) != frozen_digest(warm_model):
        raise ValueError('Warm start changed frozen branches')
    data = load_data(paths['recovery'])
    meta = json.loads(str(data['metadata']))
    integrity = validate_data(data, meta, base_meta, ReferenceCondition(base_paths['profile']))
    # The candidate lacks action_clip; use only the independently verified original control contract.
    meta = dict(meta, action_clip=base_meta['action_clip'], initial_state=base_meta['initial_state'])
    return paths, data, meta, warm, integrity


def sample_balanced(mask, episodes, generator):
    """Indices are selected before touching labels; unlabelled context is never a target."""
    mask, episodes = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(episodes)
    groups = [torch.where(mask & (episodes == episode))[0] for episode in (0, 1)]
    if any(len(group) < 32 for group in groups) or torch.any(mask & ~((episodes == 0) | (episodes == 1))):
        raise ValueError('Each recovery sampling group must have at least32 valid labels')
    return torch.cat([group[torch.randperm(len(group), generator=generator)[:32]] for group in groups])


def masked_loss(predictions, labels, mask, scale):
    valid = torch.as_tensor(mask, device=predictions.device, dtype=torch.bool)
    if not valid.any().item():
        raise ValueError('No supervised labels')
    return (scale*(predictions[valid]-labels[valid])).square().mean()


def validate_checkpoint(checkpoint, plan, frozen):
    if (checkpoint.get('schema') != SCHEMA or checkpoint.get('plan') != plan
            or checkpoint.get('observation_semantics') != SEMANTICS
            or checkpoint.get('amp_resume_compatible') is not False
            or checkpoint.get('warm_start_sha256') != WARM_SHA
            or checkpoint.get('frozen_sha256') != frozen
            or not 0 < checkpoint.get('updates', 0) <= plan['budget']['updates']):
        raise ValueError('Recovery checkpoint identity/plan/frozen contract mismatch')
