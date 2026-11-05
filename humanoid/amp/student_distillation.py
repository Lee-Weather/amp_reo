"""Bounded B1 behavior-cloning pilot, separate from AMP/PPO and its optimizers."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .dh_models import ActorCriticDHAMP
from .reference_condition import ReferenceCondition
from .teacher_dataset import load_dataset, validate_dataset

SCHEMA = 'x1_b1_bc_pilot_v1'
SOURCE_SHA = '5a9b7cfb5a1a007f19eb22ac55e656852245e2b61db3eb0bf9ac9a2c0620a7a2'


def checkpoint_directory(output, requested, source):
    if requested is None:
        return output
    directory, initial = requested.resolve(), source.resolve()
    if (directory != initial.parent or not initial.name.startswith('input_')
            or set(directory.iterdir()) != {initial}):
        raise ValueError('Platform checkpoint directory must contain only the verified input PT')
    return directory


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def trusted_load(path, expected_sha):
    """Only deserialize a content-addressed input, including on old server torch."""
    if len(expected_sha) != 64 or sha256(path) != expected_sha:
        raise ValueError('Checkpoint SHA mismatch; refusing deserialization')
    options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **options)


def build_student(checkpoint):
    cfg = checkpoint['config']
    env = cfg['env']
    settings = dict(cfg['policy'])
    exploration = dict(settings['motion_exploration'])
    names = checkpoint['metadata']['dof_names']
    exploration['indices'] = [names.index(name) for name in exploration.pop('joint_names')]
    exploration['command_index'] = env['num_observations']-env['num_single_obs']
    exploration['command_threshold'] = exploration.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    settings['motion_exploration'] = exploration
    model = ActorCriticDHAMP(env['num_observations'], env['num_privileged_obs'], env['num_actions'],
        num_short_obs=env['short_frame_stack']*env['num_single_obs'],
        num_proprio_obs=env['num_single_obs'], **settings)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def freeze_features(model):
    """First feasibility pilot updates actor MLP only; all other tensors stay exact."""
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('actor.'))
        if parameter.requires_grad:
            names.append(name)
    if not names:
        raise ValueError('Student has no actor parameters')
    return names


def frozen_digest(model):
    return state_digest({name: value for name, value in model.state_dict().items()
                         if not name.startswith('actor.')})


def validate_plan(plan):
    if plan.get('schema') != SCHEMA or plan.get('seed') != 5:
        raise ValueError('Unknown pilot schema or seed')
    budget = plan['budget']
    if (isinstance(budget['updates'], bool) or not isinstance(budget['updates'], int)
            or not 1 <= budget['updates'] <= 200 or budget['batch_size'] != 64
            or budget['learning_rate'] != 1e-4 or budget['max_grad_norm'] != 1.
            or budget['max_wall_seconds'] != 600 or budget['save_interval'] != 50
            or budget['max_raw_target_shift_rms_rad'] != .20):
        raise ValueError('Pilot budget exceeds or changes the reviewed envelope')
    required = {'checkpoint', 'dataset', 'profile', 'teacher_lock', 'teacher_bundle', 'reference'}
    if set(plan['inputs']) != required or plan['inputs']['checkpoint']['sha256'] != SOURCE_SHA:
        raise ValueError('Pilot input contract differs from the reviewed source')
    if plan.get('trainable') != 'actor_only' or plan.get('independent_validation_episodes') != 0:
        raise ValueError('Pilot scope or validation claim is invalid')
    if plan.get('acceptance') != dict(offline_raw_target_rmse_rad_max=.06, offline_rmse_relative_max=.7,
            closed_loop_steps=384, closed_loop_mean_joint_rmse_rad_max=.12, closed_loop_max_root_error_m=.35,
            zero_early_termination=True):
        raise ValueError('Pilot acceptance criteria differ from the reviewed thresholds')


def verify_inputs(plan, root):
    validate_plan(plan)
    paths = {}
    for name, item in plan['inputs'].items():
        path = (root/item['path']).resolve()
        if sha256(path) != item['sha256']:
            raise ValueError('Frozen input SHA mismatch: '+name)
        paths[name] = path
    for relative, expected in plan['source_sha256'].items():
        # Source receipt is platform-independent, unlike binary inputs.
        actual = hashlib.sha256((root/relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected:
            raise ValueError('Source changed after preparation: '+relative)
    data = load_dataset(paths['dataset'])
    integrity = validate_dataset(data)
    meta = json.loads(str(data['metadata']))
    profile = ReferenceCondition(paths['profile'])
    lock = json.loads(paths['teacher_lock'].read_text(encoding='utf-8'))
    if (meta['teacher_lock'] != lock or lock['bundle_sha256'] != sha256(paths['teacher_bundle'])
            or sha256(paths['reference']) != profile.meta['reference_sha256']
            or meta['condition_profile']['sha256'] != profile.sha256
            or meta['phase_clock'] != 'b1_finite_reference_progress_v1'
            or not meta['completed_reference'] or meta['termination_reason'] is not None
            or meta['training_updates'] != 0 or meta['seed'] != 5
            or len(data['time_s']) != profile.frames-1 or data['done'].any()):
        raise ValueError('Dataset is not the complete locked progress-conditioned demonstration')
    count = len(data['time_s'])
    for actual, expected in ((data['applied_commands'], profile.data['commands'][:count]),
                             (data['phase_features'], profile.data['features'][:count]),
                             (data['next_student_observations'][:, -2:], profile.data['features'][1:])):
        if not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError('Dataset/profile online condition mismatch')
    source = trusted_load(paths['checkpoint'], SOURCE_SHA)
    original = source['metadata']
    if (meta['dof_names'] != original['dof_names'] or meta['frame_width'] != original['policy']['num_single_obs']
            or meta['frame_stack'] != original['policy']['frame_stack']
            or abs(meta['policy_dt']-original['policy_dt']) > 1e-7
            or meta['action_scale'] != original['action_scale']
            or meta['action_clip'] != source['config']['normalization']['clip_actions']):
        raise ValueError('Student control or observation contract differs from the data')
    for a, b in ((meta['default_dof_pos'], original['default_dof_pos']),
                 (meta['hard_limits'], original['control']['hard_limits'])):
        if not np.allclose(a, b, rtol=0, atol=1e-7):
            raise ValueError('Student PD contract differs from the data')
    return paths, data, meta, source, integrity


def actor_features(model, observations):
    with torch.no_grad():
        short = observations[:, -model.num_short_obs:]
        encoded = model.long_history(observations.reshape(-1, model.in_channels, model.num_proprio_obs))
        return torch.cat((short, model.state_estimator(short), encoded), dim=-1)


def checked_backward(model, loss, maximum=1.):
    if not torch.isfinite(loss).item():
        raise ValueError('Nonfinite loss; update refused')
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all().item() for parameter in parameters):
        raise ValueError('Missing or nonfinite actor gradient; update refused')
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError('Frozen parameter received a gradient; update refused')
    norm = torch.nn.utils.clip_grad_norm_(parameters, maximum)
    if not torch.isfinite(norm).item():
        raise ValueError('Nonfinite gradient norm; update refused')
    return float(norm)


def fit_metrics(prediction, labels, default, limits, scale, clip, initial):
    raw = default+scale*prediction
    effective = torch.maximum(torch.minimum(default+scale*prediction.clamp(-clip, clip), limits[:, 1]), limits[:, 0])
    expected = default+scale*labels
    return dict(raw_target_rmse_rad=float(((raw-expected).square().mean()).sqrt()),
        effective_target_rmse_rad=float(((effective-expected).square().mean()).sqrt()),
        raw_target_shift_rms_rad=float(((prediction-initial).square().mean()).sqrt()*scale),
        raw_hard_limit_fraction=float(((raw < limits[:, 0]) | (raw > limits[:, 1])).float().mean()),
        action_clip_fraction=float((prediction.abs() > clip).float().mean()))


def save_checkpoint(path, model, optimizer, generator, step, plan, source, frozen, best):
    if path.exists():
        raise FileExistsError(path)
    if frozen_digest(model) != frozen:
        raise ValueError('Frozen student tensors changed')
    metadata = json.loads(json.dumps(source['metadata']))
    metadata['schema_version'] = SCHEMA
    metadata['training'] = dict(method='supervised_actor_only', phase_clock='b1_finite_reference_progress_v1',
                                source_checkpoint_sha256=SOURCE_SHA, amp_resume_compatible=False)
    payload = dict(schema=SCHEMA, model_state_dict=model.state_dict(), config=source['config'],
        metadata=metadata, optimizer_state_dict=optimizer.state_dict(),
        generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        updates=step, plan=plan, frozen_sha256=frozen, best=best,
        observation_semantics='b1_finite_reference_progress_v1', amp_resume_compatible=False)
    # Exclusive creation prevents replacing the source or another run's result.
    with path.open('xb') as stream:
        torch.save(payload, stream)


def restore_checkpoint(checkpoint, model, optimizer, generator, plan, frozen):
    if (checkpoint.get('schema') != SCHEMA or checkpoint.get('plan') != plan
            or checkpoint.get('observation_semantics') != 'b1_finite_reference_progress_v1'
            or checkpoint.get('amp_resume_compatible') is not False
            or checkpoint.get('frozen_sha256') != frozen
            or not 0 < checkpoint.get('updates', 0) <= plan['budget']['updates']):
        raise ValueError('Resume checkpoint contract differs from this pilot')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if frozen_digest(model) != frozen:
        raise ValueError('Resume checkpoint changed frozen tensors')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    group = optimizer.param_groups[0]
    if (len(optimizer.param_groups) != 1 or group['lr'] != plan['budget']['learning_rate']
            or group['weight_decay'] != 0 or group['betas'] != (.9, .999) or group['eps'] != 1e-8):
        raise ValueError('Resume optimizer settings differ from the fixed pilot')
    generator.set_state(checkpoint['generator_state'])
    torch.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state']:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    return checkpoint['updates'], checkpoint['best']
