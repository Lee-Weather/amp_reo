"""Causal student-observation/teacher-target dataset checks, without a simulator."""
import json
import numpy as np

from .wbt_teacher import rotation_xyzw


def canonical_action(target, default, scale, action_clip, hard_limits):
    """Keep current action/target limits, then represent the effective target."""
    raw = (np.asarray(target)-default)/scale
    clipped = np.clip(raw, -action_clip, action_clip)
    effective = np.clip(default+scale*clipped, hard_limits[:, 0], hard_limits[:, 1])
    return ((effective-default)/scale).astype(np.float32), effective


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_dataset(data):
    meta = json.loads(str(data['metadata']))
    if meta['schema'] != 'x1_teacher_student_dataset_v1':
        raise ValueError('Unknown teacher dataset schema')
    count = len(data['student_observations'])
    width, frames = meta['frame_width'], meta['frame_stack']
    n = len(meta['dof_names'])
    if width != 9+3*n+2 or frames < 2 or count < 1 or len(set(meta['dof_names'])) != n:
        raise ValueError('Student layout or row count is invalid')
    shapes = dict(student_observations=(count, frames*width), next_student_observations=(count, frames*width),
        pre_roots=(count, 13), pre_q=(count, n), pre_dq=(count, n),
        previous_actions=(count, n), applied_actions=(count, n), label_actions=(count, n),
        effective_targets=(count, n), applied_commands=(count, 3), phase_features=(count, 2),
        time_s=(count,), done=(count,))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError('Dataset shape mismatch: '+key)
    for key, value in data.items():
        if key != 'metadata' and not np.isfinite(value).all():
            raise ValueError('Nonfinite dataset array: '+key)
    if not np.isin(data['done'], [0, 1]).all() or data['done'][:-1].any():
        raise ValueError('Dataset crosses a reset boundary')
    if not np.allclose(data['time_s'], np.arange(count)*meta['policy_dt'], rtol=0, atol=1e-7):
        raise ValueError('Dataset clock is discontinuous')
    if 'episode_step' in data and not np.array_equal(data['episode_step'], np.arange(1, count+1)):
        raise ValueError('Production episode counter is discontinuous')
    obs = data['student_observations'].reshape(count, frames, width)
    after = data['next_student_observations'].reshape(count, frames, width)
    latest = obs[:, -1]
    clip = meta['observation_clip']
    default = np.asarray(meta['default_dof_pos'])
    scale = meta['observation_scales']
    angular, gravity = [], []
    for root in data['pre_roots']:
        rotation = rotation_xyzw(root[3:7])
        angular.append(rotation.T@root[10:13])
        gravity.append(rotation.T@np.array([0., 0., -1.]))
    expected = np.concatenate((data['applied_commands']*np.asarray(meta['command_scale']),
        np.asarray(angular)*scale['ang_vel'], np.asarray(gravity),
        (data['pre_q']-default)*scale['dof_pos'], data['pre_dq']*scale['dof_vel'],
        data['previous_actions'], data['phase_features']), axis=-1)
    expected = np.clip(expected, -clip, clip)
    maximum = float(np.abs(expected-latest).max())
    if maximum > 2e-5:
        raise ValueError('Student frame disagrees with pre-action state/command/history: '+str(maximum))
    if not np.allclose(obs[0, :-1], 0, rtol=0, atol=1e-7):
        raise ValueError('Initial history contains unobserved past frames')
    if not np.allclose(data['previous_actions'][0], 0, rtol=0, atol=1e-7):
        raise ValueError('Initial previous action must be zero')
    if count > 1:
        if not np.allclose(data['student_observations'][1:], data['next_student_observations'][:-1], rtol=0, atol=1e-6):
            raise ValueError('Next observation is not the next pre-action input')
        if not np.allclose(data['previous_actions'][1:], data['applied_actions'][:-1], rtol=0, atol=1e-6):
            raise ValueError('Previous action contains current/future labels')
    valid = ~data['done'].astype(bool)
    if not np.allclose(after[valid, :-1], obs[valid, 1:], rtol=0, atol=1e-6):
        raise ValueError('Historical frames were rewritten or shifted incorrectly')
    action_begin = 9+2*n
    if not np.allclose(after[valid, -1, action_begin:action_begin+n],
            np.clip(data['applied_actions'][valid], -clip, clip), rtol=0, atol=1e-6):
        raise ValueError('Next input does not contain the action actually sent')
    labels = (data['effective_targets']-default)/meta['action_scale']
    if not np.allclose(labels, data['label_actions'], rtol=0, atol=1e-6):
        raise ValueError('Label does not encode the executed effective target')
    if not np.allclose(data['applied_actions'], data['label_actions'], rtol=0, atol=2e-6):
        raise ValueError('Behavior history differs from canonical target labels')
    limits = np.asarray(meta['hard_limits'])
    if (np.any(data['effective_targets'] < limits[:, 0]-1e-6) or
            np.any(data['effective_targets'] > limits[:, 1]+1e-6)):
        raise ValueError('Teacher target lies outside current hard limits')
    return dict(rows=count, frame_width=width, frame_stack=frames, student_input_width=width*frames,
        causal_history_verified=True, previous_action_lag_verified=True,
        targets_match_executed_control=True, pre_state_frame_max_error=maximum,
        terminal_rows=int(data['done'].sum()), valid_next_history_rows=int(valid.sum()),
        limits=['Single demonstration cannot establish global input identifiability or generalization.',
            'Command/phase agreement with the expert reference needs a separate audit.',
            'Overlapping history windows are not independent train/validation examples.'])


def audit_conditions(data):
    """Compare reference intent with the actual input; do not rewrite inputs."""
    command = data['applied_commands']
    desired = data['reference_commands']
    error = np.abs(command-desired)
    observed_moving = command[:, 0] > .05
    reference_moving = desired[:, 0] > .05
    moving = observed_moving & reference_moving
    difference = np.abs((data['student_phase']-data['reference_native_phase']+.5) % 1-.5)
    return dict(command_mean_abs_error=error.mean(0).tolist(), command_max_abs_error=error.max(0).tolist(),
        phase_comparison_semantics='internal native clock versus pose-only native phase proxy; not the optional progress-conditioned input',
        command_motion_category_disagreement_fraction=float(np.mean(observed_moving != reference_moving)),
        reference_moving_while_zero_command_rows=int(np.sum(reference_moving & (command[:, 0] <= 1e-7))),
        command_moving_while_reference_quiet_rows=int(np.sum(observed_moving & ~reference_moving)),
        reference_vx_range_m_s=[float(desired[:, 0].min()), float(desired[:, 0].max())],
        moving_native_phase_circular_error_mean_cycles=None if not moving.any() else float(difference[moving].mean()),
        moving_native_phase_circular_error_p95_cycles=None if not moving.any() else float(np.quantile(difference[moving], .95)),
        limits=['Nearest native phase is a pose-only 128-bin diagnostic, not a known B1 phase label.',
                'Different quiet/moving thresholds and reference intent errors do not alone prove contradictory action labels.'])


def audit_exact_label_conflicts(data):
    """Detect exact duplicate inputs with incompatible executed targets."""
    _, inverse, counts = np.unique(data['student_observations'], axis=0, return_inverse=True, return_counts=True)
    conflicts, maximum = 0, 0.
    for group in np.flatnonzero(counts > 1):
        spread = float(np.ptp(data['effective_targets'][inverse == group], axis=0).max())
        maximum = max(maximum, spread)
        conflicts += int(spread > 1e-4)
    return dict(unique_inputs=len(counts), duplicate_input_groups=int(np.sum(counts > 1)),
        conflicting_duplicate_groups=conflicts, duplicate_target_spread_max_rad=maximum,
        limitation='No exact conflict in one trajectory does not establish information sufficiency on new states.')
