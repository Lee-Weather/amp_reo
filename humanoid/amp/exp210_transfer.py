"""Strict yaw-only refinement; keep the full policy, both Adam states and anchor."""
import copy
import hashlib
from pathlib import Path

SOURCE_SHA = '23948ebc7492c96754b8eece068be80418356206588e1e7950bc13da8decb9b0'


def expected_metadata(source):
    result = copy.deepcopy(source)
    if (result['training']['curriculum'] != 'command_scaled_phase_low_amp_v1'
            or result['training']['balance_reward_scales'] != [-.15, -.10]):
        raise ValueError('Expected the audited exp2.9 reward metadata')
    result['training'].update(curriculum='yaw_correction_low_amp_v1',balance_reward_scales=[-.15,-.30])
    return result


def validate_configuration(source, target):
    expected = copy.deepcopy(source)
    if expected['rewards']['scales']['corridor_yaw'] != -.10:
        raise ValueError('Expected source yaw penalty -0.10')
    expected['rewards']['scales']['corridor_yaw'] = -.30
    for key in ('experiment_name','run_name','max_iterations'):
        expected['runner'][key] = target['runner'][key]
    if target != expected:
        changed = [key for key in set(target)|set(expected) if target.get(key)!=expected.get(key)]
        raise ValueError('Unexpected yaw refinement configuration changes: '+str(changed))
    if target['runner']['experiment_name'] != 'x1_amp_exp210' or target['runner']['max_iterations'] != 1000:
        raise ValueError('Expected exp2.10 with exactly1000 additional updates')


def restore_yaw_continuation(runner, path, digest, source_iteration, target_iteration, iterations):
    import torch
    from humanoid.scripts.resume_amp import restore_continuation
    if digest != SOURCE_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Yaw refinement source SHA256 mismatch')
    if (source_iteration,target_iteration,iterations)!=(13500,14500,1000):
        raise ValueError('Yaw refinement must continue13500 to14500')
    source = torch.load(path,map_location='cpu')
    if runner._metadata()!=expected_metadata(source['metadata']):
        raise ValueError('Yaw refinement changed protected metadata')
    validate_configuration(source['config'],runner.config)
    # The source/target contract is checked before invoking the exact loader.
    # Only this runner instance temporarily exposes the validated source tag.
    original = runner.__dict__.get('_metadata')
    runner._metadata = lambda: copy.deepcopy(source['metadata'])
    try:
        receipt = restore_continuation(runner,path,digest,source_iteration,target_iteration,iterations)
    finally:
        if original is None:
            runner.__dict__.pop('_metadata',None)
        else:
            runner._metadata = original
    runner.session_start_iteration = source_iteration
    receipt.update(refinement='exp210_yaw_only_v1',reward_change=dict(corridor_yaw=[-.10,-.30]),
                   target_training_metadata=runner._metadata()['training'])
    return receipt
