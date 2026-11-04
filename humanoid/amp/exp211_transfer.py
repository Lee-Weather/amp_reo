"""Strict contact-domain-only full13500 refinement."""
import copy
import hashlib
from pathlib import Path

SOURCE_SHA='23948ebc7492c96754b8eece068be80418356206588e1e7950bc13da8decb9b0'


def expected_metadata(source):
    result=copy.deepcopy(source)
    if result['training']['curriculum']!='command_scaled_phase_low_amp_v1':
        raise ValueError('Expected original exp2.9 calibrated13500 source')
    result['training'].update(curriculum='contact_margin_robustness_v1',
        robustness_scope='contact_offset coverage plus unchanged source friction; original PD and rigid-body properties',
        contact_offset_randomization=dict(range_m=[.001,.020],near_fraction=.5,original_fraction=.25,
            random_fraction=.25,original_m=.01,rest_offset_m=0.,resample='static_per_environment'))
    return result


def validate_configuration(source,target):
    expected=copy.deepcopy(source)
    expected['domain_rand'].update(randomize_contact_offset=True,contact_offset_range=[.001,.020])
    for key in ('experiment_name','run_name','max_iterations'):expected['runner'][key]=target['runner'][key]
    if target!=expected:
        changed=[k for k in set(target)|set(expected) if target.get(k)!=expected.get(k)]
        raise ValueError('Unexpected contact refinement configuration changes: '+str(changed))
    if target['runner']['experiment_name']!='x1_amp_exp211' or target['runner']['max_iterations']!=1000:
        raise ValueError('Expected exp2.11 with1000 additional updates')


def restore_contact_continuation(runner,path,digest,source_iteration,target_iteration,iterations):
    import torch
    from humanoid.scripts.resume_amp import restore_continuation
    if digest!=SOURCE_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Contact refinement source SHA256 mismatch')
    if (source_iteration,target_iteration,iterations)!=(13500,14500,1000):
        raise ValueError('Contact refinement must continue13500 to14500')
    source=torch.load(path,map_location='cpu')
    if runner._metadata()!=expected_metadata(source['metadata']):raise ValueError('Protected metadata changed')
    validate_configuration(source['config'],runner.config)
    original=runner.__dict__.get('_metadata')
    runner._metadata=lambda:copy.deepcopy(source['metadata'])
    try:receipt=restore_continuation(runner,path,digest,source_iteration,target_iteration,iterations)
    finally:
        if original is None:runner.__dict__.pop('_metadata',None)
        else:runner._metadata=original
    runner.session_start_iteration=source_iteration
    receipt.update(refinement='exp211_contact_only_v1',contact_offset_range_m=[.001,.020])
    return receipt
