"""Full13500 continuation with an explicit nominal startup training cohort."""
import copy
import hashlib
from pathlib import Path
from .exp211_transfer import SOURCE_SHA,expected_metadata as contact_metadata
from .nominal_startup import training_metadata,COHORT


def expected_metadata(source):
    result=contact_metadata(source)
    result['training']=training_metadata(result['training'])
    return result


def validate_configuration(source,target):
    expected=copy.deepcopy(source)
    expected['domain_rand'].update(randomize_contact_offset=True,contact_offset_range=[.001,.020])
    expected['amp']['nominal_startup_fraction']=.25
    for key in ('experiment_name','run_name','max_iterations'):expected['runner'][key]=target['runner'][key]
    if target!=expected:raise ValueError('Unexpected nominal-startup configuration changes')
    if target['runner']['experiment_name']!='x1_amp_exp212' or target['runner']['max_iterations']!=1000:
        raise ValueError('Expected exp2.12 with1000 additional updates')


def restore_nominal_continuation(runner,path,digest,source_iteration,target_iteration,iterations):
    import torch
    from humanoid.scripts.resume_amp import restore_continuation
    if digest!=SOURCE_SHA or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Nominal startup source SHA mismatch')
    if (source_iteration,target_iteration,iterations)!=(13500,14500,1000):
        raise ValueError('Nominal startup must continue13500 to14500')
    source=torch.load(path,map_location='cpu')
    if runner._metadata()!=expected_metadata(source['metadata']):raise ValueError('Protected metadata changed')
    validate_configuration(source['config'],runner.config)
    original=runner.__dict__.get('_metadata');runner._metadata=lambda:copy.deepcopy(source['metadata'])
    try:receipt=restore_continuation(runner,path,digest,source_iteration,target_iteration,iterations)
    finally:
        if original is None:runner.__dict__.pop('_metadata',None)
        else:runner._metadata=original
    runner.session_start_iteration=source_iteration
    receipt.update(refinement='exp212_nominal_startup_v1',nominal_startup_training=copy.deepcopy(COHORT))
    return receipt
