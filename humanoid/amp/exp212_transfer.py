"""Full13500 continuation with an explicit nominal startup training cohort."""
import copy
import hashlib
from pathlib import Path
from .exp211_transfer import SOURCE_SHA,expected_metadata as contact_metadata
from .nominal_startup import training_metadata,COHORT
RECOVERY_SHA='2cce787de44fdfed1fc2894e6ea5daad073452457bad54dc95cddb374a5bc469'


def expected_metadata(source):
    if source['training']['curriculum']=='nominal_startup_coverage_v1':
        result=copy.deepcopy(source)
    else:result=contact_metadata(source)
    result['training']=training_metadata(result['training'])
    return result


def validate_configuration(source,target):
    expected=copy.deepcopy(source)
    expected['domain_rand'].update(randomize_contact_offset=True,contact_offset_range=[.001,.020])
    expected['amp']['nominal_startup_fraction']=.25
    for key in ('experiment_name','run_name','max_iterations'):expected['runner'][key]=target['runner'][key]
    if target!=expected:raise ValueError('Unexpected nominal-startup configuration changes')
    if target['runner']['experiment_name']!='x1_amp_exp212' or target['runner']['max_iterations'] not in (1000,950):
        raise ValueError('Expected exp2.12 original1000 or verified remaining950 updates')


def restore_nominal_continuation(runner,path,digest,source_iteration,target_iteration,iterations):
    import torch
    from humanoid.scripts.resume_amp import restore_continuation
    approved={(13500,14500,1000):SOURCE_SHA,(13550,14500,950):RECOVERY_SHA}
    expected=approved.get((source_iteration,target_iteration,iterations))
    if expected is None or digest!=expected or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('Nominal startup source SHA mismatch')
    source=torch.load(path,map_location='cpu')
    if runner._metadata()!=expected_metadata(source['metadata']):raise ValueError('Protected metadata changed')
    validate_configuration(source['config'],runner.config)
    original=runner.__dict__.get('_metadata');runner._metadata=lambda:copy.deepcopy(source['metadata'])
    try:receipt=restore_continuation(runner,path,digest,source_iteration,target_iteration,iterations)
    finally:
        if original is None:runner.__dict__.pop('_metadata',None)
        else:runner._metadata=original
    runner.session_start_iteration=source_iteration
    receipt.update(refinement='exp212_nominal_reset_fix_v1' if source_iteration==13550 else 'exp212_nominal_startup_v1',
        nominal_startup_training=copy.deepcopy(COHORT),nominal_reset_entrypoint='exp22.reset_idx_hook_v1')
    return receipt
