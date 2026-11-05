"""Verify the exact first uploaded full-resume output without a GPU run."""
import hashlib
import json
from pathlib import Path
import torch

import flux_ops as flux
from fetch_final_amp import final_model, validate_checkpoint
from private_training_input import request

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(1)
    state_path=ROOT/'czy/exp1/records/exp2.9_long_training.json'
    state=json.loads(state_path.read_text())
    directory=ROOT/'czy/exp1/records/exp2.9_long_evaluation'
    proof=json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert proof['verified'] and proof['full_resume_startup_verified'] and proof['task_id']==state['task_id']
    iteration=state['first_checkpoint_iteration']
    flux.ACCOUNT=state['account']
    rows=flux.call('task','model','list','--task-id',state['task_id'],'--checkpoint',str(iteration),'--limit','20')['data']['rows']
    row=final_model(rows,iteration)
    status,_,raw=request(row['policUrlDown']);receipt=proof['save_receipt']
    assert status==200 and len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
    path=ROOT/'czy/data/exp2.9'/row['fileName']
    assert not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    path.write_bytes(raw)
    validation=validate_checkpoint(path,dict(state,target_iteration=iteration))
    actual=torch.load(path,map_location='cpu',weights_only=False)
    source=torch.load(ROOT/state['source_checkpoint'],map_location='cpu',weights_only=False)
    for key,value in source['model_state_dict'].items():
        if key.startswith(('long_history.','state_estimator.','log_std')):
            assert torch.equal(actual['model_state_dict'][key],value),key
    for key,value in source['infos']['exp21']['anchor_state'].items():
        assert torch.equal(actual['infos']['exp21']['anchor_state'][key],value)
    curriculum=actual['infos']['exp21']['learning_state']['curriculum']
    assert curriculum['d_updates']==8400 and curriculum['run_start_iteration']==7500
    result=dict(verified=True,task_id=state['task_id'],iteration=iteration,commit=state['commit'],
        sha256=receipt['sha256'],bytes=len(raw),full_resume_continuity=validation['recovery_continuity'],
        all_parameters_finite=True,frozen_features_std_and_original_anchor_preserved=True,
        discriminator_updates=8400,curriculum_start_preserved=7500,observation_dim=6468,action_dim=29)
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
    current=json.loads(state_path.read_text())
    current.update(phase='running_first_checkpoint_verified',first_checkpoint_content_verification=result)
    state_path.write_text(json.dumps(current,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
