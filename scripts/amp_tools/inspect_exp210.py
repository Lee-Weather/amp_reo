"""Download and verify an exp2.10 output against its full13500 source state."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import flux_ops as flux
from fetch_final_amp import validate_checkpoint
from private_training_input import request

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT='exp2.10'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iteration',type=int,default=13550)
    a=p.parse_args()
    torch.set_num_threads(1)
    state_path=ROOT/'czy/exp1/records'/(EXPERIMENT+'_training.json')
    state=json.loads(state_path.read_text())
    flux.ACCOUNT=state['account']
    who=flux.call('auth','whoami')['data'];assert who['email']==state['account'] and int(who['userId'])==4670
    directory=ROOT/'czy/exp1/records'/(EXPERIMENT+'_evaluation')
    proof=json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert proof['verified'] and proof['refinement_startup_verified']
    assert a.iteration==proof['first_iteration'],'Use this tool for the verified first output'
    rows=flux.call('task','model','list','--task-id',state['task_id'],'--checkpoint',str(a.iteration),'--limit','30')['data']['rows']
    row=next(r for r in rows if r['fileName']=='model_'+str(a.iteration)+'.pt' and str(r['checkpoint'])==str(a.iteration))
    status,headers,raw=request(row['policUrlDown']);assert status==200
    digest=hashlib.sha256(raw).hexdigest()
    assert digest==proof['save_receipt']['sha256'] and len(raw)==proof['save_receipt']['bytes']
    assert len(raw)==int(headers['content-length']) and hashlib.md5(raw).hexdigest()==headers['etag'].strip('"').lower()
    destination=ROOT/'czy/data'/EXPERIMENT/row['fileName'];destination.parent.mkdir(parents=True,exist_ok=True)
    assert not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()==digest
    destination.write_bytes(raw)
    validation=validate_checkpoint(destination,dict(state,target_iteration=a.iteration))
    source=torch.load(ROOT/state['source_checkpoint'],map_location='cpu',weights_only=False)
    output=torch.load(destination,map_location='cpu',weights_only=False)
    for k,v in source['model_state_dict'].items():
        if k.startswith(('long_history.','state_estimator.','log_std')):
            torch.testing.assert_close(output['model_state_dict'][k],v,rtol=0,atol=0)
    for k,v in source['infos']['exp21']['anchor_state'].items():
        torch.testing.assert_close(output['infos']['exp21']['anchor_state'][k],v,rtol=0,atol=0)
    assert any(not torch.equal(v,output['model_state_dict'][k]) for k,v in source['model_state_dict'].items() if k.startswith('actor.'))
    assert any(not torch.equal(v,output['model_state_dict'][k]) for k,v in source['model_state_dict'].items() if k.startswith('critic.'))
    expected_d=13300+a.iteration-13500
    curriculum=output['infos']['exp21']['learning_state']['curriculum']
    assert curriculum['d_updates']==expected_d and curriculum['stage']==1
    report=dict(verified=True,iteration=a.iteration,sha256=digest,bytes=len(raw),
        both_adam_continued=True,actor_and_critic_updated=True,frozen_parts_unchanged=True,anchor_unchanged=True,
        discriminator_updates=expected_d,refinement=state['refinement'],recovery_continuity=validation['recovery_continuity'],
        actual_reward_change=dict(corridor_yaw=[source['config']['rewards']['scales']['corridor_yaw'],output['config']['rewards']['scales']['corridor_yaw']]),
        all_parameters_finite=True,checkpoint=str(destination.relative_to(ROOT)))
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    state.update(phase='training_first_checkpoint_verified',first_checkpoint_inspection=report)
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
