"""Verify the uploaded50-update critic-only checkpoint without retaining PT copies."""
import hashlib
import io
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import flux_ops as flux
from private_training_input import request


def main():
    p=ROOT/'czy/exp1/records/exp2.1_training.json';state=json.loads(p.read_text())
    directory=p.parent/'exp2.1_evaluation'
    evidence=json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert evidence['verified'] and evidence['first_iteration']==50
    assert evidence['task_id']==state['task_id'] and evidence['actual_commit']==state['commit']
    flux.ACCOUNT=state['account']
    rows=flux.call('task','model','list','--task-id',state['task_id'],'--checkpoint','50','--limit','30')['data']['rows']
    row=next(r for r in rows if r['fileName']=='model_50.pt' and str(r['checkpoint'])=='50')
    status,_,raw=request(row['policUrlDown']);receipt=evidence['save_receipt']
    assert status==200 and len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    source=torch.load(ROOT/state['source_checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['iter']==50 and checkpoint['training_origin']['new_iteration']==0
    assert checkpoint['training_origin']['source_sha256']==state['source_checkpoint_sha256']
    assert checkpoint['metadata']['num_obs']==6468
    assert checkpoint['metadata']['control']==source['metadata']['control']
    assert checkpoint['metadata']['training']['curriculum']=='performance_gated_native_v1'
    assert checkpoint['config']['env']['num_envs']==4096 and checkpoint['config']['seed']==5
    assert checkpoint['config']['runner']['max_iterations']==state['target_iteration']
    for name,value in checkpoint['model_state_dict'].items():
        assert torch.isfinite(value).all()
        if not name.startswith('critic.'):assert torch.equal(value,source['model_state_dict'][name]),name
    assert any(not torch.equal(v,source['model_state_dict'][k]) for k,v in checkpoint['model_state_dict'].items() if k.startswith('critic.'))
    assert not checkpoint['disc_optimizer_state_dict']['state']
    assert abs(float(checkpoint['discriminator_state_dict']['normalizer.count'])-1e-4)<1e-8
    for value in checkpoint['discriminator_state_dict'].values():assert torch.isfinite(value).all()
    steps=[float(v['step']) for v in checkpoint['optimizer_state_dict']['state'].values()]
    assert steps and min(steps)==max(steps)==400
    for entry in checkpoint['optimizer_state_dict']['state'].values():
        assert all(torch.isfinite(v).all() for v in entry.values() if torch.is_tensor(v))
    saved=checkpoint['infos']['exp21']
    assert saved['learning_state']['curriculum']['last_iteration']==50
    assert saved['learning_state']['curriculum']['stage']==0
    from humanoid.amp.dh_models import DHPolicy
    from copy import deepcopy
    # Use the complete inference module of the original architecture.
    from inspect_first_reference_checkpoint import ActorCriticDHAMP
    cfg=checkpoint['config'];policy=deepcopy(cfg['policy']);motion=policy['motion_exploration']
    names=checkpoint['metadata']['dof_names']
    motion['indices']=[names.index(n) for n in motion.pop('joint_names')]
    motion['command_index']=6468-98
    motion['command_threshold']=motion.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    model=ActorCriticDHAMP(6468,101,29,490,98,**policy);model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    torch.set_num_threads(2)
    with torch.no_grad():
        x=torch.randn(4,6468)*.1;jit=torch.jit.script(model.get_policy_module().eval())
        torch.testing.assert_close(jit(x),model.act_inference(x))
    result=dict(verified=True,task_id=state['task_id'],commit=state['commit'],iteration=50,
        bytes=len(raw),sha256=receipt['sha256'],inference_weights_identical_to_source=True,
        critic_changed=True,critic_adam_steps=400,discriminator_adam_empty=True,normalizer_unchanged=True,
        curriculum_state_saved=True,observation_dim=6468,action_dim=29,complete_policy_jit_matches=True)
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
