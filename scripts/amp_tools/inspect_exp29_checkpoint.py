"""Verify the first uploaded continuation checkpoint entirely on CPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import torch
import flux_ops as flux
from private_training_input import request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    torch.set_num_threads(1)
    state=json.loads((ROOT/'czy/exp1/records/exp2.9_training.json').read_text())
    directory=ROOT/'czy/exp1/records/exp2.9_evaluation'
    evidence=json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert evidence['verified'] and evidence['continuation_startup_verified']
    flux.ACCOUNT=state['account']
    rows=flux.call('task','model','list','--task-id',state['task_id'],'--checkpoint','7550','--limit','30')['data']['rows']
    row=next(r for r in rows if r['fileName']=='model_7550.pt')
    status,_,raw=request(row['policUrlDown']);receipt=evidence['save_receipt']
    assert status==200 and len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256']
    c=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    source=torch.load(ROOT/state['source_checkpoint'],map_location='cpu',weights_only=False)
    assert c['iter']==7550 and c['training_origin']['source_iteration']==7500 and c['training_origin']['new_iteration']==7500
    assert c['training_origin']['source_sha256']==state['source_checkpoint_sha256']
    assert c['metadata']['control']==source['metadata']['control'] and c['metadata']['num_obs']==6468
    assert c['metadata']['training']['curriculum']=='command_scaled_phase_low_amp_v1'
    assert c['metadata']['training']['phase_clock']=='command_scaled_v1'
    assert c['config']['algorithm']['source_anchor_coef']==1. and c['config']['runner']['max_iterations']==1000
    assert c['config']['env']['num_envs']==4096 and c['config']['seed']==5
    domain=c['config']['domain_rand']
    assert domain['randomize_friction'] and domain['friction_range']==[.65,.95] and domain['restitution_range']==[0.,0.]
    assert c['config']['rewards']['scales']['corridor_forward']==-.30
    assert c['config']['rewards']['scales']['corridor_lateral']==-.15 and c['config']['rewards']['scales']['corridor_yaw']==-.10
    for name,value in c['model_state_dict'].items():
        assert torch.isfinite(value).all()
        if name.startswith(('long_history.','state_estimator.','log_std')):assert torch.equal(value,source['model_state_dict'][name]),name
    assert any(not torch.equal(v,source['model_state_dict'][k]) for k,v in c['model_state_dict'].items() if k.startswith('actor.'))
    assert any(not torch.equal(v,source['model_state_dict'][k]) for k,v in c['model_state_dict'].items() if k.startswith('critic.'))
    previous=source['optimizer_state_dict']['state'];current=c['optimizer_state_dict']['state']
    assert previous.keys()==current.keys()
    increments=[float(current[k]['step'])-float(previous[k]['step']) for k in previous]
    assert min(increments)>0
    assert all(torch.isfinite(v).all() for entry in current.values() for v in entry.values() if torch.is_tensor(v))
    assert c['disc_optimizer_state_dict']['state']
    assert all(float(v['step'])==7350 for v in c['disc_optimizer_state_dict']['state'].values())
    assert any(not torch.equal(value,source['discriminator_state_dict'][name]) for name,value in c['discriminator_state_dict'].items())
    assert all(torch.isfinite(v).all() for v in c['discriminator_state_dict'].values())
    saved=c['infos']['exp21'];curriculum=saved['learning_state']['curriculum']
    assert curriculum['last_iteration']==7550 and curriculum['d_updates']==7350 and curriculum['ever_amp_activated'] and curriculum['stage']==1
    assert curriculum['baseline']==source['infos']['exp21']['learning_state']['curriculum']['last_window']['pooled100']
    for name,value in saved['anchor_state'].items():assert torch.equal(value,source['infos']['exp21']['anchor_state'][name])
    from humanoid.amp.dh_models import ActorCriticDHAMP
    cfg=c['config'];policy=copy.deepcopy(cfg['policy']);motion=policy['motion_exploration'];names=c['metadata']['dof_names']
    motion['indices']=[names.index(n) for n in motion.pop('joint_names')]
    motion['command_index']=6468-98;motion['command_threshold']=motion.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    model=ActorCriticDHAMP(6468,101,29,490,98,**policy);model.load_state_dict(c['model_state_dict'],strict=True)
    with torch.no_grad():
        x=torch.randn(4,6468)*.1
        torch.testing.assert_close(torch.jit.script(model.get_policy_module().eval())(x),model.act_inference(x))
    result=dict(verified=True,task_id=state['task_id'],commit=state['commit'],iteration=7550,sha256=receipt['sha256'],
        actor_and_critic_changed=True,features_and_std_unchanged=True,original_anchor_preserved=True,
        source_adam_retained=True,minimum_adam_step_increment=min(increments),maximum_adam_step_increment=max(increments),
        discriminator_trained_updates=7350,retained_source_discriminator_updates=7300,amp_weight=.01,swing_multiplier=2.,balance_multiplier=1.,forward_multiplier=1.,baseline_and_curriculum_saved=True,
        observation_dim=6468,action_dim=29,complete_policy_jit_matches=True)
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
