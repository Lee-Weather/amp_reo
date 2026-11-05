"""Download and inspect the first exp2.0 checkpoint in memory."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import flux_ops as flux
from private_training_input import request
from humanoid.amp.dh_models import ActorCriticDHAMP
from humanoid.amp.models import Discriminator


def main():
    state = json.loads((ROOT/'czy/exp1/records/exp2.0_training.json').read_text())
    directory = ROOT/'czy/exp1/records/exp2.0_evaluation'
    evidence = json.loads((directory/'first_checkpoint_verification.json').read_text())
    assert evidence['verified'] and evidence['first_iteration']==100
    assert evidence['task_id']==state['task_id'] and evidence['actual_commit']==state['commit']
    origin = evidence['startup_markers']['POLICY_TRANSFER_OK']
    assert origin['source_sha256']==state['source_checkpoint_sha256']
    assert origin['source_iteration']==7000 and origin['new_iteration']==0
    flux.ACCOUNT = state['account']
    who = flux.call('auth','whoami')['data']
    assert who['email']==state['account'] and int(who['userId'])==4685
    rows = flux.call('task','model','list','--task-id',state['task_id'],'--checkpoint','100','--limit','30')['data']['rows']
    row = next(r for r in rows if r['fileName']=='model_100.pt' and str(r['checkpoint'])=='100')
    status,_,raw = request(row['policUrlDown'])
    receipt = evidence['save_receipt']
    assert status==200 and len(raw)==receipt['bytes']
    assert hashlib.sha256(raw).hexdigest()==receipt['sha256']
    model = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    assert model['iter']==100
    origin = model['training_origin']
    assert origin['mode']=='policy_transfer_fresh_critic_amp_optimizers'
    assert origin['source_iteration']==7000 and origin['new_iteration']==0
    assert origin['source_sha256']==state['source_checkpoint_sha256']
    assert origin['retained']==['actor','long_history','state_estimator','log_std']
    assert origin['reset']==['critic','discriminator','normalizer','optimizers','replay']
    source = torch.load(ROOT/state['source_checkpoint'],map_location='cpu',weights_only=False)
    cfg,meta = model['config'],model['metadata']
    assert meta['control']==source['metadata']['control']
    assert meta['dof_names']==source['metadata']['dof_names']
    assert meta['policy_dt']==source['metadata']['policy_dt']
    assert abs(meta['policy_dt']-.02)<1e-8
    assert meta['num_obs']==6468 and meta['policy']['num_single_obs']==98
    assert cfg['env']['num_envs']==4096 and cfg['seed']==5
    assert cfg['runner']['max_iterations']==3000
    assert (cfg['runner']['num_steps_per_env'],cfg['algorithm']['num_learning_epochs'],
            cfg['algorithm']['num_mini_batches'],cfg['discriminator']['num_updates'])==(24,2,4,1)
    assert cfg['discriminator']['conditional_experts']
    assert cfg['amp']['expert_sampler']=='native_cycle_by_replay_category_v1'
    assert not cfg['amp']['reference_state_initialization']
    assert cfg['rewards']['scales']['raw_target_violation']==-1.
    training = meta['training']
    assert training['curriculum']=='native_full_body_v1'
    assert training['conditional_experts'] and training['initialization']=='standing_only'
    assert training['transition_seconds']==2.
    native = training['native_reference']
    assert abs(native['period_s']-1.26)<1e-8
    assert native['time_scale']==native['amplitude_scale']==1.
    assert native['command_speed']==.5953696966
    assert training['command_events']==dict(forward_vx=.5953696966,forward_probability=.8,zero_probability=.2)
    for field,path in [('correction_sha256','exp20_native_correction.json'),
                       ('sole_support_cache_sha256','exp20_sole_hulls.json')]:
        value = json.loads((ROOT/'humanoid/amp/reference_data'/path).read_text())
        assert native[field]==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for name,digest in training['reference_source_sha256'].items():
        assert hashlib.sha256((ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'/(name+'.pkl')).read_bytes()).hexdigest()==digest
    policy = copy.deepcopy(cfg['policy']); motion = policy['motion_exploration']
    motion['indices'] = [meta['dof_names'].index(n) for n in motion.pop('joint_names')]
    motion['command_index'] = 6468-98
    motion['command_threshold'] = motion.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    network = ActorCriticDHAMP(6468,101,29,490,98,**policy)
    network.load_state_dict(model['model_state_dict'],strict=True)
    disc = Discriminator(86,3,cfg['discriminator']['hidden_dims'])
    disc.load_state_dict(model['discriminator_state_dict'],strict=True)
    for values in (network.state_dict(),disc.state_dict()):
        assert all(torch.isfinite(v).all() for v in values.values())
    steps = {}
    for name in ('optimizer_state_dict','disc_optimizer_state_dict'):
        values = model[name]['state']; assert values
        assert all(torch.isfinite(v).all() for entry in values.values() for v in entry.values() if torch.is_tensor(v))
        counts = [float(entry['step']) for entry in values.values()]
        steps[name] = dict(minimum_step=min(counts),maximum_step=max(counts),parameter_states=len(counts))
    assert 100<=steps['optimizer_state_dict']['minimum_step']<=steps['optimizer_state_dict']['maximum_step']<=800
    assert steps['disc_optimizer_state_dict']['minimum_step']==steps['disc_optimizer_state_dict']['maximum_step']==100
    expected = 100*cfg['discriminator']['batch_size']*2*3
    assert abs(float(disc.normalizer.count)-expected)<1
    torch.set_num_threads(2)
    with torch.no_grad():
        inputs = torch.randn(8,6468)*.1
        exported = torch.jit.script(network.get_policy_module().eval())
        action = exported(inputs)
        assert action.shape==(8,29) and torch.isfinite(action).all()
        torch.testing.assert_close(action,network.act_inference(inputs),atol=1e-6,rtol=1e-6)
    result = dict(verified=True,task_id=state['task_id'],commit=state['commit'],iteration=100,
        checked_at=datetime.now(timezone.utc).isoformat(),sha256=receipt['sha256'],bytes=len(raw),
        observation_dim=6468,action_dim=29,actual_envs_seed_rollout_ppo_D=[4096,5,24,2,4,1],
        control_unchanged=True,training_origin=origin,native_reference=native,
        metadata_reference_hashes_verified=True,optimizer_steps=steps,normalizer_count=float(disc.normalizer.count),
        all_parameters_and_optimizer_states_finite=True,complete_policy_jit_matches=True)
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
