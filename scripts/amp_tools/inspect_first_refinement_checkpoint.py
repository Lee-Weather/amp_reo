"""Verify exp1.16 first output and retained AMP continuity entirely in memory."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flux_ops as flux
from private_training_input import request
from humanoid.amp.dh_models import ActorCriticDHAMP
from humanoid.amp.models import Discriminator


def main():
    state_path = ROOT/'czy/exp1/records/exp1.16_training.json'
    state = json.loads(state_path.read_text())
    directory = ROOT/'czy/exp1/records/exp1.16_evaluation'
    evidence = json.loads((directory/'first_checkpoint_verification.json').read_text())
    iteration = state['first_checkpoint_iteration']
    assert iteration == 1100 and evidence['verified'] and evidence['refinement_startup_verified']
    assert evidence['task_id'] == state['task_id'] and evidence['commit'] == state['commit']
    flux.ACCOUNT = state['account']
    assert flux.call('auth', 'whoami')['data']['email'] == state['account']
    rows = flux.call('task', 'model', 'list', '--task-id', state['task_id'], '--checkpoint', str(iteration), '--limit', '30')['data']['rows']
    row = next(r for r in rows if r['fileName'] == 'model_1100.pt' and str(r['checkpoint']) == '1100')
    status, _, raw = request(row['policUrlDown']); assert status == 200
    digest = hashlib.sha256(raw).hexdigest(); receipt = evidence['save_receipt']
    assert len(raw) == receipt['bytes'] and digest == receipt['sha256']
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    source_bytes = (ROOT/state['source_checkpoint']).read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == state['source_checkpoint_sha256']
    source = torch.load(io.BytesIO(source_bytes), map_location='cpu', weights_only=False)
    origin = checkpoint['training_origin']; cfg = checkpoint['config']; metadata = checkpoint['metadata']
    assert checkpoint['iter'] == iteration
    assert origin['mode'] == 'reward_refinement_preserve_policy_amp'
    assert origin['source_sha256'] == state['source_checkpoint_sha256']
    assert origin['source_iteration'] == origin['new_iteration'] == 1000
    assert origin['parent_training_origin'] == source['training_origin']
    assert origin['retained_disc_adam_step'] == 4000
    assert 'disc_optimizer' in origin['retained'] and 'policy_optimizer' in origin['reset']
    for field in ('control', 'dof_names', 'policy_dt', 'policy', 'num_obs'):
        assert metadata[field] == source['metadata'][field]
    assert metadata['num_obs'] == 6468
    assert (cfg['env']['num_envs'], cfg['seed'], cfg['runner']['max_iterations']) == (4096, 5, 3000)
    assert (cfg['runner']['num_steps_per_env'], cfg['algorithm']['num_learning_epochs'], cfg['algorithm']['num_mini_batches'], cfg['discriminator']['num_updates']) == (24, 2, 4, 4)
    assert cfg['rewards']['scales']['stand_pose'] == -1.2 and cfg['rewards']['scales']['raw_target_violation'] == -2.
    assert cfg['rewards']['scales']['reference_amplitude'] == cfg['rewards']['scales']['net_displacement'] == 0.
    assert cfg['amp']['fixed_style_weight'] == .5 and cfg['rewards']['reference_initial_weight'] == 1.
    training = metadata['training']
    assert training['curriculum'] == 'dense_phase_refinement_v1'
    assert training['reference_route'] == 'signed_per_coordinate_phase_progress_v1'
    assert training['fixed_amp_style_weight'] == .5
    assert training['command_events'] == {'forward_vx':.3, 'forward_probability':.8, 'zero_probability':.2}
    assert training['reference_source_sha256'] == source['metadata']['training']['reference_source_sha256']
    for name, digest_ref in training['reference_source_sha256'].items():
        assert hashlib.sha256((ROOT/'resources/motions/slow_walk_amp_v1.2/normalized'/(name+'.pkl')).read_bytes()).hexdigest() == digest_ref
    policy = copy.deepcopy(cfg['policy']); motion = policy['motion_exploration']
    motion['indices'] = [metadata['dof_names'].index(n) for n in motion.pop('joint_names')]
    motion['command_index'] = 6468-98
    motion['command_threshold'] = motion.pop('speed_threshold')*cfg['normalization']['obs_scales']['lin_vel']
    model = ActorCriticDHAMP(6468, 101, 29, 490, 98, **policy)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    discriminator = Discriminator(86, 3, cfg['discriminator']['hidden_dims'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
    assert all(torch.isfinite(t).all() for group in (model.state_dict(), discriminator.state_dict()) for t in group.values())
    counts = {}
    for name in ('optimizer_state_dict', 'disc_optimizer_state_dict'):
        states = checkpoint[name]['state']; assert states
        assert all(torch.isfinite(t).all() for v in states.values() for t in v.values() if isinstance(t, torch.Tensor))
        steps = [float(v['step']) for v in states.values()]
        counts[name] = dict(parameter_states=len(states), minimum_step=min(steps), maximum_step=max(steps))
    assert 1 <= counts['optimizer_state_dict']['minimum_step'] <= counts['optimizer_state_dict']['maximum_step'] <= 800
    assert counts['disc_optimizer_state_dict']['minimum_step'] == counts['disc_optimizer_state_dict']['maximum_step'] == 4400
    source_count = float(source['discriminator_state_dict']['normalizer.count'])
    count = float(discriminator.normalizer.count)
    expected_count = source_count+100*4*cfg['discriminator']['batch_size']*2*metadata['history_steps']
    assert abs(count-expected_count) < 1.
    torch.set_num_threads(2)
    with torch.no_grad():
        observations = torch.randn(4, 6468)*.1
        jit = torch.jit.script(model.get_policy_module().eval())
        actions = jit(observations)
        assert actions.shape == (4, 29) and torch.isfinite(actions).all()
        torch.testing.assert_close(actions, model.act_inference(observations), atol=1e-6, rtol=1e-6)
    result = dict(verified=True, task_id=state['task_id'], commit=state['commit'], iteration=iteration,
        bytes=len(raw), sha256=digest, actual_envs_seed_rollout_ppo_D=[4096,5,24,2,4,4],
        training_origin=origin, control_unchanged=True, optimizer_states=counts, normalizer_count=count,
        expected_normalizer_count=expected_count, retained_AMP_continuity_verified=True,
        all_parameters_and_optimizer_states_finite=True, complete_policy_jit_matches=True)
    (directory/'first_checkpoint_inspection.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
