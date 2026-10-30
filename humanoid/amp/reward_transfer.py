"""Explicit reward-change transfer: retain inference and AMP, rebuild value/PPO."""
import copy
import hashlib
import json
import os

import torch

from .dh_models import ActorCriticDHAMP


def transfer_for_reward_refinement(runner, path, expected_sha256=None):
    if (not isinstance(runner.actor_critic, ActorCriticDHAMP)
            or runner.current_learning_iteration != 0 or runner.training_origin is not None
            or runner.optimizer.state or runner.disc_optimizer.state or runner.replay.size):
        raise ValueError('Reward refinement requires a fresh DH runner')
    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    configured = runner.cfg.get('policy_transfer_source_sha256')
    if not configured or digest != configured or (expected_sha256 is not None and digest != expected_sha256):
        raise ValueError('Reward-refinement source SHA256 mismatch')
    state = torch.load(path, map_location=runner.device, weights_only=False)
    source_meta, target_meta = dict(state['metadata']), dict(runner._metadata())
    source_training = source_meta.pop('training', {})
    target_training = target_meta.pop('training', {})
    if source_meta != target_meta:
        raise ValueError('Reward-refinement robot/observation/policy/control conventions do not match')
    if (source_training.get('curriculum') != runner.cfg['policy_transfer_source_curriculum']
            or source_training.get('reference_source_sha256') != target_training.get('reference_source_sha256')
            or state['iter'] != runner.cfg['policy_transfer_source_iteration']
            or state['config']['runner']['experiment_name'] != runner.cfg['policy_transfer_source_task']):
        raise ValueError('Reward-refinement source task/iteration/reference lineage does not match')
    if state['config']['discriminator'] != runner.disc_cfg:
        raise ValueError('Cannot preserve AMP Adam after changing discriminator settings')
    # Expert semantics, noise and observation scales must also remain exact.
    for section in ('policy', 'control', 'env', 'asset', 'sim', 'normalization', 'noise', 'domain_rand'):
        if state['config'][section] != runner.config[section]:
            raise ValueError('Reward refinement changed protected configuration: '+section)
    amp = dict(runner.config['amp'])
    amp.pop('fixed_style_weight', None)
    if amp != state['config']['amp']:
        raise ValueError('Reward refinement changed AMP expert or initialization semantics')
    for name, fresh in (('model_state_dict', runner.actor_critic.state_dict()),
                        ('discriminator_state_dict', runner.discriminator.state_dict())):
        loaded = state[name]
        if loaded.keys() != fresh.keys():
            raise ValueError('Reward-refinement model components differ')
        for key, tensor in loaded.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != fresh[key].shape
                    or tensor.dtype != fresh[key].dtype or not torch.isfinite(tensor).all()):
                raise ValueError('Invalid reward-refinement tensor: '+key)
    if state['discriminator_state_dict']['normalizer.count'] <= 0:
        raise ValueError('Reward refinement requires a trained AMP normalizer')
    # Validate loaded Adam on an isolated network before changing the runner.
    staged_network = copy.deepcopy(runner.discriminator.network)
    staged_adam = torch.optim.Adam(staged_network.parameters(), lr=runner.disc_cfg['learning_rate'],
                                   weight_decay=runner.disc_cfg['weight_decay'])
    staged_adam.load_state_dict(state['disc_optimizer_state_dict'])
    if len(staged_adam.state) != len(list(staged_network.parameters())):
        raise ValueError('AMP Adam state is incomplete')
    for group in staged_adam.param_groups:
        if (group['lr'] != runner.disc_cfg['learning_rate']
                or group['weight_decay'] != runner.disc_cfg['weight_decay']):
            raise ValueError('AMP Adam hyperparameters differ from the retained discriminator configuration')
    steps = []
    for parameter, values in staged_adam.state.items():
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            tensor = values.get(key)
            expected_shape = torch.Size([]) if key == 'step' else parameter.shape
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected_shape
                    or (key != 'step' and tensor.dtype != parameter.dtype) or not torch.isfinite(tensor).all()):
                raise ValueError('Invalid AMP Adam state: '+key)
        steps.append(float(values['step']))
    if min(steps) <= 0 or len(set(steps)) != 1:
        raise ValueError('AMP Adam steps do not describe a trained discriminator')
    fresh = runner.actor_critic.state_dict()
    fresh.update({k: v for k, v in state['model_state_dict'].items() if not k.startswith('critic.')})
    runner.actor_critic.load_state_dict(fresh, strict=True)
    runner.discriminator.load_state_dict(state['discriminator_state_dict'], strict=True)
    runner.disc_optimizer.load_state_dict(staged_adam.state_dict())
    runner.optimizer = torch.optim.Adam(runner.actor_critic.parameters(), lr=runner.alg_cfg['learning_rate'])
    runner.current_learning_iteration = state['iter']
    runner.training_origin = dict(mode='reward_refinement_preserve_policy_amp',
        source_file=os.path.basename(path), source_sha256=digest, source_iteration=state['iter'],
        new_iteration=state['iter'], parent_training_origin=state.get('training_origin'),
        retained=['actor', 'long_history', 'state_estimator', 'log_std', 'discriminator', 'normalizer', 'disc_optimizer'],
        reset=['critic', 'policy_optimizer', 'replay', 'simulator_episode_buffers'],
        retained_disc_adam_step=steps[0], retained_normalizer_count=float(state['discriminator_state_dict']['normalizer.count']))
    print('REWARD_REFINEMENT_OK '+json.dumps(dict(runner.training_origin,
        learning_rate=runner.optimizer.param_groups[0]['lr'])), flush=True)
