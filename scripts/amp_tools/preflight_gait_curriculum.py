"""CPU checks for exp1.15: real checkpoint transfer and production reward counterexamples."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(Path(__file__).resolve().parent)]
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.amp.teacher_transfer import apply_mlp_teacher
from humanoid.amp.gait_guidance import GaitWindow, MotionScales, normalized_score, span_improvement
from humanoid.amp.gait_rewards import motion_velocity_reward
from humanoid.amp.motion import URDFKinematics, quat_inverse_rotate
from humanoid.amp.phase_reference import PhaseReference
from preflight_reference_transfer import dictionary
from reference_task_fixtures import configuration114
from test_amp import library, URDF
from test_exp110 import DHFakeEnv
from test_exp3 import definitions


def configuration115():
    ns = configuration114('114')
    definitions('humanoid/envs/x1/x1_amp_exp115_config.py', ns)
    definitions('humanoid/envs/x1/x1_amp_exp115t_config.py', ns)
    return ns


def audit(lib, settings, standing):
    class Parent:
        def _reference_weight(self): return 1.
        def _reference_valid(self): return self.commands[:, 0].abs() > .05
        def compute_reward(self):
            self.reference_q, self.reference_dq = self.reference_guide.sample(self.commands[:, 0], self.gait_phase)
            self.reference_feet = self.reference_foot_fk.key_positions(self.reference_q).flatten(-2)
            self.values = {k: getattr(self, '_reward_'+k)() for k in
                           ('reference_pose', 'reference_velocity', 'reference_feet',
                            'reference_amplitude', 'reference_support', 'net_displacement')}
            self.extras = {'training_metrics': {}, 'amp_style_multiplier': torch.ones(len(self.commands))}
    ns = dict(torch=torch, X1AMPExp114Env=Parent, quat_inverse_rotate=quat_inverse_rotate,
              GaitWindow=GaitWindow, MotionScales=MotionScales, normalized_score=normalized_score,
              span_improvement=span_improvement, motion_velocity_reward=motion_velocity_reward)
    definitions('humanoid/envs/x1/x1_amp_exp115_env.py', ns)
    env = ns['X1AMPExp115Env']()
    env.cfg, env.device = settings, 'cpu'
    env.commands = torch.tensor([[.3, 0., 0.]]*6)
    env.num_envs, env.dt = 6, .02
    env.reference_guide = PhaseReference(lib, standing, settings.amp.reference_cycles)
    env.reference_foot_fk = URDFKinematics(URDF, lib.dof_names, ['left_ankle_roll_link', 'right_ankle_roll_link'])
    env.reference_foot_columns = list(range(68, 74))
    env.motion_scales = MotionScales(env.reference_guide, env.reference_foot_fk, env.reference_guide.pitch_indices, [.3])
    env.guidance_window = GaitWindow(6, 101, 'cpu')
    env.root_states = torch.zeros(6, 13); env.root_states[:, 6] = 1.
    env.base_quat = env.root_states[:, 3:7]
    env.rigid_state = torch.zeros(6, 2, 13); env.feet_indices = torch.tensor([0, 1])
    env.contact_forces = torch.zeros(6, 2, 3); env.contact_forces[:, :, 2] = 100.
    env.amp_history = torch.zeros(6, 3, 86)
    sums = {k: torch.zeros(6) for k in ('reference_pose', 'reference_velocity', 'reference_feet',
                                      'reference_amplitude', 'reference_support', 'net_displacement')}
    mean_q = env.reference_guide.sample(torch.full((256,), .3), torch.arange(256)/256)[0].mean(0)
    for step in range(401):
        time = step*.02
        env.gait_phase = env.reference_guide.phase_rate(env.commands[:, 0])*time
        q, dq = env.reference_guide.sample(env.commands[:, 0], env.gait_phase)
        q[0] = standing; dq[0] = 0.  # quiet standing
        q[1] = mean_q; dq[1] = 0.    # best constant average-cycle pose
        q[2] = standing+.001*math.sin(time*12); dq[2] = .012*math.cos(time*12)
        q[3] = standing; dq[3] = 0.  # sliding with locked legs
        # Index4: gait in place. Index5: demonstrated gait plus net progression.
        env.dof_pos, env.dof_vel = q, dq
        env.root_states[:, 0] = 0.; env.root_states[3, 0] = .3*time; env.root_states[5, 0] = .3*time
        feet = env.reference_foot_fk.key_positions(q)
        env.rigid_state[:, :, :3] = feet+env.root_states[:, None, :3]
        env.amp_history[:, -1, 68:74] = feet.flatten(-2)
        lift = feet[:, :, 2]-env.motion_scales.tables['feet_min_z'][0]
        env.sole_height = lift.clamp(min=0); env.sole_height[:4] = 0.
        env.contact_forces[:, :, 2] = torch.where(env.sole_height > .015, 0., 100.)
        env.compute_reward()
        if step >= 200:
            for key in sums: sums[key] += env.values[key]/201
    cases = ('standing', 'constant_cycle_mean', 'tiny_jitter', 'locked_leg_sliding', 'gait_in_place', 'reference_walk')
    result = {name: {k: float(v[i]) for k, v in sums.items()} for i, name in enumerate(cases)}
    for case in result.values():
        case['weighted_new_guidance'] = sum(v*getattr(settings.rewards.scales, k) for k, v in case.items())
    assert abs(result['standing']['weighted_new_guidance']) < 1e-6
    assert abs(result['constant_cycle_mean']['weighted_new_guidance']) < 1e-6
    assert abs(result['tiny_jitter']['weighted_new_guidance']) < .1
    assert abs(result['locked_leg_sliding']['reference_pose']) < 1e-6
    assert abs(result['locked_leg_sliding']['reference_amplitude']) < 1e-6
    assert abs(result['gait_in_place']['net_displacement']) < 1e-6
    assert result['reference_walk']['weighted_new_guidance'] > result['gait_in_place']['weighted_new_guidance']+1.9
    assert result['reference_walk']['reference_amplitude'] > .9
    # Actual command sampler; reset history cannot leak a previous gait.
    commands = env._draw_commands(20000)
    zero = float((commands[:, 0] == 0).float().mean())
    assert .18 < zero < .22 and set(commands[:, 0].tolist()) <= {0., float(torch.tensor(.3))}
    env.guidance_window.reset(torch.tensor([5]))
    assert env.guidance_window.count[5] == 0
    result['scope'] = 'CPU kinematic reward counterexamples, not simulated physical rollouts or total return'
    result['zero_command_sampling_fraction'] = zero
    result['verified'] = True
    return result


def check(route, source_path, teacher_path):
    torch.set_num_threads(2); torch.manual_seed(5)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    ns = configuration115()
    settings = ns['X1AMPExp'+route+'Cfg']()
    cfg = dictionary(ns['X1AMPExp'+route+'CfgPPO']())
    cfg.update(dictionary(settings))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert digest == cfg['runner']['policy_transfer_source_sha256'] and source['iter'] == 5300
    for section in ('control', 'asset', 'sim', 'noise', 'normalization'):
        assert dictionary(getattr(settings, section)) == source['config'][section], section
    lib = library(source['metadata']['dof_names'])
    def environment():
        env = DHFakeEnv(lib)
        env.dof_names = list(source['metadata']['dof_names'])
        env.cfg.env.num_single_obs = 98
        env.cfg.control = settings.control
        env.num_obs, env.num_privileged_obs = 6468, 101
        env.obs, env.critic = torch.zeros(4, 6468), torch.zeros(4, 101)
        env.dt = source['metadata']['policy_dt']
        env.default_dof_pos = torch.tensor(source['metadata']['default_dof_pos'])[None]
        env.checkpoint_control_metadata = lambda: copy.deepcopy(source['metadata']['control'])
        env.checkpoint_training_metadata = lambda: {'curriculum': 'normalized_forward_guidance_v1'}
        return env
    target = AMPOnPolicyRunner(environment(), cfg)
    fresh_critic = copy.deepcopy(target.actor_critic.critic.state_dict())
    fresh_disc = copy.deepcopy(target.discriminator.state_dict())
    with contextlib.redirect_stdout(io.StringIO()): target.warm_start(str(source_path), digest)
    baseline = copy.deepcopy(target.actor_critic)
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    if route.endswith('t'):
        with contextlib.redirect_stdout(io.StringIO()): apply_mlp_teacher(target, teacher_path, cfg['runner']['teacher_sha256'])
    assert not target.optimizer.state and not target.disc_optimizer.state and target.replay.size == 0
    for key, value in fresh_critic.items(): assert torch.equal(value, target.actor_critic.critic.state_dict()[key])
    for key, value in fresh_disc.items(): assert torch.equal(value, target.discriminator.state_dict()[key])
    for prefix in ('long_history.', 'state_estimator.', 'log_std'):
        for key, value in baseline.state_dict().items():
            if key.startswith(prefix): assert torch.equal(value, target.actor_critic.state_dict()[key])
    maximum = 0.
    from humanoid.amp.models import mlp
    actor = mlp(480, teacher['config']['policy']['actor_hidden_dims'], 29)
    actor.load_state_dict({k[6:]:v for k,v in teacher['model_state_dict'].items() if k.startswith('actor.')})
    with torch.no_grad():
        for _ in range(16):
            obs = torch.randn(256, 66, 98)*.1
            core = obs[:, -5:, :96].reshape(256, 480)
            actual = target.actor_critic.act_inference(obs.flatten(1))
            expected = actor(core) if route.endswith('t') else baseline.act_inference(obs.flatten(1))
            maximum = max(maximum, float((actual-expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    batch = target.collect_rollout()
    metrics = target.update_policy(batch)
    metrics.update(target.update_discriminator(batch['amp']))
    assert all(math.isfinite(v) for v in metrics.values())
    target.current_learning_iteration = 1
    with tempfile.TemporaryDirectory(prefix='amp-forward-cpu-') as temp, contextlib.redirect_stdout(io.StringIO()):
        path = str(Path(temp)/'model_1.pt'); target.save(path)
        resumed = AMPOnPolicyRunner(environment(), cfg); resumed.load(path)
        for key, value in target.actor_critic.state_dict().items(): assert torch.equal(value, resumed.actor_critic.state_dict()[key])
        assert resumed.training_origin == target.training_origin
    jit = torch.jit.script(copy.deepcopy(target.actor_critic.get_policy_module()).eval())
    with torch.no_grad(): jit_error = float((jit(batch['obs'])-target.actor_critic.act_inference(batch['obs'])).abs().max())
    assert jit_error <= 1e-6
    standing = torch.tensor(source['metadata']['control']['standing_state'][13:42])
    rewards = audit(lib, settings, standing)
    return dict(verified=True, scope='CPU, production checkpoints/networks/PPO/AMP/JIT; synthetic physics',
                route=route, source_iteration=5300, source_sha256=digest,
                teacher_sha256=cfg['runner'].get('teacher_sha256'), inference_max_error=maximum,
                finite_inference_samples=4096, fresh_critic_discriminator_optimizers_replay=True,
                dh_history_estimator_exploration_preserved=True, complete_jit_max_error=jit_error,
                update_metrics=metrics, reward_audit=rewards, formal_gpu_physics_verified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route', choices=('115', '115t'), required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = check(a.route, a.source.resolve(), a.teacher.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
