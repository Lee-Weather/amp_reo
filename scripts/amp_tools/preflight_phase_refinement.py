"""CPU validation of exp1.16 transfer, phase reward counterexamples and exports."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from humanoid.amp.runner import AMPOnPolicyRunner
from humanoid.amp.motion import MotionLibrary, URDFKinematics, quat_inverse_rotate
from humanoid.amp.grounding import FootGrounding
from humanoid.amp.phase_reference import PhaseReference
from humanoid.amp.gait_guidance import GaitWindow, MotionScales, normalized_score, span_improvement
from humanoid.amp.gait_rewards import motion_velocity_reward
from humanoid.amp.phase_rewards import phase_quality, phase_progress, phase_velocity_credit, phase_support_credit
from preflight_gait_curriculum import configuration115
from preflight_reference_transfer import dictionary
from test_amp import library, URDF
from test_exp110 import DHFakeEnv
from test_exp3 import definitions


def configuration116():
    ns = configuration115()
    definitions('humanoid/envs/x1/x1_amp_exp116_config.py', ns)
    return ns


def environment(lib, settings, metadata=None):
    env = DHFakeEnv(lib)
    env.dof_names = list(lib.dof_names)
    env.cfg.env = settings.env
    env.cfg.control = settings.control
    env.num_obs, env.num_privileged_obs = 6468, 101
    env.obs, env.critic = torch.zeros(4, 6468), torch.zeros(4, 101)
    original_step = env.step
    def step(actions):
        obs, critic, reward, done, info = original_step(actions)
        # FakeEnv uses -100 terminal rows to test generic GAE. That is not a
        # meaningful DH velocity target. Keep synthetic features/labels in
        # realistic scaled units for this numerical transfer/update check.
        obs.mul_(.05)
        frames = obs.view(4, 66, 98)
        frames[:, :, 0] = .6
        frames[:, :, 8] = -1.
        frames[:, :, 96] = 0.; frames[:, :, 97] = 1.
        critic[:, :98] = frames[:, -1]
        critic[:, 98:] = 0.; critic[:, 98] = .6
        info['terminal_critic_obs'] = critic.clone()
        return obs, critic, reward, done, info
    env.step = step
    if metadata:
        env.dt = metadata['policy_dt']
        env.default_dof_pos = torch.tensor(metadata['default_dof_pos'])[None]
        env.checkpoint_control_metadata = lambda: copy.deepcopy(metadata['control'])
    training = {'curriculum': 'dense_phase_refinement_v1'}
    if metadata:
        training['reference_source_sha256'] = metadata['training']['reference_source_sha256']
    env.checkpoint_training_metadata = lambda: copy.deepcopy(training)
    return env


def reward_audit(lib, cfg, standing):
    class Parent:
        def _reference_weight(self): return 1.
        def _reference_valid(self): return self.commands[:, 0].abs() > .05
        def compute_reward(self):
            self.reference_q, self.reference_dq = self.reference_guide.sample(self.commands[:, 0], self.gait_phase)
            self.reference_feet = self.reference_foot_fk.key_positions(self.reference_q).flatten(-2)
            self.values = {k: (getattr(self, '_reward_'+k)() if getattr(cfg.rewards.scales, k) else torch.zeros(self.num_envs))
                           for k in ('reference_pose', 'reference_velocity', 'reference_feet',
                                     'reference_amplitude', 'reference_support', 'net_displacement')}
            self.extras = {'training_metrics': {}, 'amp_style_multiplier': torch.ones(self.num_envs)}
        def step(self, physical):
            q, dq, root, feet, height = physical
            self.dof_pos.copy_(q); self.dof_vel.copy_(dq); self.root_states.copy_(root)
            self.rigid_state[:, :, :3] = feet+root[:, None, :3]
            self.sole_height = height
            self.contact_forces[:, :, 2] = torch.where(height > .01, 0., 100.)
            self.compute_reward()
            self.episode_length_buf.add_(1)
            return self.values
    ns = dict(torch=torch, X1AMPExp114Env=Parent, quat_inverse_rotate=quat_inverse_rotate,
              GaitWindow=GaitWindow, MotionScales=MotionScales, normalized_score=normalized_score,
              span_improvement=span_improvement, motion_velocity_reward=motion_velocity_reward,
              phase_quality=phase_quality, phase_progress=phase_progress,
              phase_velocity_credit=phase_velocity_credit, phase_support_credit=phase_support_credit)
    definitions('humanoid/envs/x1/x1_amp_exp115_env.py', ns)
    definitions('humanoid/envs/x1/x1_amp_exp116_env.py', ns)
    env = ns['X1AMPExp116Env']()
    cases = ('standing', 'constant_cycle_mean', 'tiny_jitter', 'locked_leg_sliding',
             'wrong_phase', 'fast_reference_4x', 'small_reference_20pct', 'reference_in_place', 'reference_walk')
    env.cfg, env.device, env.dt, env.num_envs = cfg, 'cpu', .02, len(cases)
    env.commands = torch.tensor([[.3, 0., 0.]]*env.num_envs)
    env.episode_length_buf = torch.ones(env.num_envs, dtype=torch.long)
    env.reference_guide = PhaseReference(lib, standing, cfg.amp.reference_cycles)
    env.reference_foot_fk = URDFKinematics(URDF, lib.dof_names, ['left_ankle_roll_link', 'right_ankle_roll_link'])
    ground = FootGrounding(URDF, env.reference_foot_fk, .002)
    env.motion_scales = MotionScales(env.reference_guide, env.reference_foot_fk, env.reference_guide.pitch_indices, [.3])
    env.guidance_window = GaitWindow(env.num_envs, 101, 'cpu')
    env.dof_pos = standing.repeat(env.num_envs, 1); env.dof_vel = torch.zeros_like(env.dof_pos)
    env.root_states = torch.zeros(env.num_envs, 13); env.root_states[:, 6] = 1.
    env.base_quat = env.root_states[:, 3:7]
    env.feet_indices = torch.tensor([0, 1]); env.rigid_state = torch.zeros(env.num_envs, 2, 13)
    env.contact_forces = torch.zeros(env.num_envs, 2, 3)
    env.previous_guidance_q = env.dof_pos.clone(); env.previous_guidance_feet = torch.zeros(env.num_envs, 6)
    env.previous_guidance_command = env.commands[:, 0].clone()
    env.guidance_transition_valid = torch.zeros(env.num_envs, dtype=torch.bool)
    env.rigid_state[:, :, :3] = env.reference_foot_fk.key_positions(env.dof_pos)
    mean = env.reference_guide.sample(torch.full((256,), .3), torch.arange(256)/256)[0].mean(0)
    env.dof_pos[1] = mean
    sums = {k: torch.zeros(env.num_envs) for k in ('reference_pose', 'reference_velocity', 'reference_feet', 'reference_support')}
    early = None
    for step in range(401):
        time = step*env.dt
        env.gait_phase = env.reference_guide.phase_rate(env.commands[:, 0])*time
        q, dq = env.reference_guide.sample(env.commands[:, 0], env.gait_phase)
        q[0] = standing; dq[0] = 0.
        q[1] = mean; dq[1] = 0.
        q[2] = standing+.001*math.sin(time*12); dq[2] = .012*math.cos(time*12)
        q[3] = standing; dq[3] = 0.
        q[4], dq[4] = [v[0] for v in env.reference_guide.sample(torch.tensor([.3]), env.gait_phase[4:5]+.5)]
        q[5], dq[5] = [v[0] for v in env.reference_guide.sample(torch.tensor([.3]), env.gait_phase[5:6]*4.)]
        dq[5] *= 4.
        q[6] = standing+.2*(q[6]-standing); dq[6] *= .2
        root = torch.zeros_like(env.root_states); root[:, 6] = 1.
        root[3, 0] = .3*time; root[8, 0] = .3*time
        feet, orientations = env.reference_foot_fk.key_transforms(q)
        root[:, :3] = ground.align(root[:, :3], root[:, 3:7], q)
        heights = []
        for i, vertices in ground.feet:
            # Root orientation is identity in this kinematic fixture.
            x, y, z, w = orientations[:, i].unbind(-1)
            zrow = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
            heights.append((zrow@vertices.T).min(-1).values+feet[:, i, 2]+root[:, 2])
        env.step((q, dq, root, feet, torch.stack(heights, -1)))
        if step == 5:
            early = sum(float(env.values[k][-1])*getattr(cfg.rewards.scales, k) for k in sums)
            assert not bool(env.guide_ready[-1])
        if step >= 200:
            for key in sums: sums[key] += env.values[key]/201
    result = {label: {k: float(v[i]) for k, v in sums.items()} for i, label in enumerate(cases)}
    for value in result.values():
        value['weighted_guidance'] = sum(v*getattr(cfg.rewards.scales, k) for k, v in value.items())
    for label in ('standing', 'constant_cycle_mean', 'locked_leg_sliding'):
        assert abs(result[label]['weighted_guidance']) < 1e-5, (label, result[label])
    full = result['reference_walk']['weighted_guidance']
    assert full > 0 and early > 0
    assert full > result['small_reference_20pct']['weighted_guidance']
    for label in ('tiny_jitter', 'wrong_phase', 'fast_reference_4x'):
        assert full > result[label]['weighted_guidance']+.1, (label, result[label], full)
    assert abs(result['tiny_jitter']['weighted_guidance']) < .1
    result.update(verified=True, early_reference_credit_before_full_window=early,
                  scope='New guidance only, CPU kinematics; not total return, physics stability or formal gait acceptance')
    return result


def check(source_path):
    torch.set_num_threads(2); torch.manual_seed(5)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    ns = configuration116(); settings = ns['X1AMPExp116Cfg']()
    cfg = dictionary(ns['X1AMPExp116CfgPPO']()); cfg.update(dictionary(settings))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert digest == cfg['runner']['policy_transfer_source_sha256']
    for section in ('control', 'env', 'asset', 'sim', 'noise', 'normalization', 'policy'):
        assert cfg[section] == source['config'][section], section
    for interval in ('ext_force_interval', 'push_interval'):
        cfg['domain_rand'][interval] = math.ceil(cfg['domain_rand'][interval+'_s']/source['metadata']['policy_dt'])
    lib = MotionLibrary(ROOT/'resources/motions/slow_walk_amp_v1.2/normalized', URDF,
                        source['metadata']['dof_names'], settings.amp.key_body_names,
                        dt=source['metadata']['policy_dt'], history_steps=settings.amp.history_steps,
                        weights=settings.amp.motion_weights, limit_tolerance=settings.amp.limit_tolerance,
                        ground_clearance=settings.amp.ground_clearance)
    def make(): return environment(lib, settings, source['metadata'])
    runner = AMPOnPolicyRunner(make(), cfg)
    fresh_critic = copy.deepcopy(runner.actor_critic.critic.state_dict())
    with contextlib.redirect_stdout(io.StringIO()): runner.warm_start(str(source_path), digest)
    assert runner.current_learning_iteration == 1000 and not runner.optimizer.state and runner.replay.size == 0
    for k, v in source['model_state_dict'].items():
        if not k.startswith('critic.'): assert torch.equal(v, runner.actor_critic.state_dict()[k]), k
    for k, v in fresh_critic.items(): assert torch.equal(v, runner.actor_critic.critic.state_dict()[k])
    for k, v in source['discriminator_state_dict'].items(): assert torch.equal(v, runner.discriminator.state_dict()[k])
    assert min(float(v['step']) for v in runner.disc_optimizer.state.values()) == 4000
    for original, retained in zip(source['disc_optimizer_state_dict']['state'].values(), runner.disc_optimizer.state_dict()['state'].values()):
        assert original.keys() == retained.keys()
        for k, v in original.items(): assert torch.equal(v, retained[k])
    smoke_tree = ast.parse((ROOT/'humanoid/scripts/smoke_amp.py').read_text(encoding='utf-8'))
    equal_function = next(node for node in smoke_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'equal_state')
    smoke_main = next(node for node in smoke_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    warm_block = next(node for node in smoke_main.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Attribute) and node.test.attr == 'warm_start')
    smoke_tree.body = [equal_function, warm_block]
    namespace = dict(torch=torch, runner=runner, initial_iteration=1000,
                     args=SimpleNamespace(warm_start=str(source_path)), train_cfg=ns['X1AMPExp116CfgPPO']())
    exec(compile(smoke_tree, str(ROOT/'humanoid/scripts/smoke_amp.py'), 'exec'), namespace)
    for iteration, expected in ((1000, 1e-5), (1099, 1e-5), (1100, 1e-4)):
        runner.current_learning_iteration = iteration
        assert runner._policy_learning_rate_cap() == expected
    runner.current_learning_iteration = 1000
    # Execute the production train() function with a CPU registry fixture:
    # verified co-located PT input, real migration, exactly3000 NEW updates.
    with tempfile.TemporaryDirectory(prefix='phase_refinement_entry_') as temp:
        folder = Path(temp); staged = folder/'input_exp115t_model_1000.pt'
        staged.write_bytes(source_path.read_bytes())
        arguments = SimpleNamespace(task='x1_amp_exp116', checkpoint_dir=str(folder),
            warm_start=str(staged), warm_start_sha256=digest, warm_start_url_b64=None, resume=False)
        bootstrap = AMPOnPolicyRunner(make(), cfg)
        learned = []
        def make_runner(**kwargs):
            bootstrap.warm_start(str(staged), digest)
            return bootstrap, ns['X1AMPExp116CfgPPO'](), None
        def learn(num_learning_iterations, init_at_random_ep_len):
            assert num_learning_iterations == 3000 and not init_at_random_ep_len
            assert bootstrap.current_learning_iteration == 1000
            assert bootstrap.training_origin['mode'] == 'reward_refinement_preserve_policy_amp'
            assert not bootstrap.optimizer.state and bootstrap.disc_optimizer.state
            learned.append(True)
        bootstrap.learn = learn
        tree = ast.parse((ROOT/'humanoid/scripts/train.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train']
        namespace = {'task_registry': SimpleNamespace(make_env=lambda **kwargs:(bootstrap.env, settings), make_alg_runner=make_runner)}
        exec(compile(tree, str(ROOT/'humanoid/scripts/train.py'), 'exec'), namespace)
        with contextlib.redirect_stdout(io.StringIO()): namespace['train'](arguments)
        assert learned == [True]
    source_model = copy.deepcopy(runner.actor_critic)
    source_model.load_state_dict(source['model_state_dict'], strict=True)
    error = 0.
    with torch.no_grad():
        for _ in range(16):
            obs = torch.randn(256, 6468)*.1
            expected = source_model.act_inference(obs)
            actual = runner.actor_critic.act_inference(obs)
            assert torch.isfinite(actual).all()
            error = max(error, float((actual-expected).abs().max()))
    batch = runner.collect_rollout(); values = runner.update_policy(batch); values.update(runner.update_discriminator(batch['amp']))
    assert all(math.isfinite(v) for v in values.values())
    assert min(float(v['step']) for v in runner.disc_optimizer.state.values()) == 4004
    assert float(runner.discriminator.normalizer.count) > float(source['discriminator_state_dict']['normalizer.count'])
    runner.current_learning_iteration = 1001
    with tempfile.TemporaryDirectory(prefix='phase_refinement_cpu_') as temp, contextlib.redirect_stdout(io.StringIO()):
        path = str(Path(temp)/'model_1001.pt'); runner.save(path)
        resumed = AMPOnPolicyRunner(make(), cfg); resumed.load(path)
        assert resumed.current_learning_iteration == 1001 and resumed.training_origin == runner.training_origin
        assert resumed._policy_learning_rate_cap() == 1e-5
        for k, v in runner.actor_critic.state_dict().items(): assert torch.equal(v, resumed.actor_critic.state_dict()[k])
    jit = torch.jit.script(copy.deepcopy(runner.actor_critic.get_policy_module()).eval())
    with torch.no_grad(): jit_error = float((jit(batch['obs'])-runner.actor_critic.act_inference(batch['obs'])).abs().max())
    assert jit_error <= 1e-6
    audit = reward_audit(lib, settings, torch.tensor(source['metadata']['control']['standing_state'][13:42]))
    files = ['humanoid/amp/phase_rewards.py', 'humanoid/amp/reward_transfer.py', 'humanoid/amp/runner.py',
             'humanoid/amp/replay.py', 'humanoid/envs/__init__.py', 'humanoid/envs/x1/x1_amp_exp116_config.py',
             'humanoid/envs/x1/x1_amp_exp116_env.py', 'humanoid/scripts/play_amp.py', 'humanoid/scripts/smoke_amp.py',
             'scripts/amp_tools/preflight_phase_refinement.py', 'tests/test_exp116.py', 'tests/test_playback.py']
    return dict(verified=True, source_sha256=digest, source_iteration=1000, planned_final_iteration=4000,
        inference_samples=4096, inference_max_error=error, complete_jit_max_error=jit_error,
        inference_weights_preserved=True, critic_and_ppo_adam_fresh=True, amp_network_normalizer_adam_preserved=True,
        disc_adam_step_before=4000, disc_adam_step_after=4004, update_metrics=values, reward_audit=audit,
        actual_train_entrypoint_verified=True, actual_smoke_transfer_checks_verified=True, retained_disc_adam_moments_exact=True,
        tested_files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        scope='CPU production checkpoint/network/PPO/AMP/export with synthetic physics; no GPU run or cloud task',
        formal_gait_acceptance=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = check(a.source.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__': main()
